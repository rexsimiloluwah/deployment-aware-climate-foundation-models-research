from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_harness.config import dataset_config_path, load_yaml, model_config_path
from eval_harness.datasets import build_dataset, stable_seed
from eval_harness.evaluation import segmentation_metrics
from eval_harness.models.foundation import check_foundation_model
from eval_harness.models.prithvi import (
    adapt_batch_to_prithvi,
    apply_prithvi_lora,
    load_prithvi_eo_v2_300m,
    lora_parameter_summary,
    prithvi_feature_map,
)
from eval_harness.models.terramind import (
    adapt_batch_to_terramind,
    apply_terramind_lora,
    load_terramind_backbone,
    terramind_feature_map,
)
from eval_harness.training.local_pilot import collate_segmentation, make_balanced_label_budget_indices


class LinearProbeSegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.head(features)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def run_command(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def choose_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware_summary(device: str) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "selected_device": device,
        "nvidia_smi": run_command(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        )
        if torch.cuda.is_available()
        else "cuda unavailable",
        "terratorch_installed": importlib.util.find_spec("terratorch") is not None,
    }


def clear_accelerator(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def parameter_summary(backbone: nn.Module, head: nn.Module) -> dict[str, int]:
    backbone_total = sum(p.numel() for p in backbone.parameters())
    backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    head_total = sum(p.numel() for p in head.parameters())
    head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    return {
        "backbone_total_params": backbone_total,
        "backbone_trainable_params": backbone_trainable,
        "head_total_params": head_total,
        "head_trainable_params": head_trainable,
        "total_trainable_params": backbone_trainable + head_trainable,
    }


def memory_summary(device: str) -> dict[str, float | None]:
    if device == "cuda":
        return {"peak_accelerator_memory_gb": torch.cuda.max_memory_allocated() / 1024**3}
    if device == "mps" and hasattr(torch, "mps"):
        try:
            return {"peak_accelerator_memory_gb": torch.mps.current_allocated_memory() / 1024**3}
        except Exception:
            return {"peak_accelerator_memory_gb": None}
    return {"peak_accelerator_memory_gb": None}


def load_backbone(
    model_name: str,
    adapter: str,
    experiment: dict[str, Any],
    load_model_weights: bool,
    device: str,
) -> tuple[nn.Module, dict[str, Any]]:
    model_cfg = load_yaml(model_config_path(model_name))
    lora_cfg = experiment.get("adaptation", {}).get("lora", {})
    if model_name == "prithvi_eo_v2_300m":
        loaded = load_prithvi_eo_v2_300m(
            load_weights=load_model_weights,
            device=device,
            freeze=(adapter == "linear_probe"),
        )
        backbone = loaded.model
        metadata = {"checkpoint_path": str(loaded.checkpoint_path) if loaded.checkpoint_path else None}
        if adapter == "lora":
            backbone = apply_prithvi_lora(
                backbone,
                rank=int(lora_cfg.get("rank", 8)),
                alpha=int(lora_cfg.get("alpha", 16)),
                dropout=float(lora_cfg.get("dropout", 0.05)),
            ).to(device)
            metadata.update({f"lora_{k}": v for k, v in lora_parameter_summary(backbone).items()})
        return backbone, metadata

    if model_cfg.get("model_family") == "terramind":
        loaded = load_terramind_backbone(
            model_name,
            pretrained=load_model_weights,
            device=device,
            freeze=(adapter == "linear_probe"),
        )
        backbone = loaded.model
        metadata = {"checkpoint_path": "managed_by_terratorch", "terramind_backbone": loaded.backbone_name}
        if adapter == "lora":
            backbone = apply_terramind_lora(
                backbone,
                rank=int(lora_cfg.get("rank", 8)),
                alpha=int(lora_cfg.get("alpha", 16)),
                dropout=float(lora_cfg.get("dropout", 0.05)),
            ).to(device)
            metadata.update({f"lora_{k}": v for k, v in lora_parameter_summary(backbone).items()})
        return backbone, metadata

    raise ValueError(f"No backbone loader for {model_name}")


def adapt_inputs(model_name: str, dataset_name: str, x: torch.Tensor, device: str, image_size: int):
    if model_name == "prithvi_eo_v2_300m":
        return adapt_batch_to_prithvi(dataset_name, x.to(device), image_size=image_size)
    if model_name.startswith("terramind"):
        return {
            key: value.to(device)
            for key, value in adapt_batch_to_terramind(dataset_name, x.to(device), image_size=image_size).items()
        }
    raise ValueError(model_name)


def extract_features(model_name: str, backbone: nn.Module, adapted_inputs):
    if model_name == "prithvi_eo_v2_300m":
        return prithvi_feature_map(backbone, adapted_inputs)
    if model_name.startswith("terramind"):
        return terramind_feature_map(backbone, adapted_inputs)
    raise ValueError(model_name)


def load_dataset_bundles(dataset_names: list[str]) -> dict[str, dict[str, Any]]:
    bundles = {}
    for dataset_name in dataset_names:
        cfg = load_yaml(dataset_config_path(dataset_name))
        splits = {split: build_dataset(cfg, split=split) for split in ["train", "val", "test"]}
        bundles[dataset_name] = {"cfg": cfg, "splits": splits}
    return bundles


def checkpoint_probe(model_names: list[str], load_model_weights: bool) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        cfg = load_yaml(model_config_path(model_name))
        started = time.perf_counter()
        try:
            result = check_foundation_model(cfg, load_weights=load_model_weights).as_dict()
            rows.append(
                {
                    "model": model_name,
                    "ok": True,
                    "seconds": time.perf_counter() - started,
                    "weight_files": result.get("weight_files"),
                    "checkpoint_load_ok": result.get("checkpoint_load_ok"),
                    "checkpoint_summary": result.get("checkpoint_summary"),
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "ok": False,
                    "seconds": time.perf_counter() - started,
                    "weight_files": None,
                    "checkpoint_load_ok": None,
                    "checkpoint_summary": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


def run_one_cell(
    dataset_bundles: dict[str, dict[str, Any]],
    experiment: dict[str, Any],
    dataset_name: str,
    model_name: str,
    adapter: str,
    budget: float,
    seed: int,
    batch_size: int,
    image_size: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
    load_model_weights: bool,
    run_training_step: bool,
    device: str,
) -> dict[str, Any]:
    clear_accelerator(device)
    cfg = dataset_bundles[dataset_name]["cfg"]
    train_ds = dataset_bundles[dataset_name]["splits"]["train"]
    val_ds = dataset_bundles[dataset_name]["splits"]["val"]
    indices = make_balanced_label_budget_indices(
        train_ds,
        cfg,
        budget,
        stable_seed(f"{dataset_name}:{model_name}:{adapter}:{seed}:{budget}"),
    )
    train_loader = DataLoader(
        Subset(train_ds, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_segmentation(batch, image_size=image_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_segmentation(batch, image_size=image_size),
    )

    load_started = time.perf_counter()
    backbone, model_metadata = load_backbone(model_name, adapter, experiment, load_model_weights, device)
    load_seconds = time.perf_counter() - load_started

    first_batch = next(iter(train_loader))
    with torch.set_grad_enabled(adapter == "lora"):
        first_features = extract_features(
            model_name,
            backbone,
            adapt_inputs(model_name, dataset_name, first_batch["x"], device, image_size),
        )
    head = LinearProbeSegmentationHead(first_features.shape[1], int(cfg["num_classes"])).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad],
        lr=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    losses = []
    train_started = time.perf_counter()
    if run_training_step:
        backbone.train(adapter == "lora")
        head.train()
        for step, batch in enumerate(train_loader):
            if max_train_batches is not None and step >= max_train_batches:
                break
            adapted = adapt_inputs(model_name, dataset_name, batch["x"], device, image_size)
            y = batch["y"].to(device)
            with torch.set_grad_enabled(adapter == "lora"):
                features = extract_features(model_name, backbone, adapted)
            logits = head(features, output_size=y.shape[-2:])
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    train_seconds = time.perf_counter() - train_started

    y_true, y_pred = [], []
    backbone.eval()
    head.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if max_val_batches is not None and step >= max_val_batches:
                break
            adapted = adapt_inputs(model_name, dataset_name, batch["x"], device, image_size)
            features = extract_features(model_name, backbone, adapted)
            logits = head(features, output_size=batch["y"].shape[-2:])
            y_true.append(batch["y"].numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
    metrics = segmentation_metrics(
        np.concatenate(y_true, axis=0),
        np.concatenate(y_pred, axis=0),
        int(cfg["num_classes"]),
        class_names=cfg.get("class_names"),
    )

    row = {
        "dataset": dataset_name,
        "model": model_name,
        "adapter": adapter,
        "label_budget": budget,
        "seed": seed,
        "ok": True,
        "device": device,
        "num_train_examples": len(indices),
        "batch_size": batch_size,
        "foundation_image_size": image_size,
        "feature_shape": tuple(first_features.shape),
        "model_load_seconds": load_seconds,
        "train_seconds": train_seconds,
        "train_batches": len(losses),
        "train_loss_last": losses[-1] if losses else None,
        **parameter_summary(backbone, head),
        **memory_summary(device),
        **model_metadata,
        **metrics,
    }
    del backbone, head, optimizer, first_features
    clear_accelerator(device)
    return row


def parse_csv(value: str | None, cast=str) -> list:
    if value is None:
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def optional_int(value: str) -> int | None:
    return None if value.lower() in {"none", "full", "null"} else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run foundation-model adaptation experiments headlessly.")
    parser.add_argument("--experiment", default="config/experiment.yaml")
    parser.add_argument("--run-label", default="ftw_l4_validation_v1")
    parser.add_argument("--datasets", default="ftw_africa")
    parser.add_argument("--models", default="prithvi_eo_v2_300m,terramind_v1_base,terramind_v1_large")
    parser.add_argument("--adapters", default="linear_probe,lora")
    parser.add_argument("--label-budgets", default="0.10,0.25")
    parser.add_argument("--seeds", default="1234")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-train-batches", type=optional_int, default=32)
    parser.add_argument("--max-val-batches", type=optional_int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--no-model-weights", action="store_true")
    parser.add_argument("--skip-training-step", action="store_true")
    parser.add_argument("--skip-checkpoint-probe", action="store_true")
    parser.add_argument("--raise-on-first-failure", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str((ROOT / ".matplotlib-cache").resolve()))
    os.chdir(ROOT)

    experiment = load_yaml(ROOT / args.experiment)
    dataset_names = parse_csv(args.datasets)
    model_names = parse_csv(args.models)
    adaptation_methods = parse_csv(args.adapters)
    label_budgets = parse_csv(args.label_budgets, float)
    seeds = parse_csv(args.seeds, int)
    device = choose_device(args.device)
    load_model_weights = not args.no_model_weights
    run_training_step = not args.skip_training_step

    artifact_dir = ROOT / "artifacts" / args.run_label
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_path = artifact_dir / "results.jsonl"
    readiness_path = artifact_dir / "readiness.csv"
    summary_path = artifact_dir / "summary.json"
    checkpoint_path = artifact_dir / "checkpoint_probe.csv"

    print(f"RUN_LABEL={args.run_label}")
    print(f"DEVICE={device}")
    print(f"RESULTS={results_path}")

    hardware = hardware_summary(device)
    print(json.dumps(hardware, indent=2, default=str))

    dataset_bundles = load_dataset_bundles(dataset_names)
    checkpoint_df = pd.DataFrame()
    if not args.skip_checkpoint_probe:
        checkpoint_df = checkpoint_probe(model_names, load_model_weights)
        checkpoint_df.to_csv(checkpoint_path, index=False)
        print(checkpoint_df.to_string(index=False))

    results: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        for model_name in model_names:
            for adapter in adaptation_methods:
                for budget in label_budgets:
                    for seed in seeds:
                        print(f"RUN {dataset_name} | {model_name} | {adapter} | budget={budget} | seed={seed}")
                        started = time.perf_counter()
                        try:
                            row = run_one_cell(
                                dataset_bundles,
                                experiment,
                                dataset_name,
                                model_name,
                                adapter,
                                budget,
                                seed,
                                args.batch_size,
                                args.image_size,
                                args.max_train_batches,
                                args.max_val_batches,
                                load_model_weights,
                                run_training_step,
                                device,
                            )
                            row["wall_seconds"] = time.perf_counter() - started
                        except Exception as exc:
                            row = {
                                "dataset": dataset_name,
                                "model": model_name,
                                "adapter": adapter,
                                "label_budget": budget,
                                "seed": seed,
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "wall_seconds": time.perf_counter() - started,
                            }
                            print("FAILED:", row["error"])
                            clear_accelerator(device)
                            if args.raise_on_first_failure:
                                raise
                        results.append(row)
                        with results_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, default=str) + "\n")
                        print(json.dumps(row, indent=2, default=str))

    results_df = pd.DataFrame(results)
    readiness = (
        results_df.assign(status=lambda df: np.where(df["ok"], "runnable", "failed"))
        .groupby(["dataset", "model", "adapter", "status"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    readiness.to_csv(readiness_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "hardware": hardware,
                "experiment": experiment,
                "run_label": args.run_label,
                "dataset_names": dataset_names,
                "model_names": model_names,
                "adaptation_methods": adaptation_methods,
                "label_budgets": label_budgets,
                "seeds": seeds,
                "max_train_batches": args.max_train_batches,
                "max_val_batches": args.max_val_batches,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "checkpoint_rows": checkpoint_df.to_dict("records"),
                "results_path": str(results_path),
                "readiness_path": str(readiness_path),
            },
            handle,
            indent=2,
            default=str,
        )
    print(readiness.to_string(index=False))
    print(f"Saved artifacts to {artifact_dir}")


if __name__ == "__main__":
    main()

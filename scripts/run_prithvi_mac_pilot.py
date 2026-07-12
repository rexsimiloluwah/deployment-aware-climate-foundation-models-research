from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from eval_harness.config import dataset_config_path, load_yaml
from eval_harness.datasets import build_dataset, stable_seed
from eval_harness.evaluation import segmentation_metrics
from eval_harness.models.prithvi import (
    adapt_ftw_batch_to_prithvi,
    load_prithvi_eo_v2_300m,
    prithvi_feature_map,
)
from eval_harness.training.local_pilot import collate_segmentation, make_balanced_label_budget_indices


def choose_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PrithviSegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.head(features)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def loader(dataset, indices, batch_size: int, image_size: int, shuffle: bool) -> DataLoader:
    ds = Subset(dataset, list(indices)) if indices is not None else dataset
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_segmentation(batch, image_size=image_size),
    )


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    dataset_cfg = load_yaml(dataset_config_path(args.dataset))
    train_ds = build_dataset(dataset_cfg, split="train")
    val_ds = build_dataset(dataset_cfg, split="val")
    indices = make_balanced_label_budget_indices(
        train_ds,
        dataset_cfg,
        args.label_budget,
        stable_seed(f"{args.dataset}:{args.seed}:{args.label_budget}:prithvi"),
    )

    train_loader = loader(train_ds, indices, args.batch_size, args.image_size, shuffle=True)
    val_loader = loader(val_ds, None, args.batch_size, args.image_size, shuffle=False)

    started_load = time.perf_counter()
    load_result = load_prithvi_eo_v2_300m(
        load_weights=not args.no_weights,
        cache_dir=args.model_cache,
        device=device,
        freeze=True,
    )
    model_load_seconds = time.perf_counter() - started_load
    backbone = load_result.model

    batch = next(iter(train_loader))
    x_prithvi = adapt_ftw_batch_to_prithvi(batch["x"].to(device), image_size=args.prithvi_image_size)

    started_forward = time.perf_counter()
    with torch.no_grad():
        features = prithvi_feature_map(backbone, x_prithvi)
    first_forward_seconds = time.perf_counter() - started_forward

    head = PrithviSegmentationHead(features.shape[1], int(dataset_cfg["num_classes"])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    train_seconds = 0.0
    if not args.forward_only:
        started_train = time.perf_counter()
        head.train()
        for _epoch in range(args.max_epochs):
            for step, train_batch in enumerate(train_loader):
                if args.limit_train_batches is not None and step >= args.limit_train_batches:
                    break
                x = adapt_ftw_batch_to_prithvi(train_batch["x"].to(device), image_size=args.prithvi_image_size)
                y = (
                    F.interpolate(
                        train_batch["y"][:, None].float(),
                        size=(args.prithvi_image_size, args.prithvi_image_size),
                        mode="nearest",
                    )
                    .squeeze(1)
                    .long()
                    .to(device)
                )
                with torch.no_grad():
                    feature_map = prithvi_feature_map(backbone, x)
                logits = head(feature_map, output_size=(args.prithvi_image_size, args.prithvi_image_size))
                loss = criterion(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
        train_seconds = time.perf_counter() - started_train

    eval_metrics = {}
    if not args.skip_eval:
        y_true, y_pred = [], []
        head.eval()
        with torch.no_grad():
            for step, val_batch in enumerate(val_loader):
                if args.limit_val_batches is not None and step >= args.limit_val_batches:
                    break
                x = adapt_ftw_batch_to_prithvi(val_batch["x"].to(device), image_size=args.prithvi_image_size)
                feature_map = prithvi_feature_map(backbone, x)
                logits = head(feature_map, output_size=(args.prithvi_image_size, args.prithvi_image_size))
                preds = logits.argmax(dim=1).cpu().numpy()
                y = (
                    F.interpolate(
                        val_batch["y"][:, None].float(),
                        size=(args.prithvi_image_size, args.prithvi_image_size),
                        mode="nearest",
                    )
                    .squeeze(1)
                    .long()
                    .numpy()
                )
                y_pred.append(preds)
                y_true.append(y)
        if y_true:
            eval_metrics = segmentation_metrics(
                np.concatenate(y_true, axis=0),
                np.concatenate(y_pred, axis=0),
                int(dataset_cfg["num_classes"]),
                class_names=dataset_cfg.get("class_names"),
            )

    result = {
        "dataset": args.dataset,
        "model": "prithvi_eo_v2_300m",
        "adapter": "linear_probe_head",
        "label_budget": args.label_budget,
        "seed": args.seed,
        "device": device,
        "torch_version": torch.__version__,
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "num_train_examples": len(indices),
        "batch_size": args.batch_size,
        "source_image_size": args.image_size,
        "prithvi_image_size": args.prithvi_image_size,
        "model_load_seconds": model_load_seconds,
        "first_forward_seconds": first_forward_seconds,
        "train_seconds": train_seconds,
        "train_loss_last": train_losses[-1] if train_losses else None,
        "train_loss_mean": float(np.mean(train_losses)) if train_losses else None,
        "input_shape": list(x_prithvi.shape),
        "feature_shape": list(features.shape),
        "head_trainable_params": sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad),
        "checkpoint_path": str(load_result.checkpoint_path) if load_result.checkpoint_path else None,
        "missing_keys": load_result.missing_keys[:20],
        "unexpected_keys": load_result.unexpected_keys[:20],
        "notes": (
            "FTW-to-Prithvi adapter maps PlanetScope B/G/R/NIR windows onto Prithvi HLS bands; "
            "this is a Mac pipeline smoke test, not a final modality-matched experiment."
        ),
        **eval_metrics,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Mac-local Prithvi + FTW foundation-model pilot.")
    parser.add_argument("--dataset", default="ftw_africa")
    parser.add_argument("--label-budget", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--prithvi-image-size", type=int, default=224)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--limit-train-batches", type=int, default=1)
    parser.add_argument("--limit-val-batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-cache", default="data/hf_cache/models")
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--output", default="artifacts/prithvi_mac_pilot/results.jsonl")
    args = parser.parse_args()

    result = run(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

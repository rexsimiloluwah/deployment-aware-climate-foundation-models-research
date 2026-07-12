from __future__ import annotations

import json
import tarfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F

from eval_harness.adapters import adapter_metadata, apply_adaptation_strategy, count_parameters, memory_summary
from eval_harness.config import dataset_config_path, load_yaml
from eval_harness.datasets import build_dataset, make_label_budget_indices, stable_seed
from eval_harness.datasets.registry import _read_tif_from_tar
from eval_harness.evaluation import segmentation_metrics


class TinySegmentationCNN(nn.Module):
    """Small local model for CPU/MPS end-to-end harness validation."""

    def __init__(self, in_channels: int, num_classes: int, width: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(width, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def choose_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collate_segmentation(batch: list[dict], image_size: int) -> dict:
    x = torch.tensor(np.stack([item["x"] for item in batch]), dtype=torch.float32)
    y = torch.tensor(np.stack([item["y"] for item in batch]), dtype=torch.long)
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        y = (
            F.interpolate(y[:, None].float(), size=(image_size, image_size), mode="nearest")
            .squeeze(1)
            .long()
        )
    return {
        "id": [item["id"] for item in batch],
        "country": [item.get("country", item["id"].split("/")[0]) for item in batch],
        "x": x,
        "y": y,
    }


def dataset_patch_stats(dataset, dataset_cfg: dict) -> list[dict]:
    cache_path = (
        Path("data")
        / "cache"
        / "patch_stats"
        / f"{dataset_cfg['name']}_{getattr(dataset, 'split', 'split')}.csv"
    )
    if cache_path.exists():
        return pd_read_records(cache_path)

    stats = []
    class_names = dataset_cfg.get("class_names", [])
    if hasattr(dataset, "items") and dataset.items and "members" in dataset.items[0]:
        by_tar: dict[Path, list[tuple[int, dict]]] = {}
        for idx, item in enumerate(dataset.items):
            by_tar.setdefault(Path(item["tar_path"]), []).append((idx, item))
        for tar_path, indexed_items in by_tar.items():
            with tarfile.open(tar_path) as tar:
                for idx, item in indexed_items:
                    y = _read_tif_from_tar(tar, item["members"]["label"])
                    if y.ndim == 3:
                        y = y[0]
                    stats.append(_mask_stats_row(idx, f"{item['country']}/{item['id']}", item["country"], y, class_names))
    else:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            stats.append(
                _mask_stats_row(
                    idx,
                    sample["id"],
                    sample.get("country", sample["id"].split("/")[0]),
                    sample["y"],
                    class_names,
                )
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd_write_records(stats, cache_path)
    return stats


def _mask_stats_row(idx: int, sample_id: str, country: str, y: np.ndarray, class_names: list[str]) -> dict:
    total = int(y.size)
    row = {
        "idx": idx,
        "id": sample_id,
        "country": country,
        "has_foreground": bool(np.any(y > 0)),
    }
    for label, class_name in enumerate(class_names):
        row[f"{class_name}_fraction"] = float(np.sum(y == label) / total)
    return row


def pd_read_records(path: Path) -> list[dict]:
    import pandas as pd

    return pd.read_csv(path).to_dict("records")


def pd_write_records(records: list[dict], path: Path) -> None:
    import pandas as pd

    pd.DataFrame(records).to_csv(path, index=False)


def make_balanced_label_budget_indices(dataset, dataset_cfg: dict, budget: float, seed: int) -> list[int]:
    n = len(dataset)
    target_n = max(1, int(round(n * budget)))
    if dataset_cfg["name"] != "ftw_africa":
        return make_label_budget_indices(n, budget, seed)

    stats = dataset_patch_stats(dataset, dataset_cfg)
    rng = np.random.default_rng(seed)
    selected: list[int] = []

    countries = sorted({row["country"] for row in stats})
    for country in countries:
        country_rows = [row for row in stats if row["country"] == country]
        rich_rows = [
            row
            for row in country_rows
            if row.get("field_interior_fraction", 0.0) > 0.01 or row.get("field_boundary_fraction", 0.0) > 0.001
        ]
        source = rich_rows or country_rows
        selected.append(int(rng.choice([row["idx"] for row in source])))

    remaining_n = max(0, target_n - len(set(selected)))
    if remaining_n:
        weights = []
        candidates = []
        selected_set = set(selected)
        for row in stats:
            if row["idx"] in selected_set:
                continue
            candidates.append(row["idx"])
            weights.append(
                1.0
                + 5.0 * row.get("field_interior_fraction", 0.0)
                + 20.0 * row.get("field_boundary_fraction", 0.0)
            )
        probs = np.asarray(weights, dtype="float64")
        probs = probs / probs.sum()
        selected.extend(rng.choice(candidates, size=min(remaining_n, len(candidates)), replace=False, p=probs).tolist())

    return sorted(set(int(idx) for idx in selected))[:target_n]


def _loader(dataset, indices: Iterable[int] | None, batch_size: int, image_size: int, shuffle: bool) -> DataLoader:
    ds = Subset(dataset, list(indices)) if indices is not None else dataset
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_segmentation(batch, image_size=image_size),
    )


def evaluate_segmentation(model: nn.Module, loader: DataLoader, dataset_cfg: dict, device: str) -> dict[str, float]:
    y_true, y_pred = [], []
    countries, ids = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.append(preds)
            y_true.append(batch["y"].numpy())
            countries.extend(batch["country"])
            ids.extend(batch["id"])

    y_true_np = np.concatenate(y_true, axis=0)
    y_pred_np = np.concatenate(y_pred, axis=0)
    metrics = segmentation_metrics(
        y_true_np,
        y_pred_np,
        int(dataset_cfg["num_classes"]),
        class_names=dataset_cfg.get("class_names"),
    )

    for country in sorted(set(countries)):
        mask = np.asarray([c == country for c in countries])
        country_metrics = segmentation_metrics(
            y_true_np[mask],
            y_pred_np[mask],
            int(dataset_cfg["num_classes"]),
            class_names=dataset_cfg.get("class_names"),
        )
        for key, value in country_metrics.items():
            if key.startswith("iou") or key in {"mean_iou", "foreground_iou", "pixel_accuracy"}:
                metrics[f"{country}_{key}"] = value
    return metrics


def run_local_segmentation_experiment(
    dataset_name: str,
    method: str,
    label_budget: float,
    seed: int,
    batch_size: int = 2,
    image_size: int = 128,
    max_epochs: int = 1,
    limit_train_batches: int | None = 8,
    limit_val_batches: int | None = 8,
    device: str = "auto",
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = choose_device(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dataset_cfg = load_yaml(dataset_config_path(dataset_name))
    train_ds = build_dataset(dataset_cfg, split="train")
    val_ds = build_dataset(dataset_cfg, split="val")
    indices = make_balanced_label_budget_indices(train_ds, dataset_cfg, label_budget, stable_seed(f"{dataset_name}:{seed}:{label_budget}"))

    train_loader = _loader(train_ds, indices, batch_size=batch_size, image_size=image_size, shuffle=True)
    val_loader = _loader(val_ds, None, batch_size=batch_size, image_size=image_size, shuffle=False)

    model = TinySegmentationCNN(
        in_channels=int(dataset_cfg["input"]["shape"][0]),
        num_classes=int(dataset_cfg["num_classes"]),
        width=16,
    )
    model = apply_adaptation_strategy(model, method).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    started = time.perf_counter()
    losses = []
    model.train()
    for _epoch in range(max_epochs):
        for step, batch in enumerate(train_loader):
            if limit_train_batches is not None and step >= limit_train_batches:
                break
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - started

    if limit_val_batches is not None:
        val_indices = list(range(min(len(val_ds), batch_size * limit_val_batches)))
        eval_loader = _loader(val_ds, val_indices, batch_size=batch_size, image_size=image_size, shuffle=False)
    else:
        eval_loader = val_loader
    metrics = evaluate_segmentation(model, eval_loader, dataset_cfg, device=device)

    selected_countries = []
    if hasattr(train_ds, "items"):
        selected_countries = [train_ds.items[idx].get("country") for idx in indices if idx < len(train_ds.items)]

    return {
        "dataset": dataset_name,
        "model": "tiny_segmentation_cnn_local",
        "model_family": "local_pilot",
        "model_is_foundation_model": False,
        "adapter": method,
        **adapter_metadata(method, is_foundation_model=False),
        "label_budget": label_budget,
        "seed": seed,
        "num_train_examples": len(indices),
        "selected_country_counts": json.dumps({c: selected_countries.count(c) for c in sorted(set(selected_countries))}),
        "image_size": image_size,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "train_seconds": elapsed,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_mean": float(np.mean(losses)) if losses else None,
        **memory_summary(device),
        **params,
        **metrics,
    }

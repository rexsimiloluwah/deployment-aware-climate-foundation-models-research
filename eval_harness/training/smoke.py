from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from eval_harness.adapters import adapter_metadata, apply_adaptation_strategy, count_parameters, memory_summary
from eval_harness.datasets import build_synthetic_dataset, make_label_budget_indices
from eval_harness.evaluation import classification_metrics, segmentation_metrics
from eval_harness.models import build_smoke_model


def _collate(batch: list[dict]) -> dict:
    return {
        "id": [item["id"] for item in batch],
        "x": torch.tensor(np.stack([item["x"] for item in batch]), dtype=torch.float32),
        "y": torch.tensor(np.stack([item["y"] for item in batch])),
    }


def run_smoke_experiment(dataset_cfg: dict, method: str, label_budget: float, seed: int, batch_size: int = 8) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.reset_peak_memory_stats()
    else:
        device = "cpu"
    train_ds = build_synthetic_dataset(dataset_cfg, "train", seed=seed, smoke_size=64)
    val_ds = build_synthetic_dataset(dataset_cfg, "val", seed=seed, smoke_size=24)
    indices = make_label_budget_indices(len(train_ds), label_budget, seed)
    train_loader = DataLoader(Subset(train_ds, indices), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    model = build_smoke_model(dataset_cfg)
    model = apply_adaptation_strategy(model, method).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    started = time.perf_counter()
    model.train()
    for batch in train_loader:
        x = batch["x"].to(device)
        y = batch["y"].long().to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["x"].to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.append(preds)
            y_true.append(batch["y"].numpy())

    y_true_np = np.concatenate(y_true, axis=0)
    y_pred_np = np.concatenate(y_pred, axis=0)
    if dataset_cfg["task_type"] == "classification":
        metrics = classification_metrics(y_true_np, y_pred_np)
    else:
        metrics = segmentation_metrics(y_true_np, y_pred_np, int(dataset_cfg["num_classes"]))

    return {
        "dataset": dataset_cfg["name"],
        "model_family": "smoke_test",
        "model_is_foundation_model": False,
        "adapter": method,
        **adapter_metadata(method, is_foundation_model=False),
        "label_budget": label_budget,
        "seed": seed,
        "num_train_examples": len(indices),
        "train_seconds": elapsed,
        **memory_summary(device),
        **params,
        **metrics,
    }

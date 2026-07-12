from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def metric_column_for_dataset(dataset: str) -> str:
    if "ftw" in dataset:
        return "iou"
    return "macro_f1"


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for dataset, group in df.groupby("dataset"):
        metric = metric_column_for_dataset(dataset)
        rows.append(
            group.groupby(["adapter", "label_budget"], as_index=False)
            .agg({metric: "mean", "train_seconds": "mean", "trainable_params": "mean"})
            .assign(primary_metric=metric, dataset=dataset)
        )
    return pd.concat(rows, ignore_index=True)


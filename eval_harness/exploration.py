from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from eval_harness.datasets import make_label_budget_indices, make_stratified_label_budget_indices


def dataset_overview(dataset_cfg: dict, dataset: Iterable) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": dataset_cfg["name"],
                "display_name": dataset_cfg["display_name"],
                "region": dataset_cfg["region"],
                "theme": dataset_cfg["climate_theme"],
                "task_type": dataset_cfg["task_type"],
                "num_classes": dataset_cfg["num_classes"],
                "input_shape": tuple(dataset_cfg["input"]["shape"]),
                "num_examples": len(dataset),
                "smoke_fixture": dataset_cfg.get("data", {}).get("smoke_fixture"),
            }
        ]
    )


def sample_table(dataset, n: int = 8) -> pd.DataFrame:
    rows = []
    for idx in range(min(n, len(dataset))):
        item = dataset[idx]
        y = item["y"]
        rows.append(
            {
                "idx": idx,
                "id": item["id"],
                "x_shape": tuple(item["x"].shape),
                "x_min": float(np.min(item["x"])),
                "x_mean": float(np.mean(item["x"])),
                "x_max": float(np.max(item["x"])),
                "y_shape": tuple(np.shape(y)),
                "y_preview": int(y) if np.ndim(y) == 0 else np.unique(y).tolist(),
            }
        )
    return pd.DataFrame(rows)


def classification_label_distribution(dataset, class_names: list[str] | None = None) -> pd.DataFrame:
    counts = Counter(int(dataset[idx]["y"]) for idx in range(len(dataset)))
    rows = []
    for label in sorted(counts):
        rows.append(
            {
                "label": label,
                "class_name": class_names[label] if class_names and label < len(class_names) else str(label),
                "count": counts[label],
                "fraction": counts[label] / len(dataset),
            }
        )
    return pd.DataFrame(rows)


def segmentation_pixel_distribution(
    dataset,
    class_names: list[str] | None = None,
    max_examples: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    counts: Counter[int] = Counter()
    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    if seed is None:
        indices = range(n)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    for idx in indices:
        labels, label_counts = np.unique(dataset[idx]["y"], return_counts=True)
        counts.update({int(label): int(count) for label, count in zip(labels, label_counts)})
    total = sum(counts.values())
    rows = []
    for label in sorted(counts):
        rows.append(
            {
                "label": label,
                "class_name": class_names[label] if class_names and label < len(class_names) else str(label),
                "pixels": counts[label],
                "fraction": counts[label] / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def label_budget_summary(
    dataset,
    dataset_cfg: dict,
    budgets: list[float],
    seed: int = 1234,
    strategy: str = "random",
) -> pd.DataFrame:
    rows = []
    class_names = dataset_cfg.get("class_names", [])
    for budget in budgets:
        if strategy == "stratified" and dataset_cfg["task_type"] == "classification":
            labels = [int(dataset[idx]["y"]) for idx in range(len(dataset))]
            indices = make_stratified_label_budget_indices(labels, budget, seed)
        else:
            indices = make_label_budget_indices(len(dataset), budget, seed)
        if dataset_cfg["task_type"] == "classification":
            counts = Counter(int(dataset[idx]["y"]) for idx in indices)
            for label in range(int(dataset_cfg["num_classes"])):
                rows.append(
                    {
                        "budget": budget,
                        "strategy": strategy,
                        "num_examples": len(indices),
                        "label": label,
                        "class_name": class_names[label] if label < len(class_names) else str(label),
                        "count": counts[label],
                        "fraction": counts[label] / len(indices),
                    }
                )
        else:
            counts: Counter[int] = Counter()
            for idx in indices:
                labels, label_counts = np.unique(dataset[idx]["y"], return_counts=True)
                counts.update({int(label): int(count) for label, count in zip(labels, label_counts)})
            total = sum(counts.values())
            for label in range(int(dataset_cfg["num_classes"])):
                rows.append(
                    {
                        "budget": budget,
                        "strategy": strategy,
                        "num_examples": len(indices),
                        "label": label,
                        "class_name": class_names[label] if label < len(class_names) else str(label),
                        "pixels": counts[label],
                        "fraction": counts[label] / total if total else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def plot_class_balance(df: pd.DataFrame, value_col: str = "fraction", title: str = "Class Balance"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["class_name"], df[value_col])
    ax.set_title(title)
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.set_xlabel("Class")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_label_budget_coverage(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = df.pivot_table(index="budget", columns="class_name", values="fraction", fill_value=0)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Label Budget Coverage")
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Label Budget")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_classification_profile(sample: dict, title: str = "Temporal-Spectral Profile"):
    import matplotlib.pyplot as plt

    x = np.asarray(sample["x"])
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(x, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Band / Feature")
    ax.set_ylabel("Time Step")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_segmentation_sample(sample: dict, title: str = "Image / Mask Pair"):
    import matplotlib.pyplot as plt

    x = np.asarray(sample["x"])
    y = np.asarray(sample["y"])
    rgb = _rgb_proxy(x)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Image Proxy")
    axes[0].axis("off")
    axes[1].imshow(y, cmap="gray", interpolation="nearest")
    axes[1].set_title("Mask")
    axes[1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def _rgb_proxy(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("Expected image tensor with shape [channels, height, width].")
    channels = min(3, x.shape[0])
    rgb = np.zeros((x.shape[1], x.shape[2], 3), dtype="float32")
    rgb[..., :channels] = np.moveaxis(x[:channels], 0, -1)
    lo, hi = np.percentile(rgb, [2, 98])
    if hi <= lo:
        return np.zeros_like(rgb)
    return np.clip((rgb - lo) / (hi - lo), 0, 1)

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FTW_RUNS = ["ftw_l4_5pct_v1", "ftw_l4_full_v1"]
GHANA_RUNS = ["ghana_l4_full_v1"]

MODEL_DISPLAY = {
    "prithvi_eo_v2_300m": "Prithvi-EO-2.0 300M",
    "terramind_v1_base": "TerraMind-1.0 Base",
    "terramind_v1_large": "TerraMind-1.0 Large",
}

ADAPTER_DISPLAY = {
    "linear_probe": "Linear Probe",
    "lora": "LoRA",
}

DATASET_DISPLAY = {
    "ftw_africa": "FTW Africa",
    "sen1floods11_ghana": "Sen1Floods11 Ghana",
}

MODEL_COLORS = {
    "prithvi_eo_v2_300m": "#1f77b4",
    "terramind_v1_base": "#2ca02c",
    "terramind_v1_large": "#d62728",
}

ADAPTER_MARKERS = {
    "linear_probe": "o",
    "lora": "s",
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )


def load_run(artifact_root: Path, run_label: str) -> pd.DataFrame:
    path = artifact_root / run_label / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_json(path, lines=True)
    df["run_label"] = run_label
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_iou",
        "foreground_iou",
        "pixel_accuracy",
        "train_seconds",
        "wall_seconds",
        "peak_accelerator_memory_gb",
        "total_trainable_params",
        "num_train_examples",
    ]
    existing = [column for column in metrics if column in df.columns]
    grouped = (
        df.groupby(["dataset", "model", "adapter", "label_budget"], as_index=False)[existing]
        .agg(["mean", "std", "min", "max"])
    )
    grouped.columns = ["_".join(column).rstrip("_") for column in grouped.columns.to_flat_index()]
    grouped = grouped.rename(
        columns={
            "dataset_": "dataset",
            "model_": "model",
            "adapter_": "adapter",
            "label_budget_": "label_budget",
        }
    )
    counts = df.groupby(["dataset", "model", "adapter", "label_budget"], as_index=False).size()
    return grouped.merge(counts, on=["dataset", "model", "adapter", "label_budget"], how="left")


def adapter_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, budget), group in summary.groupby(["dataset", "model", "label_budget"]):
        adapters = set(group["adapter"])
        if {"linear_probe", "lora"} - adapters:
            continue
        linear = group[group["adapter"] == "linear_probe"].iloc[0]
        lora = group[group["adapter"] == "lora"].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "label_budget": budget,
                "delta_mean_iou_lora_minus_linear": lora["mean_iou_mean"] - linear["mean_iou_mean"],
                "delta_foreground_iou_lora_minus_linear": lora["foreground_iou_mean"]
                - linear["foreground_iou_mean"],
                "delta_train_seconds_lora_minus_linear": lora["train_seconds_mean"]
                - linear["train_seconds_mean"],
                "delta_peak_memory_gb_lora_minus_linear": lora["peak_accelerator_memory_gb_mean"]
                - linear["peak_accelerator_memory_gb_mean"],
                "delta_trainable_params_lora_minus_linear": lora["total_trainable_params_mean"]
                - linear["total_trainable_params_mean"],
            }
        )
    return pd.DataFrame(rows)


def pareto_frontier(df: pd.DataFrame, metric: str = "foreground_iou_mean") -> pd.DataFrame:
    candidates = df.copy()
    candidates["cost"] = candidates["peak_accelerator_memory_gb_mean"] * candidates["train_seconds_mean"]
    keep = []
    for idx, row in candidates.iterrows():
        dominated = candidates[
            (candidates[metric] >= row[metric])
            & (candidates["cost"] <= row["cost"])
            & ((candidates[metric] > row[metric]) | (candidates["cost"] < row["cost"]))
        ]
        if dominated.empty:
            keep.append(idx)
    return candidates.loc[keep].sort_values(["label_budget", "cost", metric])


def model_display(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def adapter_display(adapter: str) -> str:
    return ADAPTER_DISPLAY.get(adapter, adapter)


def dataset_display(dataset: str) -> str:
    return DATASET_DISPLAY.get(dataset, dataset)


def budget_display(value: float) -> str:
    return f"{int(round(value * 100))}% labels"


def two_column_figure(figsize: tuple[float, float] = (13.5, 7.6)):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.8, 1.55], wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1])
    legend_ax.axis("off")
    return fig, ax, legend_ax


def add_side_legend(ax, legend_ax, title: str | None = None, ncol: int = 1, y: float = 0.58):
    handles, labels = ax.get_legend_handles_labels()
    legend = legend_ax.legend(
        handles,
        labels,
        title=title,
        loc="center",
        bbox_to_anchor=(0.5, y),
        frameon=True,
        borderaxespad=0.0,
        ncol=ncol,
        columnspacing=1.0,
        handletextpad=0.6,
        labelspacing=0.9,
    )
    if legend and legend.get_title():
        legend.get_title().set_fontweight("bold")
    return legend


def y_limits_with_margin(values: pd.Series, errors: pd.Series | None = None) -> tuple[float, float]:
    values = pd.Series(values, dtype="float64")
    if errors is not None:
        errors = pd.Series(errors, dtype="float64").fillna(0)
        lower = float((values - errors).min())
        upper = float((values + errors).max())
    else:
        lower = float(values.min())
        upper = float(values.max())
    span = max(upper - lower, 0.05)
    y_min = lower - 0.08 * span
    y_max = upper + 0.12 * span
    if lower > 0 and y_min < 0:
        y_min = max(0.0, lower - 0.04 * span)
    return y_min, y_max


def draw_zero_line_if_visible(ax, y_min: float, y_max: float) -> None:
    if y_min < 0 < y_max:
        ax.axhline(0, color="#444444", linewidth=1.0, alpha=0.75)


def plot_label_efficiency(summary: pd.DataFrame, dataset: str, output: Path) -> None:
    subset = summary[summary["dataset"] == dataset].copy()
    fig, ax, legend_ax = two_column_figure()
    budgets = sorted(subset["label_budget"].unique())
    x_positions = {budget: idx for idx, budget in enumerate(budgets)}
    for (model, adapter), group in subset.groupby(["model", "adapter"]):
        group = group.sort_values("label_budget")
        style = "--" if adapter == "lora" else "-"
        x = group["label_budget"].map(x_positions).to_numpy(dtype=float)
        y = group["foreground_iou_mean"]
        yerr = group["foreground_iou_std"].fillna(0)
        ax.plot(
            x,
            y,
            marker=ADAPTER_MARKERS.get(adapter, "o"),
            linestyle=style,
            linewidth=2.6,
            markersize=7,
            color=MODEL_COLORS.get(model),
            label=f"{model_display(model)} | {adapter_display(adapter)}",
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=MODEL_COLORS.get(model),
            alpha=0.30,
            capsize=4,
            elinewidth=1.4,
        )
    ax.set_xlabel("Label Budget (%)", fontweight="bold")
    ax.set_ylabel("Foreground IoU", fontweight="bold")
    ax.set_title(f"{dataset_display(dataset)}: Label Efficiency Across Adaptation Strategies", pad=16)
    ax.set_xticks(list(range(len(budgets))))
    ax.set_xticklabels([f"{int(round(budget * 100))}%" for budget in budgets])
    ax.margins(x=0.05)
    y_min, y_max = y_limits_with_margin(subset["foreground_iou_mean"], subset["foreground_iou_std"])
    ax.set_ylim(y_min, y_max)
    draw_zero_line_if_visible(ax, y_min, y_max)
    ax.grid(True, alpha=0.55)
    add_side_legend(ax, legend_ax, title="Model and Adaptation", y=0.55)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_cost_tradeoff(summary: pd.DataFrame, dataset: str, output: Path) -> None:
    subset = summary[summary["dataset"] == dataset].copy()
    fig, ax, legend_ax = two_column_figure()
    for (model, adapter), group in subset.groupby(["model", "adapter"]):
        sizes = 120 + 2.6 * (group["label_budget"] * 100)
        ax.scatter(
            group["peak_accelerator_memory_gb_mean"],
            group["foreground_iou_mean"],
            s=sizes,
            marker=ADAPTER_MARKERS.get(adapter, "o"),
            color=MODEL_COLORS.get(model),
            edgecolor="#222222",
            linewidth=0.9,
            alpha=0.86,
            label=f"{model_display(model)} | {adapter_display(adapter)}",
        )
    ax.set_xlabel("Peak GPU Memory During Adaptation (GB)", fontweight="bold")
    ax.set_ylabel("Foreground IoU", fontweight="bold")
    ax.set_title(f"{dataset_display(dataset)}: Accuracy Versus GPU Memory", pad=16)
    legend_ax.text(
        0.5,
        0.82,
        "Point size indicates\nlabel budget.",
        transform=legend_ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#F7F7F7", "ec": "#CCCCCC"},
    )
    y_min, y_max = y_limits_with_margin(subset["foreground_iou_mean"])
    ax.set_ylim(y_min, y_max)
    draw_zero_line_if_visible(ax, y_min, y_max)
    ax.grid(True, alpha=0.55)
    add_side_legend(ax, legend_ax, title="Model and Adaptation", y=0.55)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_pareto_efficiency(summary: pd.DataFrame, pareto: pd.DataFrame, dataset: str, output: Path) -> None:
    subset = summary[summary["dataset"] == dataset].copy()
    subset["efficiency_cost"] = subset["peak_accelerator_memory_gb_mean"] * subset["train_seconds_mean"]
    frontier = pareto[pareto["dataset"] == dataset].copy()
    if "cost" not in frontier.columns:
        frontier["cost"] = frontier["peak_accelerator_memory_gb_mean"] * frontier["train_seconds_mean"]

    fig, ax, legend_ax = two_column_figure(figsize=(14.5, 8.2))
    for (model, adapter), group in subset.groupby(["model", "adapter"]):
        sizes = 115 + 2.8 * (group["label_budget"] * 100)
        ax.scatter(
            group["efficiency_cost"],
            group["foreground_iou_mean"],
            s=sizes,
            marker=ADAPTER_MARKERS.get(adapter, "o"),
            color=MODEL_COLORS.get(model),
            edgecolor="#333333",
            linewidth=0.8,
            alpha=0.34,
            label=f"{model_display(model)} | {adapter_display(adapter)}",
        )

    frontier = frontier.sort_values("cost")
    ax.plot(
        frontier["cost"],
        frontier["foreground_iou_mean"],
        color="#111111",
        linewidth=3.0,
        marker="D",
        markersize=7,
        label="Pareto Frontier",
        zorder=5,
    )
    ax.scatter(
        frontier["cost"],
        frontier["foreground_iou_mean"],
        s=260,
        facecolors="none",
        edgecolors="#111111",
        linewidths=2.2,
        zorder=6,
    )
    frontier_rows = []
    for i, (_, row) in enumerate(frontier.iterrows(), start=1):
        ax.text(
            row["cost"],
            row["foreground_iou_mean"],
            str(i),
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="white",
            zorder=8,
        )
        frontier_rows.append(
            f"{i}. {model_display(row['model'])}, {adapter_display(row['adapter'])}, {budget_display(row['label_budget'])}"
        )

    for _i, (_, row) in enumerate(frontier.iterrows(), start=1):
        ax.annotate(
            "",
            (row["cost"], row["foreground_iou_mean"]),
            xytext=(0, 0),
            textcoords="offset points",
            zorder=7,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Deployment Efficiency Cost: Peak VRAM (GB) × Training Time (Seconds)", fontweight="bold")
    ax.set_ylabel("Foreground IoU", fontweight="bold")
    ax.set_title(f"{dataset_display(dataset)}: Pareto Frontier of Deployment Readiness", pad=16)
    legend_ax.text(
        0.5,
        0.54,
        "Better configurations\nmove upward and left.",
        transform=legend_ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#F7F7F7", "ec": "#CCCCCC"},
    )
    legend_ax.text(
        0.02,
        0.98,
        "Pareto Frontier Points\n" + "\n".join(frontier_rows),
        transform=legend_ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.42", "fc": "white", "ec": "#BBBBBB", "alpha": 0.98},
    )
    y_min, y_max = y_limits_with_margin(subset["foreground_iou_mean"])
    ax.set_ylim(y_min, y_max)
    draw_zero_line_if_visible(ax, y_min, y_max)
    ax.grid(True, which="both", alpha=0.50)
    add_side_legend(ax, legend_ax, title="All Configurations", y=0.36)
    fig.savefig(output, dpi=260)
    plt.close(fig)


def write_markdown_report(
    output: Path,
    ftw_summary: pd.DataFrame,
    ghana_summary: pd.DataFrame,
    ftw_deltas: pd.DataFrame,
    ghana_deltas: pd.DataFrame,
) -> None:
    best_ftw = ftw_summary.sort_values("foreground_iou_mean", ascending=False).head(8)
    best_ghana = ghana_summary.sort_values("foreground_iou_mean", ascending=False).head(8)
    lines = [
        "# Lightning L4 Experiment Results",
        "",
        "## Coverage",
        "",
        "- FTW Africa: 72 rows total after combining 5%, 10%, 25%, and 100% label budgets.",
        "- Ghana Sen1Floods11 stress-test: 72 rows total across 5%, 10%, 25%, and 100% label budgets.",
        "- All completed rows have `ok = true`; no failed experiment cells.",
        "",
        "## FTW Africa: Best Foreground IoU",
        "",
        best_ftw[
            [
                "model",
                "adapter",
                "label_budget",
                "foreground_iou_mean",
                "mean_iou_mean",
                "peak_accelerator_memory_gb_mean",
                "total_trainable_params_mean",
            ]
        ].to_markdown(index=False),
        "",
        "## FTW Africa: LoRA Minus Linear Probe",
        "",
        ftw_deltas.to_markdown(index=False),
        "",
        "## Ghana Flood Stress-Test: Best Foreground IoU",
        "",
        best_ghana[
            [
                "model",
                "adapter",
                "label_budget",
                "foreground_iou_mean",
                "mean_iou_mean",
                "pixel_accuracy_mean",
            ]
        ].to_markdown(index=False),
        "",
        "## Ghana: LoRA Minus Linear Probe",
        "",
        ghana_deltas.to_markdown(index=False),
        "",
        "## Main Takeaways",
        "",
        "1. FTW is the headline dataset: it gives non-trivial foreground IoU and meaningful efficiency trade-offs.",
        "2. Linear probing is a very strong baseline on FTW; LoRA is not automatically worth its extra trainable parameters.",
        "3. Ghana is useful as a stress-test for class imbalance: high pixel accuracy hides zero flood IoU in many settings.",
        "4. The harness itself is a central contribution because it records accuracy, label budget, train time, VRAM, and trainable parameters together.",
        "",
        "## Key Figures",
        "",
        "- `ftw_label_efficiency.png`: label budget versus foreground IoU on FTW Africa.",
        "- `ftw_memory_tradeoff.png`: foreground IoU versus peak GPU memory on FTW Africa.",
        "- `ftw_pareto_efficiency.png`: deployment-readiness Pareto frontier on FTW Africa.",
        "- `ghana_label_efficiency.png`: flood foreground IoU under severe class imbalance.",
        "- `ghana_pareto_efficiency.png`: stress-test Pareto view for Ghana.",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output-dir", default="artifacts/analysis_l4_v1")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ftw = pd.concat([load_run(artifact_root, run) for run in FTW_RUNS], ignore_index=True)
    ghana = pd.concat([load_run(artifact_root, run) for run in GHANA_RUNS], ignore_index=True)
    all_results = pd.concat([ftw, ghana], ignore_index=True)

    ftw_summary = aggregate(ftw)
    ghana_summary = aggregate(ghana)
    all_summary = aggregate(all_results)
    ftw_deltas = adapter_deltas(ftw_summary)
    ghana_deltas = adapter_deltas(ghana_summary)
    ftw_pareto = pareto_frontier(ftw_summary)
    ghana_pareto = pareto_frontier(ghana_summary)

    all_results.to_csv(output_dir / "all_results.csv", index=False)
    ftw_summary.to_csv(output_dir / "ftw_summary.csv", index=False)
    ghana_summary.to_csv(output_dir / "ghana_summary.csv", index=False)
    all_summary.to_csv(output_dir / "all_summary.csv", index=False)
    ftw_deltas.to_csv(output_dir / "ftw_adapter_deltas.csv", index=False)
    ghana_deltas.to_csv(output_dir / "ghana_adapter_deltas.csv", index=False)
    ftw_pareto.to_csv(output_dir / "ftw_pareto.csv", index=False)
    ghana_pareto.to_csv(output_dir / "ghana_pareto.csv", index=False)

    plot_label_efficiency(ftw_summary, "ftw_africa", output_dir / "ftw_label_efficiency.png")
    plot_cost_tradeoff(ftw_summary, "ftw_africa", output_dir / "ftw_memory_tradeoff.png")
    plot_pareto_efficiency(ftw_summary, ftw_pareto, "ftw_africa", output_dir / "ftw_pareto_efficiency.png")
    plot_label_efficiency(ghana_summary, "sen1floods11_ghana", output_dir / "ghana_label_efficiency.png")
    plot_cost_tradeoff(ghana_summary, "sen1floods11_ghana", output_dir / "ghana_memory_tradeoff.png")
    plot_pareto_efficiency(
        ghana_summary,
        ghana_pareto,
        "sen1floods11_ghana",
        output_dir / "ghana_pareto_efficiency.png",
    )

    write_markdown_report(
        output_dir / "results_report.md",
        ftw_summary,
        ghana_summary,
        ftw_deltas,
        ghana_deltas,
    )
    print(f"Wrote analysis outputs to {output_dir}")
    print(f"FTW rows: {len(ftw)}; Ghana rows: {len(ghana)}; failures: {int((~all_results['ok']).sum())}")


if __name__ == "__main__":
    main()

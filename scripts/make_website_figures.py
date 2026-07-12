"""Publication-quality figures for the lightning-talk website Results section.

Figures (one or more per result):
  1a. EO label efficiency (linear probe vs LoRA on FTW Africa)
  1b. EO deployment Pareto frontier (segmentation quality vs cost)
  2.  Aurora forecasting scale (Small vs Large, absolute RMSE by lead time)
  3.  Aurora cheap adaptation (Small pretrained vs +MOS vs +LoRA vs Large)

Design choices (dataviz best practice):
- Okabe-Ito colorblind-safe categorical palette; hues assigned by entity, fixed order.
- Legends live OUTSIDE the axes, to the right, so they never occlude data.
- Larger fonts + compact figure sizes so text stays readable when the site scales them down.
- Absolute RMSE leads (monotonic, physical); persistence-relative skill is secondary.
- No em dashes anywhere in titles/labels.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "aurora_greater_horn_final_v1"
FTW = ROOT / "artifacts" / "analysis_l4_v1" / "ftw_summary.csv"
OUT = ROOT / "website" / "public" / "assets"

OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
}
INK = "#1a1a1a"
GRID = "#d9d9d9"

MODEL_COLOR = {
    "prithvi_eo_v2_300m": OKABE["blue"],
    "terramind_v1_base": OKABE["orange"],
    "terramind_v1_large": OKABE["purple"],
}
MODEL_LABEL = {
    "prithvi_eo_v2_300m": "Prithvi-EO-2.0",
    "terramind_v1_base": "TerraMind Base",
    "terramind_v1_large": "TerraMind Large",
}

VAR_LABEL = {
    "2m_temperature": "2m Temperature (K)",
    "10m_u_component_of_wind": "10m U Wind (m/s)",
    "10m_v_component_of_wind": "10m V Wind (m/s)",
    "mean_sea_level_pressure": "Mean Sea-Level Pressure (Pa)",
}


def base_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#666666",
        "axes.linewidth": 1.0,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12.5,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
        "savefig.bbox": "tight",
        "savefig.dpi": 180,
    })


def side_legend(ax) -> None:
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, handlelength=2.2)


def fig_eo_label_efficiency() -> None:
    df = pd.read_csv(FTW)
    df = df[df["model"] == "prithvi_eo_v2_300m"].copy()
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    styles = {
        "linear_probe": (OKABE["blue"], "-", "o", "Linear Probe (12.3K params)"),
        "lora": (OKABE["vermillion"], "--", "s", "LoRA (3.7M params)"),
    }
    for adapter, (color, ls, marker, label) in styles.items():
        g = df[df["adapter"] == adapter].sort_values("label_budget")
        x = (g["label_budget"] * 100).to_numpy()
        y = g["foreground_iou_mean"].to_numpy()
        err = g["foreground_iou_std"].fillna(0).to_numpy()
        ax.plot(x, y, ls, color=color, marker=marker, linewidth=2.6, markersize=8, label=label, zorder=3)
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.13, linewidth=0)
    ax.set_xlabel("Label Budget (%)")
    ax.set_ylabel("Foreground IoU")
    ax.set_title("Prithvi-EO-2.0 on FTW Africa: Linear Probe vs LoRA")
    ax.set_xticks([5, 10, 25, 100])
    ax.grid(True, alpha=0.6)
    ax.annotate("25% labels reach the\n100% label accuracy",
                xy=(25, 0.346), xytext=(40, 0.27),
                fontsize=11, color=INK,
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.3})
    side_legend(ax)
    fig.savefig(OUT / "fig_eo_label_efficiency.png")
    plt.close(fig)


def fig_eo_pareto() -> None:
    df = pd.read_csv(FTW)
    df = df.copy()
    df["cost"] = df["peak_accelerator_memory_gb_mean"] * df["train_seconds_mean"]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    # Non-dominated (Pareto) set: no other point has higher IoU AND lower cost.
    keep = []
    for i, r in df.iterrows():
        dominated = df[(df["foreground_iou_mean"] >= r["foreground_iou_mean"]) & (df["cost"] <= r["cost"]) &
                       ((df["foreground_iou_mean"] > r["foreground_iou_mean"]) | (df["cost"] < r["cost"]))]
        if dominated.empty:
            keep.append(i)
    frontier = df.loc[keep].sort_values("cost")

    marker = {"linear_probe": "o", "lora": "s"}
    for model, g in df.groupby("model"):
        for adapter, gg in g.groupby("adapter"):
            ax.scatter(gg["cost"], gg["foreground_iou_mean"], s=70, color=MODEL_COLOR.get(model, "#888"),
                       marker=marker.get(adapter, "o"), edgecolor="white", linewidth=0.8, alpha=0.85, zorder=3)
    ax.plot(frontier["cost"], frontier["foreground_iou_mean"], color=INK, linewidth=2.0,
            marker="D", markersize=8, markerfacecolor="white", markeredgecolor=INK, zorder=4,
            label="Pareto frontier")

    ax.set_xscale("log")
    ax.set_xlabel("Deployment cost: peak memory (GB) x training time (s), log scale")
    ax.set_ylabel("Foreground IoU")
    ax.set_title("FTW Africa: Deployment Pareto Frontier")
    ax.grid(True, which="both", alpha=0.5)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=INK, lw=2.0, marker="D", markerfacecolor="white", label="Pareto frontier")]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLOR[m], markersize=10, label=MODEL_LABEL[m]) for m in MODEL_COLOR]
    handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", markersize=10, label="Linear Probe (circle)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#555", markersize=10, label="LoRA (square)"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.annotate("better", xy=(df["cost"].min() * 1.1, df["foreground_iou_mean"].max()),
                fontsize=10.5, color="#777", ha="left")
    fig.savefig(OUT / "fig_eo_pareto.png")
    plt.close(fig)


def _load_aurora() -> pd.DataFrame:
    return pd.read_csv(FINAL / "summary.csv")


def fig_aurora_scale() -> None:
    s = _load_aurora()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.4))
    series = {
        ("aurora_small", "pretrained"): (OKABE["blue"], "Aurora Small (113M)"),
        ("aurora_large", "pretrained"): (OKABE["vermillion"], "Aurora Large (1.26B, fine-tuned)"),
    }
    for ax, var in zip(axes.ravel(), VAR_LABEL):
        for (model, adapter), (color, label) in series.items():
            g = s[(s["model"] == model) & (s["adapter"] == adapter) & (s["variable"] == var)].sort_values("lead_time_hours")
            x = g["lead_time_hours"].to_numpy()
            y = g["weighted_rmse_mean"].to_numpy()
            err = g["weighted_rmse_std"].fillna(0).to_numpy()
            ax.plot(x, y, "-o", color=color, linewidth=2.4, markersize=6, label=label, zorder=3)
            ax.fill_between(x, y - err, y + err, color=color, alpha=0.13, linewidth=0)
        ax.set_title(VAR_LABEL[var], fontsize=13.5)
        ax.set_xlabel("Lead Time (hours)")
        ax.set_ylabel("Weighted RMSE")
        ax.set_xticks([6, 12, 24, 48])
        ax.grid(True, alpha=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Forecast Error vs Lead Time by Model Scale (Greater Horn of Africa)", fontsize=15.5, fontweight="bold")
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_aurora_scale.png")
    plt.close(fig)


def fig_aurora_adaptation() -> None:
    s = _load_aurora()
    var = "2m_temperature"
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    series = [
        ("aurora_small", "pretrained", OKABE["blue"], "-", "o", "Aurora Small (no adaptation)"),
        ("aurora_small", "linear_calibration", OKABE["orange"], "-", "^", "Aurora Small + MOS (32 params)"),
        ("aurora_small", "lora", OKABE["green"], "-", "s", "Aurora Small + LoRA (541K params)"),
        ("aurora_large", "pretrained", OKABE["vermillion"], "--", "D", "Aurora Large (reference, 1.26B)"),
    ]
    for model, adapter, color, ls, marker, label in series:
        g = s[(s["model"] == model) & (s["adapter"] == adapter) & (s["variable"] == var)].sort_values("lead_time_hours")
        ax.plot(g["lead_time_hours"], g["weighted_rmse_mean"], ls, color=color, marker=marker,
                linewidth=2.6, markersize=8, label=label, zorder=3)
    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("2m Temperature Weighted RMSE (K)")
    ax.set_title("Cheap Adaptation of Aurora Small (2m Temperature)")
    ax.set_xticks([6, 12, 24, 48])
    ax.grid(True, alpha=0.6)
    ax.annotate("", xy=(45, 0.965), xytext=(45, 1.435),
                arrowprops={"arrowstyle": "<->", "color": "#555555", "lw": 1.6})
    ax.text(46.5, 1.2, "gap to\nLarge", fontsize=10.5, color="#555", va="center")
    side_legend(ax)
    fig.savefig(OUT / "fig_aurora_adaptation.png")
    plt.close(fig)


def main() -> None:
    base_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_eo_label_efficiency()
    fig_eo_pareto()
    fig_aurora_scale()
    fig_aurora_adaptation()
    print(f"Wrote figures to {OUT}")
    for name in ["fig_eo_label_efficiency.png", "fig_eo_pareto.png", "fig_aurora_scale.png", "fig_aurora_adaptation.png"]:
        print(" -", name)


if __name__ == "__main__":
    main()

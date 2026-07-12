from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "lightning_talk_report_v1"
FTW_SUMMARY = ROOT / "artifacts" / "analysis_l4_v1" / "ftw_summary.csv"
AURORA_CANDIDATES = [
    ROOT / "artifacts" / "aurora_greater_horn_final_v1",
    ROOT / "artifacts" / "aurora_greater_horn_adapted_v1",
    ROOT / "artifacts" / "aurora_greater_horn_global_v1",
]


def aurora_artifact_is_reportable(path: Path) -> bool:
    summary = path / "summary.csv"
    overall = path / "model_overall.csv"
    results = path / "results.jsonl"
    if not (summary.exists() and overall.exists() and results.exists()):
        return False
    summary_df = pd.read_csv(summary)
    required = ["weighted_rmse_mean", "weighted_mae_mean", "inference_seconds_mean", "peak_memory_gb_mean"]
    return bool(np.isfinite(summary_df[required].to_numpy(dtype=float)).all())


AURORA_DIR = next((path for path in AURORA_CANDIDATES if path.exists() and aurora_artifact_is_reportable(path)), None)
if AURORA_DIR is None:
    valid_names = ", ".join(path.name for path in AURORA_CANDIDATES)
    raise FileNotFoundError(
        "No valid Aurora artifact directory found. Re-run the fixed global-input Aurora experiment first. "
        f"Expected one of: {valid_names}. Do not use aurora_greater_horn_defensible_v1; it used cropped input."
    )
AURORA_RESULTS = AURORA_DIR / "results.jsonl"
AURORA_SUMMARY = AURORA_DIR / "summary.csv"
AURORA_OVERALL = AURORA_DIR / "model_overall.csv"
PERSISTENCE_RESULTS = ROOT / "artifacts" / "aurora_greater_horn_persistence_v2" / "results.jsonl"
if not PERSISTENCE_RESULTS.exists():
    raise FileNotFoundError(
        "Missing persistence baseline required for Aurora reporting: "
        f"{PERSISTENCE_RESULTS}. Run scripts/run_forecasting_persistence_baseline.py first."
    )


MODEL_NAMES = {
    "prithvi_eo_v2_300m": "Prithvi-EO-2.0 300M",
    "terramind_v1_base": "TerraMind-1.0 Base",
    "terramind_v1_large": "TerraMind-1.0 Large",
    "aurora_small": "Aurora Small",
    "aurora_large": "Aurora Large (0.25 Fine-Tuned)",
}
ADAPTER_NAMES = {"linear_probe": "Linear Probe", "linear_calibration": "Linear Calibration (MOS)", "lora": "LoRA"}
VAR_NAMES = {
    "2m_temperature": "2m Temperature",
    "10m_u_component_of_wind": "10m U Wind",
    "10m_v_component_of_wind": "10m V Wind",
    "mean_sea_level_pressure": "Mean Sea-Level Pressure",
}
COLORS = {
    "aurora_small": "#1f77b4",
    "aurora_large": "#d62728",
    "linear_calibration": "#9467bd",
    "linear_probe": "#1f77b4",
    "lora": "#2ca02c",
}


def ensure_adapter_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "adapter" not in df.columns:
        df["adapter"] = "pretrained"
    if "trainable_params" not in df.columns:
        df["trainable_params"] = 0
    if "trainable_fraction" not in df.columns:
        df["trainable_fraction"] = 0.0
    return df


def model_label(model: str, adapter: str = "pretrained") -> str:
    if model == "aurora_small" and adapter == "lora":
        return "Aurora Small + LoRA"
    if model == "aurora_small" and adapter == "linear_calibration":
        return "Aurora Small + Linear Calibration (MOS)"
    return MODEL_NAMES.get(model, model)


def load_aurora_results() -> pd.DataFrame:
    rows = [json.loads(line) for line in AURORA_RESULTS.read_text().splitlines() if line.strip()]
    return ensure_adapter_columns(pd.DataFrame(rows))


def load_persistence_results() -> pd.DataFrame:
    rows = [json.loads(line) for line in PERSISTENCE_RESULTS.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def title_case_axis(ax) -> None:
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.tick_params(axis="both", labelsize=10)


def save_aurora_rmse_plot(summary: pd.DataFrame) -> Path:
    summary = ensure_adapter_columns(summary)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), constrained_layout=True)
    axes = axes.ravel()
    for ax, variable in zip(axes, VAR_NAMES):
        sub = summary[summary["variable"] == variable].copy()
        for (model, adapter), group in sub.groupby(["model", "adapter"]):
            group = group.sort_values("lead_time_hours")
            label = model_label(model, adapter)
            color = COLORS.get(adapter) if adapter in COLORS else COLORS.get(model)
            ax.plot(
                group["lead_time_hours"],
                group["weighted_rmse_mean"],
                marker="o",
                linewidth=2.6,
                markersize=6,
                color=color,
                label=label,
            )
            ax.fill_between(
                group["lead_time_hours"].to_numpy(),
                (group["weighted_rmse_mean"] - group["weighted_rmse_std"]).to_numpy(),
                (group["weighted_rmse_mean"] + group["weighted_rmse_std"]).to_numpy(),
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        ax.set_title(VAR_NAMES[variable])
        ax.set_xlabel("Forecast Lead Time (Hours)")
        ax.set_ylabel("Weighted RMSE")
        ax.grid(True, alpha=0.35)
        title_case_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, -0.035),
        fontsize=11,
    )
    fig.suptitle("WeatherBench2 Greater Horn: Forecast Error Across Lead Times", fontsize=17, fontweight="bold")
    path = OUTPUT / "aurora_rmse_by_lead_time.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def aurora_deployment_scores(summary: pd.DataFrame) -> pd.DataFrame:
    scored = ensure_adapter_columns(summary)
    persistence = load_persistence_results()
    persistence_summary = (
        persistence.groupby(["variable", "lead_time_hours"], as_index=False)
        .agg(persistence_rmse=("weighted_rmse", "mean"))
    )
    scored = scored.merge(persistence_summary, on=["variable", "lead_time_hours"], how="left")
    if scored["persistence_rmse"].isna().any():
        missing = scored[scored["persistence_rmse"].isna()][["variable", "lead_time_hours"]].drop_duplicates()
        raise ValueError(f"Persistence baseline is missing variable/lead tasks:\n{missing}")
    scored["rmse_relative_to_persistence"] = scored["weighted_rmse_mean"] / scored["persistence_rmse"]

    # Persistence skill changes character with lead time (stiff bar at 6h, trivial to beat at 48h),
    # so emit the full per-variable / per-lead breakdown and a per-lead aggregate. The single
    # cross-variable, cross-lead scalar below is only a summary dot for the tradeoff scatter.
    detail_cols = [
        "model",
        "adapter",
        "variable",
        "lead_time_hours",
        "weighted_rmse_mean",
        "persistence_rmse",
        "rmse_relative_to_persistence",
    ]
    detail = scored[detail_cols].sort_values(["model", "adapter", "variable", "lead_time_hours"])
    detail.to_csv(OUTPUT / "aurora_skill_by_variable_lead.csv", index=False)
    by_lead = (
        scored.groupby(["model", "adapter", "lead_time_hours"], as_index=False)
        .agg(mean_rmse_relative_to_persistence=("rmse_relative_to_persistence", "mean"))
        .sort_values(["model", "adapter", "lead_time_hours"])
    )
    by_lead.to_csv(OUTPUT / "aurora_skill_by_lead.csv", index=False)

    scores = (
        scored.groupby(["model", "adapter"])
        .agg(
            mean_rmse_relative_to_persistence=("rmse_relative_to_persistence", "mean"),
            inference_seconds_mean=("inference_seconds_mean", "mean"),
            peak_memory_gb_mean=("peak_memory_gb_mean", "mean"),
            total_params=("total_params", "first"),
            trainable_params=("trainable_params", "first"),
            trainable_fraction=("trainable_fraction", "first"),
        )
        .reset_index()
    )
    scores.to_csv(OUTPUT / "aurora_deployment_scores.csv", index=False)
    return scores


def save_aurora_efficiency_plot(summary: pd.DataFrame) -> Path:
    scores = aurora_deployment_scores(summary)
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    for _, row in scores.iterrows():
        model = row["model"]
        adapter = row["adapter"]
        size = 180 + 0.00000015 * row["total_params"]
        color = COLORS.get(adapter) if adapter in COLORS else COLORS.get(model)
        ax.scatter(
            row["peak_memory_gb_mean"],
            row["mean_rmse_relative_to_persistence"],
            s=size,
            color=color,
            edgecolor="#222222",
            linewidth=1.2,
            alpha=0.86,
            label=model_label(model, adapter),
        )
        offset = (9, 18) if model == "aurora_small" and adapter != "linear_calibration" else (-175, 8)
        ax.annotate(
            model_label(model, adapter),
            (row["peak_memory_gb_mean"], row["mean_rmse_relative_to_persistence"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )
    ax.axhline(1.0, color="#666666", linewidth=1.4, linestyle="--", alpha=0.8)
    ax.set_xlabel("Peak GPU Memory During Forecasting (GB)")
    ax.set_ylabel("Mean RMSE Relative to Persistence")
    ax.set_title("Aurora Forecasting Deployment Tradeoff")
    ax.grid(True, alpha=0.35)
    ax.set_xscale("log")
    ax.set_xlim(0.45, max(scores["peak_memory_gb_mean"]) * 1.30)
    ax.set_xticks([0.5, 1.0, 2.0, 5.0])
    ax.set_xticklabels(["0.5", "1", "2", "5"])
    y_min = max(0.0, scores["mean_rmse_relative_to_persistence"].min() - 0.08)
    y_max = max(1.05, scores["mean_rmse_relative_to_persistence"].max() + 0.08)
    ax.set_ylim(y_min, y_max)
    ax.text(
        0.03,
        0.92,
        "Better configurations move downward and left.\nPoint size indicates parameter count.",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#F7F7F7", "ec": "#CCCCCC"},
    )
    title_case_axis(ax)
    path = OUTPUT / "aurora_deployment_tradeoff.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def save_framework_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6.8), constrained_layout=True)
    ax.axis("off")

    boxes = [
        (0.05, 0.68, 0.22, 0.18, "Axis 1\nEarth Observation", "FTW Africa\nPrithvi + TerraMind"),
        (0.05, 0.28, 0.22, 0.18, "Axis 2\nWeather Forecasting", "WeatherBench2 HRES T0\nAurora Small: MOS/LoRA + Large"),
        (0.39, 0.48, 0.22, 0.20, "Shared Harness", "Load data -> run model -> score task\nrecord memory, time, params"),
        (0.73, 0.48, 0.22, 0.20, "Deployment Readiness", "Accuracy + data needs + compute cost\nWhich model is practical?"),
    ]
    for x, y, w, h, header, body in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor="#F8FAFC", edgecolor="#263238", linewidth=1.6)
        )
        ax.text(x + w / 2, y + h * 0.65, header, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=10.5)

    arrows = [
        ((0.27, 0.77), (0.39, 0.60)),
        ((0.27, 0.37), (0.39, 0.56)),
        ((0.61, 0.58), (0.73, 0.58)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#263238"})

    ax.text(
        0.5,
        0.93,
        "Two-Axis Evaluation Framework for Deployable Climate Foundation Models",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.12,
        "Central question: when does adaptation or model scale justify its extra deployment cost?",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    path = OUTPUT / "two_axis_framework.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def best_ftw_rows(ftw: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    best = ftw.sort_values("foreground_iou_mean", ascending=False).iloc[0]
    prithvi_linear_25 = ftw[
        (ftw["model"] == "prithvi_eo_v2_300m")
        & (ftw["adapter"] == "linear_probe")
        & (ftw["label_budget"] == 0.25)
    ].iloc[0]
    prithvi_lora_100 = ftw[
        (ftw["model"] == "prithvi_eo_v2_300m")
        & (ftw["adapter"] == "lora")
        & (ftw["label_budget"] == 1.0)
    ].iloc[0]
    return best, prithvi_linear_25, prithvi_lora_100


def write_report(ftw: pd.DataFrame, aurora: pd.DataFrame, aurora_overall: pd.DataFrame, figures: list[Path]) -> Path:
    aurora = ensure_adapter_columns(aurora)
    aurora_overall = ensure_adapter_columns(aurora_overall)
    best, prithvi_linear_25, prithvi_lora_100 = best_ftw_rows(ftw)
    small = aurora_overall[(aurora_overall["model"] == "aurora_small") & (aurora_overall["adapter"] == "pretrained")].iloc[0]
    large = aurora_overall[(aurora_overall["model"] == "aurora_large") & (aurora_overall["adapter"] == "pretrained")].iloc[0]
    calibration_rows = aurora_overall[
        (aurora_overall["model"] == "aurora_small") & (aurora_overall["adapter"] == "linear_calibration")
    ]
    lora_rows = aurora_overall[(aurora_overall["model"] == "aurora_small") & (aurora_overall["adapter"] == "lora")]
    deployment_scores = aurora_deployment_scores(aurora)
    small_score = deployment_scores[
        (deployment_scores["model"] == "aurora_small") & (deployment_scores["adapter"] == "pretrained")
    ].iloc[0]
    large_score = deployment_scores[
        (deployment_scores["model"] == "aurora_large") & (deployment_scores["adapter"] == "pretrained")
    ].iloc[0]
    ftw_rows = int(ftw["size"].sum())
    aurora_rows = len(load_aurora_results())
    calibration_sentence = ""
    if not calibration_rows.empty:
        calibration = calibration_rows.iloc[0]
        calibration_score = deployment_scores[
            (deployment_scores["model"] == "aurora_small")
            & (deployment_scores["adapter"] == "linear_calibration")
        ].iloc[0]
        calibration_param_count = int(calibration.get("calibration_parameter_count", calibration["trainable_params"]))
        calibration_sentence = (
            f" Aurora Small + Linear Calibration (MOS) used {calibration['peak_memory_gb_mean']:.2f} GB peak GPU "
            f"memory, learned {calibration_param_count:,} calibration parameters, and achieved mean RMSE relative "
            f"to persistence {calibration_score['mean_rmse_relative_to_persistence']:.3f}."
        )
    lora_sentence = ""
    if not lora_rows.empty:
        lora = lora_rows.iloc[0]
        lora_score = deployment_scores[
            (deployment_scores["model"] == "aurora_small") & (deployment_scores["adapter"] == "lora")
        ].iloc[0]
        lora_sentence = (
            f" Aurora Small + LoRA used {lora['peak_memory_gb_mean']:.2f} GB peak GPU memory, "
            f"{int(lora['trainable_params']):,} trainable parameters, and achieved mean RMSE relative "
            f"to persistence {lora_score['mean_rmse_relative_to_persistence']:.3f}."
        )
    has_adapted_aurora = not calibration_rows.empty and not lora_rows.empty
    forecasting_axis_phrase = (
        "weather forecasting regional adaptation and model scale"
        if has_adapted_aurora
        else "weather forecasting model scale, with Aurora adaptation still being finalized"
    )
    forecasting_method = (
        "For weather forecasting, we adapt an existing forecasting task to the Greater Horn region: Aurora Small "
        "pretrained versus Aurora Small + Linear Calibration (MOS) versus Aurora Small + LoRA, with Aurora Large "
        "kept as a frozen larger-checkpoint baseline."
        if has_adapted_aurora
        else "For weather forecasting, the completed artifact compares Aurora Small and Aurora Large using global "
        "WeatherBench2 input and Greater Horn regional scoring. The adapted Aurora Small + Linear Calibration (MOS) "
        "and Aurora Small + LoRA runs are the remaining Modal A100 experiments."
    )
    forecasting_models = (
        "Aurora Small pretrained checkpoint, Aurora Small + Linear Calibration (MOS), Aurora Small + LoRA, and "
        "Aurora Large / 0.25 fine-tuned checkpoint."
        if has_adapted_aurora
        else "Aurora Small pretrained checkpoint and Aurora Large / 0.25 fine-tuned checkpoint; adapted Aurora "
        "Small runs are pending."
    )
    forecasting_adaptation = (
        "Linear Calibration (MOS), fitted as per-variable/per-lead affine output correction on the train split, "
        "and LoRA on Aurora Small using WeatherBench training windows."
        if has_adapted_aurora
        else "pending: Linear Calibration (MOS) and LoRA on Aurora Small."
    )
    forecasting_discussion = (
        "Aurora Small + Linear Calibration (MOS) tests whether a tiny output-level correction is enough to remove "
        "regional bias, while Aurora Small + LoRA tests whether updating a small number of model weights is worth "
        "the additional memory and training cost. Aurora Large remains the frozen scale baseline."
        if has_adapted_aurora
        else "The current WeatherBench2 artifact proves the global-input evaluation gate and the scale baseline. "
        "The remaining step is to compare cheap output calibration against LoRA adaptation on a larger GPU."
    )

    lines = [
        "# Deployable Earth Foundation Models for African Climate Resilience",
        "",
        "## Background and Motivation",
        "",
        "Earth foundation models are increasingly useful for climate-relevant geospatial tasks such as agricultural mapping, flood monitoring, and weather forecasting. But in many African deployment settings, the limiting question is not only whether a model is accurate. It is whether the model can be adapted with limited labels, run on realistic GPUs, and produce useful outputs without excessive memory, training time, or engineering complexity.",
        "",
        f"This work evaluates deployment readiness across two complementary axes: Earth observation adaptation and {forecasting_axis_phrase}. The goal is to move beyond headline accuracy and ask which foundation-model choices are practically worth using for African climate resilience.",
        "",
        "## Research Questions",
        "",
        "1. For Earth observation, when do lightweight adaptation strategies such as Linear Probe and LoRA provide enough accuracy to be useful under limited labels and compute?",
        "2. For weather forecasting, can cheap regional adaptation of Aurora Small close the gap to Aurora Large, and when is LoRA worth its extra deployment cost over Linear Calibration (MOS)?",
        "3. Can one evaluation harness compare deployment readiness across both Earth observation and weather forecasting adaptation settings?",
        "",
        "## Significance",
        "",
        "For African climate applications, deployment cost is not a side detail. Training time affects iteration speed, peak memory affects who can run the model, and trainable parameters affect whether adaptation is feasible for local teams. A model that is slightly more accurate but much harder to adapt or deploy may be the wrong choice for operational resilience work.",
        "",
        "## Methodology",
        "",
        f"The evaluation framework records both task performance and deployment cost. For Earth observation, we run an adaptation experiment: Linear Probe versus LoRA under label-budget constraints. {forecasting_method}",
        "",
        "![Two-axis framework](two_axis_framework.png)",
        "",
        "Evaluation metrics include foreground IoU for segmentation, weighted RMSE/MAE for forecasting, peak GPU memory, inference/training time, and parameter counts.",
        "",
        "## Experiments",
        "",
        "**Axis 1: Earth observation adaptation**",
        "",
        "- Dataset: FTW Africa, using Kenya, Rwanda, and South Africa field-boundary segmentation data.",
        "- Models: Prithvi-EO-2.0 300M, TerraMind-1.0 Base, TerraMind-1.0 Large.",
        "- Adaptation: Linear Probe and LoRA.",
        "- Label budgets: 5%, 10%, 25%, and 100%.",
        "- Seeds: 3 per configuration.",
        "",
        "**Axis 2: weather forecasting adaptation and model scale**",
        "",
        "- Dataset: WeatherBench2 HRES T0, subset to the Greater Horn of Africa.",
        f"- Models: {forecasting_models}",
        f"- Adaptation: {forecasting_adaptation}",
        "- Initialization times: 64 evenly sampled validation times from January 1 to June 28, 2022.",
        "- Lead times: 6, 12, 24, and 48 hours.",
        "- Variables: 2m temperature, 10m u wind, 10m v wind, and mean sea-level pressure.",
        "",
        "## Results",
        "",
        f"On FTW Africa, the best configuration was **{MODEL_NAMES[best['model']]} + {ADAPTER_NAMES[best['adapter']]}** at {int(best['label_budget'] * 100)}% labels, with foreground IoU {best['foreground_iou_mean']:.3f}, peak memory {best['peak_accelerator_memory_gb_mean']:.2f} GB, and {int(best['total_trainable_params_mean']):,} trainable parameters.",
        "",
        f"A notable deployment result is that **Prithvi Linear Probe at 25% labels** reached foreground IoU {prithvi_linear_25['foreground_iou_mean']:.3f}, close to the 100% label result, while keeping trainable parameters to {int(prithvi_linear_25['total_trainable_params_mean']):,}. Prithvi LoRA at 100% labels used {int(prithvi_lora_100['total_trainable_params_mean']):,} trainable parameters and {prithvi_lora_100['peak_accelerator_memory_gb_mean']:.2f} GB peak memory, but did not dominate the linear probe result.",
        "",
        "![FTW label efficiency](../analysis_l4_v1/ftw_label_efficiency.png)",
        "",
        "![FTW Pareto efficiency](../analysis_l4_v1/ftw_pareto_efficiency.png)",
        "",
        f"On WeatherBench2 Greater Horn, Aurora Small used {small['peak_memory_gb_mean']:.2f} GB peak GPU memory and {int(small['total_params']):,} parameters. Aurora Large used {large['peak_memory_gb_mean']:.2f} GB and {int(large['total_params']):,} parameters. Using mean RMSE relative to the persistence baseline, Aurora Small averaged {small_score['mean_rmse_relative_to_persistence']:.3f}, while Aurora Large averaged {large_score['mean_rmse_relative_to_persistence']:.3f}; lower is better and values below 1 beat persistence.{calibration_sentence}{lora_sentence}",
        "",
        "![Aurora RMSE by lead time](aurora_rmse_by_lead_time.png)",
        "",
        "![Aurora deployment tradeoff](aurora_deployment_tradeoff.png)",
        "",
        "## Discussion",
        "",
        "The FTW results suggest that more adaptive methods are not automatically better. Linear probing, especially with Prithvi, produced a strong Pareto point: high segmentation performance with very low trainable-parameter cost. LoRA increased the adaptation surface and memory footprint, but its value depended strongly on the model and label budget.",
        "",
        f"The WeatherBench2 results sharpen the deployment-cost argument from a forecasting perspective. {forecasting_discussion} Because the forecasting variables have different physical units, we interpret the deployment tradeoff using both variable-specific RMSE curves and RMSE relative to a persistence baseline rather than a raw cross-variable RMSE average.",
        "",
        "Together, the two axes support the central hypothesis: the best foundation model strategy depends on the joint tradeoff between accuracy, available labels, memory, runtime, model scale, and deployment context.",
        "",
        "## Conclusion",
        "",
        "This work presents a deployment-readiness evaluation harness for African climate foundation models. Across Earth observation and weather forecasting, the results show that compact checkpoints and lightweight adaptation can be strong practical baselines. Larger models and heavier adaptation methods should be justified by measured gains, not by scale alone.",
        "",
        "## Limitations",
        "",
        "- The FTW experiment uses an African regional subset rather than the full global FTW dataset.",
        "- The WeatherBench2 experiment evaluates 64 validation initializations over roughly six months, not a full multi-year operational benchmark.",
        "- The Aurora adaptation axis is currently limited to Aurora Small; Aurora Large adaptation is deferred to future work because it is the scale baseline for this lightning-talk design.",
        "- Metrics focus on average error and segmentation quality; extreme-event metrics are still needed.",
        "",
        "## Future Work",
        "",
        "- Extend Aurora adaptation to Aurora Large after the Aurora Small calibration-versus-LoRA comparison is complete.",
        "- Expand WeatherBench2 evaluation to a full validation year and include seasonal breakdowns.",
        "- Add extreme-event metrics for heat, heavy rainfall, wind, and flood-relevant thresholds.",
        "- Add active learning and semi-supervised adaptation for the Earth observation axis.",
        "- Package the harness as a reusable benchmark for African climate foundation models.",
        "",
        "## Artifacts",
        "",
        f"- FTW summary rows: {len(ftw)} aggregated configurations from {ftw_rows} seed-level runs.",
        f"- Aurora forecasting rows: {aurora_rows} from `{AURORA_DIR.name}`.",
    ]

    path = OUTPUT / "lightning_talk_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ftw = pd.read_csv(FTW_SUMMARY)
    aurora = pd.read_csv(AURORA_SUMMARY)
    aurora_overall = pd.read_csv(AURORA_OVERALL)
    figures = [
        save_aurora_rmse_plot(aurora),
        save_aurora_efficiency_plot(aurora),
        save_framework_diagram(),
    ]
    report = write_report(ftw, aurora, aurora_overall, figures)
    print(f"Wrote {report}")
    for figure in figures:
        print(f"Wrote {figure}")


if __name__ == "__main__":
    main()

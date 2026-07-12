from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Aurora results file: {path}")
    return pd.read_json(path, lines=True)


def label(row: pd.Series) -> str:
    adapter = row.get("adapter", "pretrained")
    if row["model"] == "aurora_small" and adapter == "lora":
        return "Aurora Small + LoRA"
    if row["model"] == "aurora_small" and adapter == "linear_calibration":
        return "Aurora Small + Linear Calibration (MOS)"
    if row["model"] == "aurora_small":
        return "Aurora Small"
    if row["model"] == "aurora_large":
        return "Aurora Large"
    return f"{row['model']} + {adapter}"


def summarize(df: pd.DataFrame, persistence: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = df[df["ok"] == True].copy()  # noqa: E712
    if ok.empty:
        raise ValueError("No successful Aurora rows found.")
    optional_columns = [
        "adaptation_train_seconds",
        "calibration_parameter_count",
        "calibration_fit_seconds",
        "lora_train_steps",
        "lora_train_seconds",
        "lora_initial_loss",
        "lora_final_loss",
    ]
    for column in optional_columns:
        if column not in ok.columns:
            ok[column] = None
    ok["adaptation_train_seconds"] = ok["adaptation_train_seconds"].fillna(0.0)
    ok["calibration_parameter_count"] = ok["calibration_parameter_count"].fillna(0)

    group_cols = ["model", "adapter", "variable", "lead_time_hours"]
    summary = (
        ok.groupby(group_cols, as_index=False)
        .agg(
            weighted_rmse_mean=("weighted_rmse", "mean"),
            weighted_rmse_std=("weighted_rmse", "std"),
            weighted_mae_mean=("weighted_mae", "mean"),
            weighted_mae_std=("weighted_mae", "std"),
            inference_seconds_mean=("inference_seconds_for_rollout", "mean"),
            peak_memory_gb_mean=("peak_accelerator_memory_gb", "mean"),
            total_params=("total_params", "first"),
            trainable_params=("trainable_params", "first"),
            trainable_fraction=("trainable_fraction", "first"),
            size=("weighted_rmse", "size"),
            adaptation_train_seconds=("adaptation_train_seconds", "first"),
            calibration_parameter_count=("calibration_parameter_count", "first"),
            calibration_fit_seconds=("calibration_fit_seconds", "first"),
            lora_train_steps=("lora_train_steps", "first"),
            lora_train_seconds=("lora_train_seconds", "first"),
            lora_initial_loss=("lora_initial_loss", "first"),
            lora_final_loss=("lora_final_loss", "first"),
        )
        .fillna({"weighted_rmse_std": 0.0, "weighted_mae_std": 0.0})
    )
    summary["display_name"] = summary.apply(label, axis=1)

    overall = (
        ok.groupby(["model", "adapter"], as_index=False)
        .agg(
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_weighted_mae=("weighted_mae", "mean"),
            inference_seconds_mean=("inference_seconds_for_rollout", "mean"),
            peak_memory_gb_mean=("peak_accelerator_memory_gb", "mean"),
            total_params=("total_params", "first"),
            trainable_params=("trainable_params", "first"),
            trainable_fraction=("trainable_fraction", "first"),
            rows=("weighted_rmse", "size"),
            adaptation_train_seconds=("adaptation_train_seconds", "first"),
            calibration_parameter_count=("calibration_parameter_count", "first"),
            calibration_fit_seconds=("calibration_fit_seconds", "first"),
            lora_train_steps=("lora_train_steps", "first"),
            lora_train_seconds=("lora_train_seconds", "first"),
            lora_initial_loss=("lora_initial_loss", "first"),
            lora_final_loss=("lora_final_loss", "first"),
        )
    )
    overall["display_name"] = overall.apply(label, axis=1)

    scored = summary.copy()
    if persistence is not None and not persistence.empty:
        persistence_summary = (
            persistence.groupby(["variable", "lead_time_hours"], as_index=False)
            .agg(persistence_rmse=("weighted_rmse", "mean"))
        )
        scored = scored.merge(persistence_summary, on=["variable", "lead_time_hours"], how="left")
        if scored["persistence_rmse"].isna().any():
            missing = scored[scored["persistence_rmse"].isna()][["variable", "lead_time_hours"]].drop_duplicates()
            raise ValueError(f"Persistence baseline is missing variable/lead tasks:\n{missing}")
        scored["deployment_score"] = scored["weighted_rmse_mean"] / scored["persistence_rmse"]
        score_name = "mean_rmse_relative_to_persistence"
    else:
        scored["best_rmse_for_task"] = scored.groupby(["variable", "lead_time_hours"])["weighted_rmse_mean"].transform("min")
        scored["deployment_score"] = scored["weighted_rmse_mean"] / scored["best_rmse_for_task"]
        score_name = "mean_relative_rmse_to_best_model"
    deployment = (
        scored.groupby(["model", "adapter"], as_index=False)
        .agg(
            deployment_score=("deployment_score", "mean"),
            inference_seconds_mean=("inference_seconds_mean", "mean"),
            peak_memory_gb_mean=("peak_memory_gb_mean", "mean"),
            total_params=("total_params", "first"),
            trainable_params=("trainable_params", "first"),
            trainable_fraction=("trainable_fraction", "first"),
            adaptation_train_seconds=("adaptation_train_seconds", "first"),
            calibration_parameter_count=("calibration_parameter_count", "first"),
        )
    )
    deployment = deployment.rename(columns={"deployment_score": score_name})
    deployment["display_name"] = deployment.apply(label, axis=1)
    return summary, overall, deployment


def write_report(output_dir: Path, summary: pd.DataFrame, overall: pd.DataFrame, deployment: pd.DataFrame) -> Path:
    score_column = (
        "mean_rmse_relative_to_persistence"
        if "mean_rmse_relative_to_persistence" in deployment.columns
        else "mean_relative_rmse_to_best_model"
    )
    score_label = (
        "mean RMSE relative to persistence"
        if score_column == "mean_rmse_relative_to_persistence"
        else "mean relative RMSE to best model"
    )
    best = deployment.sort_values([score_column, "peak_memory_gb_mean"]).iloc[0]
    lines = [
        "# Aurora WeatherBench2 Greater Horn Results",
        "",
        "## Configurations",
        "",
    ]
    for _, row in overall.sort_values(["model", "adapter"]).iterrows():
        lines.append(
            f"- {row['display_name']}: rows={int(row['rows'])}, "
            f"parameters={int(row['total_params']):,}, trainable={int(row['trainable_params']):,}, "
            f"peak memory={row['peak_memory_gb_mean']:.2f} GB."
        )
    lines.extend(
        [
            "",
            "## Deployment Score",
            "",
            f"{score_label.capitalize()} normalizes each variable/lead-time task; lower is better.",
            "",
        ]
    )
    for _, row in deployment.sort_values(score_column).iterrows():
        lines.append(
            f"- {row['display_name']}: {score_label}={row[score_column]:.3f}, "
            f"peak memory={row['peak_memory_gb_mean']:.2f} GB, trainable params={int(row['trainable_params']):,}."
        )
    lines.extend(
        [
            "",
            f"Best deployment score: **{best['display_name']}**.",
            "",
            "## Files",
            "",
            "- `summary.csv`: variable and lead-time metrics.",
            "- `model_overall.csv`: overall model/adapter metrics.",
            "- `aurora_deployment_scores.csv`: normalized deployment tradeoff table.",
        ]
    )
    path = output_dir / "results_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Aurora WeatherBench2 experiment results.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--persistence-artifact-dir", default="artifacts/aurora_greater_horn_persistence_v2")
    args = parser.parse_args()

    output_dir = Path(args.artifact_dir)
    persistence_path = Path(args.persistence_artifact_dir) / "results.jsonl"
    persistence = load_results(persistence_path) if persistence_path.exists() else None
    df = load_results(output_dir / "results.jsonl")
    summary, overall, deployment = summarize(df, persistence=persistence)
    summary.to_csv(output_dir / "summary.csv", index=False)
    overall.to_csv(output_dir / "model_overall.csv", index=False)
    deployment.to_csv(output_dir / "aurora_deployment_scores.csv", index=False)
    report = write_report(output_dir, summary, overall, deployment)
    status = {
        "artifact_dir": str(output_dir),
        "rows": len(df),
        "successful_rows": int(df["ok"].sum()) if "ok" in df else len(df),
        "summary": str(output_dir / "summary.csv"),
        "overall": str(output_dir / "model_overall.csv"),
        "deployment": str(output_dir / "aurora_deployment_scores.csv"),
        "report": str(report),
    }
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

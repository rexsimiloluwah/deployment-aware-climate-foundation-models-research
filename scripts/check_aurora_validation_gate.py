from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_json(path, lines=True)


def mean_rmse(df: pd.DataFrame, *, model: str | None, adapter: str | None, variable: str, lead: int) -> float:
    subset = df[(df["variable"] == variable) & (df["lead_time_hours"] == lead)].copy()
    if model is not None:
        subset = subset[subset["model"] == model]
    if adapter is not None and "adapter" in subset:
        subset = subset[subset["adapter"] == adapter]
    if subset.empty:
        label = f"model={model}, adapter={adapter}, variable={variable}, lead={lead}"
        raise ValueError(f"No rows found for {label}")
    return float(subset["weighted_rmse"].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fixed Aurora global-input experiment before using it.")
    parser.add_argument("--aurora-artifact-dir", required=True)
    parser.add_argument("--persistence-artifact-dir", required=True)
    parser.add_argument("--t2m-small-max-rmse", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    aurora_dir = Path(args.aurora_artifact_dir)
    persistence_dir = Path(args.persistence_artifact_dir)
    aurora = read_jsonl(aurora_dir / "results.jsonl")
    persistence = read_jsonl(persistence_dir / "results.jsonl")
    aurora_ok = aurora[aurora["ok"] == True].copy() if "ok" in aurora else aurora.copy()  # noqa: E712

    small_t2m_6h = mean_rmse(
        aurora_ok,
        model="aurora_small",
        adapter="pretrained",
        variable="2m_temperature",
        lead=6,
    )
    persistence_t2m_6h = mean_rmse(
        persistence,
        model="persistence_baseline",
        adapter=None,
        variable="2m_temperature",
        lead=6,
    )

    comparable = (
        aurora_ok[
            (aurora_ok["adapter"] == "pretrained")
            & (aurora_ok["model"].isin(["aurora_small", "aurora_large"]))
        ]
        .groupby(["model", "variable", "lead_time_hours"], as_index=False)["weighted_rmse"]
        .mean()
    )
    wide = comparable.pivot_table(
        index=["variable", "lead_time_hours"],
        columns="model",
        values="weighted_rmse",
    ).dropna()
    if wide.empty or {"aurora_small", "aurora_large"} - set(wide.columns):
        raise ValueError("Need both aurora_small and aurora_large pretrained rows for validation.")
    large_beats_small = wide["aurora_large"] < wide["aurora_small"]
    large_win_fraction = float(large_beats_small.mean())

    checks = {
        "small_beats_persistence_t2m_6h": small_t2m_6h < persistence_t2m_6h,
        "large_beats_small_on_most_variable_lead_tasks": large_win_fraction > 0.5,
    }
    if args.t2m_small_max_rmse is not None:
        checks["small_t2m_6h_below_absolute_threshold"] = small_t2m_6h <= args.t2m_small_max_rmse

    result = {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "metrics": {
            "aurora_small_t2m_6h_rmse": small_t2m_6h,
            "persistence_t2m_6h_rmse": persistence_t2m_6h,
            "aurora_large_win_fraction_vs_small": large_win_fraction,
            "num_variable_lead_tasks": int(len(wide)),
        },
        "aurora_artifact_dir": str(aurora_dir),
        "persistence_artifact_dir": str(persistence_dir),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

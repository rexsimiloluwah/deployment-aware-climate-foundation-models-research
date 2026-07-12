from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_harness.config import dataset_config_path, load_yaml
from eval_harness.forecasting import open_forecasting_dataset, weighted_mae, weighted_rmse
from eval_harness.forecasting.regions import first_existing_coord


def parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def variable_or_skip(ds, variable: str):
    if variable in ds:
        return ds[variable]
    compact = variable.replace("_", "")
    for candidate in ds.data_vars:
        if candidate.replace("_", "").lower() == compact.lower():
            return ds[candidate]
    raise KeyError(f"Variable {variable!r} not found. Available variables include: {list(ds.data_vars)[:30]}")


def aurora_matched_init_indices(num_times: int, max_lead_hours: int, max_times: int) -> list[int]:
    """Init-time indices matching the Aurora runner's even sampling, so persistence is the skill
    reference over the SAME period/sample the model is scored on (not just the first N steps).

    Aurora builds each batch from input steps (start_idx, start_idx+1) and verifies against
    start_idx+1+step, so the forecast's initial state is at index start_idx+1. We return those
    start_idx+1 indices here so persistence carries forward the same states at the same times.
    """
    max_steps = max_lead_hours // 6
    max_start_exclusive = num_times - max_steps - 1
    if max_start_exclusive <= 0:
        raise ValueError(f"Not enough time points ({num_times}) for max rollout steps ({max_steps}).")
    n = min(max_times, max_start_exclusive)
    start_indices = sorted({int(round(value)) for value in np.linspace(0, max_start_exclusive - 1, n)})
    return [start_idx + 1 for start_idx in start_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a persistence baseline on the forecasting dataset.")
    parser.add_argument("--dataset", default="weatherbench2_hres_t0_greater_horn")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--variables", default="2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind")
    parser.add_argument("--lead-times-hours", default="6,12,24,48")
    parser.add_argument(
        "--max-times",
        type=int,
        default=64,
        help="Number of evenly-sampled initialization times, matching the Aurora runner.",
    )
    parser.add_argument("--run-label", default="aurora_greater_horn_persistence_v1")
    args = parser.parse_args()

    cfg = load_yaml(dataset_config_path(args.dataset))
    ds = open_forecasting_dataset(cfg, split=args.split, subset=True)
    time_coord = cfg.get("data", {}).get("time_coord", "time")
    lat_coord = first_existing_coord(ds, cfg.get("data", {}).get("latitude_coord_candidates", ["latitude", "lat"]))

    variables = parse_csv(args.variables)
    lead_times = parse_csv(args.lead_times_hours, int)
    # Sample the same even initialization times over the whole split that the Aurora runner uses,
    # so persistence is a period-matched reference rather than a mid-January-only baseline.
    init_indices = aurora_matched_init_indices(
        num_times=int(ds.sizes[time_coord]),
        max_lead_hours=max(lead_times),
        max_times=args.max_times,
    )
    rows = []
    output_dir = ROOT / "artifacts" / args.run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for variable in variables:
        da = variable_or_skip(ds, variable)
        for lead in lead_times:
            step = lead // 6
            # Score each initialization separately, then average, to match the Aurora runner's
            # aggregation (mean of per-init RMSE, not a pooled RMSE over all times at once).
            per_init_rmse = []
            per_init_mae = []
            for idx in init_indices:
                prediction = da.isel({time_coord: idx})
                target = da.isel({time_coord: idx + step})
                per_init_rmse.append(weighted_rmse(prediction, target, latitude_coord=lat_coord))
                per_init_mae.append(weighted_mae(prediction, target, latitude_coord=lat_coord))
            row = {
                "dataset": args.dataset,
                "region": cfg["region"],
                "model": "persistence_baseline",
                "adapter": "none",
                "variable": variable,
                "lead_time_hours": lead,
                "split": args.split,
                "num_initialization_times": len(init_indices),
                "weighted_rmse": float(np.mean(per_init_rmse)),
                "weighted_mae": float(np.mean(per_init_mae)),
                "ok": True,
            }
            rows.append(row)

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    summary = {
        "run_label": args.run_label,
        "dataset": args.dataset,
        "split": args.split,
        "variables": variables,
        "lead_times_hours": lead_times,
        "initialization_sampling": "even",
        "num_initialization_times": len(init_indices),
        "wall_seconds": time.perf_counter() - started,
        "results_path": str(results_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

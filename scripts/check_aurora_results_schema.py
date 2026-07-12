from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COMMON_REQUIRED = {
    "ok",
    "dataset",
    "region",
    "model",
    "adapter",
    "variable",
    "lead_time_hours",
    "initialization_index",
    "initialization_sample_order",
    "initialization_time",
    "inference_seconds_for_rollout",
    "peak_accelerator_memory_gb",
    "device",
    "precision",
    "adapter_implementation",
    "checkpoint_file",
    "model_load_seconds",
    "total_params",
    "trainable_params",
    "trainable_fraction",
    "adaptation_train_seconds",
    "weighted_rmse",
    "weighted_mae",
}
LORA_REQUIRED = {"lora_train_steps", "lora_train_seconds", "lora_initial_loss", "lora_final_loss"}
CALIBRATION_REQUIRED = {"calibration_parameter_count", "calibration_fit_seconds", "calibration_parameters_path"}


def check_row(row: dict, line_number: int) -> list[str]:
    errors = []
    missing = sorted(COMMON_REQUIRED - set(row))
    if missing:
        errors.append(f"line {line_number}: missing common fields {missing}")
    nulls = sorted(key for key in COMMON_REQUIRED if key in row and row[key] is None)
    if nulls:
        errors.append(f"line {line_number}: null common fields {nulls}")
    for metric in ("weighted_rmse", "weighted_mae"):
        if metric in row:
            try:
                finite = math.isfinite(float(row[metric]))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                errors.append(f"line {line_number}: non-finite {metric}={row[metric]}")
    if row.get("adapter") == "lora":
        missing_lora = sorted(LORA_REQUIRED - set(row))
        null_lora = sorted(key for key in LORA_REQUIRED if key in row and row[key] is None)
        if missing_lora or null_lora:
            errors.append(f"line {line_number}: incomplete LoRA fields missing={missing_lora} null={null_lora}")
    if row.get("adapter") == "linear_calibration":
        missing_cal = sorted(CALIBRATION_REQUIRED - set(row))
        null_cal = sorted(key for key in CALIBRATION_REQUIRED if key in row and row[key] is None)
        if missing_cal or null_cal:
            errors.append(
                f"line {line_number}: incomplete Linear Calibration (MOS) fields "
                f"missing={missing_cal} null={null_cal}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Aurora result JSONL deployment schema.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--require-adapters", default="")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise FileNotFoundError(path)

    errors = []
    adapters = set()
    rows = 0
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        rows += 1
        row = json.loads(line)
        adapters.add(row.get("adapter"))
        errors.extend(check_row(row, line_number))

    required_adapters = {item.strip() for item in args.require_adapters.split(",") if item.strip()}
    missing_adapters = sorted(required_adapters - adapters)
    if missing_adapters:
        errors.append(f"missing required adapters {missing_adapters}; observed {sorted(adapters)}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps({"results": str(path), "rows": rows, "adapters": sorted(adapters), "ok": True}, indent=2))


if __name__ == "__main__":
    main()

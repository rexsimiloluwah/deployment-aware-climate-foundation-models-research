from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_row(row: dict) -> dict:
    normalized = dict(row)
    adapter = normalized.get("adapter", "pretrained")
    normalized.setdefault("adapter", adapter)
    normalized.setdefault("adaptation_train_seconds", 0.0)
    normalized.setdefault("calibration_parameter_count", 0)
    normalized.setdefault("calibration_fit_seconds", None)
    if adapter == "linear_calibration":
        normalized.setdefault("adapter_implementation", "linear_calibration_mos")
        normalized["trainable_params"] = int(normalized.get("calibration_parameter_count", 0))
        total_params = float(normalized.get("total_params", 0))
        normalized["trainable_fraction"] = float(normalized["trainable_params"] / total_params) if total_params else 0.0
    return normalized


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def copy_supporting_files(adapted_dir: Path, output_dir: Path) -> None:
    for name in [
        "aurora_small_linear_calibration_mos.json",
        "linear_calibration_training_log.jsonl",
        "aurora_small_lora_adapter.pt",
        "lora_checkpoint_load_audit.json",
        "lora_training_log.jsonl",
    ]:
        source = adapted_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Aurora pretrained baselines with adapted Aurora rows.")
    parser.add_argument("--pretrained-dir", default="artifacts/aurora_greater_horn_global_v1")
    parser.add_argument("--adapted-dir", default="artifacts/aurora_greater_horn_adapted_v1")
    parser.add_argument("--output-dir", default="artifacts/aurora_greater_horn_final_v1")
    args = parser.parse_args()

    pretrained_dir = Path(args.pretrained_dir)
    adapted_dir = Path(args.adapted_dir)
    output_dir = Path(args.output_dir)
    pretrained_rows = [normalize_row(row) for row in read_jsonl(pretrained_dir / "results.jsonl")]
    adapted_rows = [normalize_row(row) for row in read_jsonl(adapted_dir / "results.jsonl")]

    rows = pretrained_rows + adapted_rows
    write_jsonl(output_dir / "results.jsonl", rows)
    copy_supporting_files(adapted_dir, output_dir)
    summary = {
        "pretrained_dir": str(pretrained_dir),
        "adapted_dir": str(adapted_dir),
        "output_dir": str(output_dir),
        "pretrained_rows": len(pretrained_rows),
        "adapted_rows": len(adapted_rows),
        "total_rows": len(rows),
        "adapters": sorted({row.get("adapter") for row in rows}),
    }
    (output_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_harness.config import dataset_config_path, load_yaml, model_config_path
from eval_harness.forecasting import check_aurora_readiness, summarize_forecasting_dataset


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Aurora forecasting-axis readiness.")
    parser.add_argument("--dataset", default="weatherbench2_hres_t0_greater_horn")
    parser.add_argument("--models", default="aurora_small,aurora_large")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_cfg = load_yaml(dataset_config_path(args.dataset))
    model_cfgs = [load_yaml(model_config_path(name)) for name in parse_csv(args.models)]

    readiness = {
        "dataset": summarize_forecasting_dataset(dataset_cfg, split=args.split).as_dict(),
        "models": [check_aurora_readiness(cfg).as_dict() for cfg in model_cfgs],
    }

    text = json.dumps(readiness, indent=2, default=str)
    print(text)
    if args.output:
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

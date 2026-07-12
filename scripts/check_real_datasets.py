from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_harness.config import dataset_config_path, load_yaml
from eval_harness.datasets import DatasetNotReadyError, build_dataset
from eval_harness.exploration import dataset_overview, sample_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=["sen1floods11_ghana", "ftw_africa"])
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    any_not_ready = False
    for dataset_name in args.datasets:
        cfg = load_yaml(dataset_config_path(dataset_name))
        print(f"\nDATASET: {dataset_name}")
        try:
            ds = build_dataset(cfg, split=args.split)
        except DatasetNotReadyError as exc:
            any_not_ready = True
            print("NOT READY")
            print(exc)
            continue
        print("READY")
        print(dataset_overview(cfg, ds).to_string(index=False))
        print(sample_table(ds, n=5).to_string(index=False))
    if any_not_ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_harness.config import dataset_config_path, load_yaml
from eval_harness.datasets import build_dataset
from eval_harness.exploration import (
    classification_label_distribution,
    dataset_overview,
    label_budget_summary,
    sample_table,
    segmentation_pixel_distribution,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["sen1floods11_ghana", "ftw_africa"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--smoke-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    cfg = load_yaml(dataset_config_path(args.dataset))
    ds = build_dataset(cfg, args.split, seed=args.seed, smoke_size=args.smoke_size)
    print("\nOVERVIEW")
    print(dataset_overview(cfg, ds).to_string(index=False))
    print("\nSAMPLES")
    print(sample_table(ds).to_string(index=False))
    print("\nDISTRIBUTION")
    if cfg["task_type"] == "classification":
        print(classification_label_distribution(ds, cfg.get("class_names")).to_string(index=False))
    elif cfg["task_type"] == "segmentation":
        print(segmentation_pixel_distribution(ds, cfg.get("class_names")).to_string(index=False))
    else:
        print("Unlabeled inspection dataset: no label distribution.")
    print("\nLABEL BUDGETS")
    print(label_budget_summary(ds, cfg, [0.05, 0.10, 0.25, 1.00], seed=args.seed).to_string(index=False))
    if cfg["task_type"] == "classification":
        print("\nSTRATIFIED LABEL BUDGETS")
        print(
            label_budget_summary(ds, cfg, [0.05, 0.10, 0.25, 1.00], seed=args.seed, strategy="stratified").to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()

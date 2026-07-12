from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_harness.config import load_yaml
from eval_harness.training import run_local_segmentation_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="config/experiment.local.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    experiment = load_yaml(args.experiment)
    run_id = args.run_id or experiment["run_id"]
    out_dir = Path("artifacts") / run_id / "tiny_segmentation_cnn_local"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    for dataset_name in experiment["datasets"]:
        for method in experiment["adaptation"]["methods"]:
            for budget in experiment["label_budgets"]:
                for seed in experiment["seeds"]:
                    result = run_local_segmentation_experiment(
                        dataset_name=dataset_name,
                        method=method,
                        label_budget=float(budget),
                        seed=int(seed),
                        batch_size=int(experiment.get("batch_size", 2)),
                        image_size=int(experiment.get("image_size", 128)),
                        max_epochs=int(experiment.get("max_epochs", 1)),
                        limit_train_batches=experiment.get("limit_train_batches", 8),
                        limit_val_batches=experiment.get("limit_val_batches", 8),
                        device=experiment.get("device", "auto"),
                    )
                    result["run_id"] = run_id
                    with results_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(result) + "\n")
                    print(result)

    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()

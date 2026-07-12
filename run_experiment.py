from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_harness.config import dataset_config_path, load_yaml
from eval_harness.readiness import check_experiment_ready
from eval_harness.training import run_smoke_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="config/experiment.validation.yaml")
    parser.add_argument("--model", default="smoke_toy")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-readiness-check", action="store_true")
    args = parser.parse_args()

    experiment = load_yaml(args.experiment)
    run_id = args.run_id or experiment["run_id"]
    if args.model != "smoke_toy" and not args.skip_readiness_check:
        results = check_experiment_ready(experiment, load_weights=False)
        failed = [result for result in results if not result.ready]
        if failed:
            messages = "\n".join(f"- {result.kind} {result.name}: {result.error}" for result in failed)
            raise SystemExit(
                "Experiment is not ready. Run `uv run python scripts/check_experiment_readiness.py`.\n"
                + messages
            )
    if args.model != "smoke_toy":
        raise SystemExit(
            f"Model {args.model!r} is configured but not trainable yet. "
            "The real checkpoint smoke test is implemented; the next step is model-family-specific forward/adaptation code."
        )
    out_dir = Path("artifacts") / run_id / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    seed = int(experiment.get("seed", 1234))
    batch_size = int(experiment.get("batch_size", 8))
    for dataset_name in experiment["datasets"]:
        dataset_cfg = load_yaml(dataset_config_path(dataset_name))
        for method in experiment["adaptation"]["methods"]:
            for label_budget in experiment["label_budgets"]:
                result = run_smoke_experiment(dataset_cfg, method, float(label_budget), seed, batch_size=batch_size)
                result.update({"model": args.model, "run_id": run_id})
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                print(result)


if __name__ == "__main__":
    main()

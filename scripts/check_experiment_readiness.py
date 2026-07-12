from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_harness.config import load_yaml
from eval_harness.readiness import check_experiment_ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="config/experiment.yaml")
    parser.add_argument("--load-model-weights", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    experiment = load_yaml(args.experiment)
    results = check_experiment_ready(experiment, load_weights=args.load_model_weights)
    if args.json:
        print(json.dumps([result.as_dict() for result in results], indent=2))
    else:
        for result in results:
            status = "READY" if result.ready else "NOT READY"
            print(f"{result.kind.upper():7s} {result.name:28s} {status}")
            if result.error:
                print(f"  {result.error}")
            elif result.kind == "model":
                print(f"  repo={result.details.get('hf_repo')} weights={result.details.get('weight_files')}")
            elif result.kind == "dataset":
                for split, info in result.details.get("splits", {}).items():
                    print(f"  {split}: n={info['num_examples']} x={info['x_shape']} y={info['y_shape']}")
    if not all(result.ready for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()


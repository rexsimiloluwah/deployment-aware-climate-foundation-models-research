from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_harness.datasets import make_label_budget_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.25, 1.00])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="artifacts/label_splits.json")
    args = parser.parse_args()

    splits = {str(budget): make_label_budget_indices(args.n, budget, args.seed) for budget in args.budgets}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


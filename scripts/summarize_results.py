from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import load_results, summarize_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl")
    args = parser.parse_args()
    summary = summarize_results(load_results(args.results_jsonl))
    print(summary.to_string(index=False) if not summary.empty else "No results found.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_harness.config import load_yaml, model_config_path
from eval_harness.models import check_foundation_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["prithvi_eo_v2_300m", "terramind_v1_base", "terramind_v1_large"])
    parser.add_argument("--load-weights", action="store_true")
    args = parser.parse_args()

    load_weights = args.load_weights or os.environ.get("LOAD_MODEL_WEIGHTS") == "1"
    for model_name in args.models:
        cfg = load_yaml(model_config_path(model_name))
        result = check_foundation_model(cfg, load_weights=load_weights)
        print(result.as_dict())


if __name__ == "__main__":
    main()

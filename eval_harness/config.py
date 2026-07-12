from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def dataset_config_path(name: str) -> Path:
    return PROJECT_ROOT / "config" / "datasets" / f"{name}.yaml"


def model_config_path(name: str) -> Path:
    return PROJECT_ROOT / "config" / "models" / f"{name}.yaml"


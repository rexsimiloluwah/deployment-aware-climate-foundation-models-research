from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eval_harness.config import dataset_config_path, load_yaml, model_config_path
from eval_harness.datasets import DatasetNotReadyError, build_dataset
from eval_harness.models import check_foundation_model


@dataclass
class CheckResult:
    name: str
    kind: str
    ready: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "ready": self.ready,
            "details": self.details,
            "error": self.error,
        }


def check_dataset_ready(dataset_name: str, splits: list[str] | None = None) -> CheckResult:
    cfg = load_yaml(dataset_config_path(dataset_name))
    splits = splits or ["train", "val", "test"]
    details: dict[str, Any] = {"splits": {}}
    try:
        for split in splits:
            dataset = build_dataset(cfg, split=split)
            sample = dataset[0]
            details["splits"][split] = {
                "num_examples": len(dataset),
                "sample_id": sample["id"],
                "x_shape": list(sample["x"].shape),
                "y_shape": list(getattr(sample["y"], "shape", ())),
            }
    except (DatasetNotReadyError, FileNotFoundError, ValueError, ImportError) as exc:
        return CheckResult(name=dataset_name, kind="dataset", ready=False, details=details, error=str(exc))
    return CheckResult(name=dataset_name, kind="dataset", ready=True, details=details)


def check_model_ready(model_name: str, load_weights: bool = False) -> CheckResult:
    cfg = load_yaml(model_config_path(model_name))
    try:
        result = check_foundation_model(cfg, load_weights=load_weights).as_dict()
    except Exception as exc:
        return CheckResult(name=model_name, kind="model", ready=False, error=str(exc))
    ready = bool(result["access_ok"])
    if load_weights:
        ready = ready and bool(result["checkpoint_load_ok"])
    return CheckResult(name=model_name, kind="model", ready=ready, details=result)


def check_experiment_ready(experiment_cfg: dict, load_weights: bool = False) -> list[CheckResult]:
    results: list[CheckResult] = []
    for dataset_name in experiment_cfg["datasets"]:
        results.append(check_dataset_ready(dataset_name))
    for model_name in experiment_cfg["models"]:
        results.append(check_model_ready(model_name, load_weights=load_weights))
    return results

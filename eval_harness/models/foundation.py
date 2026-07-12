from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download


@dataclass(frozen=True)
class FoundationModelSmokeResult:
    model: str
    display_name: str
    hf_repo: str
    is_foundation_model: bool
    access_ok: bool
    config_files: list[str]
    weight_files: list[str]
    parameter_count_estimate: int | None
    checkpoint_size_estimate: str | None
    load_weights: bool
    local_files: list[str]
    checkpoint_load_ok: bool | None
    checkpoint_summary: dict[str, Any] | None
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "display_name": self.display_name,
            "hf_repo": self.hf_repo,
            "is_foundation_model": self.is_foundation_model,
            "access_ok": self.access_ok,
            "config_files": self.config_files,
            "weight_files": self.weight_files,
            "parameter_count_estimate": self.parameter_count_estimate,
            "checkpoint_size_estimate": self.checkpoint_size_estimate,
            "load_weights": self.load_weights,
            "local_files": self.local_files,
            "checkpoint_load_ok": self.checkpoint_load_ok,
            "checkpoint_summary": self.checkpoint_summary,
            "notes": self.notes,
        }


def smoke_check_foundation_model(model_cfg: dict, load_weights: bool = False) -> FoundationModelSmokeResult:
    """Check that a foundation-model checkpoint is reachable.

    By default this avoids downloading large weights. Set load_weights=True on a GPU VM
    when you intentionally want the heavyweight smoke test.
    """

    repo_id = model_cfg["hf_repo"]
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id)
    config_files = sorted(
        file
        for file in files
        if file.endswith((".json", ".yaml", ".yml", ".py", ".txt", ".md")) and not file.startswith(".")
    )
    weight_files = sorted(
        file for file in files if file.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt"))
    )
    local_files: list[str] = []
    for file in config_files[:8]:
        try:
            local_files.append(hf_hub_download(repo_id=repo_id, filename=file))
        except Exception:
            # Some repos include code files referenced by metadata but not directly downloadable without trust settings.
            continue
    checkpoint_load_ok: bool | None = None
    checkpoint_summary: dict[str, Any] | None = None
    if load_weights:
        expected_file = model_cfg.get("checkpoint_file")
        files_to_load = [expected_file] if expected_file else weight_files
        for file in files_to_load:
            local_path = hf_hub_download(repo_id=repo_id, filename=file)
            local_files.append(local_path)
            checkpoint_summary = inspect_torch_checkpoint(Path(local_path))
            checkpoint_load_ok = True
    notes = (
        "Metadata/config smoke test only. Set LOAD_MODEL_WEIGHTS=1 on a GPU VM to download checkpoint weights."
        if not load_weights
        else "Checkpoint weight files downloaded. Add model-family-specific constructors before training."
    )
    return FoundationModelSmokeResult(
        model=model_cfg["model"],
        display_name=model_cfg["display_name"],
        hf_repo=repo_id,
        is_foundation_model=bool(model_cfg["is_foundation_model"]),
        access_ok=True,
        config_files=config_files,
        weight_files=weight_files,
        parameter_count_estimate=model_cfg.get("parameter_count_estimate"),
        checkpoint_size_estimate=model_cfg.get("checkpoint_size_estimate"),
        load_weights=load_weights,
        local_files=[str(Path(path)) for path in local_files],
        checkpoint_load_ok=checkpoint_load_ok,
        checkpoint_summary=checkpoint_summary,
        notes=notes,
    )


def check_foundation_model(model_cfg: dict, load_weights: bool = False) -> FoundationModelSmokeResult:
    """Public alias for checking real foundation-model checkpoint availability."""

    return smoke_check_foundation_model(model_cfg, load_weights=load_weights)


def inspect_torch_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        tensor_count = 0
        parameter_count = 0
        for value in checkpoint.values():
            if torch.is_tensor(value):
                tensor_count += 1
                parameter_count += value.numel()
            elif isinstance(value, dict):
                for nested_value in value.values():
                    if torch.is_tensor(nested_value):
                        tensor_count += 1
                        parameter_count += nested_value.numel()
        return {
            "checkpoint_type": "dict",
            "top_level_keys": keys[:25],
            "num_top_level_keys": len(keys),
            "tensor_count": tensor_count,
            "parameter_count_from_tensors": parameter_count or None,
        }
    if torch.is_tensor(checkpoint):
        return {
            "checkpoint_type": "tensor",
            "shape": list(checkpoint.shape),
            "parameter_count_from_tensors": checkpoint.numel(),
        }
    return {
        "checkpoint_type": type(checkpoint).__name__,
        "parameter_count_from_tensors": None,
    }

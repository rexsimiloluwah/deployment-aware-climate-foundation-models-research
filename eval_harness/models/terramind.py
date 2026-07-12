from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


TERRAMIND_BACKBONES = {
    "terramind_v1_base": "terramind_v1_base",
    "terramind_v1_large": "terramind_v1_large",
}


@dataclass(frozen=True)
class TerraMindLoadResult:
    model: nn.Module
    model_name: str
    backbone_name: str
    modalities: tuple[str, ...]
    bands: dict[str, list[str]]


def load_terramind_backbone(
    model_name: str,
    *,
    pretrained: bool = True,
    device: str | torch.device = "cpu",
    modalities: tuple[str, ...] = ("S2L2A",),
    merge_method: str = "mean",
    freeze: bool = True,
) -> TerraMindLoadResult:
    """Load a TerraMind backbone through TerraTorch's registry."""

    try:
        from terratorch import BACKBONE_REGISTRY
    except ImportError as exc:
        raise ImportError(
            "TerraMind requires TerraTorch. Run `uv sync --extra geo --extra gcp`, "
            "then restart the active notebook kernel."
        ) from exc

    if model_name not in TERRAMIND_BACKBONES:
        raise ValueError(f"Unsupported TerraMind model {model_name!r}")

    backbone_name = TERRAMIND_BACKBONES[model_name]
    bands = {"S2L2A": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]}
    model = BACKBONE_REGISTRY.build(
        backbone_name,
        pretrained=pretrained,
        modalities=list(modalities),
        bands=bands,
        merge_method=merge_method,
    )
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.eval()
    model.to(device)
    return TerraMindLoadResult(
        model=model,
        model_name=model_name,
        backbone_name=backbone_name,
        modalities=modalities,
        bands=bands,
    )


def adapt_batch_to_terramind(dataset_name: str, x: torch.Tensor, image_size: int = 224) -> dict[str, torch.Tensor]:
    if dataset_name == "sen1floods11_ghana":
        return {"S2L2A": adapt_sen1floods11_to_s2l2a(x, image_size=image_size)}
    if dataset_name == "ftw_africa":
        return {"S2L2A": adapt_ftw_to_s2l2a(x, image_size=image_size)}
    raise ValueError(f"No TerraMind adapter is configured for dataset {dataset_name!r}")


def adapt_sen1floods11_to_s2l2a(x: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Map Sen1Floods11 Sentinel-2 tensors to a 6-band S2L2A-like TerraMind input."""

    if x.ndim != 4 or x.shape[1] < 13:
        raise ValueError(f"Expected Sen1Floods11 tensor with shape [B, 13, H, W], got {tuple(x.shape)}")
    x = F.interpolate(x.float(), size=(image_size, image_size), mode="bilinear", align_corners=False)
    # B02, B03, B04, B08, B11, B12 from common S2 ordering.
    selected = x[:, [1, 2, 3, 7, 11, 12]]
    return selected


def adapt_ftw_to_s2l2a(x: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Map FTW PlanetScope windows to a 6-band S2L2A-like TerraMind input.

    FTW has B/G/R/NIR for two windows. TerraMind S2L2A expects Sentinel-2 style
    bands, so this is a compatibility adapter for pipeline testing: use window B
    as BLUE/GREEN/RED/NIR_NARROW and fill unavailable SWIR bands with 0.
    """

    if x.ndim != 4 or x.shape[1] != 8:
        raise ValueError(f"Expected FTW tensor with shape [B, 8, H, W], got {tuple(x.shape)}")
    x = F.interpolate(x.float(), size=(image_size, image_size), mode="bilinear", align_corners=False)
    window_b = x[:, 4:8] * 10000.0
    out = torch.zeros((x.shape[0], 6, image_size, image_size), dtype=x.dtype, device=x.device)
    out[:, 0] = window_b[:, 0]  # BLUE
    out[:, 1] = window_b[:, 1]  # GREEN
    out[:, 2] = window_b[:, 2]  # RED
    out[:, 3] = window_b[:, 3]  # NIR_NARROW
    return out


def terramind_feature_map(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    model = _unwrap_peft_model(model)
    output = model(inputs)
    output = _select_feature_tensor(output)
    if output.ndim == 4:
        return output
    if output.ndim != 3:
        raise ValueError(f"Expected TerraMind output [B, tokens, channels] or [B, C, H, W], got {tuple(output.shape)}")
    b, tokens, channels = output.shape
    side = int(tokens**0.5)
    if side * side != tokens:
        raise ValueError(f"Cannot reshape {tokens} TerraMind tokens into a square feature map")
    return output.transpose(1, 2).reshape(b, channels, side, side)


def _select_feature_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        tensors = [_select_feature_tensor(value) for value in output.values()]
        return _last_feature_tensor(tensors)
    if isinstance(output, (list, tuple)):
        tensors = [_select_feature_tensor(value) for value in output]
        return _last_feature_tensor(tensors)
    raise TypeError(f"Could not find a TerraMind tensor output in {type(output).__name__}")


def _last_feature_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("TerraMind returned an empty feature collection")
    preferred = [tensor for tensor in tensors if tensor.ndim in {3, 4}]
    return preferred[-1] if preferred else tensors[-1]


def apply_terramind_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
) -> nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("Real TerraMind LoRA requires `peft`. Run `uv sync`.") from exc

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=list(target_modules),
    )
    return get_peft_model(model, config)


def _unwrap_peft_model(model: nn.Module) -> nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        return get_base_model()
    return model

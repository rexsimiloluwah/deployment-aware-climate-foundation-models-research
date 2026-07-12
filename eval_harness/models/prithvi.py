from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import functional as F


PRITHVI_REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
PRITHVI_CHECKPOINT_FILE = "Prithvi_EO_V2_300M.pt"


@dataclass(frozen=True)
class PrithviLoadResult:
    model: nn.Module
    config: dict[str, Any]
    config_path: Path
    code_path: Path
    checkpoint_path: Path | None
    missing_keys: list[str]
    unexpected_keys: list[str]


def load_prithvi_eo_v2_300m(
    *,
    load_weights: bool = True,
    cache_dir: str | Path = "data/hf_cache/models",
    device: str | torch.device = "cpu",
    freeze: bool = True,
) -> PrithviLoadResult:
    """Load the real Prithvi-EO-2.0 300M architecture and optionally its HF checkpoint."""

    cache_dir = Path(cache_dir)
    config_path = Path(hf_hub_download(PRITHVI_REPO_ID, "config.json", cache_dir=cache_dir))
    code_path = Path(hf_hub_download(PRITHVI_REPO_ID, "prithvi_mae.py", cache_dir=cache_dir))

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    pretrained_cfg = dict(config["pretrained_cfg"])
    prithvi_module = _load_prithvi_module(code_path)
    model = prithvi_module.PrithviMAE(**pretrained_cfg)

    checkpoint_path = None
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    if load_weights:
        checkpoint_path = Path(
            hf_hub_download(PRITHVI_REPO_ID, PRITHVI_CHECKPOINT_FILE, cache_dir=cache_dir)
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.eval()
    model.to(device)
    return PrithviLoadResult(
        model=model,
        config=config,
        config_path=config_path,
        code_path=code_path,
        checkpoint_path=checkpoint_path,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


def apply_prithvi_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
) -> nn.Module:
    """Attach real PEFT LoRA adapters to Prithvi transformer layers."""

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("Real LoRA requires `peft`. Run `uv sync` in this project environment.") from exc

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=list(target_modules),
    )
    return get_peft_model(model, config)


def lora_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
    }


def adapt_ftw_batch_to_prithvi(x: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Map FTW's two 4-band PlanetScope windows to Prithvi's 6-band, 4-frame input.

    FTW provides window_a/window_b as blue, green, red, and NIR-like channels after
    division by 10000. Prithvi was trained on HLS DNs for B02, B03, B04, B05, B06,
    and B07 across four frames. For a Mac smoke test we keep the real tensors but
    use a pragmatic compatibility adapter: repeat the two observed windows to four
    frames, map B/G/R/NIR to B02/B03/B04/B05, and fill SWIR channels with zeros.
    """

    if x.ndim != 4 or x.shape[1] != 8:
        raise ValueError(f"Expected FTW tensor with shape [B, 8, H, W], got {tuple(x.shape)}")

    x = F.interpolate(x.float(), size=(image_size, image_size), mode="bilinear", align_corners=False)
    x_dn = x * 10000.0
    window_a = x_dn[:, 0:4]
    window_b = x_dn[:, 4:8]
    frames_4band = torch.stack([window_a, window_b, window_a, window_b], dim=2)

    out = torch.zeros(
        (x.shape[0], 6, 4, image_size, image_size),
        dtype=x.dtype,
        device=x.device,
    )
    out[:, 0:4] = frames_4band
    return normalize_prithvi_hls(out)


def adapt_sen1floods11_batch_to_prithvi(x: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Map Sen1Floods11 Sentinel-2 scenes to Prithvi's 6-band, 4-frame input.

    Sen1Floods11 Sentinel-2 tensors are treated as common Sentinel-2 band order:
    B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12.
    Prithvi expects B02, B03, B04, B05, B06, B07, so this adapter selects
    channels 1 through 6 and repeats the single scene across four frames.
    """

    if x.ndim != 4 or x.shape[1] < 7:
        raise ValueError(f"Expected Sen1Floods11 tensor with shape [B, >=7, H, W], got {tuple(x.shape)}")

    x = F.interpolate(x.float(), size=(image_size, image_size), mode="bilinear", align_corners=False)
    six_bands = x[:, 1:7]
    out = torch.stack([six_bands, six_bands, six_bands, six_bands], dim=2)
    return normalize_prithvi_hls(out)


def adapt_batch_to_prithvi(dataset_name: str, x: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    if dataset_name == "ftw_africa":
        return adapt_ftw_batch_to_prithvi(x, image_size=image_size)
    if dataset_name == "sen1floods11_ghana":
        return adapt_sen1floods11_batch_to_prithvi(x, image_size=image_size)
    raise ValueError(f"No Prithvi adapter is configured for dataset {dataset_name!r}")


def normalize_prithvi_hls(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], device=x.device)
    std = torch.tensor([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], device=x.device)
    return (x - mean.view(1, 6, 1, 1, 1)) / std.view(1, 6, 1, 1, 1)


def prithvi_feature_map(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the last Prithvi encoder feature as an image-like tensor."""

    feature_model = _unwrap_peft_model(model)
    features = feature_model.forward_features(x)
    image_features = feature_model.encoder.prepare_features_for_image_model([features[-1]])
    return image_features[-1]


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _unwrap_peft_model(model: nn.Module) -> nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        return get_base_model()
    return model


def _load_prithvi_module(code_path: Path):
    module_name = "_eval_harness_prithvi_mae"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Prithvi code from {code_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

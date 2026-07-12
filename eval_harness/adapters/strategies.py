from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def apply_adaptation_strategy(model: nn.Module, method: str) -> nn.Module:
    if method == "linear_probe":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("head.")
        return model
    if method == "lora":
        # Smoke-model placeholder: keep encoder trainable until EFM-specific LoRA targets are wired.
        for param in model.parameters():
            param.requires_grad = True
        return model
    raise ValueError(f"Unknown adaptation method: {method}")


def adapter_metadata(method: str, is_foundation_model: bool = False) -> dict[str, object]:
    if method == "linear_probe":
        return {
            "adapter_implementation": "linear_probe",
            "adapter_is_placeholder": False,
        }
    if method == "lora":
        return {
            "adapter_implementation": "lora" if is_foundation_model else "full_model_trainable_smoke_placeholder",
            "adapter_is_placeholder": not is_foundation_model,
            "adapter_note": (
                "Real LoRA requires a foundation-model backbone with target modules. "
                "For the toy smoke model, this placeholder simply makes all parameters trainable."
            )
            if not is_foundation_model
            else "LoRA adapter applied to foundation-model target modules.",
        }
    raise ValueError(f"Unknown adaptation method: {method}")


def memory_summary(device: str) -> dict[str, object]:
    if torch.cuda.is_available():
        return {
            "device": device,
            "cuda_available": True,
            "mps_available": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "memory_backend": "cuda",
            "memory_note": "peak_vram_gb is reported from torch.cuda.max_memory_allocated().",
        }
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    return {
        "device": device,
        "cuda_available": False,
        "mps_available": mps_available,
        "peak_vram_gb": None,
        "memory_backend": "mps" if mps_available and device == "mps" else "cpu",
        "memory_note": "peak_vram_gb is None because CUDA is not available. PyTorch only reports this VRAM metric through CUDA.",
    }

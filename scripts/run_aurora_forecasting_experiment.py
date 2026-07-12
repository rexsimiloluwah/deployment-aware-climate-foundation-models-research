from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_harness.config import dataset_config_path, load_yaml, model_config_path
from eval_harness.forecasting import open_forecasting_dataset
from eval_harness.forecasting.regions import RegionBounds
from eval_harness.forecasting.weatherbench import resolve_static_uri


SURFACE_TO_AURORA = {
    "2m_temperature": "2t",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "mean_sea_level_pressure": "msl",
}
ATMOS_TO_AURORA = {
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "specific_humidity": "q",
}
REQUIRED_AURORA_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
VALID_ADAPTERS = {"pretrained", "lora", "linear_calibration"}
COMMON_RESULT_FIELDS = {
    "ok",
    "dataset",
    "region",
    "model",
    "adapter",
    "variable",
    "lead_time_hours",
    "initialization_index",
    "initialization_sample_order",
    "initialization_time",
    "inference_seconds_for_rollout",
    "peak_accelerator_memory_gb",
    "device",
    "precision",
    "adapter_implementation",
    "checkpoint_file",
    "model_load_seconds",
    "total_params",
    "trainable_params",
    "trainable_fraction",
    "adaptation_train_seconds",
    "weighted_rmse",
    "weighted_mae",
}


def parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_model_adapter_specs(models: str, model_adapters: str | None) -> list[tuple[str, str]]:
    if model_adapters:
        specs = []
        for item in parse_csv(model_adapters):
            if ":" not in item:
                raise ValueError(
                    f"Model-adapter spec {item!r} must use MODEL:ADAPTER, for example aurora_small:lora."
                )
            model_name, adapter = item.split(":", 1)
            if adapter.strip() not in VALID_ADAPTERS:
                raise ValueError(f"Unsupported adapter {adapter!r}. Valid adapters are {sorted(VALID_ADAPTERS)}.")
            specs.append((model_name.strip(), adapter.strip()))
        return specs
    return [(model_name, "pretrained") for model_name in parse_csv(models)]


def choose_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_memory(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(device: str) -> float | None:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    return None


def autocast_context(device: str, precision: str):
    enabled = device == "cuda" and precision == "bf16"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def parameter_counts(model: torch.nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_fraction": float(trainable / total) if total else 0.0,
    }


def serialize_checkpoint_load_result(result: Any) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "return_type": type(result).__name__,
        "missing_keys": [],
        "unexpected_keys": [],
        "raw_repr": repr(result),
    }
    for key in ("missing_keys", "unexpected_keys"):
        value = getattr(result, key, None)
        if value is not None:
            audit[key] = [str(item) for item in value]
    if isinstance(result, tuple) and len(result) >= 2:
        audit["missing_keys"] = [str(item) for item in result[0]]
        audit["unexpected_keys"] = [str(item) for item in result[1]]
    return audit


def write_lora_load_audit(output_dir: Path | None, audit: dict[str, Any]) -> None:
    print("Aurora LoRA checkpoint load audit:")
    print(json.dumps(audit, indent=2, default=str))
    if output_dir is not None:
        (output_dir / "lora_checkpoint_load_audit.json").write_text(
            json.dumps(audit, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def load_checkpoint_capturing_keys(model: Any, repo: str, checkpoint: str, strict: bool) -> dict[str, Any]:
    """Load an Aurora checkpoint and capture the REAL ``load_state_dict`` incompatible keys.

    ``model.load_checkpoint(...)`` loads in place and returns ``None``, so auditing its return
    value tells us nothing. We temporarily shadow ``model.load_state_dict`` to capture the
    ``_IncompatibleKeys`` it produces, which is what actually reveals whether ``strict=False``
    silently dropped weights.
    """
    captured: dict[str, Any] = {"missing_keys": [], "unexpected_keys": [], "captured": False}
    original_load_state_dict = model.load_state_dict

    def capturing(state_dict, strict=True, *args, **kwargs):  # noqa: ANN001
        result = original_load_state_dict(state_dict, strict=strict, *args, **kwargs)
        missing = getattr(result, "missing_keys", None)
        unexpected = getattr(result, "unexpected_keys", None)
        if missing is not None or unexpected is not None:
            captured["missing_keys"] = [str(key) for key in (missing or [])]
            captured["unexpected_keys"] = [str(key) for key in (unexpected or [])]
            captured["captured"] = True
        return result

    model.load_state_dict = capturing
    try:
        model.load_checkpoint(repo, checkpoint, strict=strict)
    finally:
        model.load_state_dict = original_load_state_dict

    non_lora_missing = [key for key in captured["missing_keys"] if "lora" not in key.lower()]
    captured["lora_missing_key_count"] = len(captured["missing_keys"]) - len(non_lora_missing)
    captured["non_lora_missing_keys"] = non_lora_missing
    captured["non_lora_missing_key_count"] = len(non_lora_missing)
    return captured


def freeze_non_lora_parameters(model: torch.nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        should_train = "lora" in name.lower()
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError(
            "Aurora LoRA was requested, but no parameters with 'lora' in their name were found. "
            "Check the installed microsoft-aurora version and AuroraSmallPretrained(use_lora=True) support."
        )
    return int(trainable)


def sort_lat_desc(ds: xr.Dataset) -> xr.Dataset:
    if float(ds.latitude.values[0]) < float(ds.latitude.values[-1]):
        return ds.sortby("latitude", ascending=False)
    return ds


def crop_to_patch_multiple(ds: xr.Dataset, patch_size: int = 4) -> xr.Dataset:
    lat_size = int(ds.sizes["latitude"])
    lon_size = int(ds.sizes["longitude"])
    cropped_lat = (lat_size // patch_size) * patch_size
    cropped_lon = (lon_size // patch_size) * patch_size
    if cropped_lat == 0 or cropped_lon == 0:
        raise ValueError(f"Region too small for patch size {patch_size}: {lat_size} x {lon_size}")
    return ds.isel(latitude=slice(0, cropped_lat), longitude=slice(0, cropped_lon))


def tensor_from_dataarray(da: xr.DataArray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(da.values).copy()).float()


def region_indices(
    lat: torch.Tensor, lon: torch.Tensor, bounds: RegionBounds
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index arrays selecting a lat/lon box out of a (global) Aurora grid, for regional scoring.

    Aurora is a global model: it must be run on the full grid so that lateral inflow is
    resolved. We therefore crop the *predictions and targets*, not the model input.
    """
    lat_np = lat.detach().cpu().numpy()
    lon_np = lon.detach().cpu().numpy()
    lat_mask = (lat_np >= bounds.latitude_min) & (lat_np <= bounds.latitude_max)
    lon_min, lon_max = bounds.longitude_min, bounds.longitude_max
    if float(lon_np.min()) >= 0 and lon_min < 0:
        lon_min %= 360
        lon_max %= 360
    lon_mask = (lon_np >= lon_min) & (lon_np <= lon_max)
    lat_idx = torch.from_numpy(np.nonzero(lat_mask)[0]).long()
    lon_idx = torch.from_numpy(np.nonzero(lon_mask)[0]).long()
    if lat_idx.numel() == 0 or lon_idx.numel() == 0:
        raise ValueError(f"Region crop selected an empty grid for bounds {bounds}.")
    return lat_idx, lon_idx


def crop_hw(tensor: torch.Tensor, lat_idx: torch.Tensor, lon_idx: torch.Tensor) -> torch.Tensor:
    """Select the region box from the last two (lat, lon) dims of a tensor."""
    return tensor.index_select(-2, lat_idx.to(tensor.device)).index_select(-1, lon_idx.to(tensor.device))


def align_static_to_dynamic(static_ds: xr.Dataset, ds: xr.Dataset) -> xr.Dataset:
    static_ds = sort_lat_desc(static_ds)
    return static_ds.sel(latitude=ds.latitude, longitude=ds.longitude)


def static_path_from_config(cfg: dict) -> Path:
    static_uri, _ = resolve_static_uri(cfg)
    if static_uri is None:
        raise ValueError("No Aurora static field URI configured.")
    path = Path(static_uri)
    if not path.is_absolute():
        path = ROOT / path
    return path


def validate_aurora_inputs(ds: xr.Dataset) -> None:
    missing_surface = [name for name in SURFACE_TO_AURORA if name not in ds]
    missing_atmos = [name for name in ATMOS_TO_AURORA if name not in ds]
    if missing_surface or missing_atmos:
        raise ValueError(
            "WeatherBench2 dataset is missing variables required by Aurora. "
            f"Missing surface={missing_surface}, missing atmospheric={missing_atmos}."
        )
    if "level" not in ds.coords and "level" not in ds.dims:
        raise ValueError("WeatherBench2 dataset must include a `level` coordinate for Aurora atmospheric inputs.")
    available_levels = {int(level) for level in ds.level.values}
    missing_levels = [level for level in REQUIRED_AURORA_LEVELS if level not in available_levels]
    if missing_levels:
        raise ValueError(
            "WeatherBench2 dataset does not include all Aurora pressure levels. "
            f"Missing levels={missing_levels}; available levels={sorted(available_levels)}."
        )


_SANITIZE_WARNED: set[str] = set()


def sanitize_input_tensor(tensor: torch.Tensor, name: str, max_nan_fraction: float = 0.05) -> torch.Tensor:
    """Fill sparse non-finite pixels in an Aurora input field with its finite mean.

    A single NaN anywhere in a global input poisons the whole attention and produces an all-NaN
    forecast. HRES T0 has occasional sparse gaps (e.g. specific humidity / static fields) that are
    safe to impute; but a pervasively non-finite field is a structural problem we must NOT silently
    paper over, so we cap the fill fraction and raise beyond it.
    """
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return tensor
    nan_fraction = 1.0 - float(finite_mask.float().mean().item())
    if nan_fraction > max_nan_fraction:
        raise ValueError(
            f"Aurora input '{name}' is {nan_fraction:.1%} non-finite in this time slice "
            f"(> {max_nan_fraction:.0%} cap); this is a structural data problem, not sparse gaps."
        )
    finite_vals = tensor[finite_mask]
    fill = float(finite_vals.mean().item()) if finite_vals.numel() else 0.0
    if name not in _SANITIZE_WARNED:
        print(f"[input sanitize] '{name}': filled {nan_fraction:.3%} non-finite pixels with finite mean {fill:.4g}", flush=True)
        _SANITIZE_WARNED.add(name)
    return torch.nan_to_num(tensor, nan=fill, posinf=fill, neginf=fill)


def build_batch(ds: xr.Dataset, static_ds: xr.Dataset, start_idx: int):
    from aurora import Batch, Metadata

    ds = sort_lat_desc(ds)
    static_ds = sort_lat_desc(static_ds)
    surf_vars = {
        aurora_name: sanitize_input_tensor(
            tensor_from_dataarray(ds[source].isel(time=slice(start_idx, start_idx + 2))), source
        ).unsqueeze(0)
        for source, aurora_name in SURFACE_TO_AURORA.items()
    }
    atmos_vars = {
        aurora_name: sanitize_input_tensor(
            tensor_from_dataarray(ds[source].isel(time=slice(start_idx, start_idx + 2))), source
        ).unsqueeze(0)
        for source, aurora_name in ATMOS_TO_AURORA.items()
    }
    static_vars = {
        "z": sanitize_input_tensor(
            tensor_from_dataarray(static_ds["z"].isel(valid_time=0) if "valid_time" in static_ds["z"].dims else static_ds["z"]),
            "static_z",
        ),
        "lsm": sanitize_input_tensor(
            tensor_from_dataarray(static_ds["lsm"].isel(valid_time=0) if "valid_time" in static_ds["lsm"].dims else static_ds["lsm"]),
            "static_lsm",
        ),
        "slt": sanitize_input_tensor(
            tensor_from_dataarray(static_ds["slt"].isel(valid_time=0) if "valid_time" in static_ds["slt"].dims else static_ds["slt"]),
            "static_slt",
        ),
    }
    time_value = ds.time.values.astype("datetime64[s]").tolist()[start_idx + 1]
    metadata = Metadata(
        lat=tensor_from_dataarray(ds.latitude),
        lon=tensor_from_dataarray(ds.longitude),
        time=(time_value,),
        atmos_levels=tuple(int(level) for level in ds.level.values),
    )
    return Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, metadata=metadata)


def load_aurora_model(model_name: str, adapter: str, device: str, output_dir: Path | None = None):
    from aurora import Aurora, AuroraSmallPretrained

    cfg = load_yaml(model_config_path(model_name))
    checkpoint = cfg["checkpoint_file"]
    started = time.perf_counter()
    if model_name == "aurora_small":
        model = AuroraSmallPretrained(use_lora=(adapter == "lora"))
    elif model_name == "aurora_large":
        if adapter == "lora":
            raise ValueError("LoRA adaptation is currently supported only for aurora_small.")
        model = Aurora()
    else:
        raise ValueError(f"Unsupported Aurora model: {model_name}")
    strict = adapter != "lora"
    repo = cfg.get("hf_repo", "microsoft/aurora")
    if adapter == "lora":
        audit = {
            "model": model_name,
            "adapter": adapter,
            "strict": strict,
            "checkpoint_file": checkpoint,
            **load_checkpoint_capturing_keys(model, repo, checkpoint, strict),
        }
        write_lora_load_audit(output_dir, audit)
        if not audit["captured"]:
            raise RuntimeError(
                "Could not capture Aurora LoRA checkpoint load keys; cannot verify strict=False did not "
                "silently drop weights. Inspect the aurora load_checkpoint implementation."
            )
        if audit["non_lora_missing_key_count"] > 0:
            raise RuntimeError(
                f"strict=False dropped {audit['non_lora_missing_key_count']} non-LoRA weights while loading the "
                f"Aurora LoRA model, so the backbone is not fully initialized (this is what produced NaN forecasts). "
                f"First missing: {audit['non_lora_missing_keys'][:8]}"
            )
        freeze_non_lora_parameters(model)
    else:
        model.load_checkpoint(repo, checkpoint, strict=strict)
        for param in model.parameters():
            param.requires_grad_(False)
    model.to(device)
    model.eval()
    return model, {
        "adapter_implementation": adapter,
        "checkpoint_file": checkpoint,
        "model_load_seconds": time.perf_counter() - started,
        "adaptation_train_seconds": 0.0,
        "calibration_parameter_count": 0,
        "calibration_fit_seconds": None,
        **parameter_counts(model),
    }


def tensor_metrics(pred: torch.Tensor, target: torch.Tensor, lat: torch.Tensor) -> dict[str, float]:
    pred = pred.float().cpu()
    target = target.float().cpu()
    weights = torch.cos(torch.deg2rad(lat.float().cpu()))
    while weights.ndim < pred.ndim:
        weights = weights.unsqueeze(-1)
    weights = torch.broadcast_to(weights, pred.shape)
    mask = torch.isfinite(pred) & torch.isfinite(target) & torch.isfinite(weights)
    if not bool(mask.any()):
        raise ValueError("No finite prediction/target pixels are available for metric computation.")
    error = pred - target
    weights = weights[mask]
    error = error[mask]
    weight_sum = weights.sum().clamp_min(1e-12)
    return {
        "weighted_rmse": float(torch.sqrt((weights * error.square()).sum() / weight_sum).item()),
        "weighted_mae": float((weights * error.abs()).sum().item() / weight_sum.item()),
    }


def broadcast_lat_weights(values: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
    weights = torch.cos(torch.deg2rad(lat.float().to(values.device)))
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return torch.broadcast_to(weights, values.shape)


def update_calibration_stats(
    stats: dict[tuple[str, int], dict[str, float]],
    variable: str,
    lead: int,
    prediction: torch.Tensor,
    target: torch.Tensor,
    lat: torch.Tensor,
) -> None:
    pred = prediction.float().cpu()
    truth = target.float().cpu()
    weights = broadcast_lat_weights(pred, lat.float().cpu())
    mask = torch.isfinite(pred) & torch.isfinite(truth) & torch.isfinite(weights)
    if not bool(mask.any()):

        def _frac(tensor: torch.Tensor) -> float:
            return float(torch.isfinite(tensor).float().mean().item())

        raise ValueError(
            f"No finite pixels available for calibration stats: {variable}, lead={lead}. "
            f"Diagnostics -> pred finite_frac={_frac(pred):.3f} shape={tuple(pred.shape)}; "
            f"target finite_frac={_frac(truth):.3f} shape={tuple(truth.shape)}; "
            f"weights finite_frac={_frac(weights):.3f}. "
            f"(pred all-NaN => model forward diverged on train split; target all-NaN => train data/grid problem.)"
        )
    pred = pred[mask]
    truth = truth[mask]
    weights = weights[mask]
    key = (variable, lead)
    entry = stats.setdefault(key, {"sum_w": 0.0, "sum_x": 0.0, "sum_y": 0.0, "sum_xx": 0.0, "sum_xy": 0.0})
    entry["sum_w"] += float(weights.sum().item())
    entry["sum_x"] += float((weights * pred).sum().item())
    entry["sum_y"] += float((weights * truth).sum().item())
    entry["sum_xx"] += float((weights * pred.square()).sum().item())
    entry["sum_xy"] += float((weights * pred * truth).sum().item())


def solve_affine_calibration(stats: dict[tuple[str, int], dict[str, float]]) -> dict[tuple[str, int], dict[str, float]]:
    params: dict[tuple[str, int], dict[str, float]] = {}
    for key, values in stats.items():
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite calibration statistics for {key}: {values}")
        sum_w = values["sum_w"]
        denominator = (sum_w * values["sum_xx"]) - values["sum_x"] ** 2
        if abs(denominator) < 1e-12:
            slope = 1.0
            intercept = (values["sum_y"] - values["sum_x"]) / max(sum_w, 1e-12)
        else:
            slope = ((sum_w * values["sum_xy"]) - (values["sum_x"] * values["sum_y"])) / denominator
            intercept = (values["sum_y"] - slope * values["sum_x"]) / max(sum_w, 1e-12)
        params[key] = {"slope": float(slope), "intercept": float(intercept)}
        if not math.isfinite(params[key]["slope"]) or not math.isfinite(params[key]["intercept"]):
            raise ValueError(f"Non-finite calibration parameters for {key}: {params[key]}")
    return params


def calibration_to_jsonable(params: dict[tuple[str, int], dict[str, float]]) -> list[dict[str, Any]]:
    return [
        {"variable": variable, "lead_time_hours": lead, **values}
        for (variable, lead), values in sorted(params.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def apply_calibration(
    prediction: torch.Tensor,
    calibration_params: dict[tuple[str, int], dict[str, float]] | None,
    variable: str,
    lead: int,
) -> torch.Tensor:
    if not calibration_params:
        return prediction
    values = calibration_params[(variable, lead)]
    return prediction * values["slope"] + values["intercept"]


def validate_result_row(row: dict[str, Any]) -> None:
    missing = sorted(COMMON_RESULT_FIELDS - set(row))
    if missing:
        raise ValueError(f"Aurora result row is missing required fields: {missing}")
    null_forbidden = [
        key
        for key in COMMON_RESULT_FIELDS
        if key not in {"calibration_fit_seconds"} and row.get(key) is None
    ]
    if null_forbidden:
        raise ValueError(f"Aurora result row has null required values: {sorted(null_forbidden)}")
    for key in ("weighted_rmse", "weighted_mae"):
        if not math.isfinite(float(row[key])):
            raise ValueError(f"Aurora result row has non-finite {key}: {row[key]}")
    if row["adapter"] == "linear_calibration":
        for key in ("calibration_fit_seconds", "calibration_parameter_count"):
            if row.get(key) is None:
                raise ValueError(f"Linear Calibration (MOS) row is missing {key}.")
    if row["adapter"] == "lora":
        for key in ("lora_train_steps", "lora_train_seconds", "lora_initial_loss", "lora_final_loss"):
            if row.get(key) is None:
                raise ValueError(f"LoRA row is missing {key}.")


def write_result_row(output_path: Path, row: dict[str, Any]) -> None:
    validate_result_row(row)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def one_step_surface_loss(pred, ds: xr.Dataset, start_idx: int, device: str, bounds: RegionBounds) -> torch.Tensor:
    target_idx = start_idx + 2
    losses = []
    sorted_ds = sort_lat_desc(ds)
    lat_idx, lon_idx = region_indices(pred.metadata.lat, pred.metadata.lon, bounds)
    for source, aurora_name in SURFACE_TO_AURORA.items():
        target = tensor_from_dataarray(sorted_ds[source].isel(time=target_idx)).to(device)
        target = crop_hw(target, lat_idx, lon_idx)
        prediction = crop_hw(pred.surf_vars[aurora_name][0, 0], lat_idx, lon_idx)
        mask = torch.isfinite(prediction) & torch.isfinite(target)
        if not bool(mask.any()):
            raise ValueError(f"No finite pixels available for LoRA loss: {source}.")
        valid_target = target[mask]
        valid_prediction = prediction[mask]
        scale = valid_target.detach().std().clamp_min(1e-6)
        losses.append(((valid_prediction - valid_target) / scale).square().mean())
    return torch.stack(losses).mean()


def train_lora_adapter(
    model: torch.nn.Module,
    ds: xr.Dataset,
    static_ds: xr.Dataset,
    train_indices: list[int],
    device: str,
    precision: str,
    learning_rate: float,
    output_dir: Path,
    bounds: RegionBounds,
) -> dict[str, Any]:
    from aurora import rollout

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters are available for Aurora LoRA training.")

    model.train()
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    rows = []
    started = time.perf_counter()
    for step, start_idx in enumerate(train_indices):
        clear_memory(device)
        batch = build_batch(ds, static_ds, start_idx).to(device)
        optimizer.zero_grad(set_to_none=True)
        # The LoRA training forward runs at `precision` (default fp32). bf16 autocast with gradients
        # enabled can overflow softmax/LayerNorm to NaN even when bf16 eval is fine; fp32 is the stable
        # default. The first step is decisive: if the base weights are loaded and LoRA is 0-initialized,
        # step 0 must be finite, so a non-finite first step is a hard error, not a skippable outlier.
        with autocast_context(device, precision):
            pred = next(iter(rollout(model, batch, steps=1)))
            loss = one_step_surface_loss(pred, ds, start_idx, device, bounds)
        if not torch.isfinite(loss):
            raise ValueError(
                f"Non-finite Aurora LoRA loss at step={step}, initialization_index={start_idx}. "
                f"If step==0, check lora_checkpoint_load_audit.json (base-weight load) and try --lora-train-precision fp32."
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        row = {
            "step": step,
            "initialization_index": start_idx,
            "initialization_time": str(ds.time.values[start_idx + 1]),
            "loss": float(loss.detach().cpu().item()),
            "peak_accelerator_memory_gb": peak_memory_gb(device),
        }
        rows.append(row)
        with (output_dir / "lora_training_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    adapter_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    adapter_path = output_dir / "aurora_small_lora_adapter.pt"
    torch.save(adapter_state, adapter_path)
    model.eval()
    lora_train_seconds = time.perf_counter() - started
    return {
        "adapter_checkpoint": str(adapter_path),
        "lora_train_steps": len(train_indices),
        "lora_train_seconds": lora_train_seconds,
        "adaptation_train_seconds": lora_train_seconds,
        "lora_final_loss": rows[-1]["loss"] if rows else None,
        "lora_initial_loss": rows[0]["loss"] if rows else None,
    }


def fit_linear_calibration(
    model: torch.nn.Module,
    ds: xr.Dataset,
    static_ds: xr.Dataset,
    train_indices: list[int],
    lead_times: list[int],
    device: str,
    precision: str,
    output_dir: Path,
    bounds: RegionBounds,
) -> dict[str, Any]:
    from aurora import rollout

    started = time.perf_counter()
    max_steps = max(lead_times) // 6
    stats: dict[tuple[str, int], dict[str, float]] = {}
    model.eval()
    sorted_ds = sort_lat_desc(ds)
    rows = []
    for sample_order, start_idx in enumerate(train_indices):
        clear_memory(device)
        batch = build_batch(sorted_ds, static_ds, start_idx).to(device)
        with torch.inference_mode(), autocast_context(device, precision):
            preds = [pred.to("cpu") for pred in rollout(model, batch, steps=max_steps)]
        lat_idx, lon_idx = region_indices(preds[0].metadata.lat, preds[0].metadata.lon, bounds)
        region_lat = preds[0].metadata.lat.index_select(0, lat_idx)
        for lead in lead_times:
            step = lead // 6
            pred = preds[step - 1]
            target_idx = start_idx + 1 + step
            for source, aurora_name in SURFACE_TO_AURORA.items():
                target = tensor_from_dataarray(sorted_ds[source].isel(time=target_idx))
                pred_region = crop_hw(pred.surf_vars[aurora_name][0, 0], lat_idx, lon_idx)
                target_region = crop_hw(target, lat_idx, lon_idx)
                update_calibration_stats(stats, source, lead, pred_region, target_region, region_lat)
        rows.append(
            {
                "sample_order": sample_order,
                "initialization_index": start_idx,
                "initialization_time": str(sorted_ds.time.values[start_idx + 1]),
                "peak_accelerator_memory_gb": peak_memory_gb(device),
            }
        )
    params = solve_affine_calibration(stats)
    parameter_count = len(params) * 2
    fit_seconds = time.perf_counter() - started
    params_path = output_dir / "aurora_small_linear_calibration_mos.json"
    payload = {
        "adapter": "linear_calibration",
        "adapter_label": "Linear Calibration (MOS)",
        "model": "aurora_small",
        "fit_split": "train",
        "form": "y_calibrated = slope * y_pred + intercept",
        "parameter_count": parameter_count,
        "lead_times_hours": lead_times,
        "variables": list(SURFACE_TO_AURORA),
        "num_train_initialization_times": len(train_indices),
        "fit_seconds": fit_seconds,
        "parameters": calibration_to_jsonable(params),
    }
    params_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    with (output_dir / "linear_calibration_training_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    return {
        "calibration_params": params,
        "calibration_parameters_path": str(params_path),
        "calibration_parameter_count": parameter_count,
        "calibration_fit_seconds": fit_seconds,
        "adaptation_train_seconds": fit_seconds,
    }


def run_model(
    model_name: str,
    adapter: str,
    ds: xr.Dataset,
    static_ds: xr.Dataset,
    lead_times: list[int],
    initialization_indices: list[int],
    device: str,
    precision: str,
    output_path: Path,
    bounds: RegionBounds,
    train_ds: xr.Dataset | None = None,
    train_static_ds: xr.Dataset | None = None,
    train_indices: list[int] | None = None,
    lora_learning_rate: float = 1e-4,
    lora_train_precision: str = "fp32",
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    from aurora import rollout

    clear_memory(device)
    if adapter == "linear_calibration" and model_name != "aurora_small":
        raise ValueError("Linear Calibration (MOS) is currently defined only for aurora_small.")
    model, model_meta = load_aurora_model(model_name, adapter, device, output_dir=output_dir)
    calibration_params = None
    if adapter == "lora":
        if train_ds is None or train_static_ds is None or train_indices is None or output_dir is None:
            raise ValueError("Aurora LoRA requires train_ds, train_static_ds, train_indices, and output_dir.")
        model_meta.update(
            train_lora_adapter(
                model=model,
                ds=train_ds,
                static_ds=train_static_ds,
                train_indices=train_indices,
                device=device,
                precision=lora_train_precision,
                learning_rate=lora_learning_rate,
                output_dir=output_dir,
                bounds=bounds,
            )
        )
    if adapter == "linear_calibration":
        if train_ds is None or train_static_ds is None or train_indices is None or output_dir is None:
            raise ValueError(
                "Aurora Linear Calibration (MOS) requires train_ds, train_static_ds, train_indices, and output_dir."
            )
        calibration_meta = fit_linear_calibration(
            model=model,
            ds=train_ds,
            static_ds=train_static_ds,
            train_indices=train_indices,
            lead_times=lead_times,
            device=device,
            precision=precision,
            output_dir=output_dir,
            bounds=bounds,
        )
        calibration_params = calibration_meta.pop("calibration_params")
        model_meta.update(calibration_meta)
        model_meta["adapter_implementation"] = "linear_calibration_mos"
        model_meta["trainable_params"] = int(model_meta["calibration_parameter_count"])
        model_meta["trainable_fraction"] = float(model_meta["trainable_params"] / model_meta["total_params"])
    max_steps = max(lead_times) // 6
    rows = []
    sorted_ds = sort_lat_desc(ds)
    for sample_order, start_idx in enumerate(initialization_indices):
        batch = build_batch(ds, static_ds, start_idx)
        batch = batch.to(device)
        started = time.perf_counter()
        with torch.inference_mode(), autocast_context(device, precision):
            preds = [pred.to("cpu") for pred in rollout(model, batch, steps=max_steps)]
        inference_seconds = time.perf_counter() - started
        lat_idx, lon_idx = region_indices(preds[0].metadata.lat, preds[0].metadata.lon, bounds)
        region_lat = preds[0].metadata.lat.index_select(0, lat_idx)
        for lead in lead_times:
            step = lead // 6
            pred = preds[step - 1]
            target_idx = start_idx + 1 + step
            for source, aurora_name in SURFACE_TO_AURORA.items():
                target = tensor_from_dataarray(sorted_ds[source].isel(time=target_idx))
                pred_region = crop_hw(pred.surf_vars[aurora_name][0, 0], lat_idx, lon_idx)
                pred_region = apply_calibration(pred_region, calibration_params, source, lead)
                target_region = crop_hw(target, lat_idx, lon_idx)
                metrics = tensor_metrics(pred_region, target_region, region_lat)
                row = {
                    "ok": True,
                    "dataset": "weatherbench2_hres_t0_greater_horn",
                    "region": "Greater Horn of Africa",
                    "model": model_name,
                    "adapter": adapter,
                    "variable": source,
                    "lead_time_hours": lead,
                    "initialization_index": start_idx,
                    "initialization_sample_order": sample_order,
                    "initialization_time": str(ds.time.values[start_idx + 1]),
                    "inference_seconds_for_rollout": inference_seconds,
                    "peak_accelerator_memory_gb": peak_memory_gb(device),
                    "device": device,
                    "precision": precision,
                    **model_meta,
                    **metrics,
                }
                rows.append(row)
                write_result_row(output_path, row)
    model.to("cpu")
    del model
    clear_memory(device)
    return rows


def choose_initialization_indices(
    num_times: int,
    max_steps: int,
    max_initialization_times: int,
    sampling: str,
) -> list[int]:
    max_start_exclusive = num_times - max_steps - 1
    if max_start_exclusive <= 0:
        raise ValueError(f"Not enough time points ({num_times}) for max rollout steps ({max_steps}).")
    n = min(max_initialization_times, max_start_exclusive)
    if sampling == "first":
        return list(range(n))
    if sampling == "even":
        return sorted({int(round(value)) for value in np.linspace(0, max_start_exclusive - 1, n)})
    raise ValueError(f"Unsupported initialization sampling: {sampling}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aurora forecasts on WeatherBench2 HRES T0 Greater Horn subset.")
    parser.add_argument("--dataset", default="weatherbench2_hres_t0_greater_horn")
    parser.add_argument("--models", default="aurora_small,aurora_large")
    parser.add_argument(
        "--model-adapters",
        default=None,
        help=(
            "Comma-separated MODEL:ADAPTER specs. Example: "
            "aurora_small:pretrained,aurora_small:linear_calibration,aurora_small:lora,aurora_large:pretrained"
        ),
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--train-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--lead-times-hours", default="6,12,24,48")
    parser.add_argument("--max-initialization-times", type=int, default=4)
    parser.add_argument(
        "--lora-train-steps",
        type=int,
        default=16,
        help="Train-split initialization times used for LoRA and Linear Calibration (MOS).",
    )
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--initialization-sampling", default="even", choices=["first", "even"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument(
        "--lora-train-precision",
        default="fp32",
        choices=["fp32", "bf16"],
        help="Precision for the LoRA training forward/backward. fp32 is more numerically stable than bf16-with-grad.",
    )
    parser.add_argument("--run-label", default="aurora_greater_horn_v1")
    args = parser.parse_args()

    device = choose_device(args.device)
    cfg = load_yaml(dataset_config_path(args.dataset))
    output_dir = ROOT / "artifacts" / args.run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    summary_path = output_dir / "summary.json"

    # Aurora is a global model: feed the full global grid (subset=False) so lateral inflow is
    # resolved, then crop predictions/targets to the region for scoring. Cropping the INPUT to a
    # limited-area window has no boundary forcing and makes forecasts worse than persistence.
    bounds = RegionBounds.from_config(cfg)
    ds = sort_lat_desc(open_forecasting_dataset(cfg, split=args.split, subset=False))
    validate_aurora_inputs(ds)
    static_uri = static_path_from_config(cfg)
    static_ds = align_static_to_dynamic(xr.open_dataset(static_uri), ds)
    lead_times = parse_csv(args.lead_times_hours, int)
    model_specs = parse_model_adapter_specs(args.models, args.model_adapters)
    max_steps = max(lead_times) // 6
    initialization_indices = choose_initialization_indices(
        num_times=int(ds.sizes["time"]),
        max_steps=max_steps,
        max_initialization_times=args.max_initialization_times,
        sampling=args.initialization_sampling,
    )
    needs_lora = any(adapter == "lora" for _, adapter in model_specs)
    needs_calibration = any(adapter == "linear_calibration" for _, adapter in model_specs)
    needs_train_adapter = any(adapter in {"lora", "linear_calibration"} for _, adapter in model_specs)
    train_ds = None
    train_static_ds = None
    train_indices = None
    if needs_train_adapter:
        train_ds = sort_lat_desc(open_forecasting_dataset(cfg, split=args.train_split, subset=False))
        validate_aurora_inputs(train_ds)
        train_static_ds = align_static_to_dynamic(xr.open_dataset(static_uri), train_ds)
        train_max_steps = max_steps if needs_calibration else 1
        train_indices = choose_initialization_indices(
            num_times=int(train_ds.sizes["time"]),
            max_steps=train_max_steps,
            max_initialization_times=args.lora_train_steps,
            sampling=args.initialization_sampling,
        )

    all_rows = []
    started = time.perf_counter()
    for model_name, adapter in model_specs:
        all_rows.extend(
            run_model(
                model_name=model_name,
                adapter=adapter,
                ds=ds,
                static_ds=static_ds,
                lead_times=lead_times,
                initialization_indices=initialization_indices,
                device=device,
                precision=args.precision,
                output_path=results_path,
                bounds=bounds,
                train_ds=train_ds,
                train_static_ds=train_static_ds,
                train_indices=train_indices,
                lora_learning_rate=args.lora_learning_rate,
                lora_train_precision=args.lora_train_precision,
                output_dir=output_dir,
            )
        )

    summary = {
        "run_label": args.run_label,
        "dataset": args.dataset,
        "split": args.split,
        "input_domain": "global",
        "scoring_domain": "region_crop",
        "region_bounds": {
            "latitude_min": bounds.latitude_min,
            "latitude_max": bounds.latitude_max,
            "longitude_min": bounds.longitude_min,
            "longitude_max": bounds.longitude_max,
        },
        "model_specs": [{"model": model_name, "adapter": adapter} for model_name, adapter in model_specs],
        "lead_times_hours": lead_times,
        "max_initialization_times": args.max_initialization_times,
        "adaptation_train_steps": args.lora_train_steps if needs_train_adapter else 0,
        "lora_train_steps": args.lora_train_steps if needs_lora else 0,
        "lora_learning_rate": args.lora_learning_rate if needs_lora else None,
        "initialization_sampling": args.initialization_sampling,
        "actual_initialization_times": len(initialization_indices),
        "first_initialization_time": str(ds.time.values[initialization_indices[0] + 1]),
        "last_initialization_time": str(ds.time.values[initialization_indices[-1] + 1]),
        "rows": len(all_rows),
        "wall_seconds": time.perf_counter() - started,
        "device": device,
        "precision": args.precision,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "results_path": str(results_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

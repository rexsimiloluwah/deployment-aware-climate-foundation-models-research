from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import xarray as xr

from .regions import RegionBounds, first_existing_coord, subset_region


@dataclass(frozen=True)
class ForecastingDatasetReadiness:
    dataset: str
    display_name: str
    uri: str | None
    uri_source: str
    can_open: bool
    region_subset_ok: bool | None
    dims: dict[str, int] | None
    data_vars: list[str] | None
    static_uri: str | None
    static_fields_ok: bool
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_dataset_uri(cfg: dict) -> tuple[str | None, str]:
    data_cfg = cfg.get("data", {})
    env_name = data_cfg.get("uri_env")
    if env_name and os.environ.get(env_name):
        return os.environ[env_name], f"env:{env_name}"
    if data_cfg.get("uri"):
        return data_cfg["uri"], "config:data.uri"
    return None, f"missing:{env_name or 'data.uri'}"


def resolve_static_uri(cfg: dict) -> tuple[str | None, str]:
    data_cfg = cfg.get("data", {})
    env_name = data_cfg.get("static_uri_env")
    if env_name and os.environ.get(env_name):
        return os.environ[env_name], f"env:{env_name}"
    if data_cfg.get("static_uri"):
        return data_cfg["static_uri"], "config:data.static_uri"
    return None, f"missing:{env_name or 'data.static_uri'}"


def static_fields_available(cfg: dict) -> tuple[str | None, bool]:
    uri, _ = resolve_static_uri(cfg)
    if not uri:
        return None, False
    if uri.startswith(("gs://", "s3://", "http://", "https://")):
        return uri, True
    path = Path(uri)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return str(path), path.exists()


def open_xarray_uri(uri: str, storage_options: dict | None = None) -> xr.Dataset:
    storage_options = storage_options or {}
    if uri.endswith(".zarr") or uri.startswith("gs://") or uri.startswith("s3://"):
        return xr.open_zarr(uri, storage_options=storage_options, consolidated=True)
    path = Path(uri)
    if path.suffix in {".nc", ".netcdf"} or path.exists():
        return xr.open_dataset(uri)
    return xr.open_zarr(uri, storage_options=storage_options, consolidated=True)


def open_forecasting_dataset(cfg: dict, split: str | None = None, subset: bool = True) -> xr.Dataset:
    uri, _ = resolve_dataset_uri(cfg)
    if not uri:
        env_name = cfg.get("data", {}).get("uri_env", "WEATHERBENCH2_HRES_T0_URI")
        raise ValueError(f"No WeatherBench2/HRES T0 URI configured. Set {env_name} or data.uri in the dataset config.")

    ds = open_xarray_uri(uri, storage_options=cfg.get("data", {}).get("storage_options"))
    data_cfg = cfg.get("data", {})
    time_coord = data_cfg.get("time_coord", "time")
    if split:
        split_cfg = cfg["splits"][split]
        ds = ds.sel({time_coord: slice(split_cfg["start"], split_cfg["end"])})
    if subset:
        lat = first_existing_coord(ds, data_cfg.get("latitude_coord_candidates", ["latitude", "lat"]))
        lon = first_existing_coord(ds, data_cfg.get("longitude_coord_candidates", ["longitude", "lon"]))
        ds = subset_region(ds, RegionBounds.from_config(cfg), lat, lon)
    return ds


def summarize_forecasting_dataset(cfg: dict, split: str | None = None) -> ForecastingDatasetReadiness:
    uri, uri_source = resolve_dataset_uri(cfg)
    static_uri, static_ok = static_fields_available(cfg)
    try:
        ds = open_forecasting_dataset(cfg, split=split, subset=True)
        dims = {name: int(size) for name, size in ds.sizes.items()}
        return ForecastingDatasetReadiness(
            dataset=cfg["name"],
            display_name=cfg["display_name"],
            uri=uri,
            uri_source=uri_source,
            can_open=True,
            region_subset_ok=True,
            dims=dims,
            data_vars=sorted(list(ds.data_vars)),
            static_uri=static_uri,
            static_fields_ok=static_ok,
            error=None,
        )
    except Exception as exc:
        return ForecastingDatasetReadiness(
            dataset=cfg["name"],
            display_name=cfg["display_name"],
            uri=uri,
            uri_source=uri_source,
            can_open=False,
            region_subset_ok=None,
            dims=None,
            data_vars=None,
            static_uri=static_uri,
            static_fields_ok=static_ok,
            error=f"{type(exc).__name__}: {exc}",
        )

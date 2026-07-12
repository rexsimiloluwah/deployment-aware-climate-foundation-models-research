from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import xarray as xr


@dataclass(frozen=True)
class RegionBounds:
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float

    @classmethod
    def from_config(cls, cfg: dict) -> "RegionBounds":
        bounds = cfg["region_bounds"]
        return cls(
            latitude_min=float(bounds["latitude_min"]),
            latitude_max=float(bounds["latitude_max"]),
            longitude_min=float(bounds["longitude_min"]),
            longitude_max=float(bounds["longitude_max"]),
        )


def first_existing_coord(dataset: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise KeyError(f"None of these coordinate candidates exist in dataset: {list(candidates)}")


def subset_region(
    dataset: xr.Dataset,
    bounds: RegionBounds,
    latitude_coord: str,
    longitude_coord: str,
) -> xr.Dataset:
    """Subset a lat/lon xarray dataset, handling descending latitude and 0..360 longitudes."""

    lat_values = dataset[latitude_coord].values
    lon_values = dataset[longitude_coord].values
    lat_descending = bool(lat_values[0] > lat_values[-1])

    lat_slice = (
        slice(bounds.latitude_max, bounds.latitude_min)
        if lat_descending
        else slice(bounds.latitude_min, bounds.latitude_max)
    )

    lon_min = bounds.longitude_min
    lon_max = bounds.longitude_max
    if lon_values.min() >= 0 and lon_min < 0:
        lon_min = lon_min % 360
        lon_max = lon_max % 360
    lon_slice = slice(lon_min, lon_max)

    return dataset.sel({latitude_coord: lat_slice, longitude_coord: lon_slice})

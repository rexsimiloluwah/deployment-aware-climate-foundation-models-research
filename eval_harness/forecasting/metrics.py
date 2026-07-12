from __future__ import annotations

import numpy as np
import xarray as xr


def latitude_weights(data: xr.DataArray, latitude_coord: str) -> xr.DataArray:
    weights = np.cos(np.deg2rad(data[latitude_coord]))
    return weights / weights.mean()


def weighted_rmse(
    prediction: xr.DataArray,
    target: xr.DataArray,
    latitude_coord: str = "latitude",
) -> float:
    error = prediction - target
    weights = latitude_weights(error, latitude_coord)
    value = np.sqrt((weights * error**2).mean())
    return float(value.compute() if hasattr(value, "compute") else value)


def weighted_mae(
    prediction: xr.DataArray,
    target: xr.DataArray,
    latitude_coord: str = "latitude",
) -> float:
    error = abs(prediction - target)
    weights = latitude_weights(error, latitude_coord)
    value = (weights * error).mean()
    return float(value.compute() if hasattr(value, "compute") else value)

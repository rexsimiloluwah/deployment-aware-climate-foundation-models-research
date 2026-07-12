from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_harness.config import dataset_config_path, load_yaml


def output_path_from_config(dataset_cfg: dict) -> Path:
    static_uri = dataset_cfg.get("data", {}).get("static_uri", "data/weatherbench2/hres_t0_global/static.nc")
    path = Path(static_uri)
    if not path.is_absolute():
        path = ROOT / path
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5 static fields required by Aurora.")
    parser.add_argument("--dataset", default="weatherbench2_hres_t0_greater_horn")
    parser.add_argument("--output", default=None)
    parser.add_argument("--year", default="2023")
    parser.add_argument("--month", default="01")
    parser.add_argument("--day", default="01")
    parser.add_argument("--time", default="00:00")
    parser.add_argument(
        "--global-domain",
        action="store_true",
        help="Fetch global static fields (no area crop). Required: Aurora runs on the global grid.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("Install the weather extra first: uv sync --extra weather") from exc

    dataset_cfg = load_yaml(dataset_config_path(args.dataset))
    output = Path(args.output) if args.output else output_path_from_config(dataset_cfg)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        print(f"Static fields already exist: {output}")
        return

    bounds = dataset_cfg["region_bounds"]
    area = None if args.global_domain else [
        float(bounds["latitude_max"]),
        float(bounds["longitude_min"]),
        float(bounds["latitude_min"]),
        float(bounds["longitude_max"]),
    ]

    url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    key = os.environ.get("CDSAPI_KEY")
    if not key and not (Path.home() / ".cdsapirc").exists():
        raise SystemExit("Set CDSAPI_KEY or create ~/.cdsapirc before downloading ERA5 static fields.")

    client = cdsapi.Client(url=url, key=key) if key else cdsapi.Client()
    request = {
        "product_type": "reanalysis",
        "variable": ["geopotential", "land_sea_mask", "soil_type"],
        "year": args.year,
        "month": args.month,
        "day": args.day,
        "time": args.time,
        "format": "netcdf",
    }
    if area is not None:
        request["area"] = area
    print(f"Downloading ERA5 static fields to {output}")
    print(f"Area [north, west, south, east]: {area if area is not None else 'global'}")
    client.retrieve("reanalysis-era5-single-levels", request, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

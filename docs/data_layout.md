# Real Data Layout

The harness no longer treats synthetic data as the default for dataset readiness. Real smoke tests expect manifests under `data/`.

## Western Kenya Crop Type

Expected manifest:

```text
data/western_kenya_crop_type/manifest.csv
```

Required columns:

| Column | Meaning |
|---|---|
| `id` | Stable field/sample id |
| `split` | `train`, `val`, or `test` |
| `array_path` | Path to a `.npy`, `.npz`, `.csv`, or `.pt` feature array, relative to dataset root unless absolute |
| `label` | Crop label as class name or integer id |

Expected sample tensor shape is currently `[12, 10]`: time steps x features. Update `config/datasets/western_kenya_crop_type.yaml` after inspecting the real files.

## FTW Africa

Expected manifest:

```text
data/ftw_africa/manifest.csv
```

Required columns:

| Column | Meaning |
|---|---|
| `id` | Stable patch id |
| `split` | `train`, `val`, or `test` |
| `image_path` | Path to image tensor/raster, relative to dataset root unless absolute |
| `mask_path` | Path to semantic mask tensor/raster, relative to dataset root unless absolute |

Optional columns:

| Column | Meaning |
|---|---|
| `country` | Country name/code |
| `continent` | Should equal `Africa` for Africa-only filtering |

Image paths can point to `.npy`, `.npz`, `.pt`, `.tif`, or `.tiff` files. GeoTIFF support requires `uv sync --extra geo`.

## Readiness Gate

The full experiment readiness gate fails until both manifests exist and at least one sample from every configured split can be loaded:

```bash
uv run python scripts/check_experiment_readiness.py --experiment config/experiment.yaml
```

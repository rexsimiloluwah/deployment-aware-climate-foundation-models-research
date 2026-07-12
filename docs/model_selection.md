# Foundation Model Selection

The scientific experiment will use three Earth foundation models:

| Model | Role | Estimated Size | Why It Is Included |
|---|---|---:|---|
| Prithvi-EO-2.0 300M | Primary model | 300M params | Clean first integration; EO-specific; trained on HLS/Sentinel/Landsat-style data; strong fit for crop, land, and resilience tasks. |
| TerraMind-1.0 Base | Second model | ~380M params | Similar scale but different multimodal EO design; tests whether adaptation trade-offs generalize across model families. |
| TerraMind-1.0 Large | Scaling stress test | ~950M params | Makes the deployment-efficiency question sharper by adding a substantially larger model. |

`smoke_toy` remains only in `config/experiment.validation.yaml` for local plumbing checks. It is not part of the scientific experiment.

## Why These Three

The project is not only asking whether LoRA improves accuracy. It asks whether more expressive adaptation is worth the cost under African climate-data constraints.

This model set supports that question:

- **Prithvi-EO-2.0 300M** anchors the study in a well-documented EO foundation model aligned with Sentinel/Landsat agriculture tasks.
- **TerraMind-1.0 Base** gives a second model family at roughly comparable scale.
- **TerraMind-1.0 Large** tests whether the harness captures scaling costs that matter in practice: GPU hours, peak VRAM, trainable parameters, model size, and latency.

## Execution Order

1. Finish EDA and real dataset loaders.
2. Integrate `Prithvi-EO-2.0 300M`.
3. Validate linear probe and LoRA on Prithvi.
4. Integrate `TerraMind-1.0 Base`.
5. Integrate `TerraMind-1.0 Large` after the TerraMind-base path is stable.

## Current Smoke-Test Status

Foundation model checkpoint metadata has been verified through Hugging Face:

| Model | Repo | Weight File | Status |
|---|---|---|---|
| Prithvi-EO-2.0 300M | `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` | `Prithvi_EO_V2_300M.pt` | Metadata verified |
| TerraMind-1.0 Base | `ibm-esa-geospatial/TerraMind-1.0-base` | `TerraMind_v1_base.pt` | Metadata verified |
| TerraMind-1.0 Large | `ibm-esa-geospatial/TerraMind-1.0-large` | `TerraMind_v1_large.pt` | Metadata verified |

The next smoke-test step is intentional weight download and constructor validation on a GPU VM.

## Forecasting Axis

The expanded research agenda adds a second, distinct class of Earth foundation models:

| Model | Role | Estimated Size | Why It Is Included |
|---|---|---:|---|
| Aurora Small | Forecasting adaptable model | 112.8M params | Compact Earth-system model for LoRA-style regional adaptation under constrained compute. |
| Aurora Large | Forecasting large baseline | 1.3B params | Larger Aurora-family baseline for testing whether lightweight adaptation can compete with scale. |

The locked forecasting dataset is `weatherbench2_hres_t0_greater_horn`, a WeatherBench2 HRES T0 subset over the
Greater Horn of Africa. This avoids duplicating the South Africa setup used in the related Aurora poster while keeping
the task directly relevant to African climate resilience.

The forecasting comparison is:

```text
Aurora Small pretrained
Aurora Small + LoRA
Aurora Large pretrained
```

Primary metric: latitude-weighted RMSE across 6-48 hour lead times. Deployment-readiness metrics remain model size,
trainable parameters, inference/training time, and peak accelerator memory.

## Practical Note

The full matrix is:

```text
2 datasets x 3 models x 2 adaptation methods x 4 label budgets x 3 seeds = 144 runs
```

That is feasible only if the harness remains strict about:

- one model per GPU job;
- raw JSONL artifacts;
- resumable runs;
- local analysis;
- early smoke tests before full sweeps.

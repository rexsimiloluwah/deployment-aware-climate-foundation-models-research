# Deployment-Aware Climate Foundation Models

Research question: how should Earth foundation models be adapted for African climate resilience tasks when labels, compute, and deployment resources are limited?

This repo contains an `eval_harness/` package for dataset loading, model loading, adaptation, training, forecasting, metrics, and deployment-cost logging.

## Datasets

- [Fields of The World Africa / FTW Planet](https://huggingface.co/datasets/taylor-geospatial/ftw-planet): agricultural field-boundary segmentation.
- [WeatherBench2](https://sites.research.google/weatherbench/): short-range weather forecasting over the Greater Horn of Africa.

## Models

- [Prithvi-EO-2.0 300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
- [TerraMind-1.0 Base](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
- [TerraMind-1.0 Large](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large)
- [Aurora](https://huggingface.co/microsoft/aurora)

## Status

The evaluation harness is a work in progress. Reproducibility details, experiment commands, and final result artifacts will be documented as the harness stabilizes.

Large datasets, caches, model outputs, local environments, and generated artifacts are ignored by Git.

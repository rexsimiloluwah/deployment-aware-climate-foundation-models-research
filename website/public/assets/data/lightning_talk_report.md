# Deployable Earth Foundation Models for African Climate Resilience

## Background and Motivation

Earth foundation models are increasingly useful for climate-relevant geospatial tasks such as agricultural mapping, flood monitoring, and weather forecasting. But in many African deployment settings, the limiting question is not only whether a model is accurate. It is whether the model can be adapted with limited labels, run on realistic GPUs, and produce useful outputs without excessive memory, training time, or engineering complexity.

This work evaluates deployment readiness across two complementary axes: Earth observation adaptation and weather forecasting model scale, with Aurora adaptation still being finalized. The goal is to move beyond headline accuracy and ask which foundation-model choices are practically worth using for African climate resilience.

## Research Questions

1. For Earth observation, when do lightweight adaptation strategies such as Linear Probe and LoRA provide enough accuracy to be useful under limited labels and compute?
2. For weather forecasting, can cheap regional adaptation of Aurora Small close the gap to Aurora Large, and when is LoRA worth its extra deployment cost over Linear Calibration (MOS)?
3. Can one evaluation harness compare deployment readiness across both Earth observation and weather forecasting adaptation settings?

## Significance

For African climate applications, deployment cost is not a side detail. Training time affects iteration speed, peak memory affects who can run the model, and trainable parameters affect whether adaptation is feasible for local teams. A model that is slightly more accurate but much harder to adapt or deploy may be the wrong choice for operational resilience work.

## Methodology

The evaluation framework records both task performance and deployment cost. For Earth observation, we run an adaptation experiment: Linear Probe versus LoRA under label-budget constraints. For weather forecasting, the completed artifact compares Aurora Small and Aurora Large using global WeatherBench2 input and Greater Horn regional scoring. The adapted Aurora Small + Linear Calibration (MOS) and Aurora Small + LoRA runs are the remaining Modal A100 experiments.

![Two-axis framework](two_axis_framework.png)

Evaluation metrics include foreground IoU for segmentation, weighted RMSE/MAE for forecasting, peak GPU memory, inference/training time, and parameter counts.

## Experiments

**Axis 1: Earth observation adaptation**

- Dataset: FTW Africa, using Kenya, Rwanda, and South Africa field-boundary segmentation data.
- Models: Prithvi-EO-2.0 300M, TerraMind-1.0 Base, TerraMind-1.0 Large.
- Adaptation: Linear Probe and LoRA.
- Label budgets: 5%, 10%, 25%, and 100%.
- Seeds: 3 per configuration.

**Axis 2: weather forecasting adaptation and model scale**

- Dataset: WeatherBench2 HRES T0, subset to the Greater Horn of Africa.
- Models: Aurora Small pretrained checkpoint and Aurora Large / 0.25 fine-tuned checkpoint; adapted Aurora Small runs are pending.
- Adaptation: pending: Linear Calibration (MOS) and LoRA on Aurora Small.
- Initialization times: 64 evenly sampled validation times from January 1 to June 28, 2022.
- Lead times: 6, 12, 24, and 48 hours.
- Variables: 2m temperature, 10m u wind, 10m v wind, and mean sea-level pressure.

## Results

On FTW Africa, the best configuration was **Prithvi-EO-2.0 300M + Linear Probe** at 100% labels, with foreground IoU 0.350, peak memory 1.36 GB, and 12,291 trainable parameters.

A notable deployment result is that **Prithvi Linear Probe at 25% labels** reached foreground IoU 0.346, close to the 100% label result, while keeping trainable parameters to 12,291. Prithvi LoRA at 100% labels used 3,702,787 trainable parameters and 4.12 GB peak memory, but did not dominate the linear probe result.

![FTW label efficiency](../analysis_l4_v1/ftw_label_efficiency.png)

![FTW Pareto efficiency](../analysis_l4_v1/ftw_pareto_efficiency.png)

On WeatherBench2 Greater Horn, Aurora Small used 8.68 GB peak GPU memory and 112,797,584 parameters. Aurora Large used 19.05 GB and 1,259,150,992 parameters. Using mean RMSE relative to the persistence baseline, Aurora Small averaged 0.498, while Aurora Large averaged 0.370; lower is better and values below 1 beat persistence.

![Aurora RMSE by lead time](aurora_rmse_by_lead_time.png)

![Aurora deployment tradeoff](aurora_deployment_tradeoff.png)

## Discussion

The FTW results suggest that more adaptive methods are not automatically better. Linear probing, especially with Prithvi, produced a strong Pareto point: high segmentation performance with very low trainable-parameter cost. LoRA increased the adaptation surface and memory footprint, but its value depended strongly on the model and label budget.

The WeatherBench2 results sharpen the deployment-cost argument from a forecasting perspective. The current WeatherBench2 artifact proves the global-input evaluation gate and the scale baseline. The remaining step is to compare cheap output calibration against LoRA adaptation on a larger GPU. Because the forecasting variables have different physical units, we interpret the deployment tradeoff using both variable-specific RMSE curves and RMSE relative to a persistence baseline rather than a raw cross-variable RMSE average.

Together, the two axes support the central hypothesis: the best foundation model strategy depends on the joint tradeoff between accuracy, available labels, memory, runtime, model scale, and deployment context.

## Conclusion

This work presents a deployment-readiness evaluation harness for African climate foundation models. Across Earth observation and weather forecasting, the results show that compact checkpoints and lightweight adaptation can be strong practical baselines. Larger models and heavier adaptation methods should be justified by measured gains, not by scale alone.

## Limitations

- The FTW experiment uses an African regional subset rather than the full global FTW dataset.
- The WeatherBench2 experiment evaluates 64 validation initializations over roughly six months, not a full multi-year operational benchmark.
- The Aurora adaptation axis is currently limited to Aurora Small; Aurora Large adaptation is deferred to future work because it is the scale baseline for this lightning-talk design.
- Metrics focus on average error and segmentation quality; extreme-event metrics are still needed.

## Future Work

- Extend Aurora adaptation to Aurora Large after the Aurora Small calibration-versus-LoRA comparison is complete.
- Expand WeatherBench2 evaluation to a full validation year and include seasonal breakdowns.
- Add extreme-event metrics for heat, heavy rainfall, wind, and flood-relevant thresholds.
- Add active learning and semi-supervised adaptation for the Earth observation axis.
- Package the harness as a reusable benchmark for African climate foundation models.

## Artifacts

- FTW summary rows: 24 aggregated configurations from 72 seed-level runs.
- Aurora forecasting rows: 2048 from `aurora_greater_horn_global_v1`.

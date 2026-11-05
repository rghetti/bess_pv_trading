# Provisional IDA3 target: three-day continuous comparison

Approved replay: 15–17 January 2026, farms 1/345/901/910. Completed all eight paired jobs and 2,304 MPC decisions.

| Farm | Tracking imbalance MWh | Earlier MPC MWh | Revised MPC MWh | Reduction % | Profit change EUR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.196489 | 0.486007 | 0.295164 | 39.27 | -2.98 |
| 345 | 0.013423 | 0.312854 | 0.168155 | 46.25 | +33.85 |
| 901 | 1.134779 | 1.918374 | 1.373011 | 28.43 | +61.96 |
| 910 | 0.514891 | 1.934716 | 2.255976 | -16.61 | +32.07 |

Portfolio absolute imbalance fell from 4.651951 to 4.092306 MWh (12.03% reduction). Realized operating-profit change was EUR +124.89.

The change reduced imbalance on three farms and increased it on farm 910.
Revised MPC still had more imbalance than grid tracking on every farm in this
window; tracking's portfolio total was 1.859583 MWh.

The tracking trajectories were numerically identical between target-mode cases. The MPC comparison includes changed later auction orders and state paths. This isolates the target-mode setting within the continuous policy; it does not freeze market submissions.

## Financial and execution attribution

Signed settlement change: EUR -6.10; auction revenue change: EUR +68.74; degradation-cost change: EUR -62.25; curtailment change: -1.377137 MWh; aggregate final-inventory difference: zero within numerical tolerance.

Interval groups below use local interval-start times. The earlier IDA3
submission buffer lies inside the 09:45 interval; its issued action stays locked.

| Interval start, Europe/Rome | Earlier MPC imbalance MWh | Revised MPC imbalance MWh |
| --- | ---: | ---: |
| Before 10:00 | 2.568225 | 1.401285 |
| 10:00 to 12:00 | 0.225342 | 0.438792 |
| From 12:00 | 1.858384 | 2.252229 |

The aggregate reduction is concentrated before IDA3 submission; later
intervals offset part of that gain. Subsequent auction orders also change,
so these phase totals describe the complete closed-loop outcome.

Across both cases: 0 tracking fallbacks, 0 timeouts, 11.83 seconds cumulative MPC Gurobi runtime. Batch elapsed wall time: 121.40 seconds, including preparation, auction solves and reports.

Both MPC variants finished at 50% SOC on every farm, with no physical safety
overrides. Each revised MPC used provisional targets on 120 decisions,
totalling 480 decisions across the four farms.

The independent audit checked all eight completed jobs and reported zero physical/accounting/artifact violations. The first executable MPC interval had zero provisional IDA3 quantity in every saved plan. Full submitted, provisional and planning positions are retained in checksummed horizon partitions.

## Settings and limitations

Both cases used the existing frozen reviewed-GHI selected PV bundles, latest eligible deterministic weather cycles with the six-hour lag, monthly nighttime zero policy and reviewed effective execution targets. Flagged historical-GHI replacements are counterfactual rather than independently metered production. Current-interval average PV was explicitly disclosed; future realized PV remained hidden. Model selection and current-interval average disclosure are retrospective. No fitting, calibration or GPU work was performed. The planning objective retained EUR 500/MWh absolute imbalance, throughput wear, a soft 50% midnight target and a final-study 50% planning constraint. Both cases used the same upper auction-price oracle and actual NORD signed settlement accounting. Final executed inventories remain in the output; planning targets do not reset physical state. Three development days establish a local comparison, with no statistical or full-year generalization.

The frozen model README records training with historical GHI and
validation/test with preceding-day 00 UTC forecast weather. This run instead
uses the requested updating eligible weather cycles. That irradiance-input
and forecast-vintage shift is shared by both compared cases; the pending GPU
training was not used or modified.

## Reproduction and checks

```bash
conda activate bess
PYTHONPATH=src python scripts/run_energy_research_study.py --config configs/research/provisional_ida3_targeted.yaml --check-only
PYTHONPATH=src python scripts/run_energy_research_study.py --config configs/research/provisional_ida3_targeted.yaml --run
```

The study runner requires a new run ID for a second execution. The focused licensed MPC/audit/continuous suite passed 92 tests; the expanded nine-test independent audit suite also passed (eight overlap the first suite). The legacy continuous/configuration regressions passed 62 tests, giving 155 distinct passing tests. All eight historical preflights passed. No solver checks were skipped.

Per-farm data/model input fingerprints and physical, numerical and objective
settings matched across target modes. The replayed source checksums still
match every implementation source file after the run. Tracking trajectories
were numerically identical across cases.

Raw outputs: `outputs/energy_research/provisional_ida3_jan15_17_2026_v1/`. The folder contains paired summaries, phase metrics, case contrasts, full forecasts/plans/actions and plots. Source checksums, model/data fingerprints and clocks are pinned by the manifests.

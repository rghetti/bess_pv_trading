# TimesFM IDA holiday and external feature study

This study evaluates TimesFM 3 with up to 28 complete recent calendar days for each IDA at its own gate. The fixed baseline is historical IDA prices, the four NORD load and generation forecasts, and historical spreads against the latest published, non-cancelled same-delivery earlier auction. The earlier clearing for the current day is a scoring reference, not a TimesFM input. The model's median is the point forecast; this is zero-shot inference with no fitting.

The nine variants are the baseline, the baseline plus an Italian national holiday flag, and that holiday variant with all seven nonempty combinations of gas, France and Switzerland. The holiday flag applies to each local delivery product and includes Easter, Easter Monday and 4 October from 2026. Gas is PSV on the prior day and prior week. France includes previous-day nuclear, hydro, Italian commercial exchange and balance. Switzerland includes previous-day generation, consumption and Italian and French physical flows, plus the prior-week reservoir level. Detailed source and lag definitions are in `configs/research/price_feature_selection_full_year.yaml`.

Forecast vintages for NORD are assumed available at each gate; their historical issue times are not verified. Previous-day quantities are assumed available even where a same-clock observation may postdate the gate, as authorized for retrospective analysis. No same-day realized target or price spread is used as a forecast covariate. Missing historical inputs shorten the context to a complete recent suffix of at least seven days; missing delivery covariates exclude that forecast. Cancellation and DST handling follow the existing IDA pipeline. Pinned source checksums and per-variant coverage are saved.

October 2025 through October 2026 is retrospective development, and November is reported separately without selecting on it. Reports compare all variants on the intersection of observed products, show MAE, WAPE, RMSE and other metrics, and record coverage. The winner per IDA is selected by development MAE, then WAPE, then feature name. Six plots show its three best and three worst development days by daily MAE over matched products. Pretraining may overlap the study period.

Run from the repository root in tmux, using up to four independent GPUs and one native thread per worker:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_ida_zero_shot_study --config configs/research/zero_shot_ida_feature_combinations.yaml --run --devices cuda:0 cuda:1 cuda:2 cuda:3 --batch-size 10 --threads 1 --prepare-workers 5 --report-workers 30
```

Preparation streams bundles to available GPU workers. Saved inference can be resumed with the same command and GPU list. To regenerate reports from completed forecasts:

```bash
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_ida_zero_shot_study --config configs/research/zero_shot_ida_feature_combinations.yaml --report-only --refresh-reports --report-workers 30
```

Outputs are under `outputs/ida_timesfm_holiday_external/<IDA>/comparison/`, with individual study bundles and reports alongside. The snapshot and model environments must already exist.

# DA TimesFM feature ablation

This zero-shot study uses TimesFM 3, up to 28 days of historical DA prices, and NORD load, total generation, solar and wind forecasts. It compares the NORD baseline against a national holiday indicator, then all combinations of previous-day gas, French quantities and Swiss quantities on top of that indicator. The precise external definitions are in `configs/research/price_feature_selection_full_year.yaml`. The holiday indicator is 1 for each local product on an Italian national holiday, including Easter and Easter Monday, and 0 otherwise; it is supplied for history and the known delivery horizon.

October 2025 through October 2026 is retrospective development; November is reserved. The forecast is the median of nine quantiles. Historical NORD vintages and the availability of previous-day external observations are assumed; same-clock prior-day observations can be later than the DA gate. Missing historical values shorten the context to a complete suffix of at least seven days; missing delivery covariates exclude that day. Results can overlap foundation-model pretraining. Every variant is scored on common products, with row counts, MAE, WAPE, RMSE, bias and other scores saved in the comparison. DA hourly history is expanded to quarter hours according to the established input workflow.

Run from the repository root in tmux:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_da_zero_shot_studies --config configs/research/zero_shot_da_holiday_ablation.yaml --run --devices cuda:0 cuda:1 cuda:2 cuda:3 --batch-size 10 --threads 1 --prepare-workers 5 --report-workers 30
```

GPU inference starts as soon as bundles are ready; independent input preparation and reporting use Python processes. To regenerate reports from completed inference use the same command with `--report-only --report-workers 30` instead of `--run` and the GPU options. Outputs are in `outputs/da_timesfm_holiday_external/`; the combined report is in `comparison/summary.md`. Existing input bundles and predictions are checked and reused; resume with the same device list.

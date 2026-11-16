# TimesFM realized NORD covariate study

This is a retrospective zero-shot inference study for DA, IDA1, IDA2 and IDA3, using 28 days of price history. DA uses NORD forecasts. IDAs use NORD forecasts and a reconstructed historical cancellation-aware spread. The existing price-only bundles are paired with their original pinned forecast snapshot, while offshore forecasts and actual observations come from a separate newly pinned snapshot. This recovers days excluded by the previous forecast-complete bundles. Both manifest checksums and the actual-source checksum are recorded for every generated bundle.

Three variants are evaluated per auction:

1. `baseline`: unchanged input sequences, with TimesFM 3 selected from the baseline bundle.
2. `realized_forecast`: replace the historical portions of load, solar, onshore-wind and offshore-wind forecast covariates by their corresponding realized NORD series. Append the original delivery-day forecast to each corresponding historical series. Total generation remains its original historical and future forecast sequence.
3. `realized_forecast_hydro_gas`: the previous variant plus four historical-only covariates: pumped-hydro production, run-of-river production, reservoir-hydro production and gas-fired production. Pumping consumption is excluded. These are separate series; no synthetic hydro total is constructed.

The model directly predicts the auction price and emits nine quantiles; the median is the point forecast. Forecast prices, reference scores, MAE, WAPE, RMSE, bias, quantile metrics and coverage are produced by the existing reporter. The comparison uses common products across all variants independently per auction; November is reserved and reported separately. No weight training or calibration is performed. TimesFM pretraining might overlap the evaluation period. NORD forecast vintages and realized series publication times are unverified. The approved retrospective assumption treats observations for delivery dates before the target day as available at every gate, including observations which would occur after the DA gate; this is explicitly not a strictly real-time evaluation. No target-day actual is included.

For each historical observation gap, use its aligned forecast if present. For gaps that remain, interpolate consecutive missing quarter-hours only when the gap is at most two hours and an adjacent historical value is available; otherwise use zero. Interpolation is restricted to contiguous UTC quarter-hours and never crosses IDA3's overnight break or uses delivery-horizon data. For each covariate, if any delivery-horizon forecast is absent, the entire covariate is passed as past-only for that day. Coverage records per-feature forecast substitutions, interpolated values, zeros, and past-only fallback. Days are rebuilt from the price-only history bundles so missing forecast days are not excluded in advance. IDA historical spreads are reconstructed using the previous eligible same-delivery clearing; gaps in that historical spread follow the same interpolation/zero policy and are recorded.

Both realized variants include offshore wind: its historical actuals and, where the entire delivery forecast is present, the offshore delivery forecast. If its delivery forecast is missing, it is past-only. The offshore actuals and forecasts come from the newly pinned ENTSO-E snapshot; the four original NORD forecast inputs and price histories come from the earlier pinned study snapshot. An unavailable offshore series may result in an all-zero historical covariate, which is explicitly recorded and should be considered when interpreting scores. A context with fewer than seven days of valid price history is excluded. Actual observations are never taken from the delivery day.


After the current downloader has finished, pin actuals and forecasts from the catalog used by `configs/api_experiments/entsoe_nord_forecasts_actuals.yaml` (default `data/api/market.duckdb`). The snapshot output must be a fresh directory:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.pin_nord_realized_inputs --database data/api/market.duckdb
```

The source selection is `configs/research/price_search_inputs_nord_realized.yaml`; adjust its date range if needed. The initial run pins current DuckDB actual and forecast records, including their snapshot IDs and checksums. Run no other writers while pinning.

Then start the following in tmux. Bundles are prepared by separate Python processes before the GPU queue begins; inference uses up to four independent GPUs, one native thread per worker; reports are produced by separate CPU processes:

```bash
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_timesfm_realized_study --run --devices cuda:0 cuda:1 cuda:2 cuda:3 --prepare-workers 5 --report-workers 30 --threads 1 --batch-size 10
```

Change paths and variant definitions in `configs/research/timesfm_realized_study.yaml`. Rerunning validates existing bundle checksums and source identities and reuses saved predictions; keep the original GPU list for checkpoint affinity. `--prepare-only` prepares bundles without inference; `--report-only` regenerates reports from complete forecasts. Results are saved under `outputs/timesfm_realized_nord_fallback_v2/<auction>/comparison/`.

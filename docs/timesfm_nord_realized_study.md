# TimesFM realized NORD covariate study

This is a retrospective zero-shot inference study for DA, IDA1, IDA2 and IDA3, with the existing 28-day TimesFM 3 bundles as pinned baselines. DA uses the NORD forecast baseline. IDAs use NORD forecasts and the existing historical cancellation-aware spread. Baseline bundles are pinned separately from the new actual-observation snapshot: price and forecast records therefore remain those in the baseline bundle rather than being silently replaced by the newer snapshot. Both manifest checksums and the actual-source checksum are recorded for every generated bundle.

Three variants are evaluated per auction:

1. `baseline`: unchanged input sequences, with TimesFM 3 selected from the baseline bundle.
2. `realized_forecast`: replace the historical portions of load, solar and onshore-wind forecast covariates by their corresponding realized NORD series. Append the original delivery-day forecast to each corresponding historical series. Total generation remains its original historical and future forecast sequence.
3. `realized_forecast_hydro_gas`: the previous variant plus four historical-only covariates: pumped-hydro production, run-of-river production, reservoir-hydro production and gas-fired production. Pumping consumption is excluded. These are separate series; no synthetic hydro total is constructed.

The model directly predicts the auction price and emits nine quantiles; the median is the point forecast. Forecast prices, reference scores, MAE, WAPE, RMSE, bias, quantile metrics and coverage are produced by the existing reporter. The comparison uses common products across all variants independently per auction; November is reserved and reported separately. No weight training or calibration is performed. TimesFM pretraining might overlap the evaluation period. NORD forecast vintages and realized series publication times are unverified. The approved retrospective assumption treats observations for delivery dates before the target day as available at every gate, including observations which would occur after the DA gate; this is explicitly not a strictly real-time evaluation. No target-day actual is included.

A completely missing historical local day for an actual covariate is filled with zeros, as requested, and counted in coverage. Remaining missing realized values are filled first using the corresponding historical forecast for load, solar and onshore wind, then by linear interpolation within the historical context only. Hydro and gas have no matching forecast and use historical interpolation. Leading and trailing gaps use the nearest historical value, matching NumPy interpolation behavior. An entirely absent actual series consequently becomes zero for all historical days; review coverage before interpreting results. A context with fewer than seven complete days is excluded. Delivery-day forecast values must already be present in the baseline bundle: past-only fallback for missing delivery forecasts is not yet implemented. Native intervals are carried within their declared interval; duplicate or overlapping records fail. Coverage files list missing values and trimmed contexts. If the offshore series is absent it is not added to this experiment. Missing source datasets can make an entire variant ineligible, in which case there is no valid matched comparison.

After the current downloader has finished, pin actuals from the catalog used by `configs/api_experiments/entsoe_nord_forecasts_actuals.yaml` (default `data/api/market.duckdb`). The snapshot output must be a fresh directory:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.pin_nord_realized_inputs --database data/api/market.duckdb
```

The source selection is `configs/research/price_search_inputs_nord_realized.yaml`; adjust its date range if needed. The initial run pins current DuckDB records, including their snapshot IDs and checksums. Run no other writers while pinning.

Then start the following in tmux. Bundles are prepared by separate Python processes before the GPU queue begins; inference uses up to four independent GPUs, one native thread per worker; reports are produced by separate CPU processes:

```bash
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_timesfm_realized_study --run --devices cuda:0 cuda:1 cuda:2 cuda:3 --prepare-workers 5 --report-workers 30 --threads 1 --batch-size 10
```

Change paths and variant definitions in `configs/research/timesfm_realized_study.yaml`. Rerunning validates existing bundle checksums and source identities and reuses saved predictions; keep the original GPU list for checkpoint affinity. `--prepare-only` prepares bundles without inference; `--report-only` regenerates reports from complete forecasts. Results are saved under `outputs/timesfm_realized_nord/<auction>/comparison/`.

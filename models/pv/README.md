# Saved PV forecast bundles

`farm_345/residual_tcn_51/` is the farm 345 checkpoint selected from the
128-trial TCN search. The directory contains the model weights, its fitted
Ridge starting curve, the frozen feature preprocessing, and checksums in
`manifest.json`. Keep these files together.

The model was fitted on 2025 production and selected by daylight normalized
RMSE on January–April 2026. May–August 2026 was reported separately, but that
period had already been inspected in earlier research, so it is a reused
holdout. The model predicts AC MW and clips to the farm's 1.5246 MW AC rating.
Its 16 inputs use forecast weather, calendar/time, and forecast lead time.
Missing weather is filled within each forecast vintage; no missingness
indicator channels are used. National holidays are a known calendar input.

For inference, pass a forecast-view DataFrame to
`bess_pv_trading.forecasting.pv_forecast_bundle.predict_bundle(frame,
"models/pv/farm_345/residual_tcn_51", farm_id=345)` with the same forecast
columns and timestamp semantics used by the search corpus. In particular,
retain `interval_start_utc`, `delivery_day`, `auction`, `forecast_cycle_utc`,
`forecast_ghi`, `forecast_bhi`, `temperature_c`, `wind_u_10m`, and
`wind_v_10m`. The function checks the bundle hashes and returns AC MW in
the input row order. The bundle contains no measured production and does not
need the original search output directory.

This bundle is an archived candidate for later optimization experiments. The
current trading forecast selector has not been wired to this bundle format;
pointing an existing trading config at this directory alone will not activate
it. Record the bundle path and manifest hash in any future experiment that
uses it.

# Saved PV forecast bundles

The four `farm_<id>/cleaned_hpo_2025_train/` directories contain the
validation-selected forecast-input winners from the cleaned 1,136-fit study.
Farms 1, 901 and 910 selected direct TCNs; farm 345 selected scale-calibrated
pvlib on January–April validation. Training uses 2025 only. All excluded
feature/AC-target quarters are absent according to
`configs/data_quality/pv_excluded_episodes_2025_to_2026_08.csv`.

Neural directories are standalone bundles for `predict_bundle`. Farm 345's
`physical_parameters.json` instead contains the physical fit, settings and
farm metadata for the shared pvlib engine. Its
`cleaned_hpo_2025_train_tcn_alternative/` directory preserves the
validation-selected TCN family winner: its May–August daylight RMSE is
11.919% of AC rating versus 16.218% for the overall winter-validation pvlib
winner. This is a descriptive held-out comparison, not a change to selection.
Both candidates remain available for later studies.

Each directory includes `selection.json`; learned bundles also include
`manifest.json`, frozen preprocessing, Ridge reference and checksummed weights.
Satellite-trained models remain in the study output and are explicitly marked
diagnostic. Full comparison and input counts are in
`docs/results/cleaned_pv_hpo/all_farms_cleaned_16_trials/`. These artifacts do
not automatically change the existing trading forecast selector.

`farm_345/residual_tcn_51/` is the farm 345 checkpoint selected from the
128-trial TCN search. The directory contains the model weights, its fitted
Ridge starting curve, the frozen feature preprocessing, and checksums in
`manifest.json`. Keep these files together.

`farm_345/residual_tcn_48_no_wind_speed_timing/` is the matching 128-trial
feature ablation. It removes derived wind speed, forecast lead time, and the
two forecast-cycle-hour encodings from both the Ridge baseline and TCN,
leaving 12 inputs. It uses the same data splits and training settings.
Validation daylight nRMSE was 11.227% of AC rating, versus 11.292% for the
16-feature checkpoint; on the reused May–August 2026 test period the scores
were 13.756% and 13.739%, respectively. Keep both bundles for subsequent
trading comparisons; these differences are small and mixed.

The model was fitted on 2025 production and selected by daylight normalized
RMSE on January–April 2026. May–August 2026 was reported separately, but that
period had already been inspected in earlier research, so it is a reused
holdout. The model predicts AC MW and clips to the farm's 1.5246 MW AC rating.
Its 16 inputs use forecast weather, calendar/time, and forecast lead time.
Missing weather is filled within each forecast vintage; no missingness
indicator channels are used. National holidays are a known calendar input.

For inference, pass a forecast-view DataFrame to
`bess_pv_trading.forecasting.pv_forecast_bundle.predict_bundle(frame,
"models/pv/farm_345/residual_tcn_51", farm_id=345)` (or use the ablation
directory) with the same forecast
columns and timestamp semantics used by the search corpus. In particular,
retain `interval_start_utc`, `delivery_day`, `auction`, `forecast_cycle_utc`,
`forecast_ghi`, `forecast_bhi`, `temperature_c`, `wind_u_10m`, and
`wind_v_10m`. The function checks the bundle hashes and returns AC MW in
the input row order. The bundle contains no measured production and does not
need the original search output directory.

Continuous forecasting supports these portable bundles with replacement kind
`weather_bundle`, an artifact path pointing to `manifest.json`, and an explicit
availability timestamp. Replacement remains disabled by default. Record the
bundle path and manifest hash in any experiment that enables it.


## Approved per-farm research pilot

`farm_<id>/pilot_2025_train/direct_tcn/` is the easy-to-find selected pilot.
Farm 1 uses `direct_tcn_02` without timing; farm 345 uses `direct_tcn_12`
without timing; farm 901 uses `direct_tcn_timing_05`; farm 910 uses
`direct_tcn_timing_paired`. Timing means lead hours and sine/cosine of cycle
UTC hour. Check the parent `selection.json` and bundled preprocessing for the
exact features and provenance. The choices are informed by the reused
May–August 2026 report period, including farm 345's stronger summer TCN;
they preserve the main study's original validation rankings.

The 80-fit timing comparison and worst-profile review are in
[the pilot report](../../docs/results/retrained_pv_pilot/README.md).
`timing_restored_2025_train/` preserves the alternative timing-enabled
candidates. The shared `pv_forecast_pilot.models_by_farm` registry provides
paths without enabling replacement in existing trading runs. Artifact
availability is conservatively September 2026; retrospective runs must opt in.

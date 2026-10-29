# Retrospective physical PV parameter search

The run fits physical conversion parameters per farm on curated AC power from January 2025 through July 2026, then ranks candidates on August 2026 daytime RMSE. It uses historical satellite-derived GHI and the provider-confirmed historical diffuse fraction, `DHI/GHI`. GHI gaps are linearly interpolated only between existing endpoints and flagged; intervals with missing diffuse fraction remain ineligible unless GHI is zero. Temperature and wind are forecasts, not observations. For each interval, all eligible 00, 06, 12 and 18 UTC forecast cycles in their first 24 hours are used, each with weight 0.25. Their hourly instantaneous values are interpolated at quarter-hour midpoints. Historic forecast-release availability is deliberately ignored for this metadata study. Fixed metadata, tracking classification, AC rating and source-irradiance interpretations are retained. The search uses acknowledged solar-position proxies, not verified farm coordinates.

The 624 initial trials comprise the unchanged baseline for each farm; all 27 proxy/transposition combinations for each fixed farm, under MSE and Huber fitting; and 128 sampled structural configurations for each tracker farm, under both fitting losses. Faiman u0/u1, PV temperature coefficient and AC output scale are optimized continuously using bounded L-BFGS-B. The Huber threshold is 5% of AC nameplate. The initial numerical budget is 120 function evaluations and two starts. Twelve further fits refine the leading three fitted candidates per farm, using three starts and up to 600 evaluations. The search does not exhaust every tracker/bifacial combination and considers Faiman only, not other thermal model families. For farm 901 the initial sample has 234 rear-enabled versus 23 front-only trials, including the reference. Inspect `trial_plan.json`, `refinement_plan.json`, and `validation_ranking.csv` for exact settings and all scores.

## Selected physical configurations

Scores below are daytime RMSE on the four individual temperature/wind views, normalized by AC capacity. All use historical GHI and diffuse fraction and thus are not forecast-weather scores.

| Farm | Trial | Proxy | Transposition | Loss | Scale | u0 | u1 | Gamma per °C | Tracker/bifacial assumptions | Selected RMSE | Original RMSE |
|---|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 0 | Milan | isotropic | fixed reference | 0.53194 | 60.00 | 15.00 | -0.004 | fixed | 3.416% | 3.416% |
| 345 | 10010 | Genoa | isotropic | Huber | 0.54972 | 60.00 | 15.00 | -0.002 | fixed | 8.675% | 13.493% |
| 901 | 216 | Trieste | isotropic | Huber | 0.76156 | 49.74 | 12.18 | -0.002 | GCR 0.50, max 60°, no backtracking; bifaciality 0.65, albedo 0.35, height/pitch 0.50 | 7.290% | 8.192% |
| 910 | 10059 | Aosta | isotropic | MSE | 0.91250 | 10.46 | 0.67 | -0.002 | GCR 0.65, max 80°, backtracking | 13.151% | 14.027% |

The selected optimized fits for farms 345, 901 and 910 reported convergence; gamma hits its upper bound for all three, and u0 and u1 also hit their upper bounds for farm 345. Farm 1 is the unchanged reference and has no optimizer convergence result. Boundary values and the selected proxy locations are effective forecasting assumptions, not verified plant properties. Configuration selection used August and is not an independent evaluation; the different sampled structural configurations, particularly front versus rear choices, are unbalanced.

## Forecast-ensemble comparison

Each selected physical chain was applied to the existing preceding-day 00 UTC, 50-member forecast. Its seven linear quantile regressors were fitted on January 2026 ensemble summaries only. The new physical fit itself uses January observations as part of the January 2025–July 2026 parameter fitting, and was selected using August, so this is a retrospective, validation-selected comparison. Quantile curves use linear interpolation and constant tails beyond p05 and p95. The compared saved learned and physical calibrators used additional August 2–19 out-of-fold calibration data, whereas the January-only comparison refits the previous physical calibrator using the same January observations as the new physical calibrator. All methods below use matched daytime intervals over August 20–31; units are percent of AC capacity.

| Farm | New physical + January-only calibrated CRPS | Previous physical + January-only calibrated CRPS | Previous physical + original calibration CRPS | New nominal 80% coverage |
|---|---:|---:|---:|---:|
| 1 | 9.970% | 9.970% | 2.906% | 20.6% |
| 345 | 8.901% | 9.692% | 3.838% | 27.0% |
| 901 | 7.306% | 7.878% | 6.203% | 79.4% |
| 910 | 7.938% | 7.850% | 4.890% | 82.3% |

The large deterioration for farms 1 and 345 under January-only calibration is a warning about seasonal transfer. Farm 1's physical chain is unchanged and its two January-only scores are identical, showing that the difference from its original calibrated result is not a new physical conversion. Original and January-only calibrators have different histories, so their scores do not isolate physical-chain effects. `paired_forecast_metrics.csv` also contains raw-ensemble and point metrics and the previously selected learned models.

## Outputs and interpretation

Per-farm `selected_parameters.json`, full day rankings and nine PNGs are saved under `outputs/pv_realized_irradiance_search/four_vintages_2025_2026_v4/farm_<id>/`. The `training_day_plots/trial_<id>/` directory contains three literal lowest-error days, three highest-error days, and three lowest-error days with all 62 daytime intervals present. Ranking uses the four-view mean and curated targets; incomplete intervals are shown as plot gaps, not connected. Day counts are displayed. The daytime selection window is 06:00–21:30 Europe/Rome. Neither the four temperature/wind views nor their min–max shading are the 50-member weather-ensemble uncertainty distribution.

The midpoint alignment software check passed; the prepared inputs were also checked to have four distinct 0.25-weight views for each eligible interval. The previous run with start-of-quarter interpolation was stopped and its results are not used. Some corrected checkpoints were reused after an exact comparison of all eligible rows from the older and current curated corpora; the output directory contains `reused_trial_provenance.md`.

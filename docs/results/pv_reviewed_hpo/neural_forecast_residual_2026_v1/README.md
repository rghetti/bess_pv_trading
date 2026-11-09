# Statistical residual corrections to the four-cycle neural PV models

The selected correction does not improve the May–August test nRMSE for any
farm. Farm 345 selects a Ridge correction on March–April, but its test nRMSE
rises and its negative bias grows. The other farms select no correction.
The selected correction has not replaced the standard models.

| Farm | Selected correction | Validation nRMSE: baseline / selected (%) | Test nRMSE: baseline / selected (%) | Test bias: baseline / selected (% AC) |
|---:|---|---:|---:|---:|
| 1 | none | 5.564 / 5.564 | 5.783 / 5.783 | 0.534 / 0.534 |
| 345 | Ridge, alpha 1000 | 10.273 / 8.496 | 13.151 / 13.422 | -3.566 / -7.658 |
| 901 | none | 11.161 / 11.161 | 10.961 / 10.961 | 1.711 / 1.711 |
| 910 | none | 9.291 / 9.291 | 12.867 / 12.867 | 4.991 / 4.991 |

On complete measured test hours, the selected farm 345 correction reduces
quarter-weighted within-hour shape RMSE from 4.472% to 4.426% of rated AC
capacity. Each forecast cycle is an individual view, so hours can be repeated.
This small shape change accompanies worse overall test nRMSE and bias.

## Method and limitations

Each farm uses its frozen validation-selected neural model and the cached exact
forecast-cycle inputs and predictions. The correction trains on January–February
2026 and is selected against no correction on March–April, before test scoring
on May–August. There are 20,587 fitting, 27,069 selection and 56,826 test
forecast-target pairs per farm. The underlying neural weights were trained on
2025 data, but their model selection used January–April 2026. The entire 2026
reporting period has been inspected previously. Thus the correction fitting
period is outside neural weight training but is not independent of neural model
selection, and these test scores are retrospective.

The inputs are the 24 saved, standardized forecast weather and calendar features,
the neural prediction, quarter position, previous and next hourly forecast GHI
changes within the same issue, and quarter-position interactions with GHI and
predicted power. They contain no solar-position or missingness flags. At issue or
2026 forecast-context boundaries, unavailable neighboring forecasts are replaced
by the current GHI, without an added binary flag. Historical GHI never enters a
correction input. Each forecast issue is treated under the original assumed
six-hour availability mask for scoring; earlier forecast context is known after
that issue's assumed release. Hourly radiation means remain constant across the
four quarters.

For each farm, three Ridge penalties (0.1, 10, 1000), three histogram gradient
boosters (7, 15, 31 leaves), and no correction are compared using validation
nRMSE with equal weight per delivery day. The fitting objective is day-balanced
squared residual in per-unit AC power. The fitted correction is not refit on the
selection period. Predictions are bounded by zero and plant capacity, and set to
zero in configured local monthly nighttime windows. Targets include the reviewed
pvlib replacements; measured and synthetic scores are provided separately.

The score and bias columns are normalized by farm rated AC power. Row counts
represent forecast-target pairs, including repeated forecasts of physical
quarters. The `window_energy_bias_mwh` column sums those repeated pairs and is
not realized energy delivered by the farm. The within-hour analysis is
quarter-weighted on complete measured hours and removes each hour's mean before
calculating errors; it is not an estimate of attainable improvement.

- [All validation candidates](candidates.csv)
- [Metrics by target source and split](metrics.csv)
- [Monthly metrics](monthly.csv)
- [Measured within-hour shape metrics](within_hour_shape.csv)
- [Inputs, features, hashes and selected models](manifest.json)

The fitted farm 345 correction, prediction arrays and daily plots are stored
under `outputs/pv_reviewed_hpo/neural_forecast_residual_2026_v1/`. The configured
run is documented in `configs/pv_experiments/reviewed_neural_residual.yaml` and
implemented by `scripts/study_reviewed_pv_residual.py`.

# Reviewed PV HPO

Search identity: `a14fec0560210b82a45b358dce86ab2ca7d20d2dec24e1f6dd144709d1e2f579`. HPO trials and family winners were selected on validation only.
The 2026 report period was previously inspected; these are retrospective comparisons.
The 2026 scoring labels include user-requested pvlib reconstructions, not independent observations.

Training uses historical GHI and weather from all four forecast cycles over the 48 hours from issue; the first six hours are hindsight. Validation and testing score targets only after the assumed weather release.
The month-specific daylight scoring window is used. Hourly radiation means are repeated over
their four quarter-hours. The six-hour release lag is an assumption, not proof of historical availability.

No physical PV model or physical-model prediction is a candidate or input feature.
Zero and a training-only day-balanced constant are provided as statistical baselines.
Test bootstrap intervals compare each family winner with the validation-selected model and do not correct
for previous use of the report period or uncertainty in synthetic labels.

[Every validation trial](all_trials_validation.csv) · [Family/feature winners](family_feature_winners.csv) ·
[Selected model by farm](selected_by_farm.csv) · [Top three by farm](top_three_by_farm.csv) ·
[Top-three metrics](top_three_metrics.csv) · [Winner metrics](winner_metrics.csv) ·
[Monthly metrics](monthly_metrics.csv) · [Baselines](baseline_metrics.csv) ·
[Historical-GHI interpolation diagnostics](ghi_interpolation_metrics.csv) ·
[Paired test bootstrap](paired_test_bootstrap.csv)

The historical-GHI interpolation flag describes synthetic-target provenance.

| Farm | Feature set | Family | Selected trial | Validation nRMSE (% AC) |
|---|---|---|---|---:|
| 1 | extended | polynomial_ridge | polynomial_ridge_extended_048 | 4.938 |
| 345 | core | huber | huber_core_057 | 8.421 |
| 910 | extended | catboost | catboost_extended_043 | 9.353 |
| 901 | extended | catboost | catboost_extended_022 | 10.412 |

## Top three configurations per farm

Rank is based only on validation daylight nRMSE. Scores use all reviewed targets;
the linked CSV separates measured and synthetic targets for every split.

| Farm | Rank | Trial | Validation nRMSE (%) | Test nRMSE (%) | Test nMAE (%) | Test bias (%) |
|---|---:|---|---:|---:|---:|---:|
| 1 | 1 | polynomial_ridge_extended_048 | 4.938 | 6.285 | 4.335 | -1.000 |
| 1 | 2 | polynomial_ridge_extended_015 | 4.938 | 6.285 | 4.337 | -0.998 |
| 1 | 3 | polynomial_ridge_extended_039 | 4.938 | 6.287 | 4.333 | -1.004 |
| 345 | 1 | huber_core_057 | 8.421 | 13.923 | 10.387 | -7.876 |
| 345 | 2 | huber_core_014 | 8.429 | 13.920 | 10.385 | -7.870 |
| 345 | 3 | huber_core_003 | 8.441 | 13.914 | 10.383 | -7.860 |
| 910 | 1 | catboost_extended_043 | 9.353 | 11.960 | 7.482 | 2.898 |
| 910 | 2 | catboost_extended_013 | 9.369 | 11.909 | 7.380 | 2.390 |
| 910 | 3 | catboost_extended_022 | 9.373 | 11.509 | 7.336 | 0.901 |
| 901 | 1 | catboost_extended_022 | 10.412 | 11.528 | 7.376 | -0.295 |
| 901 | 2 | catboost_extended_026 | 10.427 | 11.473 | 7.123 | 0.481 |
| 901 | 3 | extra_trees_extended_035 | 10.438 | 11.414 | 7.639 | -1.203 |

Selected-model daily plots are grouped by farm, split and local delivery date under
`outputs/pv_reviewed_hpo/four_cycle_statistical_night3000_v1/plots/`. Family models, preprocessing and hashes are under each farm’s `models/` directory.

[Metrics by issue cycle and lead time](cycle_lead_metrics.csv)

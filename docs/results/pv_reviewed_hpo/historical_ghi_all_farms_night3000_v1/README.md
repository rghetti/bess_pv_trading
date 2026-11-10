# Reviewed historical-GHI PV HPO

Search identity: `e8e1b55f600928d6c139d59cb176e59ebb3e6b2a8f04146b034fc80adff2af80`. HPO trials and family winners were selected on validation only.
The 2026 report period was previously inspected; these are retrospective comparisons.
The 2026 scoring labels include user-requested pvlib reconstructions, not independent observations.

Training uses historical GHI and 3000 eligible night targets per farm, evenly spread over the year.
Validation and testing use the D-1 00 UTC
forecast GHI and the month-specific daylight scoring window. Hourly radiation means are repeated over
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

The historical-GHI interpolation flag describes a model input during training; on validation/test
it describes synthetic-target provenance because the model input is forecast GHI.

| Farm | Feature set | Family | Selected trial | Validation nRMSE (% AC) |
|---|---|---|---|---:|
| 1 | extended | bigru | bigru_extended_021 | 4.708 |
| 345 | extended | transformer | transformer_extended_001 | 7.645 |
| 910 | extended | tcn | tcn_extended_003 | 8.977 |
| 901 | extended | bilstm | bilstm_extended_021 | 10.241 |

## Top three configurations per farm

Rank is based only on validation daylight nRMSE. Scores use all reviewed targets;
the linked CSV separates measured and synthetic targets for every split.

| Farm | Rank | Trial | Validation nRMSE (%) | Test nRMSE (%) | Test nMAE (%) | Test bias (%) |
|---|---:|---|---:|---:|---:|---:|
| 1 | 1 | bigru_extended_021 | 4.708 | 6.069 | 4.006 | -1.048 |
| 1 | 2 | bigru_extended_033 | 4.709 | 5.962 | 3.940 | -0.767 |
| 1 | 3 | bilstm_extended_022 | 4.716 | 6.069 | 4.139 | -1.619 |
| 345 | 1 | transformer_extended_001 | 7.645 | 10.366 | 6.893 | -1.169 |
| 345 | 2 | transformer_extended_061 | 7.794 | 13.505 | 9.770 | 7.129 |
| 345 | 3 | transformer_core_048 | 7.884 | 12.020 | 8.829 | -2.200 |
| 910 | 1 | tcn_extended_003 | 8.977 | 10.992 | 6.630 | 1.071 |
| 910 | 2 | tcn_core_016 | 9.038 | 11.417 | 6.927 | 1.250 |
| 910 | 3 | tcn_core_045 | 9.082 | 10.929 | 6.643 | 1.431 |
| 901 | 1 | bilstm_extended_021 | 10.241 | 11.231 | 6.776 | 1.137 |
| 901 | 2 | tcn_extended_004 | 10.241 | 11.319 | 7.874 | -1.844 |
| 901 | 3 | tcn_extended_038 | 10.248 | 10.699 | 6.563 | 0.302 |

Selected-model daily plots are grouped by farm, split and local delivery date under
`outputs/pv_reviewed_hpo/historical_ghi_all_farms_night3000_v1/plots/`. Family models, preprocessing and hashes are under each farm’s `models/` directory.

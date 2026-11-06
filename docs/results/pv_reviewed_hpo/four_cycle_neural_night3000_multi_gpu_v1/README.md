# Reviewed PV HPO

Search identity: `9724bb2fc15ae4bc9f730c9fd8336be080d9b6c98deeb6ae0985a645baf81305`. HPO trials and family winners were selected on validation only.
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
| 1 | extended | bigru | bigru_extended_037 | 4.696 |
| 345 | core | bigru | bigru_core_025 | 8.167 |
| 910 | core | tcn | tcn_core_029 | 8.731 |
| 901 | extended | tcn | tcn_extended_021 | 9.854 |

## Top three configurations per farm

Rank is based only on validation daylight nRMSE. Scores use all reviewed targets;
the linked CSV separates measured and synthetic targets for every split.

| Farm | Rank | Trial | Validation nRMSE (%) | Test nRMSE (%) | Test nMAE (%) | Test bias (%) |
|---|---:|---|---:|---:|---:|---:|
| 1 | 1 | bigru_extended_037 | 4.696 | 5.783 | 3.564 | 0.534 |
| 1 | 2 | bigru_extended_046 | 4.733 | 5.960 | 3.945 | -1.150 |
| 1 | 3 | bigru_extended_053 | 4.754 | 6.018 | 4.021 | -0.807 |
| 345 | 1 | bigru_core_025 | 8.167 | 13.151 | 10.244 | -3.566 |
| 345 | 2 | transformer_extended_019 | 9.309 | 11.268 | 8.029 | -2.802 |
| 345 | 3 | bilstm_core_001 | 9.485 | 11.457 | 7.706 | -2.059 |
| 910 | 1 | tcn_core_029 | 8.731 | 12.867 | 8.145 | 4.991 |
| 910 | 2 | tcn_core_013 | 8.752 | 11.048 | 6.828 | 0.978 |
| 910 | 3 | tcn_core_063 | 8.754 | 10.934 | 6.676 | 1.131 |
| 901 | 1 | tcn_extended_021 | 9.854 | 10.961 | 6.619 | 1.711 |
| 901 | 2 | tcn_extended_046 | 9.873 | 10.814 | 7.105 | -1.044 |
| 901 | 3 | tcn_extended_038 | 9.893 | 10.875 | 7.186 | -1.160 |

Selected-model daily plots are grouped by farm, split and local delivery date under
`outputs/pv_reviewed_hpo/four_cycle_neural_night3000_multi_gpu_v1/plots/`. Family models, preprocessing and hashes are under each farm’s `models/` directory.

[Metrics by issue cycle and lead time](cycle_lead_metrics.csv)

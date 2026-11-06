# Selected four-cycle reviewed-GHI PV models

These four validation-selected bundles are the current standard PV models.
Each predicts 15-minute AC MW from a 48-hour issue context using the four
deterministic weather cycles (00, 06, 12 and 18 UTC). Training used historical
GHI; validation and test used forecast GHI after the assumed six-hour release.
The input sets contain no missing-feature binary flags.

| Farm | Model name | Architecture | Feature set | Validation nRMSE (%) | Test nRMSE (%) |
|---:|---|---|---|---:|---:|
| 1 | `bigru_extended_037` | Bidirectional GRU, 3 layers, hidden size 128 | extended | 4.696 | 5.783 |
| 345 | `bigru_core_025` | Bidirectional GRU, 3 layers, hidden size 512 | core | 8.167 | 13.151 |
| 901 | `tcn_extended_021` | Temporal Convolutional Network (TCN), 3 layers, channel width 512 | extended | 9.854 | 10.961 |
| 910 | `tcn_core_029` | Temporal Convolutional Network (TCN), 3 layers, channel width 512 | core | 8.731 | 12.867 |

The 3,000 nighttime training examples per farm were sampled as physical
quarter-hours before duplication across cycles. Predictions are clipped to
zero in the configured month-specific local night window by the inference
adapter. Reviewed pvlib replacements are counterfactual targets, not metered
production. The 2026 evaluation period has been inspected previously.

Each `farm_<id>/` contains `model.pt`, `preprocessor.json` and `result.json`.
Retrospective `prediction_mw.npy` arrays are omitted; their checksum remains
in `result.json`, and the inference adapter does not require that array.
File hashes, exact trial parameters, metrics, and source provenance are in
`manifest.json`. The source search outputs and complete plots are retained under
`outputs/pv_reviewed_hpo/four_cycle_neural_night3000_multi_gpu_v1/` and `docs/results/pv_reviewed_hpo/four_cycle_neural_night3000_multi_gpu_v1/` on the training machine.

Load these bundles with `predict_reviewed_hpo` in
`src/forecasting/inference/pv_reviewed_hpo.py`. Supply causally available
updating forecast cycles at each cutoff; the reported validation/test scores
use the configured forecast-vintage assumptions.

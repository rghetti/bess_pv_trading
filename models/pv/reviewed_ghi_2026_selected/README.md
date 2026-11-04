# Selected reviewed-GHI PV models

This directory contains the validation-selected model for each farm from the
completed `historical_ghi_all_farms_night3000_v1` search:

| Farm | Directory | Selected model |
| --- | --- | --- |
| 1 | `farm_1/` | Extended BiGRU `bigru_extended_021` |
| 345 | `farm_345/` | Extended Transformer `transformer_extended_001` |
| 901 | `farm_901/` | Extended BiLSTM `bilstm_extended_021` |
| 910 | `farm_910/` | Extended TCN `tcn_extended_003` |

Each bundle contains original model weights, fitted preprocessing and its
complete result record. `manifest.json` records source paths, selection,
metrics and file hashes. The result record includes a checksum for the original
retrospective prediction array; that array is not needed for inference and is
not included in this directory. The original search results and predictions
remain in `outputs/pv_reviewed_hpo/historical_ghi_all_farms_night3000_v1/`;
the other saved PV model weights were removed at the user's request. Source
paths in the manifest are historical provenance, not loadable model locations.

The target is quarter-hour AC MW. Training used historical GHI; validation and
test used the preceding delivery day's 00 UTC forecast, with the remaining
weather features from that forecast cycle. These are retrospective models:
the 2026 validation period was previously inspected, some labels are synthetic,
and the configured forecast-release delay is an assumption. Historical
evaluation also split neural sequences at prior excluded intervals.

The adapter `predict_reviewed_hpo` in
`src/forecasting/inference/pv_reviewed_hpo.py` loads these bundles; it requires
the forecast features, timezone-aware forecast-cycle, availability and
decision-cutoff timestamps, and an enabled monthly night policy. It clips
predictions to zero in that policy's local nighttime windows. Available later
forecast cycles are supported as requested; their performance differs from the
reported evaluation on the preceding-day cycle and must be evaluated separately.
The continuous optimization registry supports these bundles when replacement is
explicitly enabled and the model is available at the cutoff, or retrospective
use is explicitly enabled. Missing inputs and unavailable weather are errors;
the model is not silently given future weather.

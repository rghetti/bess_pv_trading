# PV forecasting modules

The forecasting code is grouped by its primary role:

- `data/`: weather, calendar, target and input preparation.
- `models/`: physical, neural and probabilistic model definitions.
- `training/`: fitting, calibration, tuning and parameter searches.
- `inference/`: loading saved models and producing forecasts.
- `evaluation/`: metrics, diagnostics, audits and model comparisons.
- `plotting/`: plots and study reports.
- `studies/`: orchestration for complete forecasting studies.
- `online/`: online adaptation and CRPS weighting.

New code should use the grouped import paths, for example
`bess_pv_trading.forecasting.models.pv` or
`bess_pv_trading.forecasting.inference.pv_neural_predict`. Legacy flat module
imports remain available as aliases so that existing serialized models can be
loaded. Historical run manifests and output artifacts are not rewritten;
source fingerprint checks may require a fresh run ID after source changes.

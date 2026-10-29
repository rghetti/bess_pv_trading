# PV forecast bundles used by continuous optimization

The selected per-farm pilot bundles are self-contained and stored at:

| Farm | Selected bundle | Model | Inputs |
| --- | --- | --- | --- |
| 1 | `farm_1/pilot_2025_train/direct_tcn/` | `direct_tcn_02` | Core weather/calendar inputs |
| 345 | `farm_345/pilot_2025_train/direct_tcn/` | `direct_tcn_12` | Core weather/calendar inputs |
| 901 | `farm_901/pilot_2025_train/direct_tcn/` | `direct_tcn_timing_05` | Core inputs, forecast lead and cycle-hour timing |
| 910 | `farm_910/pilot_2025_train/direct_tcn/` | `direct_tcn_timing_paired` | Core inputs, forecast lead and cycle-hour timing |

Each directory contains `manifest.json`, preprocessing parameters, the fitted
Ridge reference (`baseline.joblib`), and neural weights (`model.pt`). Farm 901
and 910 manifests also define the exact timing channels. All bundles predict AC
MW and clip the result to the configured farm AC rating. They contain no
measured production. The model training and selection details are summarized
in `docs/results/retrained_pv_pilot/README.md`.

## Use in the continuous optimization workflow

The shared registry is `pv.forecast_replacement.models_by_farm` in
`configs/standard.yaml`. Replacement is **disabled by default**; a study must
set `pv.forecast_replacement.enabled: true`. For example:

```yaml
include: ../standard.yaml
pv:
  forecast_replacement:
    enabled: true
    fallback: pvlib
    allow_retrospective_artifacts: false
```

The registry selects the entry matching `farm_id`. Its artifact path points to
the bundle's `manifest.json`, and `available_at_utc` controls whether it was
available by the forecast cutoff. Missing or not-yet-available artifacts use
the configured pvlib fallback. Pilot YAMLs may explicitly enable retrospective
use for comparison. The continuous runner validates the farm identity and
bundle checksums, retains the bundle's feature preprocessing, and records model
and artifact provenance per forecast. The GME oracle experiment currently
uses the configured pvlib forecast path.

For direct inference on an already constructed forecast-view DataFrame:

```python
from bess_pv_trading.forecasting.inference.pv_forecast_bundle import predict_bundle

prediction_mw = predict_bundle(
    forecast_view,
    "models/pv/farm_901/pilot_2025_train/direct_tcn",
    farm_id=901,
)
```

The frame must use the feature columns and timestamp semantics recorded in the
manifest. Typical columns include `interval_start_utc`, `delivery_day`,
`auction`, `forecast_cycle_utc`, `forecast_ghi`, `forecast_bhi`,
`temperature_c`, `wind_u_10m`, and `wind_v_10m`. The bundle checks its recorded
file hashes and returns nonnegative AC MW in input-row order.

These are retrospective pilot models: trained on 2025 data, with model choices
informed by a previously inspected 2026 report period. They are not evidence of
forecast skill available in real time during 2025. The continuous forecast
replacement remains opt-in so standard experiments continue to use pvlib.

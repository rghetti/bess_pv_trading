# Verified farm data contract

Read-only inspection on 16 October 2026 used consolidated Zarr metadata, coordinate arrays, and the matching static Parquet. Forecast values were then read only for a selected farm/vintage; native history was read for one selected farm when applying masks and establishing interpolation evidence. No full-store value scan or dataset hash is required.

## Files and observed schema

- `data/farms/pv_farms_timeseries_20250101_20260831.zarr`
- `data/farms/pv_farms_timeseries_20250101_20260831.parquet`
- Explicit variable mapping: `configs/data_schema.yaml`.

Observed dimensions are `farm_id=4`, `time=58368`, `cycle_time=2440`, `lead_time=49`. Historical UTC interval starts run from 2025-01-01 00:00 to 2026-08-31 23:45. Cycles run from 2024-12-30 00:00 to 2026-08-31 18:00 UTC. Decoded leads are `timedelta64[ns]`, zero through 48 hours; the stored lead encoding has units `hours` and integer dtype. Stored `valid_time` exactly equals cycle plus lead. There is no ensemble/member axis.

| Farm label | AC nameplate kW from Parquet | Tracking ID | Source tilt | Elevation m |
|---|---:|---:|---:|---:|
| 1 | 953.280029296875 | 0 | 15 | 25 |
| 345 | 1524.5999755859375 | 0 | 19 | 111 |
| 901 | 4160 | 3 | 60 | 3 |
| 910 | 6638 | 1 | 30 | 22 |

The small floating-point differences from the user's 953.28 and 1524.60 kW are retained. All four metadata records have zone `NORD`, source azimuth zero and `pv_type_id=0`; the type ID does not select a module technology or database. Farms remain separate plants. Joins use `farm_id` labels, independently of row order.

| Canonical field | Raw variable | Dimensions | Units/semantics |
|---|---|---|---|
| measured AC power | `power` | `(time, farm_id)` | kW, start-labeled native 15-minute mean |
| historical satellite GHI | `ghi` | `(farm_id, time)` | W/m², start-labeled historical interval |
| historical diffuse fraction | `fraction_diffuse` | `(farm_id, time)` | dimensionless DHI/GHI; provider confirmed by user on 29 October 2026; not used in the operational forecast engine |
| forecast GHI | `shortwave_radiation` | `(farm_id, cycle_time, lead_time)` | W/m², hourly mean ending at valid time |
| forecast BHI working interpretation | `direct_radiation` | same | W/m², hourly mean ending at valid time |
| air temperature | `temperature_2m` | same | °C, instantaneous |
| dew point | `dew_point_2m` | same | °C, instantaneous |
| total/low/mid/high clouds | `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high` | same | %, instantaneous |
| precipitation | `precipitation` | same | mm, hourly accumulation ending at valid time |
| snow water | `snow_water_equivalent` | same | m, instantaneous |
| visibility | `visibility` | same | km, instantaneous |
| wind u/v at 10 m | `wind_u_component_10m`, `wind_v_component_10m` | same | signed m/s, instantaneous |
| wind u/v at 100 m | `wind_u_component_100m`, `wind_v_component_100m` | same | signed m/s, instantaneous |

Attributes confirm dimensions, timezone, declared units, forecast valid-time construction, instantaneous versus interval average/accumulation, and historical interval labels. Historical satellite GHI/diffuse attributes say `mean of endpoint readings`; this records an attribute rather than independently verifying the satellite retrieval/measurement operator. The user confirmed with the data provider that `fraction_diffuse` is DHI/GHI. Multiplying the stored endpoint-averaged fraction and GHI approximates the interval-average diffuse component; a mean of ratios need not equal a ratio of means. The following electrical/radiation meanings are supplied by the user, rather than proved by public provider documentation: historical power is already AC, `total_power` is AC nameplate, shortwave is forecast GHI, direct is horizontal-direct radiation, and the wind/temperature source is not attributed to an inferred NWP provider. The user confirmed the current interpretation of metadata and irradiance inputs on 29 October 2026.

## Availability and temporal transformations

`available_at = cycle + 4 h` is a configurable assumption. An explicitly named cycle/farm publication variable overrides it. Vintage selection takes the latest cycle with publication no later than the supplied decision cutoff. Computation/submission buffers are applied by the calling market clock before supplying this cutoff. Forecast vintages are separate from an explicit deterministic member of probability one. The initial weather path is frozen until a new cycle is available; there is no stochastic model of historical forecast revisions.

Hourly radiation valid at 12:00 is redistributed uniformly over intervals starting 11:00, 11:15, 11:30, 11:45; its hourly irradiation is conserved. Precipitation is divided into four quarter-hour accumulations whose sum is the source total. Instantaneous variables interpolate between neighboring points within the same eligible vintage at each interval midpoint. Structural forecast NaNs are never interpolated across, extrapolated, backfilled, replaced by satellite observations, or changed to zero. Only configured finite negative magnitude values are floored. Raw aligned inputs and clipping flags are exported separately.

Naive source times are interpreted as UTC. Native measured power is available at interval end plus configured telemetry latency, not at interval start. A complete interval's measured AC energy is `power_kw * 0.25 / 1000` MWh, without a conversion efficiency.

## Derived observation layer and review boundaries

`data/policy.py` adds the requested policy without changing the legacy audit or source values. Version `requested-v1` defaults to anchor-based plateau matching with four native nonzero samples, `atol_kw=0.01`, `rtol=0.0001`; these are assumptions, not calibrated sensor specifications. Zero, NaN, infinite values or missing native timestamps break a plateau. Three repeated samples remain unchanged; the entire run is zeroed only when its fourth observation is available. Each subsequent sample waits for its own completion. Plateau detection occurs before interpolation.

Positive production is zeroed for interval starts in Europe/Rome 21:30 inclusive to 05:00 exclusive, including DST. This fixed clock rule is independent of Milan proxy solar position. Power and GHI interior NaNs interpolate between finite endpoints; valid values and policy-created zeros remain unchanged. Leading/trailing gaps are retained. The optional maximum-gap parameter measures elapsed time between bracketing endpoints; default null permits every internally bracketed gap. Infinities remain invalid and are not repaired as missing values. Production's clock rule is reapplied after interpolation.

Each interpolation waits until both endpoint values and their masking evidence are available. `ObservationBundle.as_of(cutoff)` returns original already completed readings until a delayed transformation is justified. The finalized treated series is for explicitly ex-post accounting. It is not certified fault-free output or unconstrained PV availability. Interpolation-created plateaus are not reclassified.

Full selected-farm history supplies boundary context before an output slice is trimmed, so an arbitrary output window does not hide a fourth plateau reading or a gap endpoint. This reads two historical arrays for one farm by default, not all forecast fields for all farms. Exports retain raw/treated values, flags, run/gap length, evidence timestamps, policy version and daily AC energy reconciliation. Imputed energy and adjustments to finite existing samples are reported separately.

An optional configured upstream review CSV preserves decisions and exclusions; it is never written. An interpolation depending on a reviewed excluded endpoint remains ineligible through `derived_review_excluded`, while the original review flag remains unchanged. The default example is exploratory and has no configured reviewed exclusion file; existing audit flags are not silently promoted to fault verdicts. Strict mode refuses remaining nonfinite/review-excluded observations and incomplete forecasts after saving diagnostic artifacts.

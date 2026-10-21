# Implemented data and PV details

The executable data layer is `src/bess_pv_trading/data/`; the actual pvlib engine is `src/bess_pv_trading/forecasting/pv.py`. The original root audit package, its data adapter, review artifacts and raw store are preserved. A root import compatibility bridge makes the original audit imports continue working alongside the new package.

## PV assumptions actually used

The production forecast is a deterministic physical conversion of the supplied
weather forecast. It uses the farm Zarr's `shortwave_radiation` (GHI),
`direct_radiation` (working assumption: horizontal direct BHI),
`temperature_2m` (air temperature in Celsius), and the two 10 m wind components.
The wind speed is $v=\sqrt{u_{10}^2+v_{10}^2}$. Geometry and nameplate power come
from the metadata and explicit YAML assumptions. Cloud cover, dew point,
precipitation, snow and visibility remain available in the aligned data but do
not enter this PV model as additional power corrections. Historical satellite
GHI and measured `power` are observation/evaluation data, not future forecast
inputs.

`configs/farms_nord.yaml` acknowledges Milan proxy coordinates (45.4642, 9.1900); supplied farm elevation remains separate. These proxy coordinates are never used as an audit fault classifier. The implementation prompt's specific section 8.2 gives source south=0, west=-90, so the example explicitly chooses positive-east azimuth: pvlib bearing `(180 - source_azimuth) % 360`. Positive-west conversion `(180 + source_azimuth) % 360` is also tested. Earlier schema text did not establish the sign; this remains explicit config rather than an inference from south-zero alone.

Fixed systems use supplied surface tilt/azimuth. Trackers use independently configured axis tilt zero, axis azimuth zero, maximum angle 60 degrees, backtracking enabled and GCR 0.35. Source module tilt is never reused as axis tilt. For farm 901 the example explicitly enables a front-side-only tracker approximation and records rear-side gain as omitted. No bifacial multiplier, module database or electrical specifications are inferred from PV type IDs.

Forecast shortwave GHI and working-interpretation horizontal direct means are redistributed uniformly; no synthetic sub-hourly cloud variability is created. The engine computes `DHI=GHI-BHI`, then uses `pvlib.irradiance.dni` at the quarter-hour midpoint with a configurable 85-degree zenith horizon threshold. Direct contribution suppressed at low sun is flagged. The independently configured DNI cap is 1400 W/m², with cap-and-flag behavior; it has no effect on the executed example. Horizontal input irradiation remains recorded independently of this derived geometry approximation. BHI above GHI invalidates the interval by default. Explicit `clip_and_flag` sensitivity is available in a separate YAML, preserving contradiction flags and original aligned inputs. An optional explicitly selected Erbs GHI-only decomposition is tested for absent direct inputs; no quiet fallback is used. `fraction_diffuse` is disabled.

`pvlib.irradiance.get_total_irradiance` uses the isotropic model and albedo 0.2. Wind is the vector magnitude of the explicitly selected 10 m components; use as a module-level proxy is acknowledged. `pvlib.temperature.faiman` uses separately configured `u0=25`, `u1=6.84`. Temperature is already Celsius. Farm metadata `total_power` is the AC nameplate in kW. `pvlib.pvsystem.pvwatts_dc` receives a derived DC reference capacity: `AC nameplate / pv.dc.reference_conversion_efficiency`. This fixed assumption is independent of the explicit or fitted DC-to-AC output coefficient. The reference is converted kW→W; the gamma coefficient is -0.004/°C and reference temperature 25°C; returned W are converted to MW. After PV and transformer conversion, forecast AC is capped at the farm AC nameplate (or a lower configured cap). Measured AC production receives no PV/transformer efficiency losses. The offline calibration command fits a through-origin MSE slope using latest-available weather forecasts and measured AC, with a chronological diagnostic holdout.

The Faiman model estimates module temperature, used here as the cell-temperature
proxy passed to PVWatts:

$$
T_{cell}\approx T_{air}+\frac{G_{POA}}{u_0+u_1v},\qquad
P_{DC}=P_{DC,nom}\frac{G_{POA}}{1000}
       \left[1+\gamma(T_{cell}-T_{ref})\right].
$$

Here $G_{POA}$ is irradiance on the oriented module surface (W/m²), obtained
from direct, diffuse and ground-reflected components using the isotropic
transposition model. $T_{ref}=25$ °C and $\gamma=-0.004$/°C in the example YAML.
Higher irradiation raises production and also heats the module; wind increases
cooling. At unchanged irradiance, raising the modeled cell temperature from
25 to 50 °C reduces the PVWatts power by 10% with this configured coefficient.
These are model assumptions, not fitted hardware parameters. See the official
[Faiman](https://pvlib-python.readthedocs.io/en/v0.16.1/reference/generated/pvlib.temperature.faiman.html),
[PVWatts DC](https://pvlib-python.readthedocs.io/en/v0.16.1/reference/generated/pvlib.pvsystem.pvwatts_dc.html)
and [transposition](https://pvlib-python.readthedocs.io/en/v0.16.1/reference/generated/pvlib.irradiance.get_total_irradiance.html)
documentation.

Predicted farm AC is modeled DC times the independent illustrative PV-output efficiency 0.96 and configured transformer efficiency (currently 1.0), with no assumed AC inverter clipping limit. This is explicitly unfitted and does not describe measured hardware. The BESS converter's 0.97 is never used for measured PV, forecast PV conversion or direct grid export. An optional tested through-origin DC-to-AC calibration helper uses only completed eligible AC targets available by the fit cutoff; it excludes interpolated targets by default and reports adjusted samples and training RMSE. No real-data calibration, held-out calibration validation or fitted PV hardware identification has been executed. The caller must additionally certify causal historical prediction inputs for calibration.

The final PV AC production clock mask is applied separately from irradiance redistribution. Long exports contain the farm, cycle/availability, source lead, UTC interval boundaries, deterministic member and origin/probability one, modeled DC diagnostic, predicted AC, input eligibility, reviewed observation status, treatment flags, policy version and assumptions ID.

## Latest available forecasts at auction decisions

The real-data GME configuration now selects
`forecast.reoptimization_pv_policy: latest_available_forecast_at_each_reoptimization`.
At each solve it selects the latest cycle whose cycle time and publication time
are no later than the auction's buffered decision cutoff. Publication defaults
to the explicitly configured cycle-plus-four-hour assumption; explicit source
publication metadata overrides it. The entire PV calculation is repeated with
the selected cycle's weather. This remains one deterministic member per solve;
successive forecast cycles are not ensemble members or known future revisions.

Only intervals not yet executed are updated. An unfinished quarter-hour also
receives the new forecast for projecting the next SOC boundary, without its
final measured PV. IDA3 then optimizes the remaining physical horizon; trades
remain restricted to its eligible products, with earlier commitments fixed.
This adds no extra market solves. The legacy
`freeze_initial_forecast_for_unobserved_delivery_intervals` policy remains
selectable for a fixed-forecast comparison.

For the 2025-01-01 study the eligible UTC cycles are 2024-12-31 06:00 for both
DA/IDA1, 2024-12-31 12:00 for IDA2, and 2025-01-01 00:00 for IDA3. Solves and
auction-plan rows record their vintage and availability clocks. Outputs include
`json/pv_forecast_updates.json` and `csv/pv_forecasts_by_auction.csv`. Every PV
comparison plot shows all distinct cycles, grouping auctions sharing a cycle.
Newer curves cover only the intervals actually predicted at their decisions.
The active-plan forecast and its source clocks are retained per executed
interval; PV error KPIs use that forecast. The detailed PV diagnostic CSV also
keeps the initial forecast as an explicitly named reference. The configured
missing-measurement fallback remains the initial forecast, independently of
these planning updates.

## Executed real-data example and interpretation

The farm-1 example predicts local delivery day 2025-10-01 at cutoff 2025-09-30 09:45 UTC. The latest eligible weather cycle is 2025-09-30 00:00 UTC, available 04:00 UTC under the four-hour assumption; the 06:00 cycle is not available until 10:00. This is a deterministic weather/PV forecast, not a meteorological ensemble or completed trading backtest.

The conservative default has four invalid quarter-hours at 2025-10-01 02:00–02:45 UTC because forecast BHI is approximately 0.005387 W/m² while GHI is zero. The default retains the contradiction rather than silently repairing it. `configs/farms_nord_radiation_sensitivity.yaml` explicitly selects clipping contradictory BHI to GHI; this produces a complete 96-product exploratory forecast with the same finite-energy total because those four intervals become zero. Both runs retain the four contradiction flags. The raw source is unchanged.

The UTC-2025-10-01 derived measurement example contains 96 native power plus 96 GHI rows. Only two satellite GHI samples (11:30 and 11:45 UTC) are interpolated; both wait until the right endpoint completes at 12:15 UTC. Raw and treated AC energies are identical at 2.255149982918054 MWh for this UTC slice. This is one example and does not imply no production adjustments elsewhere.

Forecast evaluation reindexes raw and finalized policy-treated AC observations to the exact local-day starts, including the preceding UTC evening. It reports finite-pair coverage, MAE, RMSE and bias separately for raw/treated targets, forecast vintage, member and lead bin. Missing forecast/measurement intervals and reviewed exclusions are counted. It is explicitly ex-post scoring: the target observations do not enter weather forecast construction or unfitted output conversion. Masking/interpolation effects are reported and observations can include curtailment/outages; no skill is claimed for unconstrained PV availability. There is no CRPS or ensemble-calibration claim for the current single deterministic member.

## Evidence and commands

Use the existing `bess` environment, without installation:

```bash
PYTHONPATH=src python -m bess_pv_trading inspect-data --config configs/farms_nord.yaml
PYTHONPATH=src python -m bess_pv_trading prepare-analysis-data --config configs/farms_nord.yaml
PYTHONPATH=src python -m bess_pv_trading forecast-pv --config configs/farms_nord.yaml
PYTHONPATH=src python -m bess_pv_trading forecast-pv --config configs/farms_nord_radiation_sensitivity.yaml
PYTHONPATH=src python -m pytest tests/unit/test_data_adapters.py tests/unit/test_data_policy.py tests/unit/test_pv_engine.py -q
```

Manifests, exact-grid evaluations, CSVs and non-interactive plots are grouped under `outputs/analysis/farm1_20251001/`, `outputs/pv/farm1_20251001/`, `outputs/pv/farm1_20251001_radiation_sensitivity/`. `outputs/data_pv_validation/evidence.json` records inspection and actual run results/timings, together with a before/after metadata marker check; it does not claim a full-store byte-integrity audit. Tests exercise both explicit lead encodings, named transposes/label joins, UTC and DST, native observation completion, plateau causality, delayed interpolation endpoints, raw-value preservation/idempotence, clipping allowlists, PV electrical bases, tracking/bifacial assumptions, scalar calibration and explicit multiple supplied members. External member fixtures test the interface; they are not members found in the real store.

Remaining concrete limitations are unresolved private direct-radiation provenance, proxy coordinates and tracker geometry, omitted rear-side gain, unfitted PV-output/temperature parameters, residual audit concerns outside the narrow policy, and absent observed market-price labels. A conservative-default incomplete forecast cannot be used as complete optimizer PV input; an explicitly adjusted exploratory forecast is not independent source-quality validation.

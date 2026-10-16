# Codex Implementation Task — PV+BESS Stochastic Auction Trading, Bound to the Available Farm Dataset

## 1. Mission, scope, and order of work

Work in `/home/rghetti/bess_pv_trading` on the remote machine that contains the dataset. This is a new trading/optimization implementation, not an extension of an assumed working optimizer. An existing data-audit and cloud-correlation pipeline may already be present: inspect and preserve it.

Build a working Python application for **one PV farm with a lithium-ion BESS and a shared grid point of interconnection (POI)**, participating in the Day-Ahead market and the three Intraday Auctions: DA, IDA1, IDA2, IDA3. Implement the optimization directly with `gurobipy` as a scenario-based multistage stochastic MILP maximizing **expected net operating profit**, including degradation and residual imbalance settlement.

The system must anticipate later auctions when selecting the DA offer, submit only the current actionable offer, and reoptimize using genuinely available forecasts and previously cleared auction results. Prices may be uncertain even while the available PV weather forecast is deterministic.

Scope boundaries:

- Energy trading in DA and IDA1/2/3 only. No ancillary services or continuous intraday trading.
- Price-taking and accepted fixed-quantity/self-schedule bids are explicit study assumptions.
- No endogenous market-price impact, order-book model, or price-volume bidding curves.
- No local load. The BESS may charge from PV or grid imports and discharge to the grid; PV may be curtailed.
- Initial market adapter: Italian `NORD`; keep market rules outside generic physical equations.
- Run the four farms as **four separate plants** with separate physical connections. Do not create a fictitious common POI or treat farm IDs as ensemble members.
- Each optimization experiment concerns the delivery products of **one delivery day**. Model the relevant preceding auction events, but do not add unrequested planning horizons.
- Build the deterministic backbone and a small branching stochastic example before attempting a large backtest.
- Preserve the raw dataset and existing audit through read-only adapters. Implement only the explicitly requested, configurable masking/interpolation policy in Section 3.4 as a separate derived-data layer; do not expand it into an unrequested cleaning system.

Deliver functioning code, tested configurations, reproducible commands, and a **LaTeX implementation-status document generated after the implementation work and tests**. The report must describe what actually exists, not merely repeat this specification.

## 2. Environment, repository protection, and execution rules

Read `AGENTS.md` and other local instructions first. Preserve existing instructions and audit code; do not overwrite them with this prompt. Identify conflicts with this updated task, including older descriptions of measured power as DC, the old data-directory name, or a blanket prohibition on preprocessing. The requested corrections are: measured power is AC, 97% efficiency belongs to the BESS converter, the directory is `data/farms/`, and Section 3.4 explicitly authorizes a narrow derived-data policy. Preserve the separate audit implementation and raw values.

Use the existing Conda environment `bess`. Inspect installed packages, Python version, `gurobipy`, pvlib, xarray/Zarr/Parquet readers, YAML support, and test tools. Test Gurobi availability and licensing with a tiny model. **Do not install, upgrade, downgrade, or remove packages; create environments; perform editable installs; or change licenses without permission.** Continue with independent components if a dependency is missing, and report the blocker accurately.

Do not commit or push unless requested. Do not alter Git remotes/history. Preserve existing uncommitted work. If a commit is requested, use an exaustive and meaningful message explaining changes and executed tests (do not make comparison, do not explain the code entire history, do not reference hard coded or configurable values).

Use a `src/` package and commands runnable without installation, for example:

```bash
cd /home/rghetti/bess_pv_trading
# Use the already active bess environment, or activate it normally.
PYTHONPATH=src python -m bess_pv_trading run-synthetic --config configs/synthetic_small.yaml
```

Use type hints, clear names, concise docstrings, and comments above important mathematical blocks. Keep assumptions and numerical parameters in validated YAML. Prefer vectorization and lazy reads; do not overengineer a large framework of empty classes.

The machine has 48 CPU cores and four shared GPUs, not dedicated resources. Inspect current utilization before substantial work; use conservative configurable CPU workers and Gurobi threads, avoid nested oversubscription, and do not assume a GPU is useful. Start with metadata, one farm, a handful of cycles, and short date slices. Never load or hash the entire Zarr store just to inspect its schema.

Keep secrets out of tracked files, config snapshots, source control, logs, and the LaTeX report. Use environment variables or hidden interactive CLI prompts for downloads. Unattended backtests must not block waiting for credentials.

## 3. Supplied dataset contract — preserve the distinction between facts and assumptions

The following is the user's description of the inspected store. Use it to implement the adapter. Verify names and attributes with lightweight inspection; if the actual store disagrees, report the mismatch instead of silently modifying the data or substituting a different meaning.

### 3.1 Locations and farm metadata

Data directory:

```text
/home/rghetti/bess_pv_trading/data/farms/
```

Time-series store:

```text
pv_farms_timeseries_20250101_20260831.zarr
```

The matching static-metadata `.parquet` file is in the same directory and joins by `farm_id`. Resolve and record its actual filename; do not select an arbitrary Parquet file if multiple candidates exist.

| farm_id | Installed DC capacity (kW) | Tracking ID | System type |
|---|---:|---:|---|
| 1 | 953.28 | 0 | Fixed tilt |
| 345 | 1524.60 | 0 | Fixed tilt |
| 901 | 4160.00 | 3 | Single-axis, bifacial |
| 910 | 6638.00 | 1 | Single-axis |

All farms are anonymous and associated with Northern Italy's `NORD` bidding zone. Static metadata contain zone, azimuth, tilt, PV-type ID, installed capacity, tracking-type ID, and elevation. Do not infer module technology or coefficients from an undocumented PV-type ID.

Angles are in degrees, and source azimuth 0 degrees denotes south. **The sign/direction of positive azimuth is not established by that statement.** Actual coordinates are unavailable. Milan is a provisional solar-position reference only. Tracker-axis geometry is unresolved; module azimuth/tilt metadata must not automatically be reinterpreted as axis geometry.

The files contain **no electricity prices, BESS specifications, detailed module parameters, or detailed inverter specifications**. Installed capacities in the table are still **DC nameplate capacities**, but the historical `power` series is **already AC output from the farm**. The configurable efficiency of **0.97 belongs to the bidirectional BESS inverter/converter**, in each conversion direction, not to a second conversion of measured PV output. It is a study assumption, not measured hardware metadata.

### 3.2 Historical observations

Declared coverage: **1 January 2025–31 August 2026**, with **58,368 quarter-hour intervals per farm**, all in UTC. This is the supplied record count.

| Raw variable | Meaning supplied by the user | Units | Native dimensions | Timestamp semantics |
|---|---|---|---|---|
| `power` | Average total plant AC output, already downstream of the farm's PV conversion | kW | `(time, farm_id)` | Start of `[time, time + 15 min)` |
| `ghi` | Satellite global horizontal irradiance | W/m² | `(farm_id, time)` | Start-labeled quarter-hour historical interval |
| `fraction_diffuse` | Diffuse-related dimensionless quantity; it is the fraction of ghi coming from atmosphere diffusion | 1 | `(farm_id, time)` | Same historical time axis |

Preserve the stated start-label convention. Do not assume the within-interval measurement operator for every satellite-derived field is established beyond this description.

`power` is already quarter-hourly and **already AC**. Do not multiply it by plant capacity or by 0.97. Preserve its native interval averages, except for the expressly authorized derived-data transformations in Section 3.4. Energy per complete interval is `power_kw * 0.25 / 1000` MWh on the **AC side**. Missing samples may be interpolated under the declared measurement policy; existing valid samples must not be resampled or smoothed.

Transpose arrays by **named dimensions**, not storage order. Join farm IDs by labels, not positional coincidence between Zarr and Parquet. Keep the source time coordinate UTC; if its in-memory representation is timezone-naive, interpret it as UTC rather than local time.

### 3.3 Weather forecasts

Native forecast dimensions:

```text
(farm_id, cycle_time, lead_time)
```

There are **2,440 forecast cycles**, normally six hours apart, and **49 hourly leads from 0 through 48 hours**. Valid time is:

```text
valid_time = cycle_time + lead_time
```

Inspect whether the lead coordinate is stored as timedeltas or numeric hours, then convert explicitly. Retain the original coordinate and units in provenance. Do not hard-code expected array sizes into the algorithms or infer missing cycles from the historical record count; forecast boundary coverage can differ from observation coverage.

**There is no ensemble-member dimension in this store. Overlapping forecast cycles are successive vintages, not independent ensemble members or equiprobable scenarios.** At a decision time, select an eligible vintage. Do not average different vintages and call that a probabilistic forecast. Do not count the four farms as four realizations of one plant.

| Forecast feature family | Units | Temporal meaning |
|---|---|---|
| Total, low, middle, high cloud cover | % | Instantaneous at valid time |
| Temperature and dew point at 2 m | degrees Celsius | Instantaneous at valid time |
| Shortwave radiation: forecast GHI | W/m² | Hourly mean over `[valid_time - 1 h, valid_time)` |
| Direct radiation | W/m² | Mean over `[valid_time - 1 h, valid_time)` |
| Precipitation | mm | Accumulation over `[valid_time - 1 h, valid_time)` |
| Snow-water equivalent | m | Instantaneous at valid time |
| Visibility | km | Instantaneous at valid time |
| Wind u/v at 10 m and 100 m | m/s | Instantaneous at valid time |

The descriptive feature names above are not guaranteed raw Zarr variable names. Inspect the available names and attributes and create an explicit canonical-to-raw mapping. Do not fabricate names or infer the upstream NWP provider from naming resemblance.

Radiation interpretation:

```text
shortwave radiation -> forecast GHI, hourly means; confirmed by the user
ghi                 -> historical satellite GHI, every 15 minutes
direct radiation    -> BHI, downward radiation contribution of GHI
```

Keep forecast GHI and satellite GHI in separate canonical fields; they differ in origin, temporal resolution, and availability. Do not replace hourly forecast shortwave radiation with satellite measurements during ex-ante forecasting.

Direct radiation is **not to be passed straight to pvlib as DNI** under the BHI working interpretation. Make its unresolved interpretation explicit in config and provenance. The store does not identify the original source parameters, so public provider documentation alone cannot prove its provenance.

Forecast availability is **assumed** to be:

```text
forecast_available_at = cycle_time + 4 hours
```

Make latency configurable and allow explicit availability metadata to override the assumption if subsequently supplied. All values in a cycle become usable according to this availability policy, not merely because the first valid time is in the past.

Lead-zero radiation/precipitation and forecast values outside the declared date range have documented missingness. Treat this as structural forecast missingness at the adapter boundary; do not turn these missing forecasts into zero, backfill them, or borrow future vintages. Section 3.4 clips eligible **negative finite forecasts** to zero and interpolates **missing historical measurements**; neither operation authorizes filling structural forecast gaps. Eligibility is specific to the requested model and delivery slice: an unused precipitation field or an irrelevant lead-zero gap must not invalidate an otherwise supported PV forecast.

### 3.4 Existing audit and requested masking/interpolation policy

Preserve the existing audit/cloud-correlation pipeline, its review tables, and all raw Zarr/Parquet values. Add a small, separate, versioned transformation layer for the policies below. Reuse compatible audit utilities without changing their verdicts.

**Here “mask to zero” means replacing the affected value with numeric zero in a derived series, not deleting the sample or replacing it with NaN.** Keep the original value and a reason flag. These are user-selected analysis rules, not proof that a sample is physically faulty.

Implement the following defaults, with thresholds, variables, and timezone configurable in YAML:

**A. Repeated nonzero production values**

- On each farm's native 15-minute measured `power` series, identify consecutive runs of finite, nonzero values that are almost equal.
- A run must contain **more than three measurements: four or more consecutive samples**.
- In the finalized derived analysis series, replace **the entire qualifying run** with zero, including its first three samples.
- Use configurable absolute/relative tolerances. Provide explicit initial numerical defaults, labeled assumptions, such as `atol_kw: 0.01` and `rtol: 0.0001`; do not claim these are calibrated sensor tolerances.
- Compare against the run's anchor value so many small successive changes do not incorrectly turn a gradual ramp into one long plateau. A zero, missing value, or missing timestamp breaks the run.
- Detect runs before interpolation or hourly-to-quarter-hour redistribution. Do not classify interpolation-generated or uniformly redistributed values as repeated measurements.
- Default this rule to **measured production only**. Do not silently apply it to cloud cover, wind, temperatures, satellite GHI, or forecast trajectories merely because those values repeat.
- Record run length and removed energy. Real clipping plateaus may also be caught: report that limitation without disabling the requested rule.

**B. Positive nighttime production**

- Set positive measured AC production to zero between **21:30 and 05:00**, crossing midnight.
- Interpret the requested window as **21:30 inclusive to 05:00 exclusive in `Europe/Rome`**, based on interval-start time, as an explicit configurable assumption. The source data remain stored in UTC.
- Apply the same clock-window rule to final AC PV production forecasts used by the optimizer, and record the adjustment. Do not apply a production mask to the raw satellite GHI or hourly radiation fields.
- This is a fixed clock rule, not a solar-position fault classifier. Milan proxy sunrise/sunset must not override it or justify additional fault labels.

**C. Negative nonnegative-quantity forecasts**

- Replace finite negative shortwave/GHI and direct-radiation forecasts with zero in derived forecast inputs.
- Apply the same zero floor to other configured physically nonnegative forecast magnitudes, such as precipitation, snow-water equivalent, visibility, and cloud-cover percentages.
- **Preserve valid negative temperatures, dew points, and signed wind u/v components. Preserve negative electricity-price forecasts as well.** These are not nonnegative magnitude variables.
- Use an explicit variable allowlist. Preserve NaNs, including structural lead-zero missingness; `NaN` must not become zero as a side effect of clipping.

**D. Missing historical measurements**

- Fill interior missing historical measurements, including `ghi` and `power`, with time-based **linear interpolation per farm and variable** between valid bracketing measurements.
- Retain existing finite values, including original zeros and the zeros created by A/B. Do not interpolate across zero masks as though they were gaps.
- Default to interpolating all internally bracketed gaps; expose an optional maximum gap duration and record gap lengths. Do not silently impose an extra maximum gap absent from the configured policy.
- Do not extrapolate leading/trailing gaps, cross farm boundaries, or use weather forecasts as replacement observations. Leave gaps without two valid endpoints missing and report them.
- Keep `fraction_diffuse` out of model inputs until its definition is confirmed. Its transformation policy must remain explicit if later enabled.

Apply plateau detection on the raw native production series first, then the production/forecast zero masks, then measurement-gap interpolation. Reapply the production night window after interpolation so interpolation cannot reintroduce positive nighttime production. Make this pipeline deterministic and idempotent; never overwrite raw source files.

Save raw-versus-derived values, flags such as `plateau_zero`, `nighttime_zero`, `negative_forecast_zero`, and `measurement_interpolated`, the policy version, parameters, and the change in daily production energy. Use the policy-treated values in the configured modeling/backtest pipeline, while retaining upstream review exclusions and explicitly reporting the fraction of adjusted data. Do not present treated or interpolated observations as independently verified fault-free ground truth or unconstrained available PV.

**Protect historical causality.** The first three samples of a qualifying plateau can only be retrospectively flagged once the fourth measurement is available; later samples are flagged no earlier than their own observation time. A two-sided interpolation is only available when its right endpoint and any flags affecting its endpoints are available. Record this evidence-availability time. Earlier controller decisions or price features must not see retroactive masks or interpolated values before that time. Use as-of observation views for decisions; the finalized treated series may be used for explicitly labeled ex-post accounting. Do not obtain apparent forecast skill by using later gap endpoints or later plateau detections in earlier decisions.

Other audit concerns, such as shared GHI zeros, remaining offsets or abrupt dips, are not automatically repaired by this policy. Preserve flags and report ineligible inputs rather than add new fault-detection rules. Synthetic fixtures must remain runnable independently of unresolved source-quality issues.

### 3.5 Current versus explicitly supplied probabilistic inputs

Use a common forecast interface with an explicit member axis of length one for the current store. Preserve a separate `forecast_vintage` dimension; never rename it `ensemble_member`.

The current implementation must support:

1. One deterministic weather/PV forecast and point prices.
2. One deterministic weather/PV forecast and multiple statistically generated price paths.
3. Multiple PV ensemble members **when actually supplied**, each paired with conditional price paths, using the same optimizer interfaces.

A synthetic PV ensemble is permissible for software testing only and must be labeled synthetic. If uncertainty in PV is not modeled in a particular real-data run, report exactly that. Do not manufacture 50 weather members from the deterministic cycles to meet a scenario-count example.

## 4. Software structure, configuration, and contracts

Create a maintainable package from scratch; reuse any compatible utilities already present without rewriting the audit. A suggested organization is:

```text
src/bess_pv_trading/
  cli.py, config.py, domain.py
  data/             # Read-only sources, derived masks/interpolation, adapters/cache
  forecasting/      # pvlib conversion, price features, regressions, residuals
  scenarios/        # Joint paths, information trees, conditional regeneration
  optimization/     # Physical/trading/settlement/degradation builders and solver
  simulation/       # Event clock, accepted-trade ledger, causal dispatch, replay
  reporting/        # Numerical metrics, plots, implementation-status artifacts
configs/
  synthetic_small.yaml
  farms_nord.yaml
  data_schema.yaml
  italy_nord.example.yaml
tests/
  fixtures/, unit/, integration/
docs/
  dataset_contract.md
  assumptions.md
  formulation.md
  sources.md
  implementation_status.tex
  implementation_status.bib
  implementation_status.json
```

Do not create tiny empty modules merely to match this tree. Preserve existing files and conventions where practical. Ignore large datasets, caches, credentials, result directories, and solver dumps in Git; retain small fixtures/configurations.

Define typed, solver-independent contracts for:

- `DeliveryGrid`: product/interval IDs, timezone-aware starts/ends, interval durations.
- `FarmMetadata`: confirmed fields and separately tagged assumptions.
- `ForecastBundle`: source cycle, availability, lead, interval semantics, member origin, values, units, provenance.
- `ObservationBundle`: measured AC power/satellite GHI, raw and treated values, interval boundaries, observation/evidence availability, transformation flags and review status.
- `ScenarioBundle`: joint PV/auction/imbalance trajectories and normalized probabilities.
- `InformationTree`: node parents, observation/decision times, conditional probabilities, shared decision groups.
- `PlantParameters`, `BatteryParameters`, `PhysicalState` with a state timestamp.
- `MarketState` and an immutable accepted-trade ledger.
- `OptimizationInput`, `BuiltModel`, `OptimizationResult`, `ValidationReport`.

Do not fetch data, train models, inspect Zarr, or plot while building constraints. The optimizer accepts validated in-memory objects.

### Required YAML content

Bind known fields now, rather than leaving the dataset wholly unspecified. Include the actual directory/store, IDs/capacities, native dimension names, UTC convention, historical interval-start and forecast interval-end semantics, six-hour cycle cadence, and assumed four-hour latency. Inspect and fill raw forecast-variable mappings. Leave genuinely unresolved meanings explicit, not guessed.

Provide configuration for:

- Paths, source-variable mappings, farm selection, upstream review exclusions, and strict/exploratory mode.
- Derived-data policy: plateau minimum run length and equality tolerances, variable allowlists, the 21:30–05:00 night window/timezone, negative-forecast clipping, measurement interpolation and optional maximum gap, evidence availability, and raw/treated output paths.
- Confirmed hourly shortwave/GHI semantics, assumed direct-radiation interpretation, and treatment of `fraction_diffuse` (disabled until confirmed).
- Proxy coordinates and their status, azimuth sign convention, tracker-axis assumptions, GCR/backtracking/bifacial parameters.
- PV DC nameplate capacity, DC temperature model, separate PV-output conversion/calibration and AC limit assumptions; historical power remains measured AC.
- BESS MWh/MW, bidirectional converter efficiency of 0.97 per direction, separately configured cell efficiencies or declared combined efficiencies, SoC bounds, initial state timestamp/value, degradation, and POI limits.
- Market version, timezone, gates, results publication, decision buffer, delivery-product masks, settlement and position limits.
- Price regression/features, training and validation dates, residual method/covariance, random seed, conditional branch counts.
- Scenario/member origin, probabilities, forecasts frozen or revised under a declared information policy.
- Gurobi threads/time limit/MIP gap/seed/logging; plot/output settings.

A safe real-data example may require the user to confirm BESS/POI values before a run; a synthetic example must be fully executable with declared illustrative values. If providing capacity-scaled illustrative defaults, list the formula and values in the assumption manifest. Never imply they describe installed equipment.

## 5. Timestamp alignment, forecast selection, and market products

Use MW, MWh, EUR/MWh, EUR, and fractional efficiencies in the optimizer. Keep source units in the input manifest. `delta_hours[t]` is mandatory; do not multiply already-converted MWh by the interval duration again.

### 5.1 Select vintages causally

At a decision cutoff, select the latest available cycle using `available_at <= decision_cutoff`, not nearest valid time. Apply any submission/computation buffer before selecting features. Under the assumed latency:

```text
cycle 00:00 UTC -> available 04:00 UTC
cycle 06:00 UTC -> available 10:00 UTC
cycle 12:00 UTC -> available 16:00 UTC
cycle 18:00 UTC -> available 22:00 UTC
```

These are UTC examples. Convert local auction times with `Europe/Rome`; do not hard-code UTC gate hours or confuse UTC cycles with local cycles.

After selecting a cycle, verify that it covers every required forecast interval. Default to a single selected vintage. An optional earlier-complete-vintage fallback must be explicit, causal, and logged; never silently stitch a newer unavailable cycle or final observations into coverage gaps.

When an entire older cycle is available, future forecast values within that cycle may be used for interpolation: they are already-issued predictions, not future observations. Conversely, the completed historical `power` average is available only at interval end plus declared telemetry latency, not at its start label.

An eligible later cycle may replace the current forecast at a later auction. Before its publication, the initial DA planner must not use its realized historical contents. For the initial planning model, explicitly assume frozen currently available weather forecasts when no probabilistic model of forecast revisions exists; then incorporate actual new vintages only when their assumed publication event occurs. Report that approximation.

### 5.2 Align forecasts according to variable semantics

For radiation with valid time `v`, the source interval is `[v - 1 hour, v)`. Example:

```text
valid_time = 12:00 UTC
radiation describes 11:00–12:00 UTC
maps to quarter-hours starting 11:00, 11:15, 11:30, 11:45
NOT 12:00, 12:15, 12:30, 12:45
```

Implement separate transformations:

- **Hourly mean radiation:** redistribute to quarter-hour means with conserved hourly irradiation. A uniform split is a transparent baseline; a documented daylight/solar-geometry weighting is permitted. Preserve the mean's integral; do not apply ordinary point interpolation to interval means.
- **Hourly precipitation accumulation:** preserve the total when allocating to finer intervals. It is not an instantaneous rate. This feature need not be used by the first PV model.
- **Instantaneous weather:** interpolate within the same eligible cycle to the selected model timestamps, with no unannounced extrapolation.
- **Historical power and satellite GHI:** retain the native start-labeled 15-minute grid. Apply only Section 3.4's explicit masks and linear interpolation of missing measurements; do not resample already valid values or interpolate structural forecast gaps.

Store source interval start/end and the transformation used. Converting hourly NWP into quarter-hour model inputs does not create observed sub-hourly cloud variability. Report that limitation.

For a source mean of 400 W/m² over one hour, the four derived 15-minute means must integrate to 400 Wh/m². If daylight weighting is used, normalize over the source interval and flag cases where the assumed geometry cannot allocate the source irradiation consistently. Do not overwrite or issue a fault verdict on raw data to force agreement with Milan geometry.

### 5.3 Auction calendar

Separate submission, clearing-results availability, and delivery. Verify the historical configuration using official sources [R1–R3], and store validity dates.

| Auction | Initial ordinary local gate | Eligible delivery |
|---|---|---|
| DA / MGP | D−1 12:00 | All of delivery day D |
| IDA1 / MI-A1 | D−1 15:00 | All of D |
| IDA2 / MI-A2 | D−1 22:00 | All of D |
| IDA3 / MI-A3 | D 10:00 | D 12:00–24:00 |

Do not infer results publication from the gate timestamp. Support documented publication schedules, recorded timestamps, or explicit conservative assumptions with provenance. Allow auction cancellations/missing results to make trades unavailable; do not fabricate clearing prices.

Use timezone-aware UTC internally and local `Europe/Rome` delivery days. DST days can have 92/96/100 quarter-hour products. The supplied UTC date range does not guarantee a complete first or last local market day or its preceding auction forecast coverage.

The MGP quarter-hourly regime starts with delivery on **1 October 2025** [R3]. Thus the physical dataset's January–September 2025 observations are not evidence of quarter-hourly DA prices. Either implement native hourly-product-to-quarter-hour mappings, or restrict quarter-hourly market backtests to the supported regime and report the excluded period. Never interpolate hourly prices into invented observed quarter-hour clearing prices.

With the supplied end date of August 2026, report the actual eligible intersection of plant observations after the declared policy, forecasts, market data, training history, and any upstream review exclusions. Do not claim years of homogeneous quarter-hourly auction validation that this dataset does not contain.

## 6. Optimization backbone — implement this from scratch

### 6.1 One linked stochastic program, not separate scenario optimizations

Implement the deterministic equivalent of the stochastic problem as **one Gurobi model** containing the shared decisions, scenario-dependent outcomes, and probability-weighted objective.

Do not solve one independent perfect-information model per scenario and average its bids. Do not use Gurobi's `NumScenarios`/`ScenNObj` feature as a substitute for stochastic programming: that feature computes solutions to related scenario models; it does not automatically enforce our shared decisions or nonanticipativity [R4].

The deterministic model must use the same formulation builder with one scenario and a compatible information tree. Four farms are not four scenarios: create a separate model per farm/run unless an explicit portfolio experiment is requested.

Keep battery energy and power continuous. Binary mode variables enforce physical charge/discharge exclusivity; do not discretize SoC into a lookup grid.

### 6.2 Indices and decision ownership

Let:

- `t` denote a physical delivery interval.
- `s` denote a joint-scenario leaf, with probability `p[s]`.
- `a` denote an auction in {DA, IDA1, IDA2, IDA3}.
- `n` denote an information/decision node.
- `bid_node[a, s]` identify the node where the auction-a bid is chosen along leaf s.
- `control_node[t, s]` identify the information available when the battery command for interval t is chosen.

Create signed energy decisions `trade_mwh[a, n, product]` only for eligible products and nodes. A root DA bid is shared by all leaves. Later bids are shared by leaves passing through the same node.

Create physical commands using control-node indexing, or equivalent scenario-indexed variables with explicit nonanticipativity equalities. Use the same information restrictions for mode binaries as for continuous controls.

Represent realized PV and settlement outcomes at the appropriate outcome nodes. End-of-interval battery states may be stored by leaf for readability, provided their dynamics and shared controls impose the correct consistency.

Use readable names for variables and constraints, including farm/node/interval identifiers, so infeasibility diagnostics can be interpreted.

### 6.3 AC-coupled physical model

Start with an explicit AC-coupled abstraction. The farm's measured output is already AC, and the battery connects to the common AC bus through a bidirectional inverter/converter:

```text
PV farm AC output ---------------- common AC bus ---------------- grid / POI
                                         |
                          bidirectional BESS converter
                           eta AC->DC = 0.97
                           eta DC->AC = 0.97
                                         |
                                   battery DC storage
```

The converter represents the battery's path to either PV-generated AC or the grid. Do not create two consecutive 97% conversions for one charging or discharging leg. Direct PV-to-grid flow does not pass through this BESS converter.

For each interval and scenario, let `g[t,s]` be the PV AC forecast or the measured/treated AC realization appropriate to that simulation stage. Historical `power` is not multiplied by 0.97. Define:

```text
charge_mw[t,s] >= 0
discharge_mw[t,s] >= 0
curtailment_mw[t,s] >= 0
energy_mwh[t,s]
grid_mw[t,s]                    # signed: export positive
mode_charge[t,s] in {0,1}
```

The scenario indices on battery commands and mode binaries are shorthand for their shared battery-control-node variables. Curtailment must use its separately declared observation/response nodes, not automatically the same pre-interval node: any recourse may depend on current observed PV under the declared execution abstraction, but never on unobserved future PV or prices.

Implement:

```text
eta_charge = eta_bess_converter_charge * eta_cell_charge
eta_discharge = eta_bess_converter_discharge * eta_cell_discharge

energy[t+1,s] = retention[t] * energy[t,s]
                + eta_charge * charge[t,s] * delta_hours[t]
                - discharge[t,s] * delta_hours[t] / eta_discharge

energy_min <= energy[t,s] <= energy_max

0 <= charge[t,s] <= max_charge_mw * mode_charge[t,s]
0 <= discharge[t,s] <= max_discharge_mw * (1 - mode_charge[t,s])

0 <= curtailment[t,s] <= g[t,s]

grid[t,s] = g[t,s] - curtailment[t,s]
            + discharge[t,s] - charge[t,s]

-poi_import_mw <= grid[t,s] <= poi_export_mw
```

Here `charge_mw` and `discharge_mw` are AC-bus powers, and `energy_mwh` is battery-internal stored energy. Default `eta_bess_converter_charge = eta_bess_converter_discharge = 0.97`. With no supplied cell-loss specification, allow explicitly declared cell efficiencies of 1.0; alternatively accept configured cell efficiencies. If configured charge/discharge efficiencies already include the converter, use that combined convention instead and do not multiply by 0.97 again. Validate the convention.

Apply inverter limits on the correct AC-side powers and validate consistency with the power limits. Default self-discharge may be zero if explicitly configured; otherwise convert the stated loss rate to `retention[t]` using the interval duration.

Enforce all physical bounds in every represented scenario, including the final energy state. Do not enforce battery feasibility only in expectation.

PV used to charge the battery and imported grid energy both incur the configured AC-to-battery charging losses once. Battery energy exported to the AC bus incurs the configured discharging losses once. A charge/discharge cycle has converter-only efficiency `0.97 * 0.97 = 0.9409` when cell efficiencies are 1.0. Do not apply the BESS converter efficiency to direct PV exports, measured PV output, or financial trades.

Use one signed net POI variable rather than independent unconstrained import/export variables. Introduce directional variables only when the applicable cost formulation requires them, with safeguards against artificial simultaneous transactions.

PV may charge the battery, be exported, or be curtailed. The battery may also charge from grid imports. Do not double-count these flows, and do not impose a PV-only charging restriction.

### 6.4 Initial conditions and terminal energy

Attach every energy state to its actual timestamp.

Before delivery day D starts, the battery state measured on D−1 is not automatically the battery state at D 00:00. For the first single-delivery-day experiment, permit an explicitly configured, known delivery-start energy assumption. Label it in results and use it consistently across strategies. If instead using live current state, model the intervening physical period or require a supplied boundary-state trajectory; never silently equate the two timestamps.

Once delivery starts, initialize each reoptimization with the actual simulated/measured current energy. Never reset SoC to its original daily value at an auction or forecast update.

Implement terminal treatment:

- A fixed target energy, with reachability checked;


Report salvage value separately from realized market cash flows. Apply identical terminal assumptions in comparisons. Do not force a terminal equality if it makes the declared operating problem infeasible without reporting this.

### 6.5 Commercial positions and physical settlement

Trading quantities are financial energy positions, not physical charging/discharging flows.

For each product/interval:

```text
net_position_mwh[t,s] = fixed_accepted_position_mwh[t]
                       + sum(new auction adjustments mapped to t along s)

imbalance_mwh[t,s] = grid_mw[t,s] * delta_hours[t]
                    - net_position_mwh[t,s]
```

A later IDA trade changes the position by an increment; it does not replace the earlier trade. Do not charge physical battery efficiency losses for offsetting DA/IDA financial transactions. Apply losses only to actual battery power.

Implement an accepted-trade ledger with auction, submission time, clearing/publication time, delivery product, signed accepted MWh, clearing price, and contractual cash flow.

Previously accepted trades remain fixed, while opposite new trades may offset their net position. Closed or ineligible auctions receive no decision variables.

Set explicit configurable bounds on auction trade quantities and cumulative commercial positions. POI limits alone do not bound offsetting financial DA/IDA round trips. Do not allow unbounded trading; do not silently forbid all legitimate cross-auction offsets either.

Allow physical imbalances and settle them economically. Do not impose exact physical delivery of every commercial commitment under every scenario unless a separately specified contract requires it.

Provide a settlement component with matching model-builder and numerical-evaluator implementations:

- Signed single-price settlement: `cashflow = price_imbalance * imbalance_mwh`.
- Direction-dependent settlement when verified for the configured period.

If using positive/negative imbalance components, enforce the true signed piecewise function. Do not permit profitable simultaneous positive and negative components when price ordering makes an epigraph relaxation invalid. Use tight physically/commercially derived bounds and a binary/indicator formulation where needed.

No additional grid-use fee for charging imports is the configured study assumption, not a general legal conclusion. Fees and excluded charges must remain explicit configuration fields. If market settlement rules are unresolved, use labeled synthetic economics only; do not invent historical charges.

### 6.6 Expected-profit objective

For every leaf s, construct named components:

```text
auction_revenue[s]
imbalance_cashflow[s]
battery_degradation_cost[s]
other_costs[s]
```

For accepted self-schedule bids, auction revenue is the sum of each scenario's clearing price times the appropriate shared-node trade quantity.

```text
operating_profit[s] = auction_revenue[s]
                      + imbalance_cashflow[s]
                      - battery_degradation_cost[s]
                      - other_costs[s]

maximize sum_s probability[s] * (
    operating_profit[s]
)
```

Known past contractual cash flows may be recorded as a constant for reporting, but must not be booked again or mistaken for newly optimized profit. Do not multiply traded MWh by interval duration a second time.

Price forecasts are numerical scenario coefficients, not decision variables. The objective remains linear in quantities.

Check economic boundedness under negative prices, unfavorable spreads, zero fees, and imbalance opportunities. Under linear settlement and weak coupling, uncertainty can legitimately have little decision value. Do not add arbitrary penalties to force the stochastic method to look better. by law we cannot speculate on imbalances even if the predicted outcome is positive cashflow, add a fixed penalty of configurable value to the imbalances coefficient in the cost.

### 6.7 Information structure and nonanticipativity

A collection of complete paths is not automatically a valid multistage model.

Before any auction's gate closure, its clearing price is unknown. A leaf may contain that future price for evaluating profit, but the submitted bid must be shared across leaves with the same information at submission.

Build a genuine finite information tree or explicitly shared decision groups. Distinguish auction submission, results observation, forecast availability, and physical observations. Physical control and auction events interleave: IDA3 occurs after delivery has begun.

Every node must have a parent, an information cutoff, a probability, and a list of observations actually available. Validate that child conditional probabilities sum to one and that leaf probabilities are consistent with products along their paths. Start with a tree, not an unverified recombining graph: merging histories must not erase differing battery states, accepted positions, or forecast information.

Never cluster information nodes using future PV, future prices, or whole-path identifiers that the controller cannot observe. Do not create one branch per unique floating-point price vector and then implicitly reveal every future random variable. Preserve conditional descendants representing unresolved uncertainty after observed auctions.

Start with a small hand-constructed synthetic tree whose observation structure is known exactly. Then implement conditional branching or history-only tree construction for the forecast scenario generator. Report any state aggregation as an approximation. If a supplied flat scenario bundle cannot support a defensible tree, reject the multistage configuration rather than silently granting perfect foresight.

**Physical dispatch also requires nonanticipativity.** Merely sharing auction bids while allowing each leaf to choose its entire future battery schedule independently gives an optimistic recourse model. A later MPC does not retroactively remove that optimism from DA planning.

For the first valid multistage implementation, include explicit physical decision nodes at the selected control times, with branching driven only by observations then available. A reduced planning approximation must be labeled as such and compared with a causally evaluated policy; never label free full-path recourse as executable dispatch.

The available `power` observation is a final quarter-hour average of **AC production**, not instantaneous power. Give it an availability timestamp at interval end plus configured telemetry latency. For policy-treated observations, also respect the plateau/interpolation evidence-availability timestamps from Section 3.4. No DC-to-AC efficiency factor is applied to these measurements.

An instantaneous PV measurement and the final average PV over a not-yet-delivered interval are different information. Define the control convention and latency. The non-oracle controller must not observe the realized interval energy before choosing a command that precedes that measurement.

Distinguish a pre-interval battery command from autonomous, measurement-driven PV curtailment. Curtailment may react at an explicitly defined measurement node, but must not exploit later prices or weather. If only interval-average measurements exist, document the execution approximation rather than claiming demonstrated sub-interval performance. Apply the same convention to all executable benchmarks.

### 6.8 Builder API, solving, and independent validation

Provide an API with responsibilities equivalent to:

```python
build_model(inputs: OptimizationInput) -> BuiltModel
solve_model(built: BuiltModel, settings: SolverSettings) -> OptimizationResult
validate_solution(inputs: OptimizationInput,
                  result: OptimizationResult) -> ValidationReport
```

Use separate constraint-building functions for physical balances, trading, nonanticipativity, settlement, and degradation. Retain variable handles and cost expressions for extraction without parsing variable-name strings.

Set Gurobi parameters only through validated configuration. Include threads, time limit, relative/absolute MIP gap, seed, and logging. Use MW/MWh scaling and physically tight bounds rather than enormous arbitrary big-M values.

Handle solver outcomes explicitly [R5–R6]:

- Extract solution values only when an incumbent exists (`SolCount > 0`).
- Distinguish optimal, feasible-but-time-limited, infeasible, unbounded, and no-incumbent outcomes.
- Record objective, bound, MIP gap where defined, build/solve time, sizes, and violations.
- A time limit is not a proof of infeasibility or optimality.
- For infeasibility, export the model and compute an IIS when appropriate. Diagnose ambiguous infeasible/unbounded status before interpreting it.
- Never silently relax physical constraints to manufacture a feasible result.
- Support LP/MPS export for small debugging runs; do not dump huge files by default.

Implement an independent numerical validator that recomputes energy balances, POI limits, mode feasibility, commercial positions, settlement, nonanticipativity, and objective components from extracted arrays. Tests must not merely repeat the exact same builder expressions.

Use prior solutions as warm starts only when variables can be mapped correctly by time and node. Begin with rebuilding the residual model for clarity; do not prioritize fragile in-place updates.

## 7. Battery degradation implementation

Provide a tested **constant-throughput baseline** first, with an explicit convention, such as EUR/MWh of AC charge-plus-discharge throughput:

```text
cost[s] = cost_per_mwh_throughput
          * sum_t (charge_mw[t,s] + discharge_mw[t,s]) * delta_hours[t]
```

Do not confuse charge-plus-discharge throughput, discharged-energy-only cost, and equivalent full cycles. Declare the replacement-cost/cycle-life assumptions used to set coefficients; no battery lifetime data are supplied by the farm dataset.

The preferred reference is Xu et al., *Factoring the Cycle Aging Cost of Batteries Participating in Electricity Markets* [R7]. Read the original equations before implementing its piecewise-linear marginal cycle-aging approximation. Preserve its segmented-energy accounting, depth dependence, marginal costs, ordering and initialization assumptions. Identify the implemented equations and approximation boundaries in code and documentation.

A piecewise-linear penalty on instantaneous power is not automatically a cycle-depth aging model. Do not label a constant-throughput penalty as the Xu formulation. If the reference formulation cannot be faithfully implemented or verified, leave the tested baseline operational and explicitly mark the preferred model incomplete; do not expose a misleading working configuration option.

Carry any required cycle/segment/degradation state through auctions and dispatch updates; do not reset aging history at reoptimization. Report aging cost separately from market cash flows. Do not add unrequested battery chemistry or thermal submodels.

## 8. Weather-to-PV engine for the actual inputs

Implement a pvlib-based forecast transformation using the installed library version [R8–R11]. Its operational target is **farm AC production**, because historical `power` measures AC output. Retain intermediate modeled DC predictions as diagnostics where relevant, and return AC forecasts on the market's quarter-hour grid with farm, cycle, availability, lead, model assumptions, and member origin.

### 8.1 Radiation inputs and unresolved source semantics

Do not use historical satellite `ghi` or `fraction_diffuse` as future forecast inputs. Do not substitute the next observed GHI when a weather forecast radiation value is missing.

Expose separate configuration fields:

```text
shortwave_interpretation: ghi
shortwave_interpretation_status: user_confirmed
direct_radiation_interpretation: horizontal_direct
direct_radiation_interpretation_status: user_working_assumption
satellite_ghi_resolution: 15min
forecast_ghi_resolution: 1h
```

Use hourly shortwave radiation as forecast GHI, not the quarter-hourly satellite measurements. After the authorized negative-value zero floor and interval-aware conversion, under the direct-radiation working interpretation:

```text
GHI = forecast shortwave radiation
BHI = forecast direct horizontal radiation
DHI = GHI - BHI
```

DNI must then be derived with solar geometry and suitable low-sun handling; conceptually `DNI = BHI / cos(zenith)` in daylight. Applying this identity to redistributed interval means using a representative zenith is itself an approximation. Document it, avoid division near the horizon, and use pvlib's appropriate irradiance helpers/limits where supported [R9]. **Never treat an hourly BHI mean as instantaneous DNI.**

Apply Section 3.4's zero floor to negative forecast shortwave/direct radiation without changing raw arrays. Then check remaining consistency, including BHI exceeding GHI. Do not silently introduce further corrections: a derived-model limiting rule must be explicit and logged; material contradictions make the affected interval invalid or exploratory. NaNs remain missing rather than becoming zero.

For daytime intervals with a validated GHI forecast but unusable direct radiation, an explicitly selected GHI-only decomposition model may be used as an alternative. Record which intervals use it; do not quietly fall back. `fraction_diffuse` remains disabled until its actual definition is confirmed.

### 8.2 Solar geometry, tracking, and bifaciality

Use farm-specific true coordinates only when supplied. Otherwise allow **explicitly acknowledged Milan proxy coordinates**, stored once in YAML with `location_status: proxy`. Retain supplied farm elevation separately. Proxy location is not evidence of the actual site and must not drive audit decisions.

For source azimuth `a`, pvlib uses north-zero clockwise bearings [R8]. In our data south=0 degrees and west -90 degrees.

- Farms 1/345: use fixed surface tilt/azimuth.
- Farm 910: use single-axis geometry with explicit axis tilt, axis azimuth, maximum rotation, backtracking, and GCR assumptions [R10].
- Farm 901: include a supported bifacial tracking model only with the required geometry, albedo and bifaciality assumptions. Otherwise use a clearly labeled front-side-only tracker approximation when explicitly enabled, and record omission of rear-side gain.

Do not silently choose module databases, axis direction, tracking geometry, or a bifacial multiplier. Do not multiply plant capacity by a bifacial factor and then count the same gain again in irradiance.

### 8.3 Temperature, wind, and DC output

Derive wind speeds from the supplied u/v components with the vector magnitude. Use 10 m or 100 m consistently with the chosen temperature model; a height conversion or use of 10 m wind as a module-level proxy must be documented. Temperature is already in degrees Celsius.

Use a transparent DC model compatible with limited metadata, such as pvlib's PVWatts DC model [R11], with DC nameplate capacity, plane-of-array irradiance, cell-temperature estimate, and a configurable temperature coefficient. Respect each library function's units: when it expects nameplate watts, convert the supplied kW to W and convert the returned watts explicitly back to the canonical power unit. Do not invent string/module counts or detailed electrical specifications.

Calibrate against **historical measured AC output**, using the policy-treated past observations, their flags, any upstream review exclusions, and chronological splits. Fit a conversion/calibration mapping from modeled DC to measured AC where appropriate; never compare modeled DC directly with the AC target as though their electrical bases were identical. Report how masking and interpolation affect the training set. If usable calibration data are insufficient, use only an explicitly configured PV-output model and label its assumptions; do not borrow the BESS converter efficiency.

Do not force DC output to equal nameplate capacity as an undocumented hard clipping rule. Nameplate power is a rating, not an established plant-output cap for every irradiance and temperature. Apply configured physical limits where justified and report their origin.

### 8.4 Measured AC, modeled PV conversion, and the BESS inverter

**Historical replay:** use the measured or policy-treated farm AC power directly:

```text
pv_ac_actual_mw = power_ac_kw / 1000
```

Do not apply 0.97, another PV conversion factor, or a speculative PV inverter clipping model to these already measured AC values.

**Forecast production:** if pvlib first computes DC output from DC nameplate capacity, it still needs a separate, explicit mapping to predicted farm AC output. Use an available PV inverter model with separately supplied parameters, or a documented calibration of modeled PV DC output to the historical AC target. Any fallback PV-output efficiency or DC/AC-ratio assumption must have its own YAML fields and provenance; do not silently set it equal to the battery's 0.97 efficiency. Keep PV-output calibration and inverter clipping distinguishable and avoid double-counting losses.

An AC inverter rating cannot be inferred as a confirmed fact from DC nameplate capacity. Use supplied ratings or explicit assumptions. Measured clipping/plateaus and the requested plateau-zero policy must be reported when assessing forecast error; do not claim identification of hardware limits from data that the policy has zeroed.

**Battery:** reserve the configured 97% conversion efficiency for the BESS connection in Section 6.3. PV-to-grid energy bypasses that converter. PV-to-battery and grid-to-battery energy pass through its charging direction; battery-to-grid energy passes through its discharging direction.

Apply the configured nighttime-production zero mask to final AC forecasts on the 15-minute grid, keeping unmasked modeled output as a diagnostic. Do not confuse this adjustment with the energy-conserving redistribution of hourly radiation.

Measured AC output may include curtailment or outages, and policy zeros/interpolations are analysis transformations. Do not present it as verified unconstrained PV availability. Record the availability proxy used in physical backtests and report raw-versus-treated energy where possible.

### 8.5 Outputs

Return canonical arrays and a long-format export with at least:

```text
farm_id, forecast_cycle_time, forecast_available_at,
source_lead_time, interval_start_utc, interval_end_utc,
member_id, member_origin, pv_dc_mw, pv_ac_mw,
input_eligibility, review_status, treatment_flags, policy_version, assumptions_id
```

Member probability belongs to the declared probabilistic bundle; the current deterministic member has probability one. Save diagnostics by farm, cycle and delivery day. Compare **AC forecasts with measured/policy-treated AC observations** on exactly matching intervals and score forecast vintages separately. Modeled DC is a diagnostic, not an observed target. Do not mix overlapping leads into a fictitiously enlarged independent test set.

## 9. Price forecasting and joint scenario generation

### 9.1 Implement point forecasts with honest conditional information

Start with regularized linear regression, Ridge or LASSO depending on installed dependencies. Use chronological training/validation, and fit scalers/hyperparameters on training data only. Residual uncertainty must come from rolling-origin or otherwise genuinely out-of-sample forecast errors, not optimistic in-sample residuals.

Target the zonal clearing price for each actual auction product. Candidate features:

- Lagged DA and IDA prices/spreads, including recent days and the preceding week.
- Calendar, season, delivery period, and horizon.
- Previous auctions' clearing prices, **only after their publication**.
- Timestamped load, wind, aggregate solar forecasts and residual load.
- Available forecast revisions and other documented market fundamentals.

Use a minimal lag/calendar feature set if fundamentals are unavailable, and record exactly what was omitted. Historical price labels are not present in the farm Zarr/Parquet and must be imported separately. Do not attempt to train a price regression before targets exist.

For IDA1/2/3, model spreads relative to a preceding auction, using stage-appropriate features. Forecast whole delivery-price trajectories or a well-defined collection of per-product regressions; document the choice. Same-clock-time lags should use calendar/product joins, not a blindly applied `96 * number_of_days` offset across DST or market-regime changes.

Use already available forecast quantities as predictor features. Do not train a realistic DA forecast using the next day's realized regional production, final imports, or later forecasts. A local PV trace can be a statistical weather proxy without implying price impact, but it is not a validated representation of NORD production.

If enabling a local-to-regional solar proxy, keep regional baseline/capacity/scaling coefficients distinct from plant dispatch, estimate them on eligible past data, and flag the spatial approximation. The four co-located proxy coordinates are not evidence that their actual sites experience identical weather. A deterministic local forecast does not supply a regional ensemble.

### 9.2 Conditional price paths

For each actual PV member `s` and residual-price path `r`, build:

```text
price_path[s,r] = conditional_mean_price_path(features[s])
                  + conditional_residual_price_path[s,r]

probability[s,r] = pv_probability[s] * residual_probability[r | s]
```

Maintain temporal and cross-auction relationships. In the current store `s` has length one; multiple residual paths still produce a valid **stochastic-price / deterministic-PV** run. Use that precise label. When real ensembles are supplied, retain coherent member identities and use the same contracts.

Implement a Gaussian residual baseline with estimated scale and, where supportable, regularized covariance. An explicitly labeled independent-noise baseline is acceptable, but not a claim to calibrated market dependence. Provide an empirical **whole residual-vector bootstrap** option when enough out-of-sample history exists. Sampling covariance, scaling, and bootstrap selection must use only the training/calibration history.

For sequential auction models, construct a coherent conditional chain. Avoid adding a full cross-auction residual vector on top of spread models in a way that double-counts dependence. Document which residual belongs to each conditional model and how residuals of future auctions are conditioned on observed earlier outcomes.

Generate only unknown future prices at a reoptimization. Previously observed prices and accepted cash flows are fixed. Reconditioning model inputs does not require refitting regression coefficients on every event. Refit on an explicit chronological schedule only when configured.

### 9.3 Information trees are separate from forecast paths

Do not feed a flat list of uniquely labeled future paths to the optimizer and call it a nonanticipative multistage model. Construct nodes at actual decision/observation events, with multiple unresolved descendants where uncertainty remains. Test the tree against hand-built examples before connecting the learned scenario generator.

When projecting continuous observation histories to a finite tree, cluster or classify using **observable history only** and document the approximation. Retain unresolved future dispersion. A distinct floating-point DA-price vector must not identify a complete future PV/IDA path by construction.

The current data do not provide probabilistic future forecast revisions. In initial planning, freeze current weather forecasts unless another explicitly supplied model supports such revisions. In replay, update with the actual next vintage only after `available_at`. Do not read all historical cycles to plan DA recourse with perfect knowledge of tomorrow's forecasting errors.

Prices of imbalance settlement must also be forecast or represented with a declared uncertainty/point-estimate model in executable strategies. Using future realized imbalance prices is an oracle assumption, even if auction prices are forecast correctly. Do not silently mix these modes.

Scenario probabilities must be nonnegative and sum to one; clustered representative trajectories take appropriate cluster weights. Three selected low/central/high residual trajectories are not automatically equiprobable. Keep random seeds and scenario/model provenance.

If enough data for a learned generator are not available, finish the backbone on synthetic scenarios and clearly mark the learned generator blocked. Do not disguise synthetic distributions as fitted historical forecasts.

## 10. Market providers, imports, and caching

Use Terna and ENTSO-E APIs only for datasets their official documentation actually supports. The user has credentials for these providers; authenticated GME access is not assumed. Implement or reuse a manual GME historical-price-file importer for MGP and MI-A1/2/3.

Do not assume all IDA clearing prices are exposed by Terna or ENTSO-E. Verify dataset identifiers, zone codes, units, publication semantics, revisions, and historical coverage. Do not fabricate endpoints or silently replace auctions with continuous-market indices.

At provider boundaries, normalize syntax/units/timestamps into canonical market records. Keep Section 3.4's farm transformations separate; they must not clip negative market prices or fill missing clearing prices. Validate counts and report duplicates or gaps rather than making undocumented corrections.

Use cache keys including provider, dataset, zone, product, date range, forecast vintage/revision, and schema version. Read cache first, retrieve only missing data, and save retrieval time separately from historical availability. A download timestamp today cannot establish when a forecast was available in a historical backtest.

Use secure environment credentials or hidden `getpass` prompts in interactive download commands. Unattended training/replay must fail with actionable missing-input messages rather than prompt indefinitely. Do not expose tokens in URLs, saved requests, exceptions, or documentation.

The study assumes no additional grid-use fee on imports used to charge the BESS. Keep this configurable and label it an excluded-cost assumption, not a verified universal exemption. Implement the applicable signed settlement for the chosen historical market regime, or a clearly labeled synthetic settlement for software tests. Do not invent imbalance penalties to create profitable-looking forecasts.

## 11. Event engine, accepted-trade ledger, and causal execution

Implement distinct events for:

1. Forecast/fundamental publication.
2. Auction decision/submission cutoff.
3. Auction-results publication.
4. Physical control execution and completed observation availability.
5. Delivery/imbalance settlement.

Define deterministic event ordering when timestamps coincide. Respect computation buffers and feature cutoffs; do not use a 10:00 observation in a bid that had to be finalized at 09:59.

At a results event, update prices and the accepted-trade ledger; do not change that auction's submitted offer. At the next actionable gate, rebuild the residual problem with fixed accepted positions, correctly timestamped energy/degradation state, available forecasts, and conditional future distributions. Submit only that gate's decision. Future planned adjustments are not accepted trades.

Before delivery starts, use an explicit known delivery-start energy assumption for the first daily experiment. Do not pretend a D−1 SoC is the D 00:00 SoC. During delivery, carry the actual simulated state forward and never reset it on an auction or forecast event.

IDA3 modifies only noon-onward commercial products, but physical control between its decision time and noon still affects available energy. Keep those physical intervals in the residual model even though no new IDA3 trade is allowed for them. Already elapsed intervals are immutable.

With quarter-hour average AC observations, implement a clearly specified execution convention. Battery decisions at an interval start can use only completed past measurements and published forecasts. Do not feed the interval's final AC average to that decision as if it were instantaneous telemetry. Respect evidence availability for retrospectively zeroed plateaus and two-sided interpolated observations.

Separate scheduled battery actions from a physical safety/curtailment response. If an interval-average simulator uses idealized within-interval protection, declare that approximation and give every executable strategy the same protection. Record corrections and their imbalance costs; do not claim high-frequency hardware validation. Forecast scenarios guarantee feasibility on the modeled support, not automatically outside it.

Use the declared policy-treated **AC output directly** for ex-post delivery, retaining upstream exclusions and treatment flags. Fill eligible missing measurements only with Section 3.4's linear interpolation, never with forecasts or an additional 97% conversion. Do not treat remaining unexplained offsets or policy-created values as verified physical availability.

For batches of independent delivery-day studies, record the boundary-state assumption and do not label the result continuous multi-day operation. If evaluating continuously, all overlapping auctions and delivery events must share one physical trajectory; do not independently simulate overlapping state histories.

## 12. Benchmarks, metrics, and experiment outputs

Use one physical/economic implementation for:

- **Oracle:** actual future PV/prices, explicitly non-executable, with the same admissible trades and terminal assumptions.
- **Deterministic sequential:** point forecasts; current auction optimized without valuing later trading opportunities, then revised at later gates.
- **Deterministic integrated:** point forecasts and explicit anticipation of subsequent auctions.
- **Stochastic integrated:** uncertain scenario prices and any genuinely supplied PV uncertainty, shared-node decisions, and event-driven conditioning.

Do not compare strategies on different observation masks, budgets, physical constraints, settlement conventions, or information cutoffs without making that an explicit experimental factor. Do not name two identical behaviors as separate benchmarks. An oracle's feasible incumbent is not automatically a certified upper bound; report its optimization bound/status appropriately.

Keep ex-ante expected profit separate from realized operating cash flow. The terminal-energy credit is not an auction payment. Save DA/IDA revenues, imbalance settlement, degradation, excluded charges, curtailment, SoC, and objective reconciliation separately.

For each run, save a machine-readable manifest with config, farm, data slice/review version, masking/interpolation policy and adjusted-energy totals, source vintage, availability assumptions, forecast/model fit cutoff, scenario probabilities/seed, tree sizes, code version when available, solver sizes/status/gap, and timing. Do not hash the full dataset; use available immutable dataset identifiers, file metadata, and reproducible slice identifiers.

Group results by run ID, farm, delivery day, strategy, and scenario configuration. Produce non-interactive plots of PV forecasts/observations, forecast errors by lead/issue, prices/spreads, bids and revisions, SoC/charge/discharge, actual grid exchange versus net position, profit components, and scenario-count/runtime stability. Clearly label raw and policy-treated AC observations, interpolated values, modeled DC diagnostics, AC forecasts, synthetic scenarios, and missing series.

Do not calculate ensemble calibration or CRPS as evidence of meteorological ensemble quality for a store containing only one deterministic vintage at a time. Point-forecast evaluation and stochastic-price evaluation are different outputs.

## 13. Tests and acceptance checks

Implement executable small fixtures, not only descriptions of tests. Separate tests requiring a Gurobi license from solver-independent tests; record passed, failed, skipped, and not-run cases separately.

### Dataset and timing tests

- Native `(time, farm_id)` power and `(farm_id, time)` radiation align after named transposition.
- Parquet joins remain correct when farm order is permuted; capacities convert correctly from kW to MW.
- `valid_time = cycle_time + lead_time` for both timedelta and numeric-hour lead encodings.
- Overlapping cycles remain different vintages, never ensemble members.
- Lead-zero structural forecast missingness stays missing; negative-value clipping does not turn NaNs into zero, and measurement interpolation is not applied to forecast gaps.
- A 12:00 end-labeled radiation mean maps to the four 11:00–12:00 quarter-hours.
- Hourly irradiation and precipitation totals are conserved by the selected redistribution.
- Instantaneous weather interpolation uses the same already available vintage.
- Four-hour assumed latency, explicit availability override, local/UTC conversion, gate buffers, and DST are handled correctly.
- Completed historical power becomes observable at interval end, not its start timestamp.
- Future forecast cycles, final fundamentals, and unpublished prices cannot enter earlier features.
- Audit flags/review decisions remain unchanged; no source writes occur.
- Three repeated nonzero production samples are retained; four or more almost-equal consecutive native samples zero the entire finalized run.
- Plateau tolerances, zeros/NaNs/gaps as run boundaries, and anchor-based matching work as configured; detection is farm-specific and precedes interpolation.
- Positive production is zeroed from 21:30 inclusive to 05:00 exclusive in the configured timezone, including UTC conversion and DST tests.
- Negative shortwave/direct-radiation forecasts are zeroed; negative temperatures, signed wind components, and electricity prices are preserved.
- Interior missing satellite GHI and power are linearly interpolated in time; leading/trailing gaps remain missing, and policy-created zeros remain zeros.
- Interpolation does not reintroduce positive nighttime production or manufacture plateaus for the detector.
- Transforms are deterministic/idempotent, masks carry reason flags, and raw-versus-treated energy reconciles.
- Two-sided interpolations and retroactive plateau labels are not exposed before their evidence-availability time.

### PV model tests

- `fraction_diffuse` is unused unless its definition is explicitly confirmed.
- Horizontal direct radiation is not passed as DNI; low-sun conversion is bounded/flagged.
- Derived radiation/PV transformations do not mutate raw arrays.
- Source azimuth conversion tests include both allowed sign conventions.
- Tracker axis and fixed surface orientation are distinct parameters.
- Unknown PV-type IDs do not trigger invented electrical specifications.
- Farm 901's rear-side contribution is either explicitly modeled or disclosed as omitted.
- Wind vector magnitude, Celsius temperature, DC nameplate scaling, and the separate PV-output model/calibration are correct.
- Measured power is AC and is not multiplied by 0.97; AC forecasts are evaluated against AC observations.
- The BESS converter efficiency is applied once in each physical conversion direction, not to direct PV exports; converter-only round-trip efficiency is 0.9409 when cell efficiencies are one.
- Combined-efficiency configuration cannot double-count the same 97% converter loss.
- Nighttime synthetic fixtures behave physically; Milan proxy geometry is not a fault classifier.

### Optimization and economic tests

- Energy balance, efficiency, initial/final limits, POI/inverter limits, curtailment, no simultaneous battery charging/discharging.
- Zero PV, no-BESS configuration, negative prices, saturated/empty battery, and unreachable terminal target.
- Sale of 1 MWh at EUR100/MWh followed by purchase of 0.3 MWh at EUR80/MWh leaves 0.7 MWh and EUR76 contractual cash flow, before delivery costs.
- Offsetting financial trades do not create physical cycling/efficiency losses.
- Signed imbalance costs reconcile; positive/negative decomposition cannot create artificial arbitrage.
- Position/trade limits prevent unbounded financial round trips.
- Degradation costs are nonnegative under the declared convention, and aging state is not reset on reoptimization.
- Single-scenario equivalence and unequal-probability weighting.
- Splitting one leaf into identical copies with the same total mass leaves the optimum unchanged.
- DA/IDA bids and physical controls share decisions until information genuinely diverges.
- Different latent future outcomes alone cannot produce different actions at the same node. This does not require the root optimum to stay unchanged if the predictive distribution itself changes.
- Observed DA prices do not act as perfect identifiers of unresolved entire future paths.
- Previously cleared trades and executed dispatch remain fixed.
- Solver no-incumbent, infeasible, unbounded, license-error, and time-limit outcomes are handled correctly.
- Independent numerical recomputation matches extracted objective and ledger values.

Begin with 2–4 physical intervals, one auction and a few leaves; extend to interleaved DA/IDA events and a complete ordinary/DST day. Profile build time, solve time, memory where available, variable/binary/constraint counts, number of nodes/leaves, and objective bounds. Increase scenario counts only after small tests pass. A 50-member-by-3-residual example is a possible interface/scaling test, not evidence that 50 PV members exist in this store.

## 14. Implementation sequence and runnable deliverables

Proceed incrementally; do not spend the entire task producing architecture notes.

**Milestone 1 — Inspect and establish contracts.** Read instructions, environment and Git status; check licensing; inspect Zarr/Parquet metadata lazily; record exact mappings, corrected AC/radiation semantics, assumptions and audit interfaces; create configs and domain types. Implement and test the narrow Section 3.4 transformation layer. Preserve all source data and audit code.

**Milestone 2 — Deterministic optimization backbone.** Build physical/trading/settlement equations, bounded decisions, terminal treatment, throughput degradation, solver diagnostics, and independent validation. Run an analytical synthetic case without APIs or farm files.

**Milestone 3 — Stochastic backbone.** Add probability-weighted objective, auction/control information nodes, conditional branching and nonanticipativity. Test on a hand-constructed tree before attempting empirical scenarios. Implement a small end-to-end synthetic auction replay.

**Milestone 4 — Actual weather-to-PV integration.** Bind the supplied dimensions and variable semantics; implement causal vintage selection, hourly forecast-GHI conversion, fixed/tracker/bifacial handling, modeled DC diagnostics and separately converted/calibrated AC forecasts. Apply the requested policy and compare with treated measured AC output. Use explicit assumptions for unresolved metadata, preserve the audit, and report transformation sensitivity rather than claiming certified ground truth.

**Milestone 5 — Market forecasts and replay.** Import market targets/features from verified providers or cached/manual files; fit eligible linear/spread models; generate conditional residual price paths; integrate them with the current one-member PV forecast; carry ledger/SoC through events. If market prices or usable measurements remain missing after the declared policy, mark the affected real-market evaluation blocked rather than fabricate results.

**Milestone 6 — Validate, document, and report.** Run available tests/benchmarks, implement and test the reference aging model where reproducible, save outputs and performance statistics, then generate the LaTeX status report specified below from the actual final state.

Provide CLI commands equivalent to:

```text
inspect-data --config ...
prepare-analysis-data --config ...
validate-config --config ...
run-synthetic --config ...
forecast-pv --config ...
import-market-data --config ...
train-price-models --config ...
optimize --config ...
backtest --config ...
report-implementation --config ...
```

Only advertise commands/modes that work. Commands for blocked integrations may exist with clear validation errors, but must not imply successful implementation. Document exact commands compatible with the `bess` environment and no package-installation step.

The final delivery includes code, configs tied to the actual schema, tests, a reproducible synthetic run, real-data adapter/forecast status, market-data availability status, numerical outputs/plots, and the LaTeX report. Give a concise final handoff describing changed files, commands run, observed tests/results, and remaining blockers. Do not commit/push unless asked.

## 15. Mandatory LaTeX implementation-status report

**After finishing the coding and available tests, create `docs/implementation_status.tex`.** This is a required deliverable even when some modules are blocked. It must be an English technical description of the **state of the implementation at completion**, not a proposal, literature overview, aspirational specification, or generic copy of this prompt.

Also create:

```text
docs/implementation_status.bib
docs/implementation_status.json
```

Use the JSON file, test logs, run manifests, and code inspection as traceable evidence for the report. Small generated tables may be included from a `docs/generated/` directory. Do not create reports that say "implemented" solely because an interface or TODO exists.

### 15.1 Required contents

1. **Snapshot and scope.** Date/time, repository path, Git revision if one exists and dirty-state status, actual environment/library versions, solved problem scope, and what code/tests were inspected. Do not expose credentials or private machine details beyond what is needed for reproducibility.

2. **Status matrix.** For each major component: module path/entry point, status (`implemented-and-tested`, `implemented-not-executed`, `partial`, `blocked`, or `not-implemented`), actual evidence, and blocker where applicable. Include deterministic/stochastic optimization, nonanticipativity, empirical tree generation, derived masking/interpolation, pvlib conversion, each market adapter, price regressions, residual paths, ledger/replay, throughput and reference aging, and independent validation.

3. **Dataset and provenance.** Exact file names under `data/farms/`, observed dimensions/coverage, farm IDs/DC nameplate capacities, **measured AC power basis**, hourly forecast shortwave/GHI versus quarter-hour satellite GHI, forecast vintages, and the absence of an ensemble axis. Document the masking/interpolation policy, tolerances, clock-window timezone, changed-value counts and energy, evidence availability, and retained audit exclusions. Distinguish user-confirmed meanings from attributes verified in the store and unresolved assumptions.

4. **Time semantics and causality.** State interval-start/end conventions, four-hour assumed forecast latency, local/UTC conversion, market gate/publication events, DST, observation availability, historical product regime, and one concrete traced forecast-to-auction example if executed.

5. **Weather-to-PV implementation.** Document actual pvlib functions, forecast-GHI/BHI/DNI handling, irradiance redistribution, temperature/wind model, tracker/bifacial assumptions, proxy coordinates, unconfirmed azimuth/diffuse semantics, PV-output conversion/calibration against AC measurements, and production masks. Explain that 97% is the BESS converter efficiency per direction and is not applied to measured PV or direct PV exports.

6. **Price and scenario implementation.** Actual features/targets/training splits, information cutoffs, conditional spread regressions, residual sampling/covariance, probabilities, number and origin of PV members, tree construction/conditioning, and omitted uncertainties. A deterministic PV run with price scenarios must be named exactly that.

7. **Mathematical formulation matching the code.** Define indices, units, states, continuous/binary variables, physical constraints, auction eligibility, net positions, signed settlement, nonanticipativity, initial/terminal treatment, degradation and the expected-profit objective. Mark any derived or simplified formula as such. Do not include equations for unimplemented methods as if operational.

8. **Execution/backtest architecture.** Describe how accepted trades and state are carried, when forecasts change, which decisions are implemented, how physical observations become available, and the execution/safety approximation. Distinguish a synthetic demonstration, an exploratory raw-data run, a policy-treated historical replay, and any independently reviewed historical backtest.

9. **Verification and actual results.** Exact commands, passed/failed/skipped/not-run tests, scenario counts, build/solve times, solver status/bound/gap, independent balance/accounting errors, and links or relative paths to run artifacts. No invented timings, profits, validation dates, or numerical claims. If solver tests were blocked, say so.

10. **Reproducibility and limitations.** How to rerun the demonstrated configurations; required external files/credentials without secret values; unresolved source meanings and geometry, residual data-quality limitations and masking/interpolation effects; missing market data; conditional-scenario approximations; and empirical limits of current results. Limit this section to observed limitations and concrete blockers in the requested implementation.

### 15.2 LaTeX quality and validation

Use a clean technical `article` layout with readable equations and tables. Prefer widely available packages; do not install a TeX distribution or packages. Keep paths relative to the repository/report and escape underscores, percent signs and other special characters. Use `booktabs`/`longtable` or equivalent only if available, and use a modest section hierarchy.

Cite the original aging reference and official documentation actually used through verified BibTeX entries; do not invent DOIs or cite sources not consulted. Dataset-specific claims must be attributed to the user's schema/verified local metadata, not to public pvlib or market documentation.

Use actual generated plots when available. Guard missing optional figures so the report remains compilable; write a plain explanation where evidence is absent. Do not insert placeholder plots that look like results.

If `latexmk`, `pdflatex`, or another suitable TeX engine is already installed, compile the report, inspect the log for errors/undefined references/overfull content, and fix issues within available tooling. Generate `docs/implementation_status.pdf` only after a successful build and report whether visual inspection was performed. If the compiler or bibliography tooling is absent, still deliver `.tex`, `.bib`, and `.json`, record that compilation was not executed, and provide the exact intended command. Do not claim a compiled report without producing one.

The report must be regenerated/updated after the final tests so its component statuses and numbers agree with the delivered code and run artifacts. Include even unsuccessful or blocked work honestly.

## 16. Authoritative implementation references

Use the following as starting references, checking installed versions and historical validity. Record consulted URLs, access dates, and the specific statements supported in `docs/sources.md`. Public documents cannot resolve ambiguous private dataset provenance by themselves.

- [R1] GME operating rules and technical documents: https://www.mercatoelettrico.org/en-us/Home/HowOperate
- [R2] ENTSO-E Intraday Auctions, gate closures and delivery windows: https://www.entsoe.eu/network_codes/cacm/implementation/ida/
- [R3] GME quarter-hour MGP delivery go-live notice: https://www.mercatoelettrico.org/en-us/Home/NoticesandAnnouncements/NoticesandAnnouncementsME?id=7101
- [R4] Gurobi Multiple Scenarios: https://docs.gurobi.com/projects/optimizer/en/current/features/multiscenario.html
- [R5] Gurobi solver status codes: https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html
- [R6] Gurobi model attributes: https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/model.html
- [R7] Xu et al., *Factoring the Cycle Aging Cost of Batteries Participating in Electricity Markets*: https://doi.org/10.1109/TPWRS.2017.2733339 ; author preprint https://arxiv.org/abs/1707.04567
- [R8] pvlib nomenclature/azimuth conventions: https://pvlib-python.readthedocs.io/en/stable/user_guide/extras/nomenclature.html
- [R9] pvlib irradiance/DNI conversion: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.dni.html
- [R10] pvlib single-axis tracking: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.tracking.singleaxis.html
- [R11] pvlib PVWatts DC: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html
- Terna developer catalog: https://developer.terna.it/
- ENTSO-E Transparency Platform: https://transparency.entsoe.eu/

## Final execution instruction

Inspect briefly, preserve raw inputs and the existing audit, implement the requested derived-data policy and tested mathematical core, integrate the available data, run feasible validations, and **finish by producing the LaTeX implementation-status report from the actual evidence**. A missing provider, unresolved direct-radiation meaning, unsupported geometry, or remaining unfillable measurement gap is a specific limitation or blocker to report—not permission to invent data or stop independent work. Keep measured PV power AC, reserve the 97% conversion loss for the BESS connection, and distinguish hourly forecast GHI from quarter-hour satellite GHI throughout. The required outcome is a working, honest, traceable implementation—not just scaffolding or a planned architecture.

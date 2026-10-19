# Assumptions and unresolved inputs

This records implemented configuration, not installed equipment specifications.
The implementation prompt supersedes conflicting older statements in
`AGENTS.md`: measured `power` is already AC; 0.97 belongs to the BESS converter;
the source directory is `data/farms/`; a separate derived-data policy is
authorized. Raw farm files and the original audit remain separate.

## Confirmed dataset meanings and causal clocks

- `configs/data_schema.yaml` binds the inspected Zarr and matching Parquet
  basenames `pv_farms_timeseries_20250101_20260831`. Forecasts have
  `(farm_id, cycle_time, lead_time)` dimensions. Cycles are forecast vintages,
  not probabilistic members. Farms are separate plants, not scenarios.
- DC nameplate capacities are 953.28, 1524.60, 4160 and 6638 kW for IDs
  1, 345, 901 and 910. These do not establish POI or inverter ratings.
  PV-type IDs do not establish electrical module parameters.
- Historical plant `power` is AC kW on native quarter-hour, interval-start
  labels; quarter-hour energy is `power_kw * 0.25 / 1000` MWh. No 0.97 factor
  is applied to measured power or direct PV exports.
- Forecast shortwave is hourly mean GHI over the hour ending at valid time;
  historical satellite `ghi` is a distinct quarter-hour field. Forecast direct
  radiation is treated as horizontal direct under an explicitly labeled working
  assumption. Public provider documentation cannot prove its private origin.
- Source times are UTC. Assumed forecast latency is four hours after cycle
  time, with explicit availability override support. Configured telemetry latency
  is zero; a completed production average is still unavailable before interval
  end. Feature cutoffs include the configured five-minute auction buffer.
- Delivery days use `Europe/Rome`, giving 92/96/100 quarter-hours across DST.
  Each experiment optimizes one delivery day, with a timestamped boundary state.

## Requested derived-data policy

`requested-v1` is an analysis transformation, not verified fault classification.
It first detects native nonzero production plateaus using an anchor comparison,
minimum four samples, assumed absolute tolerance 0.01 kW and relative tolerance
0.0001. It zeros the entire qualifying run. This can also remove real clipping
plateaus. It preserves the raw values and flags the change.

Positive production is zeroed from 21:30 inclusive to 05:00 exclusive in
`Europe/Rome`. This fixed clock assumption applies to measured/forecast AC
production and is independent of Milan solar geometry. It is reapplied after
measurement interpolation. The policy zero-floors only the explicitly listed
nonnegative weather magnitudes; temperatures, signed winds, negative prices and
NaNs retain their meanings.

Interior missing historical power and GHI use per-farm linear interpolation
between finite anchors, with no default maximum gap and no extrapolation.
Forecast missingness is not filled using measurements. Plateau and interpolation
evidence clocks preserve when a finalized correction actually becomes usable.
Raw-versus-treated production energy is reported; treated output is not certified
unconstrained available PV. Existing review exclusions remain active. The real
example uses exploratory mode and does not imply independent audit sign-off.

## PV geometry and output conversion

`configs/farms_nord.yaml` acknowledges Milan coordinates (45.4642 N, 9.1900 E)
as a proxy, with farm elevation kept separately. The task's later explicit
statement that source west is −90 establishes the configured east-positive,
south-zero azimuth conversion; the code also tests the opposite convention.
Metadata module orientation is kept distinct from tracker-axis geometry.

Tracker assumptions are axis tilt 0 degrees, axis azimuth 0 degrees, maximum
rotation 60 degrees, backtracking enabled and GCR 0.35. Farm 901 uses an explicitly
enabled front-side-only tracker approximation; rear gain is omitted. Albedo 0.2,
Faiman coefficients 25.0/6.84, 10 m wind as a module-level proxy, PVWatts DC
temperature coefficient −0.004 per degree Celsius and 25 C reference temperature
are model assumptions, not identified hardware parameters.

Uniform hourly-mean redistribution conserves irradiation and does not create
measured sub-hourly cloud variability. BHI is converted through solar geometry
and pvlib DNI, with an 85-degree low-sun threshold and an explicitly flagged
1400 W/m² DNI ceiling. The default invalidates BHI greater than GHI rather than
silently repairing it. The separate
`configs/farms_nord_radiation_sensitivity.yaml` explicitly enables clipping BHI
to GHI with a retained contradiction flag; results from it are exploratory
sensitivity outputs. `fraction_diffuse` remains disabled.

The default farm AC forecast uses a separately declared, unfitted PV-output
efficiency of 0.96 and no assumed AC clipping rating. This is distinct from the
BESS converter. No completed empirical AC calibration is claimed by that
configuration. Measured AC can include outages and curtailment, and policy zeros
affect any subsequent calibration/evaluation.

## Synthetic equipment and economics

All values below are software-test inputs in `configs/full_experiments/synthetic_small.yaml`;
they do not describe farm 1's actual battery, POI or operating contract.

| Quantity | Configured synthetic value |
|---|---|
| Battery internal capacity / admissible energy | 1 MWh / 0.1–0.9 MWh |
| AC charge and discharge power / converter ratings | 0.5 MW in either direction |
| POI import and export limits | 1 MW in either direction |
| Converter efficiencies / cell efficiencies | 0.97 each direction / 1.0 each direction |
| Self-discharge | Zero |
| Delivery-start energy / terminal target | 0.5 MWh / 0.5 MWh |
| AC charge-plus-discharge degradation | EUR 5 per MWh throughput |
| New auction adjustment / cumulative position bound | Each signed trade and cumulative position use POI MW × interval hours; for this 1 MW synthetic POI on 15-minute intervals each directional limit is 0.25 MWh |
| Single imbalance price / absolute-imbalance penalty | EUR 60/MWh / EUR 100/MWh |
| Grid import fee | Zero; taxes and brokerage also excluded |

No capacity-scaled real-equipment defaults are inferred. Real-market
optimization remains blocked until actual or explicitly accepted study
equipment and settlement inputs are supplied.

The synthetic PV trace has peak 0.8 MW and assumed 07:00–19:00 local daylight.
Price scenarios use a daily base EUR 70/MWh, configured daily variation,
conditional DA/IDA residual levels and branch values −1/+1 with probabilities
0.4/0.6. These finite synthetic masses need not give zero-mean innovations.
They are not estimated market uncertainty. Weather/PV is deterministic and
perfectly forecast in this test. The oracle is explicitly non-executable.

The solver uses one thread, a 30-second time limit, relative MIP gap
of 1e−4 (0.01%) and seed 17. No absolute MIP gap is configured.
No package/environment/license modifications are required.
The existing license needs execution outside the sandbox on this machine because
sandbox execution cannot expose its matching host identifier.

## Trading, information and execution conventions

The study assumes price-taking accepted fixed-quantity self-schedule trades.
Later auction trades increment earlier positions, and offsetting financial
trades incur no battery losses. Shared node-owned bids and physical controls
prevent latent future outcomes from revealing the realized path. Equal observed
price histories share information even if random draws had duplicate labels.

Only the actionable submitted offer becomes a commitment. Submitted quantities
awaiting results remain fixed pending positions; only results events add
accepted prices and contractual cash flows to the immutable ledger. Unpublished
fixed contractual cash-flow constants are excluded and identified in intermediate
residual objectives; their quantities still constrain physical settlement.

Physical battery commands are chosen before interval delivery. The synthetic
simulator uses final average PV only for the separately declared idealized
interval-average safety response and ex-post accounting. An auction cutoff
inside an already executing quarter-hour uses the committed battery command to
project its next boundary state. This uses no completed future PV observation;
the perfect deterministic synthetic-PV fixture makes its trajectory consistent.
It is not a validated high-frequency controller or a guarantee outside modeled
scenario support. Executed SoC and cumulative throughput/degradation are carried
through events, rather than reset at each auction.

## Market and model limitations retained explicitly

GME DTF03 revisions 13/14 supply auction gates and IDA result deadlines. The
13:30 local DA result time is a conservative study clock, not an official
guaranteed deadline; actual recorded results can override ordinary schedules.
Quarter-hour DA delivery begins 1 October 2025, and earlier native-hourly
commercial mapping is not implemented. IDA3 products begin at local noon.

No historical auction-price series or usable imbalance-settlement regime is
contained in the farm dataset. Manual imports require actual source fields and
publication semantics. Terna/ENTSO-E authenticated downloads and historical
settlement are not claimed operational without verified schemas and inputs.
The no-import-fee configuration is a study exclusion, not an established legal
tariff exemption. An absolute-imbalance penalty implements the requested
deterrent, rather than proving compliance with unverified settlement law.

Ridge and conditional residual/tree components have executable fixtures; real
price fitting and a historical trading backtest need genuine eligible price
targets and publication clocks. Initial planning uses the latest currently
available weather cycle. The GME replay refreshes it at each reoptimization
using only cycles published by that decision cutoff; it does not anticipate
future weather revisions or treat successive cycles as an ensemble.

Both the constant AC charge-plus-discharge `throughput` baseline and Xu et al.'s
discharge-only `xu_segmented` model are implemented and tested. Xu's Eq. (5)
prices increments of a configurable convex cycle-depth stress curve, and Eq. (7)
charges discharged AC energy from each segment. Explicit priority binaries
implement Appendix Eqs. (37)--(40): both charge and discharge use the shallowest
available segment first. Initialization uses Eq. (39) once at the assumed study
start, and the resulting cycle-memory segment vector is carried through
execution and reoptimization. It is not reconstructed from aggregate SoC.

`configs/full_experiments/synthetic_xu.yaml` separately assumes a total EUR 300,000 pack
replacement cost, eight depth segments and the paper's illustrative Eq. (24)
stress parameters 0.000524 and 2.03. These are configurable study assumptions,
not farm battery calibration or a converted historical pack-price quotation.
Depth is relative to nameplate internal capacity, rather than the operational
SoC window. Xu charging half cycles cost zero and discharging half cycles cost
the same as full cycles of that depth. Nonzero self discharge is unsupported
in Xu mode, and combining Xu with a nonzero throughput coefficient is rejected.
Historical memory before the assumed study start is unknown unless supplied.
Finite segments approximate the configured depth stress; no chemistry, thermal,
calendar-aging or capacity-fade feedback model is claimed. There is no terminal
salvage-value mode. See [the degradation equation and validation notes](xu_degradation.md).

Numerical performance, actual run profits, source-transform counts and test
counts belong in the generated implementation-status artifacts; this file does
not substitute illustrative numbers for executed evidence.

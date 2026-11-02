# January 2026 continuous PV scenario experiment

The two farm 1 configurations `continuous_2026_01_farm1_pv_scenarios_exact.yaml`
and `continuous_2026_01_farm1_pv_scenarios_penalty.yaml` use the existing
`experiment-continuous` command. Each runs an integrated policy and a perfect-realized-PV
reference solved once at day ahead, with its subsequent intraday orders retained.
Both continue physical execution against measured PV after the configured
month-specific `policy.realized_pv_night` clipping. The raw source remains
unchanged and clipping is recorded in the run's PV provenance; approved episode
exclusions are not applied to execution. The perfect-PV reference uses the same
effective realized series. Obtain experiment approval before running.
The deterministic forecast and every ensemble member also use the configured
month-specific `policy.forecast_pv_night` at model inference. The selected
weather vintage and forecast-model provenance remain separately recorded.

At each auction the optimizer receives 50 equally weighted raw weather trajectories
converted to PV by the existing PVLib baseline, plus the latest deterministic PV
forecast with the same weight. The ensemble uses the preceding delivery day's 00 UTC
cycle and the configured six-hour availability lag, which is a historical assumption.
The deterministic forecast updates at each cutoff. The configured PV conversion uses
the existing 2025 fitted coefficient as prior metadata; this study fits no parameters.
Input provenance records each cycle, conversion, value fingerprint and nighttime-only
missing-weather fills. Missing daytime forecasts fail preflight.

All auction orders and planned battery controls are shared across the 51 scenarios.
PV curtailment is interval feedback, tied for identical observed PV values; it does
not alter battery state or later orders. Retained-operation boundary SOC projections
use the deterministic forecast, while execution uses measured PV and the existing
commitment-tracking feedback, which may change battery commands. This is an idealized
interval-average controller, not a physical telemetry simulation.

The exact configuration enforces zero planned imbalance in every scenario. The
penalty configuration permits deviations, values planned imbalance settlement at
zero, and subtracts EUR 100/MWh of expected absolute deviation from the objective.
Both settle actual deviations at recorded imbalance prices. The illustrative penalty
on realized deviations is reported separately and is not a cash charge. Perfect
auction prices are used in planning for both strategies. The perfect-PV reference
does not reoptimize at intraday gates and is an information benchmark, not a
guaranteed upper bound on realized profit.

After both studies complete, produce a four-way comparison with:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python scripts/compare_pv_scenario_replays.py \
  --exact outputs/continuous_optimization/farm_1/2026_01_pv_scenarios_exact \
  --penalty outputs/continuous_optimization/farm_1/2026_01_pv_scenarios_penalty \
  --output outputs/continuous_optimization/comparisons/farm_1/2026_01_pv_scenarios_penalty100_comparison
```

The comparison requires complete runs with matching delivery days. Daily tables retain
the initial forecast objective; those objectives cover different information sets and
are not additive across repeated auction reoptimizations. Solver status, achieved gap,
time limits, failed solves and previous-plan fallbacks remain in each run's diagnostics.

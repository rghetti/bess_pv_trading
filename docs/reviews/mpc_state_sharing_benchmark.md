# Exact MPC battery-state sharing

The legacy research formulation created one SOC variable and transition equation for every PV scenario even when their battery controls were the same. Battery SOC is determined by the initial state and the history of charge/discharge controls, not by PV directly. For paths sharing a non-recombining control node, the state after that node is therefore identical. The revised formulation creates one state and transition per control node and aliases it across those paths. It checks that nodes do not merge different previous battery histories. PV, POI, signed settlement, scenario weights and conditional actions remain scenario-specific, and no scenario or physical constraint is dropped. Xu's segment memory remains pathwise constrained.

The table reports medians of five sequential one-thread Gurobi rebuilds on the specified saved reviewed-data forecast horizon. Other workloads were active, so timings are indicative for these two instances only; counts are exact before Gurobi presolve. The recorded objective and issued first action agreed across all repetitions, and all solves reached Gurobi's configured `1e-4` relative MIP tolerance.

| Saved instance | Model | Variables | Binary variables | Linear constraints | Median build s | Median solver s | Objective EUR |
|---|---|---:|---:|---:|---:|---:|---:|
| Farm 910, 2026-08-02, adaptive ECC, 51 paths × 96 products | Before | 31,670 | 2,996 | 21,677 | 3.122 | 1.067 | -6,891.237991 |
| Same | After | 29,720 | 2,996 | 19,727 | 2.784 | 0.531 | -6,891.237991 |
| Farm 1, 2026-01-15, open-loop economic ensemble, 51 paths × 96 products | Before | 20,070 | 96 | 15,089 | 2.421 | 0.583 | 854.943199 |
| Same | After | 15,220 | 96 | 10,239 | 1.815 | 0.151 | 854.943199 |

Two additional shortened adaptive horizons were compared three times each on the same reviewed farm-910 day. At 10:00 UTC (48 products), variables fell from 18,422 to 18,056 and constraints from 12,577 to 12,211; median build/solver times were 1.154/0.318 s before and 1.766/0.457 s after. At 10:30 UTC (46 products), variables fell from 17,782 to 17,460 and constraints from 12,155 to 11,833; median build/solver times were 2.333/0.546 s before and 1.362/0.282 s after. All twelve solves reached the configured gap with matching objectives and first actions within each horizon; the 10:30 action charged at 2.5 MW. The slower 10:00 measurement and fluctuating shared-machine load mean that a speedup is not guaranteed for every horizon.

The benchmark reconstructs the saved eligible PV/price forecasts, initial SOC and already submitted positions; it does not execute a new physical replay. For example, from the research repository:

```bash
python -m scripts.benchmark_adaptive_model outputs/energy_research/reviewed_farm910_august_v1 august_development__farm_910__adaptive_ecc ensemble_mpc_adaptive --output /tmp/adaptive_model_benchmark.json --repeat 5
```

The adjacent before/after JSON files retain the instance identity and measurements. Pass `--decision-index` to select a later saved MPC decision. These results do not establish the speedup or ranking for an entire month. Unit and integration checks cover shared histories, nonanticipativity, physical reconstruction, terminal constraints and Xu cycle memory. The ongoing January matrix was started before this change and remains pinned to its original source snapshot; this optimization applies to subsequent runs.

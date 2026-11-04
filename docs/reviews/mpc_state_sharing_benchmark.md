# Exact MPC battery-state sharing

The legacy research formulation created one SOC variable and transition equation for every PV scenario even when their battery controls were the same. Battery SOC is determined by the initial state and the history of charge/discharge controls, not by PV directly. For paths sharing a non-recombining control node, the state after that node is therefore identical. The revised formulation creates one state and transition per control node and aliases it across those paths. It checks that nodes do not merge different previous battery histories. PV, POI, signed settlement, scenario weights and conditional actions remain scenario-specific, and no scenario or physical constraint is dropped. Xu's segment memory remains pathwise constrained.

The table reports medians of five sequential one-thread Gurobi rebuilds on the specified saved reviewed-data forecast horizon. Other workloads were active, so timings are indicative for these two instances only; counts are exact before Gurobi presolve. The recorded objective and issued first action agreed across all repetitions, and all solves reached Gurobi's configured `1e-4` relative MIP tolerance.

| Saved instance | Model | Variables | Binary variables | Linear constraints | Median build s | Median solver s | Objective EUR |
|---|---|---:|---:|---:|---:|---:|---:|
| Farm 910, 2026-08-02, adaptive ECC, 51 paths × 96 products | Before | 31,670 | 2,996 | 21,677 | 3.122 | 1.067 | -6,891.237991 |
| Same | After | 29,720 | 2,996 | 19,727 | 2.784 | 0.531 | -6,891.237991 |
| Farm 1, 2026-01-15, open-loop economic ensemble, 51 paths × 96 products | Before | 20,070 | 96 | 15,089 | 2.421 | 0.583 | 854.943199 |
| Same | After | 15,220 | 96 | 10,239 | 1.815 | 0.151 | 854.943199 |

The benchmark reconstructs the saved eligible PV/price forecasts, initial SOC and already submitted positions; it does not execute a new physical replay. For example, from the research repository:

```bash
python -m scripts.benchmark_adaptive_model outputs/energy_research/reviewed_farm910_august_v1 august_development__farm_910__adaptive_ecc ensemble_mpc_adaptive --output /tmp/adaptive_model_benchmark.json --repeat 5
```

The adjacent before/after JSON files retain the instance identity and measurements. These results do not establish the speedup or ranking for an entire month. Unit and integration checks cover shared histories, nonanticipativity, physical reconstruction, terminal constraints and Xu cycle memory. The ongoing January matrix was started before this change and remains pinned to its original source snapshot; this optimization applies to subsequent runs.

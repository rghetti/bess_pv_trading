# Continuous MPC results on reviewed PV targets

These are closed-loop replays against each case's own grid-tracking controller;
auction positions are carried forward and reoptimized as information changes.
They are retrospective development comparisons using pvlib forecasts, not the
deferred neural-model benchmark. Results depend on the reviewed target's
synthetic replacements and assumed market/forecast information.

## October 2025, deterministic MPC

Over 1–7 October, deterministic penalty MPC increased realized cash profit
relative to tracking by EUR 44.53, 675.75, 1,157.14 and 1,408.45 for farms 1,
345, 901 and 910 respectively. Total absolute imbalance increased by 89.91 MWh
across farms. There were 2,688 MPC decisions with no tracking fallback or solver
timeout. The [independent audit](continuous_mpc_reviewed_october_v1.md) found no
physical or cash-accounting violations. Final stored energy differed, so the
cash-profit difference is not a controlled inventory-adjusted comparison.

## January 2026, four-farm development matrix

The 15–16 January matrix completed 104 jobs across 26 cases and four farms.
The following are **mean daily portfolio differences**, with all four farms
summed, relative to each case's tracking baseline:

| Controller | Profit difference EUR/day | Absolute imbalance difference MWh/day | Wear difference EUR/day |
|---|---:|---:|---:|
| Deterministic penalty MPC | +61.83 | +2.85 | +8.91 |
| Ensemble penalty MPC | +37.40 | +4.56 | +93.05 |
| Adaptive raw ensemble MPC | +150.86 | +11.78 | +128.26 |

The [deterministic comparison](../../outputs/energy_research/reviewed_january_matrix_v1/comparisons/deterministic_penalty/research_comparison.md),
[ensemble comparison](../../outputs/energy_research/reviewed_january_matrix_v1/comparisons/ensemble_penalty/research_comparison.md)
and [adaptive comparison](../../outputs/energy_research/reviewed_january_matrix_v1/comparisons/adaptive_raw/research_comparison.md)
contain per-farm and daily results. Two days do not support a statistical
generalization. The [independent all-job audit](../../outputs/energy_research/reviewed_january_matrix_audit.md)
found no physical, accounting or artifact violations. Some oracle economic
refinements were rejected on numerical validation; validated stage-one plans
were retained. The audit distinguishes those from grid-tracking fallbacks.

Examples of the generated plots are the [ensemble paired daily profit](../../outputs/energy_research/reviewed_january_matrix_v1/comparisons/ensemble_penalty/plots/paired_daily_realized_operating_profit_eur.png),
[ensemble paired daily imbalance](../../outputs/energy_research/reviewed_january_matrix_v1/comparisons/ensemble_penalty/plots/paired_daily_absolute_imbalance_mwh.png),
and [farm 910 ensemble SOC and power on 15 January](../../outputs/energy_research/reviewed_january_matrix_v1/runs/farm_910/january_development__farm_910__ensemble_penalty/continuous/plots/soc_and_planned_executed_power_2026-01-15.png).

## August 2026, farm 910

On 2–3 August, ensemble penalty MPC gained EUR 96.81 against tracking, while
the adaptive calibrated ECC case gained EUR 271.80. Both increased imbalance;
the latter used four physical safety overrides. The price-oracle case lost
EUR 62.94 and had 40 time-limited solves. Its 1% gap fallback gave the same
realized outcome on these dates. See the [audited comparison](continuous_mpc_reviewed_farm910_august_v1.md).
All 192 target intervals in this window were reconstructed daylight values or
policy-defined night zeros, so this is not independent meter-based validation.

These comparisons include changes to future auction submissions and battery
state, not just immediate physical dispatch. Tracking has actual interval PV
feedback while MPC uses its configured current PV forecast. Terminal penalties
and final stored energy differ, and no terminal inventory value is added to cash
profit. These differences can influence profit and imbalance together. See
[reviewed PV input provenance](reviewed_pv_replay_inputs.md) and the
[research results index](energy_research_results_index.md) for source details.

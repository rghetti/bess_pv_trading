# One-week MPC curtailment-recovery comparison

15–21 January 2026; four farms; continuous closed-loop trajectories. Changes are recovery minus the original MPC.

| Farm | Recovered PV MWh | Curtailment change MWh | Imbalance change MWh | Wear change EUR | Profit change EUR | Final energy change MWh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000 | +0.000 | +0.000 | +0.00 | +0.00 | +0.000 |
| 345 | 0.422 | -0.023 | -0.012 | +1.14 | +4.83 | +0.000 |
| 901 | 2.194 | -0.681 | +0.093 | +32.80 | +82.35 | +0.000 |
| 910 | 0.000 | +0.000 | +0.000 | +0.00 | +0.00 | +0.000 |

| Farm | Tracking imbalance MWh | Original MPC MWh | Recovery MPC MWh | Original final SOC % | Recovery final SOC % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.434 | 0.643 | 0.643 | 50.00 | 50.00 |
| 345 | 1.031 | 1.240 | 1.229 | 50.00 | 50.00 |
| 901 | 3.607 | 4.182 | 4.275 | 50.00 | 50.00 |
| 910 | 5.860 | 8.580 | 8.580 | 50.00 | 50.00 |

The recovery action preserves protected grid exchange at that interval; subsequent SOC, auction orders and dispatch may differ. Recovered PV is AC energy; battery energy also reflects efficiency and avoided discharge. Planning penalties are not actual charges. The one-week sample, old validation-selected PV models, current-interval average PV disclosure and perfect auction-price planning make this a retrospective development comparison. Final SOC must be inspected against the final-study target.

## Portfolio results and financial attribution

| Controller | Absolute imbalance MWh | Signed settlement EUR | Wear EUR | Realized profit EUR | Total curtailment MWh |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tracking | 10.933 | -1695.02 | 1522.19 | 36635.19 | 0.743 |
| Original MPC | 14.645 | -2337.39 | 1419.62 | 36489.92 | 1.540 |
| Recovery MPC | 14.727 | -2219.36 | 1453.56 | 36577.10 | 0.836 |

| Farm | Signed settlement change EUR | Auction revenue change EUR | Wear-cost change EUR | Profit change EUR |
| --- | ---: | ---: | ---: | ---: |
| 1 | +0.00 | +0.00 | +0.00 | +0.00 |
| 345 | +2.88 | +3.09 | +1.14 | +4.83 |
| 901 | +115.15 | +0.00 | +32.80 | +82.35 |
| 910 | +0.00 | +0.00 | +0.00 | +0.00 |

Changes compare independent closed-loop trajectories, so recovered PV, changes in aggregate curtailment, and changes in later auction orders are different quantities. Recovered energy is AC PV across the actual recovery intervals, not net extra battery energy or a guaranteed reduction in curtailment relative to the other trajectory.

Both MPC variants ended at 50% SOC on every farm. Tracking ended at 34.60%, 42.47%, 88.12% and 24.76% on farms 1, 345, 901 and 910 respectively. Cash profit assigns no value to remaining inventory, so direct comparisons with tracking are not inventory-adjusted.

## Auction-boundary projections

The table excludes the first study day, whose midnight state is supplied as an initial assumption. Values are pooled absolute energy errors across individual auction projections; three midnight auction projections per farm/day are correlated.

| MPC case | Boundary | Projections | Mean absolute energy error MWh | Fallback intervals |
| --- | --- | ---: | ---: | ---: |
| mpc_original | delivery_midnight | 72 | 0.05338 | 0 |
| mpc_original | ida3_noon | 24 | 0.10733 | 0 |
| mpc_recovery | delivery_midnight | 72 | 0.05407 | 0 |
| mpc_recovery | ida3_noon | 24 | 0.10388 | 0 |

## Validation and source data

All eight jobs completed: 5,376 MPC decisions, all reported optimal; zero MPC fallbacks, timeouts or physical safety overrides. The tracking reference had 31 physical safety overrides across its four unique trajectories. The independent accounting and physics audit found no violations and the tracking reference matched between paired cases.

The separate action audit reconstructed all 2,688 recovery-case intervals: 45 recovery intervals, 2.615388 MWh of gross AC PV recovered, no SOC-bound violations, and maximum grid-neutrality residual below 5e-16 MW. Every recovery was maximal under its protected action and battery limits.

For these dates each farm used 329 metered PV intervals and 343 monthly-night-zero intervals; no synthetic replacement interval was used. Frozen reviewed-GHI models were selected retrospectively. Current-interval average PV was disclosed; later PV remained forecast-based. Auction planning used actual-price curves. Historical timing assumptions and the small development sample limit interpretation.

The licensed software suite passed 47 tests; 22 overlapping physical/report tests and two targeted chronology tests also passed. Eight check-only preflights passed. Batch wall time was 633.6 seconds; cumulative MPC solver runtime was 39.26 seconds. No GPU use or training was part of the run.

Batch manifest SHA-256: `95dd495dadb3a075b174f25bd168eb0e0fee00ecbf0a3a1402f43f06c3ca4c1d`. The input config, complete interval outputs, plans, solver diagnostics, projection records and generated plots are preserved under `outputs/energy_research/curtailment_recovery_jan15_21_2026_v1/`. See `curtailment_recovery_one_week_audit.md` and `curtailment_recovery_one_week_action_audit.json` for independent checks.

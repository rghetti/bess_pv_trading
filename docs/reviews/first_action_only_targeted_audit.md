# Continuous MPC validation

Source batch: `/home/rghetti/bess_pv_trading_realtime_mpc/outputs/energy_research/first_action_only_jan15_17_2026_v1`.
Pinned manifest SHA-256: `f129df3ebf5e564476e87a02302cd9fa46f860bd40a66a5662be8c1ce1050d81`.
Batch status: **complete**; 16/16 jobs audited.

Differences below are target MPC minus its own continuous grid-tracking baseline. Positive profit is higher realized operating profit; positive imbalance is more absolute imbalance. These are retrospective development windows, not an untouched test set.

| Window | Farm | Controller case | Profit difference EUR | Imbalance difference MWh | Wear difference EUR | Final inventory difference MWh | Tracking fallbacks | Timeouts | Safety overrides | Rejected oracle refinements |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| january_three_days | 1 | common_costs | 5.226 | 0.033087 | -1.114 | -0.092440 | 0 | 0 | 0 | 0 |
| january_three_days | 1 | common_imbalance_only | 8.788 | 0.038316 | -0.560 | -0.092440 | 0 | 0 | 0 | 0 |
| january_three_days | 1 | first_action_costs | 7.107 | 0.083405 | -3.775 | -0.092440 | 0 | 0 | 0 | 0 |
| january_three_days | 1 | first_action_imbalance_only | 6.043 | 0.072527 | -3.301 | -0.092440 | 0 | 0 | 0 | 0 |
| january_three_days | 345 | common_costs | -55.207 | 0.049800 | -15.182 | 0.237424 | 0 | 0 | 0 | 0 |
| january_three_days | 345 | common_imbalance_only | -56.969 | 0.052888 | -22.247 | 0.237424 | 0 | 0 | 0 | 0 |
| january_three_days | 345 | first_action_costs | -21.330 | 0.100065 | -7.719 | 0.237424 | 0 | 0 | 0 | 0 |
| january_three_days | 345 | first_action_imbalance_only | -152.755 | 0.148178 | -2.867 | 0.237424 | 0 | 0 | 0 | 0 |
| january_three_days | 901 | common_costs | -4.094 | -0.006499 | -1.941 | 0.007742 | 0 | 0 | 0 | 0 |
| january_three_days | 901 | common_imbalance_only | -10.703 | -0.006499 | -3.969 | 0.007742 | 0 | 0 | 0 | 0 |
| january_three_days | 901 | first_action_costs | -7.239 | -0.020984 | -1.089 | 0.007742 | 0 | 0 | 0 | 0 |
| january_three_days | 901 | first_action_imbalance_only | -16.245 | 0.007391 | -1.557 | 0.007742 | 0 | 0 | 0 | 0 |
| january_three_days | 910 | common_costs | -106.595 | 0.862792 | 13.198 | 0.827274 | 0 | 0 | 0 | 0 |
| january_three_days | 910 | common_imbalance_only | -180.463 | 0.903990 | -13.398 | 0.827274 | 0 | 0 | 0 | 0 |
| january_three_days | 910 | first_action_costs | -127.185 | 0.873296 | 11.839 | 0.827274 | 0 | 0 | 0 | 0 |
| january_three_days | 910 | first_action_imbalance_only | -306.156 | 1.048338 | -26.996 | 0.827274 | 0 | 0 | 0 | 0 |

## Matched controller totals

| Case | Completed jobs | Profit difference EUR | Imbalance difference MWh | MPC solves | Tracking fallbacks | Timeouts | Rejected refinements |
|---|---:|---:|---:|---:|---:|---:|---:|
| common_costs | 4 | -160.671 | 0.939180 | 1152 | 0 | 0 | 0 |
| common_imbalance_only | 4 | -239.347 | 0.988695 | 1152 | 0 | 0 | 0 |
| first_action_costs | 4 | -148.647 | 1.035782 | 1152 | 0 | 0 | 0 |
| first_action_imbalance_only | 4 | -469.113 | 1.276435 | 1152 | 0 | 0 | 0 |

## Independent checks

The auditor found **0 physical/accounting/artifact violations** at a `1e-6` reconstruction tolerance. It checks SOC and power bounds, charge/discharge exclusivity, POI bounds, energy transition/continuity, PV balance, immutable ledger positions, single price publications, signed settlement, operating profit and recorded oracle imbalance caps.

| Reconstructed residual | Maximum |
|---|---:|
| quarter_hour_duration | 0 |
| energy_transition_mwh | 8.8817842e-16 |
| energy_continuity_mwh | 0 |
| grid_balance_mw | 4.4408921e-16 |
| commitment_mwh | 1.110223e-16 |
| auction_revenue_eur | 7.1054274e-15 |
| imbalance_mwh | 1.4398205e-16 |
| settlement_eur | 1.9539925e-14 |
| profit_eur | 2.8421709e-14 |
| summary_profit_eur | 9.094947e-13 |
| summary_absolute_imbalance_mwh | 2.6090241e-15 |
| soc_lower_mwh | 0 |
| soc_upper_mwh | 0 |
| power_lower_mw | 0 |
| power_upper_mw | 0 |
| simultaneous_charge_discharge_mw | 0 |
| poi_import_mw | 0 |
| poi_export_mw | 0 |
| curtailment_lower_mw | 0 |
| curtailment_upper_mw | 0 |
| publication_cash_eur | 1.8189894e-12 |

## Numerical and provenance findings


0 accepted oracle plans exceed their recorded reconstructed-MWh cap beyond the configured `2 * FeasibilityTol` allowance.


0 explicit oracle solves retain nested source metadata saying settlement was supplied only for ex-post accounting. This is a provenance-label contradiction; the top-level oracle disclosure is explicit.

## Interpretation

Grid tracking uses actual interval PV feedback. MPC current-PV information, auction-price forecasts and terminal settings depend on the resolved job configuration; consult `comparison_configuration` in the accompanying JSON and the individual controller provenance. Physical settlement-price oracles are noncausal and are not guaranteed upper bounds on realized profit under uncertain PV.

The continuous comparisons include changed battery states and consequent later auction submissions. Profit differences therefore include auction cash-flow changes as well as physical dispatch. Final stored energy can differ, and reported profit has no inventory-value adjustment. A configured terminal constraint in auction planning does not by itself guarantee that the tracking controller physically ends at the target. Cash-profit rankings can reflect terminal-inventory tradeoffs. An increase in profit alongside more imbalance is possible under signed settlement; it is not by itself evidence of a forecasting improvement or a controller bug.

Interpret findings in light of the selected dates, forecast provenance, shared prices across farms and actual sample size. Short development windows do not support precise extreme-tail estimates or independent-farm significance claims.

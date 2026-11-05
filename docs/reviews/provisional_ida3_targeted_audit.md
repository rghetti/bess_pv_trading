# Continuous MPC validation

Source batch: `/home/rghetti/bess_pv_trading_realtime_mpc/outputs/energy_research/provisional_ida3_jan15_17_2026_v1`.
Pinned manifest SHA-256: `6f73d941aa920ab914cfde1c44b6c93347c58caa2bee72377b32bd6953292e7d`.
Batch status: **complete**; 8/8 jobs audited.

Differences below are target MPC minus its own continuous grid-tracking baseline. Positive profit is higher realized operating profit; positive imbalance is more absolute imbalance. These are retrospective development windows, not an untouched test set.

| Window | Farm | Controller case | Profit difference EUR | Imbalance difference MWh | Wear difference EUR | Final inventory difference MWh | Tracking fallbacks | Timeouts | Safety overrides | Rejected oracle refinements |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| january_three_days | 1 | ida2_plan | 6.685 | 0.098674 | -4.788 | -0.092440 | 0 | 0 | 0 | 0 |
| january_three_days | 1 | submitted_only | 9.665 | 0.289517 | -0.683 | -0.092440 | 0 | 0 | 0 | 0 |
| january_three_days | 345 | ida2_plan | -21.397 | 0.154732 | -8.722 | 0.237424 | 0 | 0 | 0 | 0 |
| january_three_days | 345 | submitted_only | -55.243 | 0.299431 | -19.781 | 0.237424 | 0 | 0 | 0 | 0 |
| january_three_days | 901 | ida2_plan | -5.647 | 0.238232 | 1.279 | 0.007742 | 0 | 0 | 0 | 0 |
| january_three_days | 901 | submitted_only | -67.606 | 0.783595 | 2.910 | 0.007742 | 0 | 0 | 0 | 0 |
| january_three_days | 910 | ida2_plan | -83.414 | 1.741085 | -34.828 | 0.827274 | 0 | 0 | 0 | 0 |
| january_three_days | 910 | submitted_only | -115.482 | 1.419825 | 32.748 | 0.827274 | 0 | 0 | 0 | 0 |

## Matched controller totals

| Case | Completed jobs | Profit difference EUR | Imbalance difference MWh | MPC solves | Tracking fallbacks | Timeouts | Rejected refinements |
|---|---:|---:|---:|---:|---:|---:|---:|
| ida2_plan | 4 | -103.773 | 2.232723 | 1152 | 0 | 0 | 0 |
| submitted_only | 4 | -228.666 | 2.792368 | 1152 | 0 | 0 | 0 |

## Independent checks

The auditor found **0 physical/accounting/artifact violations** at a `1e-6` reconstruction tolerance. It checks SOC and power bounds, charge/discharge exclusivity, POI bounds, energy transition/continuity, PV balance, immutable ledger positions, single price publications, signed settlement, operating profit and recorded oracle imbalance caps.

| Reconstructed residual | Maximum |
|---|---:|
| quarter_hour_duration | 0 |
| energy_transition_mwh | 8.8817842e-16 |
| energy_continuity_mwh | 0 |
| grid_balance_mw | 4.4408921e-16 |
| commitment_mwh | 9.8879238e-17 |
| auction_revenue_eur | 1.4210855e-14 |
| imbalance_mwh | 1.4398205e-16 |
| settlement_eur | 2.0983215e-14 |
| profit_eur | 2.8421709e-14 |
| summary_profit_eur | 0 |
| summary_absolute_imbalance_mwh | 3.0531133e-15 |
| soc_lower_mwh | 0 |
| soc_upper_mwh | 0 |
| power_lower_mw | 0 |
| power_upper_mw | 0 |
| simultaneous_charge_discharge_mw | 0 |
| poi_import_mw | 0 |
| poi_export_mw | 0 |
| curtailment_lower_mw | 0 |
| curtailment_upper_mw | 0 |
| publication_cash_eur | 9.094947e-13 |

## Numerical and provenance findings


0 accepted oracle plans exceed their recorded reconstructed-MWh cap beyond the configured `2 * FeasibilityTol` allowance.


0 explicit oracle solves retain nested source metadata saying settlement was supplied only for ex-post accounting. This is a provenance-label contradiction; the top-level oracle disclosure is explicit.

## Interpretation

Grid tracking uses actual interval PV feedback. MPC current-PV information, auction-price forecasts and terminal settings depend on the resolved job configuration; consult `comparison_configuration` in the accompanying JSON and the individual controller provenance. Physical settlement-price oracles are noncausal and are not guaranteed upper bounds on realized profit under uncertain PV.

The continuous comparisons include changed battery states and consequent later auction submissions. Profit differences therefore include auction cash-flow changes as well as physical dispatch. Final stored energy can differ, and reported profit has no inventory-value adjustment. A configured terminal constraint in auction planning does not by itself guarantee that the tracking controller physically ends at the target. Cash-profit rankings can reflect terminal-inventory tradeoffs. An increase in profit alongside more imbalance is possible under signed settlement; it is not by itself evidence of a forecasting improvement or a controller bug.

Interpret findings in light of the selected dates, forecast provenance, shared prices across farms and actual sample size. Short development windows do not support precise extreme-tail estimates or independent-farm significance claims.

# Continuous MPC validation

Source batch: `/home/rghetti/bess_pv_trading_realtime_mpc/outputs/energy_research/first_interval_priority_jan15_21_2026_v1`.
Pinned manifest SHA-256: `e9177633a65cf48df58bea65b7ccf90b559fbd34fc3be0e2aba66f6967a169d0`.
Batch status: **complete**; 12/12 jobs audited.

Differences below are target MPC minus its own continuous grid-tracking baseline. Positive profit is higher realized operating profit; positive imbalance is more absolute imbalance. These are retrospective development windows, not an untouched test set.

| Window | Farm | Controller case | Profit difference EUR | Imbalance difference MWh | Wear difference EUR | Final inventory difference MWh | Tracking fallbacks | Timeouts | Safety overrides | Rejected oracle refinements |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| january_one_week | 1 | original | -3.926 | 0.208086 | -5.143 | 0.115485 | 0 | 0 | 0 | 0 |
| january_one_week | 1 | strict_first | -13.937 | 0.122294 | 3.129 | 0.115485 | 0 | 0 | 0 | 0 |
| january_one_week | 1 | weighted_k10 | -14.592 | 0.134409 | 3.138 | 0.115485 | 0 | 0 | 0 | 0 |
| january_one_week | 345 | original | 2.566 | 0.209453 | -4.349 | 0.075323 | 0 | 0 | 0 | 0 |
| january_one_week | 345 | strict_first | -10.118 | 0.129063 | 2.017 | 0.075323 | 0 | 0 | 0 | 0 |
| january_one_week | 345 | weighted_k10 | -11.969 | 0.129060 | 0.934 | 0.075323 | 0 | 0 | 0 | 0 |
| january_one_week | 901 | original | 31.339 | 0.574702 | -37.299 | -0.762389 | 0 | 0 | 0 | 0 |
| january_one_week | 901 | strict_first | 60.395 | 0.163726 | -20.089 | -0.762389 | 0 | 0 | 0 | 0 |
| january_one_week | 901 | weighted_k10 | 60.690 | 0.161664 | -20.180 | -0.762389 | 0 | 0 | 0 | 0 |
| january_one_week | 910 | original | -175.254 | 2.720549 | -55.777 | 1.261866 | 0 | 0 | 0 | 0 |
| january_one_week | 910 | strict_first | -184.583 | 1.248493 | 37.117 | 1.261866 | 0 | 0 | 0 | 0 |
| january_one_week | 910 | weighted_k10 | -179.274 | 1.154797 | 37.117 | 1.261866 | 0 | 0 | 0 | 0 |

## Matched controller totals

| Case | Completed jobs | Profit difference EUR | Imbalance difference MWh | MPC solves | Tracking fallbacks | Timeouts | Rejected refinements |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 4 | -145.275 | 3.712789 | 2688 | 0 | 0 | 0 |
| strict_first | 4 | -148.243 | 1.663576 | 2688 | 0 | 0 | 0 |
| weighted_k10 | 4 | -145.145 | 1.579930 | 2688 | 0 | 0 | 0 |

## Independent checks

The auditor found **0 physical/accounting/artifact violations** at a `1e-6` reconstruction tolerance. It checks SOC and power bounds, charge/discharge exclusivity, POI bounds, energy transition/continuity, PV balance, immutable ledger positions, single price publications, signed settlement, operating profit and recorded oracle imbalance caps.

| Reconstructed residual | Maximum |
|---|---:|
| quarter_hour_duration | 0 |
| energy_transition_mwh | 8.8817842e-16 |
| energy_continuity_mwh | 0 |
| grid_balance_mw | 4.4408921e-16 |
| commitment_mwh | 1.110223e-16 |
| auction_revenue_eur | 2.8421709e-14 |
| imbalance_mwh | 1.4398205e-16 |
| settlement_eur | 2.5757174e-14 |
| profit_eur | 5.6843419e-14 |
| summary_profit_eur | 3.6379788e-12 |
| summary_absolute_imbalance_mwh | 8.8817842e-15 |
| soc_lower_mwh | 0 |
| soc_upper_mwh | 0 |
| power_lower_mw | 0 |
| power_upper_mw | 0 |
| simultaneous_charge_discharge_mw | 0 |
| poi_import_mw | 0 |
| poi_export_mw | 0 |
| curtailment_lower_mw | 0 |
| curtailment_upper_mw | 0 |
| publication_cash_eur | 3.6379788e-12 |

## Numerical and provenance findings


0 accepted oracle plans exceed their recorded reconstructed-MWh cap beyond the configured `2 * FeasibilityTol` allowance.


0 explicit oracle solves retain nested source metadata saying settlement was supplied only for ex-post accounting. This is a provenance-label contradiction; the top-level oracle disclosure is explicit.

## Interpretation

Grid tracking uses actual interval PV feedback. MPC current-PV information, auction-price forecasts and terminal settings depend on the resolved job configuration; consult `comparison_configuration` in the accompanying JSON and the individual controller provenance. Physical settlement-price oracles are noncausal and are not guaranteed upper bounds on realized profit under uncertain PV.

The continuous comparisons include changed battery states and consequent later auction submissions. Profit differences therefore include auction cash-flow changes as well as physical dispatch. Final stored energy can differ, and reported profit has no inventory-value adjustment. A configured terminal constraint in auction planning does not by itself guarantee that the tracking controller physically ends at the target. Cash-profit rankings can reflect terminal-inventory tradeoffs. An increase in profit alongside more imbalance is possible under signed settlement; it is not by itself evidence of a forecasting improvement or a controller bug.

Interpret findings in light of the selected dates, forecast provenance, shared prices across farms and actual sample size. Short development windows do not support precise extreme-tail estimates or independent-farm significance claims.

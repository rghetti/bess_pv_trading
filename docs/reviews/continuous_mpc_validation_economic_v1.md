# Continuous MPC validation

Source batch: `/tmp/bess_research_economic_validation_snapshot/outputs/energy_research/economic_continuous_development_v1`.
Pinned manifest SHA-256: `e38599622c9f10a6d13e423a2a77bac4075e0dadb2f947ad19774af6a217cb45`.
Batch status: **complete**; 5/5 jobs audited.

Differences below are target MPC minus its own continuous grid-tracking baseline. Positive profit is higher realized operating profit; positive imbalance is more absolute imbalance. These are retrospective development windows, not an untouched test set.

| Window | Farm | Controller case | Profit difference EUR | Imbalance difference MWh | Wear difference EUR | Final inventory difference MWh | Tracking fallbacks | Timeouts | Safety overrides | Rejected oracle refinements |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| january_development | 1 | adaptive_raw | 18.397 | 2.389410 | 31.494 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 1 | adaptive_reduced | 14.573 | 1.729591 | 19.062 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 1 | causal_auctions | 27.715 | 2.163840 | 38.357 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 1 | deterministic_economic | 2.524 | 1.059346 | -8.268 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 1 | ensemble_economic | 13.244 | 2.272744 | 29.306 | 0.000000 | 0 | 0 | 0 | 0 |

## Matched controller totals

| Case | Completed jobs | Profit difference EUR | Imbalance difference MWh | MPC solves | Tracking fallbacks | Timeouts | Rejected refinements |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_raw | 1 | 18.397 | 2.389410 | 192 | 0 | 0 | 0 |
| adaptive_reduced | 1 | 14.573 | 1.729591 | 192 | 0 | 0 | 0 |
| causal_auctions | 1 | 27.715 | 2.163840 | 192 | 0 | 0 | 0 |
| deterministic_economic | 1 | 2.524 | 1.059346 | 192 | 0 | 0 | 0 |
| ensemble_economic | 1 | 13.244 | 2.272744 | 192 | 0 | 0 | 0 |

## Independent checks

The auditor found **0 physical/accounting/artifact violations** at a `1e-6` reconstruction tolerance. It checks SOC and power bounds, charge/discharge exclusivity, POI bounds, energy transition/continuity, PV balance, immutable ledger positions, single price publications, signed settlement, operating profit and recorded oracle imbalance caps.

| Reconstructed residual | Maximum |
|---|---:|
| quarter_hour_duration | 0 |
| energy_transition_mwh | 1.110223e-16 |
| energy_continuity_mwh | 0 |
| grid_balance_mw | 1.110223e-16 |
| commitment_mwh | 9.7144515e-17 |
| auction_revenue_eur | 3.5527137e-15 |
| imbalance_mwh | 1.5265567e-16 |
| settlement_eur | 2.2259972e-14 |
| profit_eur | 2.220446e-14 |
| summary_profit_eur | 0 |
| summary_absolute_imbalance_mwh | 3.9968029e-15 |
| soc_lower_mwh | 0 |
| soc_upper_mwh | 0 |
| power_lower_mw | 0 |
| power_upper_mw | 0 |
| simultaneous_charge_discharge_mw | 0 |
| poi_import_mw | 0 |
| poi_export_mw | 0 |
| curtailment_lower_mw | 0 |
| curtailment_upper_mw | 0 |
| publication_cash_eur | 2.2737368e-13 |

## Numerical and provenance findings


0 accepted oracle plans exceed their recorded reconstructed-MWh cap beyond the configured `2 * FeasibilityTol` allowance.


0 explicit oracle solves retain nested source metadata saying settlement was supplied only for ex-post accounting. This is a provenance-label contradiction; the top-level oracle disclosure is explicit.

## Interpretation

Grid tracking uses actual interval PV feedback; these MPC runs use forecast current PV. Perfect-foresight auction prices are common across members. Oracle physical controllers additionally see realized settlement prices, but retain the configured minimum-imbalance budget; an oracle controller is not a guaranteed upper bound on realized profit under uncertain PV.

The continuous comparisons include changed battery states and consequent later auction submissions. Profit differences therefore include auction cash-flow changes as well as physical dispatch. Final stored energy can differ, and reported profit has no inventory-value adjustment. These runs impose a soft 50% terminal-SOC penalty inside MPC while tracking has no equivalent soft terminal objective and the shared hard terminal target is null. Cash-profit rankings therefore do not isolate the controller from its terminal-inventory tradeoff. An increase in profit alongside more imbalance is possible under signed settlement; it is not by itself evidence of a forecasting improvement or a controller bug.

The short, selected January/August windows and shared prices across farms do not support precise extreme-tail estimates or independent-farm significance claims. Broader deterministic testing should retain paired tracking baselines, begin with the configured October 2025 week, and then include the full May 2026 month; these windows require no absent ensemble archive.

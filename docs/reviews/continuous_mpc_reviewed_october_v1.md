# Continuous MPC validation

Source batch: `/tmp/bess_research_reviewed_v1/outputs/energy_research/reviewed_october_deterministic_v1`.
Pinned manifest SHA-256: `d405e1b86af6f3465be785a4cf434b5e6607d8c46d9ea4697a36ae26234896af`.
Batch status: **complete**; 4/4 jobs audited.

Differences below are target MPC minus its own continuous grid-tracking baseline. Positive profit is higher realized operating profit; positive imbalance is more absolute imbalance. These are retrospective development windows, not an untouched test set.

| Window | Farm | Controller case | Profit difference EUR | Imbalance difference MWh | Wear difference EUR | Final inventory difference MWh | Tracking fallbacks | Timeouts | Safety overrides | Rejected oracle refinements |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| october_2025 | 1 | deterministic_penalty | 44.533 | 2.957596 | -74.855 | -0.021841 | 0 | 0 | 0 | 0 |
| october_2025 | 345 | deterministic_penalty | 675.752 | 13.351406 | -73.219 | -0.413430 | 0 | 0 | 0 | 0 |
| october_2025 | 901 | deterministic_penalty | 1157.143 | 28.842232 | -175.498 | -0.400569 | 0 | 0 | 0 | 0 |
| october_2025 | 910 | deterministic_penalty | 1408.452 | 44.754220 | -526.804 | -1.695056 | 0 | 0 | 0 | 0 |

## Matched controller totals

| Case | Completed jobs | Profit difference EUR | Imbalance difference MWh | MPC solves | Tracking fallbacks | Timeouts | Rejected refinements |
|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic_penalty | 4 | 3285.880 | 89.905454 | 2688 | 0 | 0 | 0 |

## Independent checks

The auditor found **0 physical/accounting/artifact violations** at a `1e-6` reconstruction tolerance. It checks SOC and power bounds, charge/discharge exclusivity, POI bounds, energy transition/continuity, PV balance, immutable ledger positions, single price publications, signed settlement, operating profit and recorded oracle imbalance caps.

| Reconstructed residual | Maximum |
|---|---:|
| quarter_hour_duration | 0 |
| energy_transition_mwh | 8.8817842e-16 |
| energy_continuity_mwh | 0 |
| grid_balance_mw | 8.8817842e-16 |
| commitment_mwh | 2.220446e-16 |
| auction_revenue_eur | 2.8421709e-14 |
| imbalance_mwh | 3.6082248e-16 |
| settlement_eur | 3.4028336e-14 |
| profit_eur | 5.6843419e-14 |
| summary_profit_eur | 7.2759576e-12 |
| summary_absolute_imbalance_mwh | 1.687539e-14 |
| soc_lower_mwh | 0 |
| soc_upper_mwh | 0 |
| power_lower_mw | 0 |
| power_upper_mw | 0 |
| simultaneous_charge_discharge_mw | 0 |
| poi_import_mw | 0 |
| poi_export_mw | 0 |
| curtailment_lower_mw | 0 |
| curtailment_upper_mw | 0 |
| publication_cash_eur | 0 |

## Numerical and provenance findings


0 accepted oracle plans exceed their recorded reconstructed-MWh cap beyond the configured `2 * FeasibilityTol` allowance.


0 explicit oracle solves retain nested source metadata saying settlement was supplied only for ex-post accounting. This is a provenance-label contradiction; the top-level oracle disclosure is explicit.

## Interpretation

Grid tracking uses actual interval PV feedback; these MPC runs use forecast current PV. Perfect-foresight auction prices are common across members. Oracle physical controllers additionally see realized settlement prices, but retain the configured minimum-imbalance budget; an oracle controller is not a guaranteed upper bound on realized profit under uncertain PV.

The continuous comparisons include changed battery states and consequent later auction submissions. Profit differences therefore include auction cash-flow changes as well as physical dispatch. Final stored energy can differ, and reported profit has no inventory-value adjustment. These runs impose a soft 50% terminal-SOC penalty inside MPC while tracking has no equivalent soft terminal objective and the shared hard terminal target is null. Cash-profit rankings therefore do not isolate the controller from its terminal-inventory tradeoff. An increase in profit alongside more imbalance is possible under signed settlement; it is not by itself evidence of a forecasting improvement or a controller bug.

The short, selected January/August windows and shared prices across farms do not support precise extreme-tail estimates or independent-farm significance claims. Broader deterministic testing should retain paired tracking baselines, begin with the configured October 2025 week, and then include the full May 2026 month; these windows require no absent ensemble archive.

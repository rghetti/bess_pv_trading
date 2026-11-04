# Continuous MPC validation

Source batch: `/tmp/bess_research_reviewed_v1/outputs/energy_research/reviewed_farm910_august_v1`.
Pinned manifest SHA-256: `d727d6e41aa160a1660cb577f23624ab4ab1fa62cf591f3465b78cf94b1f33e6`.
Batch status: **complete**; 4/4 jobs audited.

Differences below are target MPC minus its own continuous grid-tracking baseline. Positive profit is higher realized operating profit; positive imbalance is more absolute imbalance. These are retrospective development windows, not an untouched test set.

| Window | Farm | Controller case | Profit difference EUR | Imbalance difference MWh | Wear difference EUR | Final inventory difference MWh | Tracking fallbacks | Timeouts | Safety overrides | Rejected oracle refinements |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| august_development | 910 | adaptive_ecc | 271.803 | 26.960265 | 165.250 | 0.000000 | 0 | 0 | 4 | 0 |
| august_development | 910 | ensemble_oracle_gap_fallback | -62.935 | 7.501351 | -119.006 | 0.000000 | 0 | 40 | 0 | 0 |
| august_development | 910 | ensemble_penalty | 96.812 | 7.542783 | -163.559 | 0.000000 | 0 | 0 | 0 | 0 |
| august_development | 910 | ensemble_price_oracle | -62.935 | 7.501351 | -119.006 | 0.000000 | 0 | 40 | 0 | 0 |

## Matched controller totals

| Case | Completed jobs | Profit difference EUR | Imbalance difference MWh | MPC solves | Tracking fallbacks | Timeouts | Rejected refinements |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_ecc | 1 | 271.803 | 26.960265 | 192 | 0 | 0 | 0 |
| ensemble_oracle_gap_fallback | 1 | -62.935 | 7.501351 | 192 | 0 | 40 | 0 |
| ensemble_penalty | 1 | 96.812 | 7.542783 | 192 | 0 | 0 | 0 |
| ensemble_price_oracle | 1 | -62.935 | 7.501351 | 192 | 0 | 40 | 0 |

## Independent checks

The auditor found **0 physical/accounting/artifact violations** at a `1e-6` reconstruction tolerance. It checks SOC and power bounds, charge/discharge exclusivity, POI bounds, energy transition/continuity, PV balance, immutable ledger positions, single price publications, signed settlement, operating profit and recorded oracle imbalance caps.

| Reconstructed residual | Maximum |
|---|---:|
| quarter_hour_duration | 0 |
| energy_transition_mwh | 8.8817842e-16 |
| energy_continuity_mwh | 0 |
| grid_balance_mw | 1.3322676e-15 |
| commitment_mwh | 2.220446e-16 |
| auction_revenue_eur | 5.6843419e-14 |
| imbalance_mwh | 3.469447e-16 |
| settlement_eur | 3.8857806e-14 |
| profit_eur | 1.1368684e-13 |
| summary_profit_eur | 0 |
| summary_absolute_imbalance_mwh | 7.1054274e-15 |
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

Grid tracking uses actual interval PV feedback; these MPC runs use forecast current PV. Perfect-foresight auction prices are common across members. Oracle physical controllers additionally see realized settlement prices, but retain the configured minimum-imbalance budget; an oracle controller is not a guaranteed upper bound on realized profit under uncertain PV.

The continuous comparisons include changed battery states and consequent later auction submissions. Profit differences therefore include auction cash-flow changes as well as physical dispatch. Final stored energy can differ, and reported profit has no inventory-value adjustment. These runs impose a soft 50% terminal-SOC penalty inside MPC while tracking has no equivalent soft terminal objective and the shared hard terminal target is null. Cash-profit rankings therefore do not isolate the controller from its terminal-inventory tradeoff. An increase in profit alongside more imbalance is possible under signed settlement; it is not by itself evidence of a forecasting improvement or a controller bug.

The short, selected January/August windows and shared prices across farms do not support precise extreme-tail estimates or independent-farm significance claims. Broader deterministic testing should retain paired tracking baselines, begin with the configured October 2025 week, and then include the full May 2026 month; these windows require no absent ensemble archive.

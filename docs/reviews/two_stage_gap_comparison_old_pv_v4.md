# Continuous MPC validation

Source batch: `/tmp/bess_research_run_v4/outputs/energy_research/two_stage_gap_comparison_v4`.
Pinned manifest SHA-256: `233e51af8f5987411813bd7bfbad759ee79637105fa340a3eaa541c1159d84a0`.
Batch status: **complete**; 2/2 jobs audited.

Differences below are target MPC minus its own continuous grid-tracking baseline. Positive profit is higher realized operating profit; positive imbalance is more absolute imbalance. These are retrospective development windows, not an untouched test set.

| Window | Farm | Controller case | Profit difference EUR | Imbalance difference MWh | Wear difference EUR | Final inventory difference MWh | Tracking fallbacks | Timeouts | Safety overrides | Rejected oracle refinements |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| august_development | 1 | ensemble_oracle_gap_fallback | 72.375 | 1.071420 | -11.360 | 0.000000 | 0 | 40 | 0 | 2 |
| august_development | 1 | ensemble_price_oracle | 72.375 | 1.071420 | -11.360 | 0.000000 | 0 | 40 | 0 | 2 |

## Matched controller totals

| Case | Completed jobs | Profit difference EUR | Imbalance difference MWh | MPC solves | Tracking fallbacks | Timeouts | Rejected refinements |
|---|---:|---:|---:|---:|---:|---:|---:|
| ensemble_oracle_gap_fallback | 1 | 72.375 | 1.071420 | 192 | 0 | 40 | 2 |
| ensemble_price_oracle | 1 | 72.375 | 1.071420 | 192 | 0 | 40 | 2 |

## Independent checks

The auditor found **0 physical/accounting/artifact violations** at a `1e-6` reconstruction tolerance. It checks SOC and power bounds, charge/discharge exclusivity, POI bounds, energy transition/continuity, PV balance, immutable ledger positions, single price publications, signed settlement, operating profit and recorded oracle imbalance caps.

| Reconstructed residual | Maximum |
|---|---:|
| quarter_hour_duration | 0 |
| energy_transition_mwh | 1.110223e-16 |
| energy_continuity_mwh | 0 |
| grid_balance_mw | 1.6653345e-16 |
| commitment_mwh | 9.7144515e-17 |
| auction_revenue_eur | 7.1054274e-15 |
| imbalance_mwh | 1.7694179e-16 |
| settlement_eur | 3.4194869e-14 |
| profit_eur | 3.4638958e-14 |
| summary_profit_eur | 0 |
| summary_absolute_imbalance_mwh | 5.3290705e-15 |
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

Some oracle economic refinements were rejected by independent objective accounting checks. Their already validated minimum-imbalance stage-one plans were retained. These are distinct from grid-tracking fallbacks.

- `august_development__farm_1__ensemble_price_oracle`, `2026-08-02T18:30:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.00014472743977398837 > 0.00014442505863407101`. Tracking fallback: `False`.
- `august_development__farm_1__ensemble_price_oracle`, `2026-08-03T18:30:00+00:00`: `MPC physical validation failed: {'initial_energy': np.float64(0.0), 'energy_transition': np.float64(1.030812418686966e-07), 'energy_lower': np.float64(1.0309057443813607e-07), 'energy_upper': np.float64(-0.645), 'charge_discharge_lower': np.float64(-0.0), 'power': np.float64(-0.49999959931194243), 'exclusivity': np.float64(0.0), 'utilization_lower': np.float64(-1.0), 'utilization_upper': np.float64(0.0), 'poi_import': np.float64(-0.95328), 'poi_export': np.float64(-0.947342731915259), 'grid': np.float64(4.00688057587261e-07), 'imbalance': np.float64(1.0017201439681525e-07), 'expected_absolute': np.float64(1.0017201439681525e-07), 'expected_shortfall': np.float64(1.0017201439681525e-07)}`. Tracking fallback: `False`.
- `august_development__farm_1__ensemble_oracle_gap_fallback`, `2026-08-02T18:30:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.00014472743977398837 > 0.00014442505863407101`. Tracking fallback: `False`.
- `august_development__farm_1__ensemble_oracle_gap_fallback`, `2026-08-03T18:30:00+00:00`: `MPC physical validation failed: {'initial_energy': np.float64(0.0), 'energy_transition': np.float64(1.030812418686966e-07), 'energy_lower': np.float64(1.0309057443813607e-07), 'energy_upper': np.float64(-0.645), 'charge_discharge_lower': np.float64(-0.0), 'power': np.float64(-0.49999959931194243), 'exclusivity': np.float64(0.0), 'utilization_lower': np.float64(-1.0), 'utilization_upper': np.float64(0.0), 'poi_import': np.float64(-0.95328), 'poi_export': np.float64(-0.947342731915259), 'grid': np.float64(4.00688057587261e-07), 'imbalance': np.float64(1.0017201439681525e-07), 'expected_absolute': np.float64(1.0017201439681525e-07), 'expected_shortfall': np.float64(1.0017201439681525e-07)}`. Tracking fallback: `False`.

0 accepted oracle plans exceed their recorded reconstructed-MWh cap beyond the configured `2 * FeasibilityTol` allowance.


0 explicit oracle solves retain nested source metadata saying settlement was supplied only for ex-post accounting. This is a provenance-label contradiction; the top-level oracle disclosure is explicit.

## Interpretation

Grid tracking uses actual interval PV feedback; these MPC runs use forecast current PV. Perfect-foresight auction prices are common across members. Oracle physical controllers additionally see realized settlement prices, but retain the configured minimum-imbalance budget; an oracle controller is not a guaranteed upper bound on realized profit under uncertain PV.

The continuous comparisons include changed battery states and consequent later auction submissions. Profit differences therefore include auction cash-flow changes as well as physical dispatch. Final stored energy can differ, and reported profit has no inventory-value adjustment. These runs impose a soft 50% terminal-SOC penalty inside MPC while tracking has no equivalent soft terminal objective and the shared hard terminal target is null. Cash-profit rankings therefore do not isolate the controller from its terminal-inventory tradeoff. An increase in profit alongside more imbalance is possible under signed settlement; it is not by itself evidence of a forecasting improvement or a controller bug.

The short, selected January/August windows and shared prices across farms do not support precise extreme-tail estimates or independent-farm significance claims. Broader deterministic testing should retain paired tracking baselines, begin with the configured October 2025 week, and then include the full May 2026 month; these windows require no absent ensemble archive.

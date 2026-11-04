# Continuous MPC validation

Source batch: `/tmp/bess_research_legacy_validation_snapshot/outputs/energy_research/legacy_continuous_validation_v1`.
Pinned manifest SHA-256: `0083106877bbd3d9b899889f9344249a0d26f177e6f4194000f48c0336e40901`.
Batch status: **complete**; 32/32 jobs audited.

Differences below are target MPC minus its own continuous grid-tracking baseline. Positive profit is higher realized operating profit; positive imbalance is more absolute imbalance. These are retrospective development windows, not an untouched test set.

| Window | Farm | Controller case | Profit difference EUR | Imbalance difference MWh | Wear difference EUR | Final inventory difference MWh | Tracking fallbacks | Timeouts | Safety overrides | Rejected oracle refinements |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| august_development | 1 | deterministic_penalty | 95.853 | 1.052986 | -17.769 | 0.000000 | 0 | 0 | 0 | 0 |
| august_development | 1 | deterministic_price_oracle | 139.703 | 1.081377 | -27.472 | 0.000000 | 0 | 40 | 0 | 0 |
| august_development | 1 | ensemble_penalty | 93.595 | 1.117807 | -16.609 | 0.000000 | 0 | 0 | 0 | 0 |
| august_development | 1 | ensemble_price_oracle | 72.375 | 1.071420 | -11.360 | 0.000000 | 0 | 40 | 0 | 1 |
| august_development | 345 | deterministic_penalty | 1125.525 | 6.956439 | -35.105 | -0.081236 | 0 | 0 | 0 | 0 |
| august_development | 345 | deterministic_price_oracle | 1107.088 | 6.896966 | -32.607 | -0.081304 | 0 | 40 | 0 | 0 |
| august_development | 345 | ensemble_penalty | 1113.996 | 6.814998 | -39.652 | -0.081038 | 0 | 0 | 0 | 0 |
| august_development | 345 | ensemble_price_oracle | 1017.778 | 6.535206 | -12.636 | -0.081304 | 0 | 40 | 0 | 0 |
| august_development | 901 | deterministic_penalty | 1073.545 | 8.222661 | -51.946 | -0.094566 | 0 | 0 | 0 | 0 |
| august_development | 901 | deterministic_price_oracle | 1124.411 | 8.392434 | -54.239 | -0.094566 | 0 | 40 | 0 | 0 |
| august_development | 901 | ensemble_penalty | 1096.462 | 9.030919 | -67.194 | -0.094566 | 0 | 0 | 0 | 0 |
| august_development | 901 | ensemble_price_oracle | 1034.875 | 8.871597 | -41.833 | -0.094566 | 0 | 40 | 0 | 0 |
| august_development | 910 | deterministic_penalty | -30.076 | 2.388748 | 125.749 | 0.000000 | 0 | 0 | 0 | 0 |
| august_development | 910 | deterministic_price_oracle | -111.177 | 1.861309 | 193.799 | 0.000000 | 0 | 40 | 0 | 0 |
| august_development | 910 | ensemble_penalty | -197.331 | 1.592271 | 226.444 | 0.000000 | 0 | 0 | 0 | 0 |
| august_development | 910 | ensemble_price_oracle | -356.413 | 1.276932 | 271.035 | 0.000000 | 0 | 40 | 0 | 0 |
| january_development | 1 | deterministic_penalty | 25.476 | 0.412199 | -0.080 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 1 | deterministic_price_oracle | 27.507 | 0.387288 | -0.577 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 1 | ensemble_penalty | 17.930 | 0.471258 | 9.911 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 1 | ensemble_price_oracle | 17.664 | 0.530478 | 14.070 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 345 | deterministic_penalty | 16.405 | 1.199259 | -27.492 | -0.028491 | 0 | 0 | 0 | 0 |
| january_development | 345 | deterministic_price_oracle | 21.386 | 1.293429 | -27.492 | -0.028491 | 0 | 0 | 0 | 1 |
| january_development | 345 | ensemble_penalty | 14.347 | 1.237881 | -22.411 | -0.122134 | 0 | 0 | 0 | 0 |
| january_development | 345 | ensemble_price_oracle | 22.170 | 1.294819 | -14.866 | -0.122134 | 0 | 0 | 0 | 0 |
| january_development | 901 | deterministic_penalty | -7.485 | 1.653415 | 32.067 | 0.365256 | 0 | 0 | 0 | 0 |
| january_development | 901 | deterministic_price_oracle | 5.217 | 1.501558 | 32.069 | 0.365322 | 0 | 0 | 0 | 0 |
| january_development | 901 | ensemble_penalty | -68.024 | 2.818350 | 77.990 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 901 | ensemble_price_oracle | -140.679 | 3.270298 | 142.357 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 910 | deterministic_penalty | 89.268 | 2.441819 | 13.327 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 910 | deterministic_price_oracle | 164.312 | 2.390871 | 12.642 | 0.026137 | 0 | 0 | 0 | 0 |
| january_development | 910 | ensemble_penalty | 115.260 | 4.682327 | 120.526 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 910 | ensemble_price_oracle | 109.429 | 5.469021 | 165.739 | 0.000000 | 0 | 0 | 0 | 1 |

## Matched controller totals

| Case | Completed jobs | Profit difference EUR | Imbalance difference MWh | MPC solves | Tracking fallbacks | Timeouts | Rejected refinements |
|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic_penalty | 8 | 2388.511 | 24.327526 | 1536 | 0 | 0 | 0 |
| deterministic_price_oracle | 8 | 2478.447 | 23.805232 | 1536 | 0 | 160 | 1 |
| ensemble_penalty | 8 | 2186.234 | 27.765812 | 1536 | 0 | 0 | 0 |
| ensemble_price_oracle | 8 | 1777.200 | 28.319770 | 1536 | 0 | 160 | 2 |

## Independent checks

The auditor found **2 physical/accounting/artifact violations** at a `1e-6` reconstruction tolerance. It checks SOC and power bounds, charge/discharge exclusivity, POI bounds, energy transition/continuity, PV balance, immutable ledger positions, single price publications, signed settlement, operating profit and recorded oracle imbalance caps.

| Reconstructed residual | Maximum |
|---|---:|
| quarter_hour_duration | 0 |
| energy_transition_mwh | 1.3322676e-15 |
| energy_continuity_mwh | 0 |
| grid_balance_mw | 8.8817842e-16 |
| commitment_mwh | 2.220446e-16 |
| auction_revenue_eur | 5.6843419e-14 |
| imbalance_mwh | 3.3306691e-16 |
| settlement_eur | 3.9301895e-14 |
| profit_eur | 8.5265128e-14 |
| summary_profit_eur | 1.8189894e-12 |
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

Some oracle economic refinements were rejected by independent objective accounting checks. Their already validated minimum-imbalance stage-one plans were retained. These are distinct from grid-tracking fallbacks.

- `january_development__farm_345__deterministic_price_oracle`, `2026-01-15T15:45:00+00:00`: `MPC objective component validation failed: {'degradation_eur': 1.7763568394002505e-15, 'terminal_cost_eur': 0.0, 'import_cost_eur': 0.0, 'settlement_eur': 0.00021785916954485815}`. Tracking fallback: `False`.
- `january_development__farm_910__ensemble_price_oracle`, `2026-01-16T15:45:00+00:00`: `MPC objective component validation failed: {'degradation_eur': 0.0, 'terminal_cost_eur': 0.0, 'import_cost_eur': 0.0, 'settlement_eur': 7.948759604498035e-05}`. Tracking fallback: `False`.
- `august_development__farm_1__ensemble_price_oracle`, `2026-08-03T18:30:00+00:00`: `MPC physical validation failed: {'initial_energy': np.float64(0.0), 'energy_transition': np.float64(1.0308942833409107e-07), 'energy_lower': np.float64(1.0309057443813607e-07), 'energy_upper': np.float64(-0.645), 'charge_discharge_lower': np.float64(-0.0), 'power': np.float64(-0.49999959931194243), 'exclusivity': np.float64(0.0), 'utilization_lower': np.float64(-1.0), 'utilization_upper': np.float64(0.0), 'poi_import': np.float64(-0.95328), 'poi_export': np.float64(-0.947342731915259), 'grid': np.float64(4.00688057587282e-07), 'imbalance': np.float64(1.001720143968205e-07), 'expected_absolute': np.float64(1.0017201439682047e-07), 'expected_shortfall': np.float64(1.0017201439682047e-07)}`. Tracking fallback: `False`.

18 accepted oracle plans exceed their recorded reconstructed-MWh cap beyond the configured `2 * FeasibilityTol` allowance.

- `january_development__farm_1__ensemble_price_oracle`, `2026-01-16T15:30:00+00:00`: cap `0.00118359090909` MWh; reconstructed `0.00118474306933` MWh; excess `1.1521602e-06` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T15:45:00+00:00`: cap `0.00305320018912` MWh; reconstructed `0.00305323561527` MWh; excess `3.5426142e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T16:00:00+00:00`: cap `0.000563018825325` MWh; reconstructed `0.00056305425147` MWh; excess `3.5426145e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T16:15:00+00:00`: cap `0.000563018825325` MWh; reconstructed `0.000563054228821` MWh; excess `3.5403496e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T16:30:00+00:00`: cap `0.000563018825325` MWh; reconstructed `0.00056305425147` MWh; excess `3.5426145e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T16:45:00+00:00`: cap `0.000563018825325` MWh; reconstructed `0.000563055241899` MWh; excess `3.6416574e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T17:00:00+00:00`: cap `0.000563018825325` MWh; reconstructed `0.00056305425147` MWh; excess `3.5426145e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T17:15:00+00:00`: cap `0.000563018825325` MWh; reconstructed `0.000563054260268` MWh; excess `3.5434942e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T17:30:00+00:00`: cap `0.000563018825325` MWh; reconstructed `0.00056305428674` MWh; excess `3.5461415e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T17:45:00+00:00`: cap `0.000563015343157` MWh; reconstructed `0.000563054258908` MWh; excess `3.8915751e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T18:00:00+00:00`: cap `0.000156972574328` MWh; reconstructed `0.000157011492444` MWh; excess `3.8918116e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T18:30:00+00:00`: cap `7.87310757486e-05` MWh; reconstructed `7.87698181622e-05` MWh; excess `3.8742414e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T18:45:00+00:00`: cap `3.94272716952e-05` MWh; reconstructed `3.94589597416e-05` MWh; excess `3.1688046e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T17:00:00+00:00`: cap `0.00119119646864` MWh; reconstructed `0.00119122629495` MWh; excess `2.9826309e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T17:45:00+00:00`: cap `0.000296107278521` MWh; reconstructed `0.000296134233343` MWh; excess `2.6954822e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T18:00:00+00:00`: cap `0.000102432646477` MWh; reconstructed `0.000102474249056` MWh; excess `4.1602579e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T18:15:00+00:00`: cap `7.69002596313e-05` MWh; reconstructed `7.69268179318e-05` MWh; excess `2.65583e-08` MWh; allowance `2e-08` MWh.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T18:30:00+00:00`: cap `5.12636562632e-05` MWh; reconstructed `5.12886914923e-05` MWh; excess `2.5035229e-08` MWh; allowance `2e-08` MWh.

3072 explicit oracle solves retain nested source metadata saying settlement was supplied only for ex-post accounting. This is a provenance-label contradiction; the top-level oracle disclosure is explicit.

## Interpretation

Grid tracking uses actual interval PV feedback; these MPC runs use forecast current PV. Perfect-foresight auction prices are common across members. Oracle physical controllers additionally see realized settlement prices, but retain the configured minimum-imbalance budget; an oracle controller is not a guaranteed upper bound on realized profit under uncertain PV.

The continuous comparisons include changed battery states and consequent later auction submissions. Profit differences therefore include auction cash-flow changes as well as physical dispatch. Final stored energy can differ, and reported profit has no inventory-value adjustment. These runs impose a soft 50% terminal-SOC penalty inside MPC while tracking has no equivalent soft terminal objective and the shared hard terminal target is null. Cash-profit rankings therefore do not isolate the controller from its terminal-inventory tradeoff. An increase in profit alongside more imbalance is possible under signed settlement; it is not by itself evidence of a forecasting improvement or a controller bug.

The short, selected January/August windows and shared prices across farms do not support precise extreme-tail estimates or independent-farm significance claims. Broader deterministic testing should retain paired tracking baselines, begin with the configured October 2025 week, and then include the full May 2026 month; these windows require no absent ensemble archive.

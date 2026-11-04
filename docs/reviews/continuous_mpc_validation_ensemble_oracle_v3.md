# Continuous MPC validation

Source batch: `/tmp/bess_research_run_v3/outputs/energy_research/ensemble_oracle_corrected_v3`.
Pinned manifest SHA-256: `6371c10ffebbd00ee424a2f1da6da1f6f48792ec9486d4298f57b7ec71b0a6f9`.
Batch status: **complete**; 8/8 jobs audited.

Differences below are target MPC minus its own continuous grid-tracking baseline. Positive profit is higher realized operating profit; positive imbalance is more absolute imbalance. These are retrospective development windows, not an untouched test set.

| Window | Farm | Controller case | Profit difference EUR | Imbalance difference MWh | Wear difference EUR | Final inventory difference MWh | Tracking fallbacks | Timeouts | Safety overrides | Rejected oracle refinements |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| august_development | 1 | ensemble_price_oracle | 72.375 | 1.071420 | -11.360 | 0.000000 | 0 | 40 | 0 | 2 |
| august_development | 345 | ensemble_price_oracle | 1017.778 | 6.535206 | -12.636 | -0.081304 | 0 | 40 | 0 | 19 |
| august_development | 901 | ensemble_price_oracle | 1034.875 | 8.871597 | -41.833 | -0.094566 | 0 | 40 | 0 | 0 |
| august_development | 910 | ensemble_price_oracle | -356.413 | 1.276932 | 271.035 | 0.000000 | 0 | 40 | 0 | 0 |
| january_development | 1 | ensemble_price_oracle | 17.664 | 0.530478 | 14.070 | 0.000000 | 0 | 0 | 0 | 7 |
| january_development | 345 | ensemble_price_oracle | 22.170 | 1.294819 | -14.866 | -0.122134 | 0 | 0 | 0 | 0 |
| january_development | 901 | ensemble_price_oracle | -140.679 | 3.270298 | 142.357 | 0.000000 | 0 | 0 | 0 | 0 |
| january_development | 910 | ensemble_price_oracle | 109.429 | 5.469020 | 165.739 | 0.000000 | 0 | 0 | 0 | 1 |

## Matched controller totals

| Case | Completed jobs | Profit difference EUR | Imbalance difference MWh | MPC solves | Tracking fallbacks | Timeouts | Rejected refinements |
|---|---:|---:|---:|---:|---:|---:|---:|
| ensemble_price_oracle | 8 | 1777.199 | 28.319770 | 1536 | 0 | 160 | 29 |

## Independent checks

The auditor found **0 physical/accounting/artifact violations** at a `1e-6` reconstruction tolerance. It checks SOC and power bounds, charge/discharge exclusivity, POI bounds, energy transition/continuity, PV balance, immutable ledger positions, single price publications, signed settlement, operating profit and recorded oracle imbalance caps.

| Reconstructed residual | Maximum |
|---|---:|
| quarter_hour_duration | 0 |
| energy_transition_mwh | 8.8817842e-16 |
| energy_continuity_mwh | 0 |
| grid_balance_mw | 8.8817842e-16 |
| commitment_mwh | 2.220446e-16 |
| auction_revenue_eur | 5.6843419e-14 |
| imbalance_mwh | 2.7755576e-16 |
| settlement_eur | 3.7303494e-14 |
| profit_eur | 5.6843419e-14 |
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

- `january_development__farm_910__ensemble_price_oracle`, `2026-01-16T16:00:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.00015931752495436003 > 0.0001573679346207573`. Tracking fallback: `False`.
- `january_development__farm_1__ensemble_price_oracle`, `2026-01-16T02:30:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 7.232645552159489 > 7.232645497651272`. Tracking fallback: `False`.
- `january_development__farm_1__ensemble_price_oracle`, `2026-01-16T09:00:00+00:00`: `MPC physical validation failed: {'initial_energy': np.float64(0.0), 'energy_transition': np.float64(5.551115123125783e-17), 'energy_lower': np.float64(0.0), 'energy_upper': np.float64(-0.36412385852548945), 'charge_discharge_lower': np.float64(-0.0), 'power': np.float64(0.0), 'exclusivity': np.float64(0.0), 'utilization_lower': np.float64(-0.0), 'utilization_upper': np.float64(0.0), 'poi_import': np.float64(-0.931723822623169), 'poi_export': np.float64(-0.45326201685617695), 'grid': np.float64(8.564808886418751e-07), 'imbalance': np.float64(2.1412022216046878e-07), 'expected_absolute': np.float64(4.1984357284155505e-09), 'expected_shortfall': np.float64(3.124390845436059e-09)}`. Tracking fallback: `False`.
- `january_development__farm_1__ensemble_price_oracle`, `2026-01-16T09:30:00+00:00`: `MPC physical validation failed: {'initial_energy': np.float64(0.0), 'energy_transition': np.float64(5.551115123125783e-17), 'energy_lower': np.float64(0.0), 'energy_upper': np.float64(-0.36412385852548945), 'charge_discharge_lower': np.float64(-0.0), 'power': np.float64(0.0), 'exclusivity': np.float64(0.0), 'utilization_lower': np.float64(-0.0), 'utilization_upper': np.float64(0.0), 'poi_import': np.float64(-0.931723822623169), 'poi_export': np.float64(-0.45326201685617695), 'grid': np.float64(8.564808886418751e-07), 'imbalance': np.float64(2.1412022216046878e-07), 'expected_absolute': np.float64(4.1984357284155505e-09), 'expected_shortfall': np.float64(3.1243912192689682e-09)}`. Tracking fallback: `False`.
- `january_development__farm_1__ensemble_price_oracle`, `2026-01-16T09:45:00+00:00`: `MPC physical validation failed: {'initial_energy': np.float64(0.0), 'energy_transition': np.float64(6.245004513516506e-17), 'energy_lower': np.float64(0.0), 'energy_upper': np.float64(-0.3641238585254895), 'charge_discharge_lower': np.float64(-0.0), 'power': np.float64(0.0), 'exclusivity': np.float64(0.0), 'utilization_lower': np.float64(-0.0), 'utilization_upper': np.float64(0.0), 'poi_import': np.float64(-0.931723822623169), 'poi_export': np.float64(-0.45326201685617695), 'grid': np.float64(8.564808886418751e-07), 'imbalance': np.float64(2.1412022216046878e-07), 'expected_absolute': np.float64(4.1984357284155505e-09), 'expected_shortfall': np.float64(3.1243912075595848e-09)}`. Tracking fallback: `False`.
- `january_development__farm_1__ensemble_price_oracle`, `2026-01-16T10:00:00+00:00`: `MPC physical validation failed: {'initial_energy': np.float64(0.0), 'energy_transition': np.float64(6.245004513516506e-17), 'energy_lower': np.float64(0.0), 'energy_upper': np.float64(-0.3641238585254895), 'charge_discharge_lower': np.float64(-0.0), 'power': np.float64(0.0), 'exclusivity': np.float64(0.0), 'utilization_lower': np.float64(-0.0), 'utilization_upper': np.float64(0.0), 'poi_import': np.float64(-0.931723822623169), 'poi_export': np.float64(-0.45326201685617695), 'grid': np.float64(8.564808886418751e-07), 'imbalance': np.float64(2.1412022216046878e-07), 'expected_absolute': np.float64(4.1984357284155505e-09), 'expected_shortfall': np.float64(3.124391057506004e-09)}`. Tracking fallback: `False`.
- `january_development__farm_1__ensemble_price_oracle`, `2026-01-16T10:45:00+00:00`: `MPC physical validation failed: {'initial_energy': np.float64(0.0), 'energy_transition': np.float64(6.245004513516506e-17), 'energy_lower': np.float64(0.0), 'energy_upper': np.float64(-0.3641238585254895), 'charge_discharge_lower': np.float64(-0.0), 'power': np.float64(0.0), 'exclusivity': np.float64(0.0), 'utilization_lower': np.float64(-0.0), 'utilization_upper': np.float64(0.0), 'poi_import': np.float64(-0.931723822623169), 'poi_export': np.float64(-0.45326201685617695), 'grid': np.float64(8.564808886418751e-07), 'imbalance': np.float64(2.1412022216046878e-07), 'expected_absolute': np.float64(4.1984357284155505e-09), 'expected_shortfall': np.float64(3.124390559423526e-09)}`. Tracking fallback: `False`.
- `january_development__farm_1__ensemble_price_oracle`, `2026-01-16T11:00:00+00:00`: `MPC physical validation failed: {'initial_energy': np.float64(0.0), 'energy_transition': np.float64(5.551115123125783e-17), 'energy_lower': np.float64(0.0), 'energy_upper': np.float64(-0.36412385852548945), 'charge_discharge_lower': np.float64(-0.0), 'power': np.float64(0.0), 'exclusivity': np.float64(0.0), 'utilization_lower': np.float64(-0.0), 'utilization_upper': np.float64(0.0), 'poi_import': np.float64(-0.931723822623169), 'poi_export': np.float64(-0.45326201685617695), 'grid': np.float64(8.564808886418751e-07), 'imbalance': np.float64(2.1412022216046878e-07), 'expected_absolute': np.float64(4.1984357284155505e-09), 'expected_shortfall': np.float64(3.1243908022848127e-09)}`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T12:00:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.04090176137039289 > 0.04090173240666636`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T15:45:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.003053235623068317 > 0.003053200189123562`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T16:00:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0005630542602677925 > 0.0005630188253254581`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T16:15:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0005630558885440329 > 0.0005630188253254892`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T16:30:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0005630542592727008 > 0.0005630188253254653`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T16:45:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0005630547708371311 > 0.000563018825325441`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T17:00:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0005630542602678066 > 0.0005630188253253887`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T17:15:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0005630542602677902 > 0.0005630188253253855`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T17:30:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0005630542728761952 > 0.0005630188253253681`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T17:45:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0005630593657136808 > 0.0005630188253253681`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T18:00:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.00015701660161884112 > 0.00015697606122903846`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-02T18:45:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 3.9465526705456426e-05 > 3.944381467471862e-05`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T16:30:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0032891999744820944 > 0.0032891366779200064`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T17:15:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0008511202434787313 > 0.0008510951466709611`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T17:30:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0005488873076274157 > 0.000548863123912845`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T17:45:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0002962229021833137 > 0.0002961987184686729`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T18:00:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.0001024929724539745 > 0.00010244952064994405`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T18:15:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 7.688974665844155e-05 > 7.685439178494287e-05`. Tracking fallback: `False`.
- `august_development__farm_345__ensemble_price_oracle`, `2026-08-03T18:30:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 5.124520156286885e-05 > 5.1209846689391284e-05`. Tracking fallback: `False`.
- `august_development__farm_1__ensemble_price_oracle`, `2026-08-02T18:30:00+00:00`: `Oracle refinement exceeds reconstructed imbalance cap: 0.00014472743977398837 > 0.00014442505863407101`. Tracking fallback: `False`.
- `august_development__farm_1__ensemble_price_oracle`, `2026-08-03T18:30:00+00:00`: `MPC physical validation failed: {'initial_energy': np.float64(0.0), 'energy_transition': np.float64(1.030812418686966e-07), 'energy_lower': np.float64(1.0309057443813607e-07), 'energy_upper': np.float64(-0.645), 'charge_discharge_lower': np.float64(-0.0), 'power': np.float64(-0.49999959931194243), 'exclusivity': np.float64(0.0), 'utilization_lower': np.float64(-1.0), 'utilization_upper': np.float64(0.0), 'poi_import': np.float64(-0.95328), 'poi_export': np.float64(-0.947342731915259), 'grid': np.float64(4.00688057587261e-07), 'imbalance': np.float64(1.0017201439681525e-07), 'expected_absolute': np.float64(1.0017201439681525e-07), 'expected_shortfall': np.float64(1.0017201439681525e-07)}`. Tracking fallback: `False`.

0 accepted oracle plans exceed their recorded reconstructed-MWh cap beyond the configured `2 * FeasibilityTol` allowance.


0 explicit oracle solves retain nested source metadata saying settlement was supplied only for ex-post accounting. This is a provenance-label contradiction; the top-level oracle disclosure is explicit.

## Interpretation

Grid tracking uses actual interval PV feedback; these MPC runs use forecast current PV. Perfect-foresight auction prices are common across members. Oracle physical controllers additionally see realized settlement prices, but retain the configured minimum-imbalance budget; an oracle controller is not a guaranteed upper bound on realized profit under uncertain PV.

The continuous comparisons include changed battery states and consequent later auction submissions. Profit differences therefore include auction cash-flow changes as well as physical dispatch. Final stored energy can differ, and reported profit has no inventory-value adjustment. These runs impose a soft 50% terminal-SOC penalty inside MPC while tracking has no equivalent soft terminal objective and the shared hard terminal target is null. Cash-profit rankings therefore do not isolate the controller from its terminal-inventory tradeoff. An increase in profit alongside more imbalance is possible under signed settlement; it is not by itself evidence of a forecasting improvement or a controller bug.

The short, selected January/August windows and shared prices across farms do not support precise extreme-tail estimates or independent-farm significance claims. Broader deterministic testing should retain paired tracking baselines, begin with the configured October 2025 week, and then include the full May 2026 month; these windows require no absent ensemble archive.

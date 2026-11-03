# Farm 1 real-time MPC smoke results: 15–16 January 2026

Both requested revised comparisons completed. All MPCs used current-interval
realized AC PV as a fixed exogenous input and the selected `direct_tcn_02` bundle
for future deterministic/ensemble weather conversion. Future realized PV remained
hidden. The current-PV disclosure and model's January–April 2026 selection / October
2026 artifact availability make these retrospective comparisons.

Configuration: 0.75 MWh / 0.5 MW BESS, 90% supplied initial SOC, 7–93% bounds,
sqrt(0.90) directional efficiencies, shared throughput degradation, 500 EUR/MWh
planning imbalance weight, soft 50% ordinary-midnight inventory target at 50
EUR/MWh deviation, final-study hard 50% target, two CPU threads, no training or
GPUs. Both lower price-oracle variants used zero additional expected-imbalance
budget. All upper auction optimizers retained actual-price foresight.

## Frozen commitments: primary controller comparison

| Controller | Absolute imbalance MWh | Signed settlement EUR | Operating profit EUR | Profit vs tracking EUR | Final SOC % | Mean Gurobi seconds |
|---|---:|---:|---:|---:|---:|---:|
| track_committed_grid | 0.217316 | -44.78 | 538.25 | +0.00 | 50.677 | — |
| deterministic_mpc | 0.236596 | -33.51 | 550.53 | +12.28 | 50.000 | 0.0047 |
| deterministic_mpc_price_oracle | 0.388672 | -42.70 | 541.99 | +3.74 | 50.000 | 0.0326 |
| ensemble_mpc | 0.258667 | -28.43 | 554.73 | +16.48 | 50.000 | 0.0756 |
| ensemble_mpc_price_oracle | 0.384083 | -42.12 | 541.41 | +3.16 | 50.000 | 0.2307 |

All submitted quantities, auction revenues and trading fees reconciled across
controllers. The original common TCN auction ledger was reused after correcting
the retrospective-artifact timestamp validator; source diagnostics were preserved.

Deterministic MPC still had 8.87% more absolute imbalance than tracking across the
two dates, alongside 12.28 EUR more operating profit. Their 15 January absolute
volumes were equal at 0.217316 MWh; deterministic MPC added 0.019280 MWh on 16
January. The price-oracle variants earned less than their corresponding price-blind
MPCs in this sample. The oracle safeguard constrains forecast-scenario expected
volume at each solve; future realized PV and subsequent state paths still affect
the completed-run result. No parameters were tuned to obtain a preferred ranking.

All MPCs ended at 50% SOC; tracking ended at 50.677%, leaving MPC with 0.005078 MWh
less inventory. Reported profit excludes terminal salvage or artificial planning
penalties. These two dates provide a software smoke test rather than strategy
validation or an independent forecast-selection holdout.

## Continuous all-auction comparison: secondary

| Controller | Absolute imbalance MWh | Signed settlement EUR | Operating profit EUR | Profit vs tracking EUR | Final SOC % | Mean Gurobi seconds |
|---|---:|---:|---:|---:|---:|---:|
| track_committed_grid | 0.217316 | -44.78 | 538.25 | +0.00 | 50.677 | — |
| deterministic_mpc | 0.361619 | -51.29 | 556.16 | +17.91 | 50.000 | 0.0046 |

The continuous MPC gain of 17.91 EUR includes 24.30 EUR more auction revenue from
changed subsequent submissions. It therefore measures the complete coupled policy.
It must be interpreted separately from the frozen-ledger controller comparison.
Actual stored energy carried across midnight without resetting to projected SOC.
The three next-day midnight predictions matched realized energy on this sample;
the two IDA3-noon errors were approximately -0.055157 and -0.000773 MWh. Zero
midnight errors here do not establish general projection accuracy: battery-power
rollouts can be independent of forecast PV until protection changes an action.

## Verification and artifacts

The revised smoke executed 960 MPC decisions, with no MPC fallbacks or timeouts.
Current-step predicted versus executed absolute imbalance agreed within numerical
precision (maximum below 1e-15 MWh). Publication clocks, SOC carryover, per-solve
oracle budgets and financial reconciliation passed the saved audit.

Validation: 143 affected regression tests, 120 focused optimization/degradation
tests, 45 forecast/solver/reporting tests and 12 final plot/launcher/frozen-replay
tests passed. These suites overlap; their counts must not be added as unique
tests. Solver checks ran with the host Gurobi license and did not rely on skips.
The mathematical report PDF rebuilt successfully, and `git diff --check` passed.
A final 17-test input/launcher check passed after adding forecast-scan progress
messages; this reporting change does not alter control decisions.

The five MPC trajectories used 66.86 seconds of Gurobi runtime in total.
The recorded replay wall times were 15.72 seconds for frozen deterministic MPC,
22.50 for deterministic price-oracle MPC, 94.69 for ensemble MPC and 136.95 for
ensemble price-oracle MPC; continuous deterministic MPC took 17.24 seconds.
These replay timings exclude shared data preparation and final reporting, so
they are not end-to-end experiment timings. Mean per-decision solver times are
shown in the tables above.

Raw interval, full-plan, first-action, scenario, solver, provenance and plot files
remain under `outputs/realtime_mpc/farm_1/farm1_2026_01_15_16_mpc_tcn_current_pv/`.
Those outputs are git-ignored; the tables and validation audit beside this README
retain compact numerical evidence. Run manifests record the working-tree source
checksums and data/model evidence. Final reporting-only plot corrections include
the closing midnight boundary and the three-day detail cap.

The earlier pvlib/forecast-only run remains under its original run ID and its
comparison CSV is retained here. Its commitment ledger differs, so its totals
are not an isolated causal comparison of the forecasting or information changes.

The full January 2026 study is planned in
`docs/realtime_mpc_january_2026_study_plan.md`; it has not been executed. The new
comparison launcher defaults to plan-only and requires `--run` for execution.

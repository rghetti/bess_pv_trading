# First-interval MPC priority: one-week comparison

15–21 January 2026; four farms; continuous closed-loop cases. All use planned IDA3 and disable recovery.

| Farm | Tracker imbalance MWh | Original MPC MWh | Weight 10 MWh | Strict first MWh | Original profit EUR | Weight 10 profit EUR | Strict first profit EUR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4344 | 0.6425 | 0.5688 | 0.5567 | 2499.14 | 2488.47 | 2489.13 |
| 345 | 1.0308 | 1.2403 | 1.1599 | 1.1599 | 5299.99 | 5285.45 | 5287.31 |
| 901 | 3.6073 | 4.1820 | 3.7690 | 3.7711 | 11966.81 | 11996.16 | 11995.87 |
| 910 | 5.8599 | 8.5804 | 7.0147 | 7.1084 | 16723.98 | 16719.96 | 16714.65 |

Cash profit excludes planning penalties and terminal values; final inventory must be compared. Later auction orders and battery states can differ. This is a retrospective seven-day development sample with old PV models, disclosed current interval-average PV and actual auction-price curves.

The per-case daily and whole-run CSVs and `first_priority_objective_audit.json` retain the detailed results. The independent physical and accounting audit passed.

## Pooled outcomes

| Case | Absolute imbalance MWh | Signed settlement EUR | Auction revenue EUR | Wear EUR | Profit EUR | Curtailment MWh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Grid tracking | 10.9325 | -1695.02 | 39852.40 | 1522.19 | 36635.19 | 0.7426 |
| original | 14.6453 | -2337.39 | 40246.93 | 1419.62 | 36489.92 | 1.5398 |
| weighted_k10 | 12.5124 | -1828.35 | 39861.59 | 1543.20 | 36490.05 | 1.4718 |
| strict_first | 12.5961 | -1847.07 | 39878.38 | 1544.37 | 36486.95 | 1.4554 |

## Paired MPC differences against original

| Farm | Case | Imbalance change MWh | Settlement change EUR | Auction revenue change EUR | Wear change EUR | Profit change EUR | Final SOC change % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | strict_first | -0.0858 | +14.00 | -15.74 | +8.27 | -10.01 | +0.000 |
| 1 | weighted_k10 | -0.0737 | +11.26 | -13.65 | +8.28 | -10.67 | +0.000 |
| 345 | strict_first | -0.0804 | +12.00 | -18.32 | +6.37 | -12.68 | +0.000 |
| 345 | weighted_k10 | -0.0804 | +12.01 | -21.26 | +5.28 | -14.53 | +0.000 |
| 901 | strict_first | -0.4110 | +111.38 | -65.12 | +17.21 | +29.06 | -0.000 |
| 901 | weighted_k10 | -0.4130 | +111.35 | -64.88 | +17.12 | +29.35 | -0.000 |
| 910 | strict_first | -1.4721 | +352.93 | -269.37 | +92.89 | -9.33 | +0.000 |
| 910 | weighted_k10 | -1.5658 | +374.42 | -285.55 | +92.89 | -4.02 | +0.000 |

Cash profit is actual auction revenue and signed settlement less executed wear, imports and fees. It excludes planning penalties and terminal values. Final SOC is shown in the source table and CSV; cash profit does not value leftover battery inventory. Continuous trajectories can change later auction submissions, so these differences include both dispatch and later trading effects.

## First-stage and solver checks

| Case | MPC decisions | First-stage certified | Second stage used | Refinements rejected | Stage 1 only | Tracking fallbacks | Time-limited solves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 2688 | 0 | 0 | 0 | 0 | 0 | 0 |
| weighted_k10 | 2688 | 0 | 0 | 0 | 0 | 0 | 0 |
| strict_first | 2688 | 2688 | 2617 | 71 | 0 | 0 | 0 |

The first-stage minimum in strict-first is certified only within recorded solver gap and numerical tolerances. The independent objective check reconstructed first-interval weights and strict caps from saved diagnostics. All 71 refinements were rejected by independent physical validation at the configured numerical tolerance. A rejected refinement retains its validated first-stage action and does not use the intended second-stage horizon solution in that decision. The separate independent physical/financial audit checked all jobs and reported no violations; tracker trajectories matched across the three cases for each farm.

## Provenance and limitations

This is a seven-day retrospective development comparison with older reviewed-GHI models (BiGRU 1, Transformer 345, BiLSTM 901 and TCN 910), not an annual result or model selection test. Current interval-average PV was disclosed to the MPC; future PV remained forecast-based. Auction planning used actual auction-price curves, and settlement used actual NORD prices. Provisional IDA3 targets were included; curtailment recovery was disabled to isolate objective changes. The modeled first-stage cap and Gurobi limits do not become realized charges.

| Farm | Realized target provenance across 672 intervals |
| --- | --- |
| 1 | metered_ac: 329, monthly_night_zero: 343 |
| 345 | metered_ac: 329, monthly_night_zero: 343 |
| 901 | metered_ac: 329, monthly_night_zero: 343 |
| 910 | metered_ac: 329, monthly_night_zero: 343 |

Batch elapsed time: 652.3 seconds; summed MPC Gurobi runtime: 52.34 seconds. Two CPU workers and no GPU training were used. The licensed regression suite passed 104 tests; the objective-report test passed; all 12 input preflights passed.

Manifest SHA-256: `e9177633a65cf48df58bea65b7ccf90b559fbd34fc3be0e2aba66f6967a169d0`. Full plans, actions, cash flows, solver status, daily plots, paired CSVs, exact resolved configs and forecast provenance are in `outputs/energy_research/first_interval_priority_jan15_21_2026_v1/`. See `first_interval_priority_one_week_audit.md` for the independent audit.

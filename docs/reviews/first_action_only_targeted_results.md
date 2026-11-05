# Shared-first-action ensemble MPC: targeted results

Four farms, 15–17 January 2026; sixteen closed-loop jobs, 4,608 MPC decisions and 51 PV trajectories per decision.

| Case | Absolute imbalance MWh | Signed settlement EUR | Auction revenue EUR | Degradation EUR | Realized profit EUR |
| --- | ---: | ---: | ---: | ---: | ---: |
| common_costs | 2.798763 | -350.17 | 11507.25 | 514.81 | 10642.27 |
| first_action_costs | 2.895365 | -386.73 | 11560.13 | 519.11 | 10654.30 |
| common_imbalance_only | 2.848278 | -360.32 | 11403.59 | 479.68 | 10563.60 |
| first_action_imbalance_only | 3.136017 | -431.91 | 11250.87 | 485.13 | 10333.83 |

The common tracking baseline had 1.859583 MWh of absolute imbalance. Its per-interval physical and financial trajectories matched across all four case configurations.

| Farm | Tracking MWh | Common, costs MWh | Shared first, costs MWh | Common, imbalance only MWh | Shared first, imbalance only MWh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.196489 | 0.229576 | 0.279895 | 0.234806 | 0.269016 |
| 345 | 0.013423 | 0.063223 | 0.113488 | 0.066311 | 0.161601 |
| 901 | 1.134779 | 1.128280 | 1.113795 | 1.128280 | 1.142170 |
| 910 | 0.514891 | 1.377684 | 1.388188 | 1.418881 | 1.563230 |

Profit attribution compares each shared-first controller with the common-schedule controller using the same objective. A positive wear-cost change reduces profit; signed settlement may move either way as the timing and sign of deviations change.

| Farm | Objective | Imbalance change MWh | Settlement change EUR | Auction revenue change EUR | Wear-cost change EUR | Profit change EUR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | costs | +0.050319 | -3.18 | +2.40 | -2.66 | +1.88 |
| 1 | imbalance only | +0.034211 | -4.38 | -1.10 | -2.74 | -2.74 |
| 345 | costs | +0.050265 | -6.21 | +47.55 | +7.46 | +33.88 |
| 345 | imbalance only | +0.095290 | -11.80 | -64.60 | +19.38 | -95.79 |
| 901 | costs | -0.014485 | +2.88 | -5.18 | +0.85 | -3.15 |
| 901 | imbalance only | +0.013890 | -0.96 | -2.17 | +2.41 | -5.54 |
| 910 | costs | +0.010504 | -30.06 | +8.11 | -1.36 | -20.59 |
| 910 | imbalance only | +0.144349 | -54.45 | -84.85 | -13.60 | -125.69 |

## Solver and audit

Batch wall time 1120.2 seconds. Fallbacks 0; time-limited solves 0; physical safety overrides 0; cumulative MPC Gurobi runtime 799.47 seconds. Build-time-limit failures: 0.

Independent audit: 0 physical/accounting/artifact violations; 2304 shared-first scenario horizons audited. The deterministic member remained the stored auction shadow path; every checked first interval had zero provisional order. Per-farm physical settings and input provenance matched across the four cases after excluding the intentionally different dispatch policy. The complete input fingerprints differ between common and adaptive dispatch because the policy is included in the fingerprint.

## Interpretation and provenance

The shared-first models treat each forecast PV path as revealed after the first command for planning. That relaxes future nonanticipativity and may make continuation values optimistic. Only the current shared action executes, followed by feedback and replanning. Raw 50 weather members plus the updated deterministic member use uniform 1/51 weights; the weights are not calibrated probabilities and do not establish out-of-sample constraint guarantees.

The pure objective minimizes expected absolute imbalance MWh and includes no wear, import, terminal preference or EUR 500 multiplier. Executed wear, imports, signed settlement and fees remain in realized profit. The cost objective includes the configured wear and soft terminal preference. Hard physical constraints and the final-study SOC target apply in every case and scenario. MPC uses the provisional IDA2 plan for IDA3 until submission.

Auction decisions reoptimize from each controller’s own state and use the existing actual-price oracle; therefore profit and imbalance include changes in later submitted quantities. The current-interval average PV is disclosed to the MPC retrospectively, and future actual PV remains hidden. The existing selected reviewed-GHI models were trained with historical GHI and evaluated on preceding-day forecasts; this run applies updating eligible weather cycles. Reviewed targets can include explicitly flagged counterfactual synthetic and historical-GHI fills. For these dates each farm used 141 metered intervals and 147 configured night-zero intervals, with no synthetic replacement intervals. The three-day window does not establish annual performance or significance. No training, calibration or GPU work was performed.

The shared-root scenario structure is a known two-stage stochastic MPC formulation; [van der Meer, Wang and Munkhammar (2021)](https://www.diva-portal.org/smash/get/diva2%3A1502295/FULLTEXT01.pdf) and [Groß (2023), section 2.3](https://publications.syscop.de/Gross2023a.pdf) provide closely related PV-and-battery examples.

## Artifacts and checks

Configuration: `configs/research/first_action_only_targeted.yaml`. Raw reports, paired daily plots, full plans/scenarios, forecast provenance and solution diagnostics are under `outputs/energy_research/first_action_only_jan15_17_2026_v1/`. A repeat requires a new run ID. The focused licensed MPC/integration/audit suite passed 104 tests; updated subsets of 32, 12, 5 and 4 tests also passed with overlapping coverage. All sixteen data preflights passed.

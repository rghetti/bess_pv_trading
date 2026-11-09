# Results: deterministic six-worker K=25 batch

**Status:** Complete; all six distinct controller trajectories completed without batch failures.

**Period:** 1 January–29 August 2026 inclusive, 241 Italian local delivery days.

**Batch:** `outputs/energy_research/deterministic_six_workers_v2`.

This is the **earlier soft-terminal-SOC formulation**, not the newer hard-daily-SOC run. Each farm has one grid tracker, an equal-weight deterministic MPC and a K=25 deterministic MPC. The tracker result is shared by the two MPC comparisons and is reported once here.

## Physical results

PV production is the curated replay input, which includes synthetic counterfactual values. Both percentages use total replay PV production as their denominator. Absolute imbalance sums the magnitude of each interval's deviation; signed imbalance retains the direction.

| Farm | Controller | PV production (MWh) | Absolute imbalance (MWh) | Imbalance / PV | Signed imbalance (MWh) | PV curtailment (MWh) | Curtailment / PV |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Grid tracker | 677.35 | 55.37 | 8.17% | -53.41 | 21.68 | 3.20% |
| 1 | MPC equal weights | 677.35 | 56.38 | 8.32% | -56.38 | 26.26 | 3.88% |
| 1 | MPC K=25 | 677.35 | 54.92 | 8.11% | -54.92 | 25.31 | 3.74% |
| 345 | Grid tracker | 1,310.74 | 91.48 | 6.98% | -87.67 | 258.55 | 19.73% |
| 345 | MPC equal weights | 1,310.74 | 103.32 | 7.88% | -103.32 | 267.09 | 20.38% |
| 345 | MPC K=25 | 1,310.74 | 95.08 | 7.25% | -95.08 | 264.69 | 20.19% |

## Revenue, imbalance cash flow and battery wear

Auction revenue is net signed cash from all four auctions, including purchases. Settlement cash flow is signed: a negative value is a net cost. Profit equals auction cash plus settlement cash, less degradation, import fees and trading fees. The MPC's planning penalty is not charged as realized cash.

| Farm | Controller | Auction cash (€) | Imbalance cash flow (€) | Degradation (€) | Fees and import charges (€) | Realized operating profit (€) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Grid tracker | 179,709.42 | -9,591.15 | 6,252.64 | 0.00 | 163,865.63 |
| 1 | MPC equal weights | 179,678.75 | -9,595.20 | 6,117.00 | 0.00 | 163,966.55 |
| 1 | MPC K=25 | 179,585.84 | -9,684.87 | 6,179.86 | 0.00 | 163,721.11 |
| 345 | Grid tracker | 291,612.29 | -15,614.26 | 10,151.21 | 0.00 | 265,846.81 |
| 345 | MPC equal weights | 292,590.53 | -17,607.61 | 9,548.10 | 0.00 | 265,434.82 |
| 345 | MPC K=25 | 291,684.51 | -16,636.35 | 9,828.27 | 0.00 | 265,219.89 |

### Auction cash breakdown

Values are signed net cash by auction. Negative values can represent net purchases or the effect of negative prices.

| Farm | Controller | DA (€) | IDA1 (€) | IDA2 (€) | IDA3 (€) |
|---|---|---:|---:|---:|---:|
| 1 | Grid tracker | 20,664.85 | -32,410.71 | 137,249.71 | 54,205.56 |
| 1 | MPC equal weights | 20,657.69 | -32,401.57 | 137,181.94 | 54,240.69 |
| 1 | MPC K=25 | 20,645.39 | -32,417.50 | 137,175.40 | 54,182.55 |
| 345 | Grid tracker | 36,206.59 | -49,599.90 | 227,341.59 | 77,664.01 |
| 345 | MPC equal weights | 36,253.55 | -50,094.56 | 226,739.42 | 79,692.12 |
| 345 | MPC K=25 | 36,233.24 | -50,012.52 | 226,715.56 | 78,748.24 |

### Imbalance direction and settlement

Positive and negative MWh below are nonnegative magnitudes; the cash columns retain their actual signs. Absolute settlement exposure sums the absolute value of each interval's settlement cash flow.

| Farm | Controller | Positive imbalance (MWh) | Negative imbalance (MWh) | Positive cash (€) | Negative cash (€) | Absolute exposure (€) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Grid tracker | 0.98 | 54.39 | 2.21 | -9,593.36 | 9,595.74 |
| 1 | MPC equal weights | 0.00 | 56.38 | 0.00 | -9,595.20 | 9,595.36 |
| 1 | MPC K=25 | 0.00 | 54.92 | 0.00 | -9,684.87 | 9,685.03 |
| 345 | Grid tracker | 1.91 | 89.57 | 11.77 | -15,626.03 | 15,637.80 |
| 345 | MPC equal weights | 0.00 | 103.32 | 0.00 | -17,607.61 | 17,607.61 |
| 345 | MPC K=25 | 0.00 | 95.08 | 0.00 | -16,636.35 | 16,636.35 |

## Battery and solver

Daily minimum counts use actual end-of-day SOC within 0.01 percentage points of the configured 7% minimum. The final SOC is at the end of the full replay. MPC fallback counts concern physical controller solves, while auction fallback counts concern market-planning solves.

| Farm | Controller | Charge (MWh) | Discharge (MWh) | Mean daily end SOC | Days at minimum | Final SOC | MPC fallbacks | Auction fallbacks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Grid tracker | 119.07 | 107.18 | 10.86% | 195/241 | 47.65% | 0 | 0 |
| 1 | MPC equal weights | 116.50 | 104.85 | 10.32% | 196/241 | 50.00% | 0 | 0 |
| 1 | MPC K=25 | 117.70 | 105.93 | 10.24% | 199/241 | 50.00% | 0 | 0 |
| 345 | Grid tracker | 193.22 | 174.11 | 18.61% | 151/241 | 28.07% | 0 | 0 |
| 345 | MPC equal weights | 181.85 | 163.66 | 14.94% | 156/241 | 50.00% | 0 | 0 |
| 345 | MPC K=25 | 187.18 | 168.46 | 15.04% | 153/241 | 50.00% | 0 | 0 |

## K=25 versus equal-weight MPC

Differences below are K=25 minus equal-weight MPC; negative imbalance or curtailment differences mean a reduction. Both are continuous trajectories, so auction submissions can differ.

| Farm | Absolute imbalance change (MWh) | Curtailment change (MWh) | Settlement cash change (€) | Degradation change (€) | Profit change (€) |
|---|---:|---:|---:|---:|---:|
| 1 | -1.46 | -0.95 | -89.67 | 62.85 | -245.44 |
| 345 | -8.24 | -2.40 | 971.26 | 280.17 | -214.93 |

## Interpretation and limitations

- The auction planner enforced the 50% terminal SOC only on the final replay day. The MPC used a soft 50% daily target at €50/MWh of deviation, a €500/MWh planning imbalance weight, and, in the K=25 case, a 25-times current-interval weight. The tracker did not have an independent daily SOC target.
- The BiGRU model bundles were selected retrospectively on inspected 2026 data. Weather forecasts use the latest eligible issue with an assumed six-hour availability lag; current-interval realized PV is disclosed to the MPC. The reviewed targets include synthetic values.
- Auction planning uses actual clearing-price curves as a retrospective price oracle. Actual imbalance prices are used to calculate realized settlement cash flow; they are not given to the deterministic MPC.
- Realized profit does not include an end-of-replay inventory adjustment. The tracker and MPC can end with different SOC, so profit differences do not fully account for the remaining energy value.
- PV curtailment has no separate direct cash-flow line here. Its financial effects flow through auction positions and imbalance settlement; applying one price to curtailed energy would not be a realized cash flow.
- The newer daily-hard-SOC experiment is a separate batch and is not included in these results.

## Plots and detailed records

- [Farm 1: cumulative profit, imbalance and daily SOC](../../outputs/energy_research/deterministic_six_workers_v2/summary_plots/farm_1_main_results.png)
- [Farm 345: cumulative profit, imbalance and daily SOC](../../outputs/energy_research/deterministic_six_workers_v2/summary_plots/farm_345_main_results.png)
- [Daily controller metrics](../../outputs/energy_research/deterministic_six_workers_v2/daily_metrics.csv)

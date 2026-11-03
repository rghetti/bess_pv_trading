# Full-history historical-GHI PV curation review

Preparation only: no training or new parameter fitting was performed.

Configured compute: 10 CPU workers, one native-library thread per farm worker.

Raw metered AC and raw GHI are preserved. Training uses all eligible measured quarters, including night.
Farm 345 positive production in the configured monthly night windows is excluded; valid nighttime zeros remain.
Validation and test masks use the complement of the month-specific night window, independently of proxy solar elevation.
Splits remain 2025 train, January–April 2026 validation, May–August 2026 test, with the previous one-day embargo.
Interior missing GHI is linearly interpolated within each split and flagged. Unbracketed gaps stay missing; AC is never interpolated.

## Candidate definitions

- Relative low output: observed GHI >=300 W/m², pvlib reference >=15% of AC capacity, measured/reference <=0.55, gap >=15% of capacity, for >=4 consecutive quarters.
- Absolute low output: observed GHI >=300 W/m² and measured AC <=10% of capacity for >=3 consecutive quarters.
- Zero/high-GHI episode: measured AC exactly zero and observed GHI >=100 W/m² for >=4 consecutive quarters.
- Extreme relative deficit: observed GHI >=300 W/m², reference >=25% of capacity, measured/reference <=0.25, gap >=25% of capacity, for >=8 consecutive quarters.
- Positive nighttime production: raw AC >0 kW in the configured monthly night window. At least 1% of capacity is highlighted as extreme nighttime evidence.
- Runs stop at missing/failed evidence, a non-15-minute timestamp gap or a local-day boundary. Adjacent qualifying low-output flags are grouped in one review row.

Episode end timestamps identify the final included quarter start, so the covered interval ends 15 minutes later.

The old relative detector used a cross-fitted statistical median. This review uses the frozen configured pvlib model and performs no reference fitting.
Candidates depend on model/proxy assumptions and are not confirmed faults or operator restrictions. Interpolated GHI does not trigger candidate flags.

## Authorized changes and pending decisions

Only farm 345 positive-night exclusions and farm 910 August test reconstruction are applied by this workflow.
The farm 910 August mask is the union of existing approved low-output exclusion reasons and newly detected low-output runs.
Its `test_target_mw` is replaced with finite satellite-GHI pvlib predictions. `measured_ac_raw_mw` is unchanged.
Overlapping independent exclusions still prevent scoring. No other farm/month receives synthetic labels.
Already excluded episodes are labeled `already_excluded`; other new candidates retain their current eligibility and are marked `pending_review`.
Synthetic test scores measure agreement with a physical-model reconstruction; they are not independent measured-power validation.

Curated data: `data/farms/curated/historical_ghi_20261103`. `training_eligible`, `validation_eligible`, and `test_eligible` are distinct masks.
Input GHI is historical; temperature/wind and the direct/global fraction come from the preceding-day 00 UTC weather view, with Erbs fallback for undefined fractions.
Forecast release time uses the configured six-hour assumption. Historical GHI is retrospective and is not an operational inference input.

[Focused review with plot links](PENDING_REVIEW.md) · [Pending low-output decisions](pending_low_power_episodes.csv) · [All pending decisions](pending_review_episodes.csv) · [All episodes](review_episodes.csv)

[Split/mask counts](split_summary.csv) · [Exact synthetic testing intervals](synthetic_testing_intervals.csv) · [Night production](night_production_summary.csv) · [Provenance](manifest.json)

## Counts by farm

| Farm | Low-output episodes | Extreme low/zero episodes | Pending low-output episodes | Night episodes | Training quarters | Night training quarters | Synthetic test quarters |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 10 | 15 | 4 | 34816 | 13522 | 0 |
| 345 | 52 | 2 | 4 | 495 | 30011 | 9197 | 0 |
| 901 | 24 | 4 | 6 | 4 | 34736 | 13522 | 0 |
| 910 | 107 | 53 | 7 | 5 | 34119 | 13522 | 290 |

## Month-specific scoring windows

Local quarter-hour starts in Europe/Rome. The evaluation window is half-open; all other times are night.

| Month | Evaluation window | Night window |
|---|---|---|
| 01 | 06:45–18:30 | 18:30–06:45 |
| 02 | 06:00–19:15 | 19:15–06:00 |
| 03 | 05:15–21:00 | 21:00–05:15 |
| 04 | 05:15–21:45 | 21:45–05:15 |
| 05 | 05:00–21:30 | 21:30–05:00 |
| 06 | 05:00–21:30 | 21:30–05:00 |
| 07 | 05:00–21:30 | 21:30–05:00 |
| 08 | 05:00–21:30 | 21:30–05:00 |
| 09 | 05:45–21:15 | 21:15–05:45 |
| 10 | 06:00–20:00 | 20:00–06:00 |
| 11 | 06:00–18:15 | 18:15–06:00 |
| 12 | 06:45–18:00 | 18:00–06:45 |

## Linked review days

Plots show raw AC (black), frozen historical-GHI pvlib (blue), synthetic testing values (green),
existing exclusions (gray), low-output candidates (red), and positive night observations (orange).
The lower panel shows historical and forecast GHI, with interpolated observations marked in purple.
Every low-output day has a linked plot. Additional plots cover prioritized night candidates and reconstructed August days.

| Farm | Day | Episodes | Pending | Plot |
|---|---|---:|---:|---|
| 1 | 2025-01-22 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-01-22.png) |
| 1 | 2025-02-13 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-02-13.png) |
| 1 | 2025-04-14 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-04-14.png) |
| 1 | 2025-07-13 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-07-13.png) |
| 1 | 2025-08-19 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-08-19.png) |
| 1 | 2025-08-29 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-08-29.png) |
| 1 | 2025-10-26 | 1 | 1 | CSV evidence |
| 1 | 2025-11-22 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-11-22.png) |
| 1 | 2025-11-25 | 2 | 2 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-11-25.png) |
| 1 | 2025-11-27 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-11-27.png) |
| 1 | 2025-11-28 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-11-28.png) |
| 1 | 2025-11-29 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-11-29.png) |
| 1 | 2025-12-08 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-12-08.png) |
| 1 | 2025-12-09 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-12-09.png) |
| 1 | 2025-12-12 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-12-12.png) |
| 1 | 2025-12-13 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-12-13.png) |
| 1 | 2025-12-18 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-12-18.png) |
| 1 | 2025-12-26 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-12-26.png) |
| 1 | 2025-12-27 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-12-27.png) |
| 1 | 2025-12-28 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-12-28.png) |
| 1 | 2025-12-29 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2025-12-29.png) |
| 1 | 2026-01-22 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-01-22.png) |
| 1 | 2026-01-27 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-01-27.png) |
| 1 | 2026-01-29 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-01-29.png) |
| 1 | 2026-02-17 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-02-17.png) |
| 1 | 2026-05-29 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-05-29.png) |
| 1 | 2026-06-12 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-06-12.png) |
| 1 | 2026-06-13 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-06-13.png) |
| 1 | 2026-06-14 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-06-14.png) |
| 1 | 2026-06-19 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-06-19.png) |
| 1 | 2026-07-20 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-07-20.png) |
| 1 | 2026-07-27 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-07-27.png) |
| 1 | 2026-07-28 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_1/plots/2026-07-28.png) |
| 345 | 2025-01-01 | 2 | 0 | CSV evidence |
| 345 | 2025-01-02 | 2 | 0 | CSV evidence |
| 345 | 2025-01-03 | 2 | 0 | CSV evidence |
| 345 | 2025-01-04 | 2 | 0 | CSV evidence |
| 345 | 2025-01-05 | 2 | 0 | CSV evidence |
| 345 | 2025-01-06 | 2 | 0 | CSV evidence |
| 345 | 2025-01-07 | 4 | 0 | CSV evidence |
| 345 | 2025-01-08 | 7 | 0 | CSV evidence |
| 345 | 2025-01-09 | 2 | 0 | CSV evidence |
| 345 | 2025-01-10 | 3 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-01-10.png) |
| 345 | 2025-01-11 | 2 | 0 | CSV evidence |
| 345 | 2025-01-12 | 2 | 0 | CSV evidence |
| 345 | 2025-01-13 | 2 | 0 | CSV evidence |
| 345 | 2025-01-14 | 2 | 0 | CSV evidence |
| 345 | 2025-01-15 | 2 | 0 | CSV evidence |
| 345 | 2025-01-16 | 2 | 0 | CSV evidence |
| 345 | 2025-01-17 | 2 | 0 | CSV evidence |
| 345 | 2025-01-18 | 2 | 0 | CSV evidence |
| 345 | 2025-01-19 | 2 | 0 | CSV evidence |
| 345 | 2025-01-20 | 2 | 0 | CSV evidence |
| 345 | 2025-01-21 | 2 | 0 | CSV evidence |
| 345 | 2025-01-22 | 2 | 0 | CSV evidence |
| 345 | 2025-01-23 | 2 | 0 | CSV evidence |
| 345 | 2025-01-24 | 2 | 0 | CSV evidence |
| 345 | 2025-01-25 | 2 | 0 | CSV evidence |
| 345 | 2025-01-26 | 2 | 0 | CSV evidence |
| 345 | 2025-01-27 | 2 | 0 | CSV evidence |
| 345 | 2025-01-28 | 2 | 0 | CSV evidence |
| 345 | 2025-01-29 | 2 | 0 | CSV evidence |
| 345 | 2025-01-30 | 2 | 0 | CSV evidence |
| 345 | 2025-01-31 | 2 | 0 | CSV evidence |
| 345 | 2025-02-01 | 2 | 0 | CSV evidence |
| 345 | 2025-02-02 | 2 | 0 | CSV evidence |
| 345 | 2025-02-03 | 2 | 0 | CSV evidence |
| 345 | 2025-02-04 | 2 | 0 | CSV evidence |
| 345 | 2025-02-05 | 2 | 0 | CSV evidence |
| 345 | 2025-02-06 | 2 | 0 | CSV evidence |
| 345 | 2025-02-07 | 2 | 0 | CSV evidence |
| 345 | 2025-02-08 | 2 | 0 | CSV evidence |
| 345 | 2025-02-09 | 2 | 0 | CSV evidence |
| 345 | 2025-02-10 | 2 | 0 | CSV evidence |
| 345 | 2025-02-11 | 2 | 0 | CSV evidence |
| 345 | 2025-02-12 | 2 | 0 | CSV evidence |
| 345 | 2025-02-13 | 2 | 0 | CSV evidence |
| 345 | 2025-02-14 | 2 | 0 | CSV evidence |
| 345 | 2025-02-15 | 2 | 0 | CSV evidence |
| 345 | 2025-02-16 | 2 | 0 | CSV evidence |
| 345 | 2025-02-17 | 2 | 0 | CSV evidence |
| 345 | 2025-02-18 | 2 | 0 | CSV evidence |
| 345 | 2025-02-19 | 2 | 0 | CSV evidence |
| 345 | 2025-02-20 | 2 | 0 | CSV evidence |
| 345 | 2025-02-21 | 2 | 0 | CSV evidence |
| 345 | 2025-02-22 | 2 | 0 | CSV evidence |
| 345 | 2025-02-23 | 2 | 0 | CSV evidence |
| 345 | 2025-02-24 | 2 | 0 | CSV evidence |
| 345 | 2025-02-25 | 2 | 0 | CSV evidence |
| 345 | 2025-02-26 | 2 | 0 | CSV evidence |
| 345 | 2025-02-27 | 2 | 0 | CSV evidence |
| 345 | 2025-02-28 | 2 | 0 | CSV evidence |
| 345 | 2025-03-01 | 2 | 0 | CSV evidence |
| 345 | 2025-03-02 | 2 | 0 | CSV evidence |
| 345 | 2025-03-03 | 2 | 0 | CSV evidence |
| 345 | 2025-03-04 | 2 | 0 | CSV evidence |
| 345 | 2025-03-05 | 2 | 0 | CSV evidence |
| 345 | 2025-03-06 | 2 | 0 | CSV evidence |
| 345 | 2025-03-07 | 3 | 0 | CSV evidence |
| 345 | 2025-03-08 | 7 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-03-08.png) |
| 345 | 2025-03-09 | 3 | 0 | CSV evidence |
| 345 | 2025-03-10 | 7 | 0 | CSV evidence |
| 345 | 2025-03-11 | 12 | 0 | CSV evidence |
| 345 | 2025-03-12 | 9 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-03-12.png) |
| 345 | 2025-03-13 | 10 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-03-13.png) |
| 345 | 2025-03-14 | 15 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-03-14.png) |
| 345 | 2025-03-15 | 16 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-03-15.png) |
| 345 | 2025-03-16 | 14 | 0 | CSV evidence |
| 345 | 2025-03-17 | 14 | 0 | CSV evidence |
| 345 | 2025-03-18 | 13 | 0 | CSV evidence |
| 345 | 2025-03-19 | 15 | 0 | CSV evidence |
| 345 | 2025-03-20 | 14 | 0 | CSV evidence |
| 345 | 2025-03-21 | 14 | 0 | CSV evidence |
| 345 | 2025-03-22 | 14 | 0 | CSV evidence |
| 345 | 2025-03-23 | 13 | 0 | CSV evidence |
| 345 | 2025-03-24 | 9 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-03-24.png) |
| 345 | 2025-03-28 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-03-28.png) |
| 345 | 2025-03-29 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-03-29.png) |
| 345 | 2025-03-31 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-03-31.png) |
| 345 | 2025-04-04 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-04.png) |
| 345 | 2025-04-08 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-08.png) |
| 345 | 2025-04-13 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-13.png) |
| 345 | 2025-04-14 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-14.png) |
| 345 | 2025-04-15 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-15.png) |
| 345 | 2025-04-16 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-16.png) |
| 345 | 2025-04-18 | 2 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-18.png) |
| 345 | 2025-04-19 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-19.png) |
| 345 | 2025-04-20 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-20.png) |
| 345 | 2025-04-21 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-21.png) |
| 345 | 2025-04-22 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-22.png) |
| 345 | 2025-04-25 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-04-25.png) |
| 345 | 2025-05-01 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-05-01.png) |
| 345 | 2025-05-05 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-05-05.png) |
| 345 | 2025-05-06 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-05-06.png) |
| 345 | 2025-05-07 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-05-07.png) |
| 345 | 2025-05-11 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-05-11.png) |
| 345 | 2025-05-13 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-05-13.png) |
| 345 | 2025-05-22 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-05-22.png) |
| 345 | 2025-06-16 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-06-16.png) |
| 345 | 2025-06-26 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-06-26.png) |
| 345 | 2025-07-18 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-07-18.png) |
| 345 | 2025-07-21 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-07-21.png) |
| 345 | 2025-08-30 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-08-30.png) |
| 345 | 2025-09-13 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-09-13.png) |
| 345 | 2025-09-14 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-09-14.png) |
| 345 | 2025-09-15 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-09-15.png) |
| 345 | 2025-10-06 | 2 | 0 | CSV evidence |
| 345 | 2025-10-07 | 4 | 0 | CSV evidence |
| 345 | 2025-10-08 | 1 | 0 | CSV evidence |
| 345 | 2025-10-22 | 1 | 0 | CSV evidence |
| 345 | 2025-10-24 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-10-24.png) |
| 345 | 2025-10-26 | 2 | 0 | CSV evidence |
| 345 | 2025-10-27 | 4 | 0 | CSV evidence |
| 345 | 2025-10-28 | 8 | 0 | CSV evidence |
| 345 | 2025-10-29 | 12 | 0 | CSV evidence |
| 345 | 2025-10-30 | 11 | 0 | CSV evidence |
| 345 | 2025-10-31 | 4 | 0 | CSV evidence |
| 345 | 2025-11-06 | 11 | 0 | CSV evidence |
| 345 | 2025-11-08 | 5 | 0 | CSV evidence |
| 345 | 2025-11-09 | 1 | 0 | CSV evidence |
| 345 | 2025-11-16 | 7 | 0 | CSV evidence |
| 345 | 2025-11-19 | 13 | 0 | CSV evidence |
| 345 | 2025-11-20 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2025-11-20.png) |
| 345 | 2025-11-21 | 2 | 0 | CSV evidence |
| 345 | 2025-11-22 | 2 | 0 | CSV evidence |
| 345 | 2025-11-23 | 3 | 0 | CSV evidence |
| 345 | 2025-11-26 | 2 | 0 | CSV evidence |
| 345 | 2025-11-27 | 3 | 0 | CSV evidence |
| 345 | 2025-11-28 | 3 | 0 | CSV evidence |
| 345 | 2025-11-30 | 2 | 0 | CSV evidence |
| 345 | 2025-12-01 | 2 | 0 | CSV evidence |
| 345 | 2025-12-03 | 2 | 0 | CSV evidence |
| 345 | 2025-12-05 | 4 | 0 | CSV evidence |
| 345 | 2025-12-07 | 6 | 0 | CSV evidence |
| 345 | 2025-12-10 | 2 | 0 | CSV evidence |
| 345 | 2025-12-11 | 2 | 0 | CSV evidence |
| 345 | 2025-12-20 | 9 | 0 | CSV evidence |
| 345 | 2025-12-21 | 6 | 0 | CSV evidence |
| 345 | 2025-12-22 | 3 | 0 | CSV evidence |
| 345 | 2025-12-23 | 2 | 0 | CSV evidence |
| 345 | 2025-12-24 | 2 | 0 | CSV evidence |
| 345 | 2025-12-25 | 3 | 0 | CSV evidence |
| 345 | 2025-12-26 | 3 | 0 | CSV evidence |
| 345 | 2025-12-27 | 2 | 0 | CSV evidence |
| 345 | 2026-02-03 | 1 | 0 | CSV evidence |
| 345 | 2026-02-05 | 1 | 0 | CSV evidence |
| 345 | 2026-02-08 | 1 | 0 | CSV evidence |
| 345 | 2026-02-09 | 1 | 0 | CSV evidence |
| 345 | 2026-02-11 | 1 | 0 | CSV evidence |
| 345 | 2026-02-14 | 1 | 0 | CSV evidence |
| 345 | 2026-02-16 | 1 | 0 | CSV evidence |
| 345 | 2026-02-18 | 1 | 0 | CSV evidence |
| 345 | 2026-02-23 | 1 | 0 | CSV evidence |
| 345 | 2026-03-11 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2026-03-11.png) |
| 345 | 2026-03-29 | 2 | 0 | CSV evidence |
| 345 | 2026-04-09 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2026-04-09.png) |
| 345 | 2026-04-10 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2026-04-10.png) |
| 345 | 2026-04-14 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2026-04-14.png) |
| 345 | 2026-05-06 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2026-05-06.png) |
| 345 | 2026-05-18 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2026-05-18.png) |
| 345 | 2026-05-27 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2026-05-27.png) |
| 345 | 2026-06-12 | 1 | 0 | CSV evidence |
| 345 | 2026-06-14 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_345/plots/2026-06-14.png) |
| 901 | 2025-02-15 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-02-15.png) |
| 901 | 2025-02-17 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-02-17.png) |
| 901 | 2025-03-10 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-03-10.png) |
| 901 | 2025-04-13 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-04-13.png) |
| 901 | 2025-04-23 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-04-23.png) |
| 901 | 2025-04-28 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-04-28.png) |
| 901 | 2025-04-29 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-04-29.png) |
| 901 | 2025-05-08 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-05-08.png) |
| 901 | 2025-05-13 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-05-13.png) |
| 901 | 2025-06-30 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-06-30.png) |
| 901 | 2025-07-07 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-07-07.png) |
| 901 | 2025-09-02 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-09-02.png) |
| 901 | 2025-09-12 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-09-12.png) |
| 901 | 2025-10-26 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-10-26.png) |
| 901 | 2025-11-03 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-11-03.png) |
| 901 | 2025-11-04 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-11-04.png) |
| 901 | 2025-11-05 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2025-11-05.png) |
| 901 | 2026-03-04 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2026-03-04.png) |
| 901 | 2026-03-22 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2026-03-22.png) |
| 901 | 2026-04-30 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2026-04-30.png) |
| 901 | 2026-05-01 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2026-05-01.png) |
| 901 | 2026-05-14 | 2 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2026-05-14.png) |
| 901 | 2026-06-05 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2026-06-05.png) |
| 901 | 2026-06-12 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2026-06-12.png) |
| 901 | 2026-06-13 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2026-06-13.png) |
| 901 | 2026-06-14 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2026-06-14.png) |
| 901 | 2026-07-17 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_901/plots/2026-07-17.png) |
| 910 | 2025-01-23 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-01-23.png) |
| 910 | 2025-02-21 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-02-21.png) |
| 910 | 2025-02-27 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-02-27.png) |
| 910 | 2025-03-02 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-03-02.png) |
| 910 | 2025-03-13 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-03-13.png) |
| 910 | 2025-03-23 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-03-23.png) |
| 910 | 2025-03-30 | 2 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-03-30.png) |
| 910 | 2025-04-18 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-04-18.png) |
| 910 | 2025-04-24 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-04-24.png) |
| 910 | 2025-05-01 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-05-01.png) |
| 910 | 2025-05-02 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-05-02.png) |
| 910 | 2025-05-04 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-05-04.png) |
| 910 | 2025-05-13 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-05-13.png) |
| 910 | 2025-05-18 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-05-18.png) |
| 910 | 2025-05-25 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-05-25.png) |
| 910 | 2025-06-25 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-06-25.png) |
| 910 | 2025-07-10 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-07-10.png) |
| 910 | 2025-07-24 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-07-24.png) |
| 910 | 2025-07-29 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-07-29.png) |
| 910 | 2025-08-21 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-08-21.png) |
| 910 | 2025-09-03 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-09-03.png) |
| 910 | 2025-10-12 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-10-12.png) |
| 910 | 2025-10-13 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-10-13.png) |
| 910 | 2025-10-26 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2025-10-26.png) |
| 910 | 2026-01-04 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-01-04.png) |
| 910 | 2026-01-08 | 1 | 1 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-01-08.png) |
| 910 | 2026-03-01 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-01.png) |
| 910 | 2026-03-02 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-02.png) |
| 910 | 2026-03-03 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-03.png) |
| 910 | 2026-03-06 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-06.png) |
| 910 | 2026-03-07 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-07.png) |
| 910 | 2026-03-08 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-08.png) |
| 910 | 2026-03-09 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-09.png) |
| 910 | 2026-03-10 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-10.png) |
| 910 | 2026-03-11 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-11.png) |
| 910 | 2026-03-12 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-12.png) |
| 910 | 2026-03-13 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-13.png) |
| 910 | 2026-03-14 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-14.png) |
| 910 | 2026-03-15 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-15.png) |
| 910 | 2026-03-16 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-16.png) |
| 910 | 2026-03-17 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-17.png) |
| 910 | 2026-03-19 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-19.png) |
| 910 | 2026-03-20 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-20.png) |
| 910 | 2026-03-21 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-21.png) |
| 910 | 2026-03-22 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-22.png) |
| 910 | 2026-03-23 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-23.png) |
| 910 | 2026-03-24 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-24.png) |
| 910 | 2026-03-25 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-25.png) |
| 910 | 2026-03-28 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-28.png) |
| 910 | 2026-03-29 | 3 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-29.png) |
| 910 | 2026-03-30 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-30.png) |
| 910 | 2026-03-31 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-03-31.png) |
| 910 | 2026-04-03 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-04-03.png) |
| 910 | 2026-04-05 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-04-05.png) |
| 910 | 2026-04-06 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-04-06.png) |
| 910 | 2026-04-07 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-04-07.png) |
| 910 | 2026-04-09 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-04-09.png) |
| 910 | 2026-04-10 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-04-10.png) |
| 910 | 2026-04-11 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-04-11.png) |
| 910 | 2026-04-25 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-04-25.png) |
| 910 | 2026-04-26 | 3 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-04-26.png) |
| 910 | 2026-04-27 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-04-27.png) |
| 910 | 2026-05-01 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-05-01.png) |
| 910 | 2026-05-02 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-05-02.png) |
| 910 | 2026-05-03 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-05-03.png) |
| 910 | 2026-05-04 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-05-04.png) |
| 910 | 2026-05-12 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-05-12.png) |
| 910 | 2026-05-18 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-05-18.png) |
| 910 | 2026-06-12 | 1 | 1 | CSV evidence |
| 910 | 2026-06-13 | 1 | 1 | CSV evidence |
| 910 | 2026-06-19 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-06-19.png) |
| 910 | 2026-06-30 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-06-30.png) |
| 910 | 2026-07-01 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-07-01.png) |
| 910 | 2026-07-20 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-07-20.png) |
| 910 | 2026-07-26 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-07-26.png) |
| 910 | 2026-07-27 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-07-27.png) |
| 910 | 2026-08-01 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-01.png) |
| 910 | 2026-08-02 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-02.png) |
| 910 | 2026-08-03 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-03.png) |
| 910 | 2026-08-04 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-04.png) |
| 910 | 2026-08-05 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-05.png) |
| 910 | 2026-08-06 | 2 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-06.png) |
| 910 | 2026-08-08 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-08.png) |
| 910 | 2026-08-15 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-15.png) |
| 910 | 2026-08-19 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-19.png) |
| 910 | 2026-08-22 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-22.png) |
| 910 | 2026-08-23 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-23.png) |
| 910 | 2026-08-24 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-24.png) |
| 910 | 2026-08-25 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-25.png) |
| 910 | 2026-08-26 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-26.png) |
| 910 | 2026-08-27 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-27.png) |
| 910 | 2026-08-28 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-28.png) |
| 910 | 2026-08-29 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-29.png) |
| 910 | 2026-08-30 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-30.png) |
| 910 | 2026-08-31 | 1 | 0 | [Open](../../../outputs/pv_curated_review/historical_ghi_20261103/farm_910/plots/2026-08-31.png) |

## Reproduce

Create a fresh package using the configured unused output paths:

```bash
PYTHONPATH=src python scripts/review_curated_pv.py --config configs/pv_experiments/curated_ghi_review.yaml
```

Refresh this existing fingerprint-checked review without fitting models:

```bash
PYTHONPATH=src python scripts/review_curated_pv.py --config configs/pv_experiments/curated_ghi_review.yaml --refresh-review
```

The review preserves user-entered `review_decision` and `review_notes` fields across refreshes.
Review proposals are not automatically applied to targets or exclusions.

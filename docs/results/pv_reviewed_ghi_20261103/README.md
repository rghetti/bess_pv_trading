# User-reviewed historical-GHI PV targets

Prepared data only; no model training or physical parameter fitting was performed.

[Original reviewed instructions and explicit date lists](../../../configs/pv_experiments/reviewed_ghi_user_decisions.yaml) · [All exclusions and replacement intervals](../../../configs/data_quality/pv_reviewed_ghi_decisions_2025_to_2026_08.csv)

[Reviewed rules](reviewed_rules.csv) · [Counts by farm and split](split_target_summary.csv) · [Explicit restorations](restored_days.csv) · [Dataset provenance](manifest.json)

Curated corpus: `data/farms/curated/reviewed_ghi_2025_to_2026_08`. Raw AC and GHI remain in separate unchanged columns.
`model_target_mw` contains measured or reviewed synthetic values for all splits.
Each source is identified in `model_target_source`; valid synthetic targets are included in training when requested.
Training readers must use `model_target_mw` and the supplied split masks. The CSV marks excluded raw labels; it must not be reapplied as a blanket exclusion of synthetic targets.
A full-day instruction covers every actual quarter-hour of that local day, including DST transitions.
A date-only instruction covers red low-power candidate intervals; separate prior gray exclusions remain excluded.
Prior exclusions remain in force unless the user explicitly requested their restoration or selected them for reconstruction.
Before-time and from-time rules use Europe/Rome interval starts: before is exclusive; from is inclusive.
Every pvlib replacement inside the month-specific night window is zero. Explicit zero rules also apply outside that window.
Targets with unavailable daytime pvlib or missing weather remain missing/ineligible rather than being invented.
Historical GHI and pvlib reconstructions are retrospective. Synthetic labels and scores are not independent measurements.

## Per-farm summary

| Farm | pvlib quarters | Explicit zero quarters | Prior exclusions retained | Prior quarters restored to raw | Synthetic training quarters | Synthetic validation quarters | Synthetic test quarters |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1435 | 0 | 54 | 0 | 1196 | 108 | 29 |
| 345 | 5519 | 12 | 841 | 8 | 5439 | 32 | 10 |
| 901 | 387 | 39 | 80 | 0 | 315 | 10 | 71 |
| 910 | 7485 | 0 | 475 | 5 | 545 | 2373 | 2340 |

## User-nominated rules

| Farm | Rule | Local day, month or time | Quarters matched | Replacement |
|---|---|---|---:|---|
| 1 | episode_day | 2025-01-22 | 3 | pvlib |
| 1 | episode_day | 2025-02-13 | 17 | pvlib |
| 1 | episode_day | 2025-08-19 | 12 | pvlib |
| 1 | episode_day | 2025-08-29 | 11 | pvlib |
| 1 | episode_day | 2026-01-27 | 8 | pvlib |
| 1 | episode_day | 2026-06-19 | 12 | pvlib |
| 1 | episode_day | 2026-07-27 | 8 | pvlib |
| 1 | episode_day | 2026-07-28 | 9 | pvlib |
| 1 | full_day | 2025-11-27 | 96 | pvlib |
| 1 | full_day | 2025-11-28 | 96 | pvlib |
| 1 | full_day | 2025-11-29 | 96 | pvlib |
| 1 | full_day | 2025-12-08 | 96 | pvlib |
| 1 | full_day | 2025-12-09 | 96 | pvlib |
| 1 | full_day | 2025-12-12 | 96 | pvlib |
| 1 | full_day | 2025-12-13 | 96 | pvlib |
| 1 | full_day | 2025-12-18 | 96 | pvlib |
| 1 | full_day | 2025-12-26 | 96 | pvlib |
| 1 | full_day | 2025-12-27 | 96 | pvlib |
| 1 | full_day | 2025-12-28 | 96 | pvlib |
| 1 | full_day | 2025-12-29 | 96 | pvlib |
| 1 | full_day | 2026-01-29 | 96 | pvlib |
| 1 | full_day | 2026-02-17 | 96 | pvlib |
| 1 | flagged_night_production | all history | 11 | pvlib |
| 345 | episode_day | 2025-04-19 | 4 | pvlib |
| 345 | episode_day | 2025-05-11 | 5 | pvlib |
| 345 | episode_day | 2025-06-16 | 3 | pvlib |
| 345 | episode_day | 2025-07-18 | 15 | pvlib |
| 345 | episode_day | 2025-07-21 | 3 | pvlib |
| 345 | episode_day | 2025-09-15 | 10 | pvlib |
| 345 | episode_day | 2026-04-09 | 7 | pvlib |
| 345 | episode_day | 2026-04-10 | 21 | pvlib |
| 345 | episode_day | 2026-04-14 | 4 | pvlib |
| 345 | episode_day | 2026-05-06 | 4 | pvlib |
| 345 | episode_day | 2026-05-27 | 4 | pvlib |
| 345 | full_day | 2025-03-08 | 96 | pvlib |
| 345 | full_day | 2025-03-29 | 96 | pvlib |
| 345 | full_day | 2025-04-15 | 96 | pvlib |
| 345 | full_day | 2025-04-16 | 96 | pvlib |
| 345 | full_day | 2025-04-21 | 96 | pvlib |
| 345 | full_day | 2025-05-01 | 96 | pvlib |
| 345 | full_day | 2025-05-05 | 96 | pvlib |
| 345 | full_day | 2025-06-27 | 96 | pvlib |
| 345 | full_day | 2025-09-13 | 96 | pvlib |
| 345 | full_day | 2025-09-14 | 96 | pvlib |
| 345 | full_day | 2025-11-20 | 96 | pvlib |
| 345 | before_local | 2025-10-24 before 12:00 | 48 | pvlib |
| 345 | flagged_night_production | all history | 4371 | pvlib |
| 345 | zero_from_local | 2026-06-14 from 21:00 | 12 | zero |
| 345 | restore_prior_exclusions | 2026-04-09 | 96 | measured_unless_selected |
| 901 | episode_day | 2025-03-10 | 4 | pvlib |
| 901 | episode_day | 2025-04-28 | 10 | pvlib |
| 901 | episode_day | 2025-04-29 | 4 | pvlib |
| 901 | episode_day | 2025-09-02 | 4 | pvlib |
| 901 | episode_day | 2025-09-12 | 4 | pvlib |
| 901 | episode_day | 2026-03-22 | 10 | pvlib |
| 901 | episode_day | 2026-05-01 | 35 | pvlib |
| 901 | episode_day | 2026-05-14 | 18 | pvlib |
| 901 | episode_day | 2026-06-05 | 5 | pvlib |
| 901 | episode_day | 2026-07-17 | 4 | pvlib |
| 901 | full_day | 2025-11-03 | 96 | pvlib |
| 901 | full_day | 2025-11-04 | 96 | pvlib |
| 901 | full_day | 2025-11-05 | 96 | pvlib |
| 901 | flagged_night_production | all history | 12 | pvlib |
| 901 | zero_from_local | 2026-06-12 from 20:45 | 13 | zero |
| 901 | zero_from_local | 2026-06-13 from 20:45 | 13 | zero |
| 901 | zero_from_local | 2026-06-14 from 20:45 | 13 | zero |
| 910 | episode_day | 2025-01-23 | 7 | pvlib |
| 910 | episode_day | 2025-02-21 | 14 | pvlib |
| 910 | episode_day | 2025-02-27 | 5 | pvlib |
| 910 | episode_day | 2025-03-02 | 8 | pvlib |
| 910 | episode_day | 2025-03-13 | 11 | pvlib |
| 910 | episode_day | 2025-04-18 | 15 | pvlib |
| 910 | episode_day | 2025-04-24 | 10 | pvlib |
| 910 | episode_day | 2025-05-01 | 35 | pvlib |
| 910 | episode_day | 2025-05-02 | 34 | pvlib |
| 910 | episode_day | 2025-05-04 | 4 | pvlib |
| 910 | episode_day | 2025-05-18 | 26 | pvlib |
| 910 | episode_day | 2025-05-25 | 27 | pvlib |
| 910 | episode_day | 2025-06-25 | 22 | pvlib |
| 910 | episode_day | 2025-07-10 | 7 | pvlib |
| 910 | episode_day | 2025-07-24 | 20 | pvlib |
| 910 | episode_day | 2025-08-21 | 18 | pvlib |
| 910 | episode_day | 2025-09-03 | 29 | pvlib |
| 910 | episode_day | 2025-10-12 | 13 | pvlib |
| 910 | episode_day | 2025-10-13 | 4 | pvlib |
| 910 | episode_day | 2026-04-03 | 17 | pvlib |
| 910 | episode_day | 2026-04-07 | 14 | pvlib |
| 910 | episode_day | 2026-04-09 | 10 | pvlib |
| 910 | episode_day | 2026-04-10 | 5 | pvlib |
| 910 | episode_day | 2026-04-11 | 7 | pvlib |
| 910 | episode_day | 2026-04-27 | 9 | pvlib |
| 910 | episode_day | 2026-05-02 | 50 | pvlib |
| 910 | episode_day | 2026-05-04 | 12 | pvlib |
| 910 | episode_day | 2026-05-12 | 16 | pvlib |
| 910 | episode_day | 2026-06-19 | 32 | pvlib |
| 910 | episode_day | 2026-06-30 | 25 | pvlib |
| 910 | episode_day | 2026-07-01 | 5 | pvlib |
| 910 | episode_day | 2026-07-26 | 5 | pvlib |
| 910 | episode_day | 2026-07-27 | 17 | pvlib |
| 910 | full_day | 2025-03-30 | 92 | pvlib |
| 910 | full_day | 2025-10-26 | 100 | pvlib |
| 910 | full_day | 2026-01-04 | 96 | pvlib |
| 910 | full_day | 2026-01-08 | 96 | pvlib |
| 910 | full_day | 2026-04-05 | 96 | pvlib |
| 910 | full_day | 2026-04-06 | 96 | pvlib |
| 910 | full_day | 2026-04-25 | 96 | pvlib |
| 910 | full_day | 2026-04-26 | 96 | pvlib |
| 910 | full_day | 2026-05-01 | 96 | pvlib |
| 910 | full_day | 2026-05-03 | 96 | pvlib |
| 910 | full_month | 2026-03 | 2972 | pvlib |
| 910 | full_month | 2026-08 | 2976 | pvlib |
| 910 | before_local | 2025-07-29 before 11:00 | 44 | pvlib |
| 910 | restore_prior_exclusions | 2025-03-23 | 96 | measured_unless_selected |

## Explicit restoration

| Farm | Local day | Previously excluded | Restored to measured AC | New synthetic quarters |
|---|---|---:|---:|---:|
| 345 | 2026-04-09 | 15 | 8 | 7 |
| 910 | 2025-03-23 | 5 | 5 | 0 |

The CSV records final, nonoverlapping exclusion spans. End timestamps identify the final included quarter start.
Original exclusion reasons remain in `configs/data_quality/pv_curated_ghi_exclusions_2025_to_2026_08.csv`.

To reproduce using a fresh output path:

```bash
PYTHONPATH=src python scripts/prepare_reviewed_pv.py --config configs/pv_experiments/reviewed_ghi_user_decisions.yaml
```

## Reviewed target plots


Black is raw metered AC; green is the reviewed synthetic target.
Gray shading indicates retained prior exclusions; orange indicates restored quarters.
The blue dashed line is the frozen historical-GHI pvlib reference.

| Farm | Local delivery day | Plot |
|---|---|---|
| 1 | 2025-12-18 | [Open](../../../outputs/pv_reviewed_ghi_20261103/farm_1/plots/2025-12-18.png) |
| 1 | 2026-07-27 | [Open](../../../outputs/pv_reviewed_ghi_20261103/farm_1/plots/2026-07-27.png) |
| 345 | 2026-04-09 | [Open](../../../outputs/pv_reviewed_ghi_20261103/farm_345/plots/2026-04-09.png) |
| 345 | 2026-06-14 | [Open](../../../outputs/pv_reviewed_ghi_20261103/farm_345/plots/2026-06-14.png) |
| 345 | 2025-10-24 | [Open](../../../outputs/pv_reviewed_ghi_20261103/farm_345/plots/2025-10-24.png) |
| 901 | 2026-06-12 | [Open](../../../outputs/pv_reviewed_ghi_20261103/farm_901/plots/2026-06-12.png) |
| 910 | 2025-03-23 | [Open](../../../outputs/pv_reviewed_ghi_20261103/farm_910/plots/2025-03-23.png) |
| 910 | 2025-07-29 | [Open](../../../outputs/pv_reviewed_ghi_20261103/farm_910/plots/2025-07-29.png) |
| 910 | 2026-03-29 | [Open](../../../outputs/pv_reviewed_ghi_20261103/farm_910/plots/2026-03-29.png) |
| 910 | 2026-08-02 | [Open](../../../outputs/pv_reviewed_ghi_20261103/farm_910/plots/2026-08-02.png) |

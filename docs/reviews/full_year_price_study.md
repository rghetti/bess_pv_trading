# Full-year price forecasting study

This is a prepared, **not yet fitted**, follow-on study. It does not change the
running/completed historical search or its hourly DA results. The intended
runtime config is `configs/research/price_feature_selection_full_year.yaml`;
the existing hourly-product study remains separate.

## Date and information contract

- DA, IDA1, IDA2 and IDA3 each have their own quarter-hour model and own gates.
- Training eligibility runs from **1 October 2025 through 30 September 2026**
  inclusive. The intent is one annual cycle; data gaps and availability checks
  can still remove individual products or days. It is not rolling last-N-days.
- The draft sets **October 2026 as development validation** and **November
  2026 as later test**, with November incomplete until more data arrive. These
  boundaries are provisional and can be changed before a new snapshot is made.
  A month is short for selection; do not infer year-round generalization from it.
- The draft removes the former one-day outer embargo to include 30 September
  as requested. Training labels still must be published by the first evaluation
  decision time. The latest 20% of eligible training days (minimum seven) is
  the inner selection tail; there is no extra embargo at that boundary.
- Inner winners are refitted on all eligible training rows, and development
  validation selects feature subsets/families by WAPE, then MAE. Frozen winners
  are refitted on eligible training plus validation history for final test.
  The full historical period 2026 May–August is now training, not holdout.
- Each IDA predicts its spread to the latest cleared same-product auction and
  adds the anchor back. The forecasts are point values, not calibrated price
  scenarios. DA and earlier IDA prices are inputs only when their configured
  clearing clocks precede the relevant IDA gate.
- Confirmed cancellations use the preceding cleared auction for the same
  product when it is available. The precise flags, replacement features,
  historical fallbacks and retrospective notice assumption are described in
  [the full-year proxy workflow](full_year_price_proxy_workflow.md#how-auction-cancellations-change-the-features).

## Features, candidate models and cohorts

The [complete feature and model specification](price_search_protocol_and_features.md)
lists every price lag, day curve, external field, holiday, model family and
sampled parameter range. It describes the old study's split; the split above
supersedes that section for this draft. The new study keeps those model families
and ranges, and adds **27 shared holiday indicators**: holiday × local hour
for hours 00–23, day-before-holiday, day-after-holiday, and a weekday bridge
between a holiday and a weekend. They apply across Italian national holidays;
there are no special coefficients solely for 25 April or 1 May. These are
calendar inputs, so they can be computed without future market observations.

The cohorts are intentionally separate so missing Swiss monthly data does not
silently eliminate every usable recent observation:

| Cohort | Blocks required to define its common complete-case rows | Subsets |
| --- | --- | --- |
| `price_only` | none beyond mandatory price/calendar fields | 1 |
| `nord_gas` | NORD forecasts, gas | 4 |
| `regional_previous_day` | NORD forecasts, gas, France | 8 |
| `full_history` | NORD forecasts, gas, France, Switzerland, same-day realized flows | 32 |
| `swiss_commercial_2026` | all previous blocks plus Swiss commercial data | 64 |

For each cohort, **all subsets use the same complete-case rows within that
cohort**; scores across different cohorts can cover different days. The
full-history and commercial cohorts contain explicitly retrospective same-day
features. The Italy–Switzerland DA spread contains the target DA price and is
labeled outcome-derived rather than treated as an operational input. Previous-
day feature availability is assumed per the requested research convention, not
proven by every historical source record.

There are nine families, 20 sampled hyperparameter settings per family and
subset, and four products. The maximum is **78,480 inner fits before source and
coverage checks**. Eligible groups determine the actual number. A price-only
cohort uses only price/calendar/holiday inputs; it is not a proxy for an
exogenous-feature comparison. A maximum of five CPU workers is the configured
default; no GPU is used. Model fitting is not started by preparation.

## Data and source priority

`scripts/search_price_full_year.py` creates a fresh, immutable input snapshot
from the shared DuckDB and checksums each source. For DA products it retains GME
where present and fills absent products from Energy Charts. These remain distinct
providers in the selected rows. Energy Charts values are rounded to two decimal
places; the currently overlapping GME observations differ by at most €0.005/MWh.
For French actuals, the consolidated RTE source takes priority; missing intervals
can use the separately identified provisional real-time dataset. Source snapshot
IDs, original raw tables and overlap diagnostics are retained. No IDA values are
filled with DA prices.

The NORD forecast fetch needs an ENTSO-E token. The current catalog includes only
the part downloaded before interruption; refresh it before final preparation.
Swissgrid's physical-energy workbook is published with a monthly delay, so
missing October/November observations are not evidence of a failed downloader.
The previous-day gas index was manually imported through 10 November 2026; the
current public updater does not yet fetch IGI automatically. IDA3 for 10 November
was not in the supplied workbook. Those gaps remain explicit.

## Commands after the ENTSO-E download has finished

Never run two writers against the same embedded DuckDB. Use a fresh output
directory; preparation pins the data, so updating the catalog afterward does not
change an existing snapshot. Run from the repository root in tmux:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.search_price_full_year \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --output outputs/price_feature_search_full_year_2026_cancellations_v1 \
  --workers 5 --prepare-only
```

Inspect `prepared.json` for each cohort's `training_eligible`,
`evaluation_minimum_met` and train/validation/test row counts. The minimum flag
only checks a row-count threshold; it does not prove that the entire month is
covered. A Swiss cohort can be training-eligible while its later evaluation rows
are absent. The separate `raw_sources/manifest.json` and
`inputs/manifest.json` record source snapshots and fallback decisions. A preflight
from the interrupted ENTSO-E download showed zero full-feature validation/test
rows; do not mistake successful inner training for complete source coverage.

To start **inner training only**, run the same command with `--inner-only`
instead of `--prepare-only`. It fits and checkpoints candidates without selecting
on development validation or scoring the later test. It caps each fit's CPU
threads and uses no GPU.

An existing prepared snapshot never acquires later records automatically. When
the missing evaluation data have arrived, prepare a **fresh** output directory
and compare its coverage. The checkpoint transfer command verifies identical
scientific settings, software, candidates and the exact inner-training and inner-
validation feature/target rows before copying any fit:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.transfer_full_year_checkpoints \
  --source outputs/price_feature_search_full_year_2026_cancellations_v1 \
  --destination outputs/price_feature_search_full_year_later
```

The default only reports transferable checkpoints; repeat with `--apply` to copy
them. Changed training values or unavailable earlier cohorts are skipped. After
reviewing the later snapshot, run `scripts.search_price_full_year` against that
later output with `--run` to perform validation and final test. The runner refuses
that action if any training-eligible cohort has fewer than the configured minimum
validation or test rows; this check does not establish complete monthly coverage.

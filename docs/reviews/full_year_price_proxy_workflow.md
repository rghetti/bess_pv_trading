# Full-year search and retrospective 2026 price proxies

This workflow prepares a price-proxy baseline for the later 2026 battery
studies. It supersedes the old-snapshot Ridge-only diagnostic in
`price_in_sample_baseline_2026.md`; it does not change a simulator or run farm
optimization. Wait until the ENTSO-E updater has finished before taking the
snapshot from the shared DuckDB.

## Search

The search snapshots the refreshed market catalog, including new ENTSO-E,
GME, Energy Charts, RTE and Swiss data, with checksums and source identity.
The four target groups are native 15-minute DA, IDA1, IDA2 and IDA3. Their
training window is 1 October 2025 through 30 September 2026 inclusive, with
missing rows excluded. For every cohort, the latest 20% of eligible training
days is the inner selection tail; the remaining days fit each candidate.
The search saves each candidate result incrementally and can be resumed.

There are nine estimator families, 20 sampled parameter settings per family,
and multiple feature subsets within each of five matched cohorts. The maximum
is 78,480 candidate fits; `prepared.json` contains the actual number after
coverage checks. Five is the default maximum CPU-worker count, with one native
thread per worker. Trial objectives vary by model; all candidates are compared
on inner absolute-price WAPE and then MAE. This step does **not** perform the
outer October/November validation or test.

```bash
cd /home/rghetti/bess_pv_trading
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.search_price_full_year \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --output outputs/price_feature_search_full_year_2026_cancellations_v1 \
  --workers 5 --prepare-only
```

Review each group's cohort `counts` and `training_eligible` in
`outputs/price_feature_search_full_year_2026_cancellations_v1/prepared.json`. Different cohorts
can contain different dates. Then launch in tmux:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.search_price_full_year \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --output outputs/price_feature_search_full_year_2026_cancellations_v1 \
  --workers 5 --inner-only
```

The same command resumes completed checkpoints. Add `--retry-failed` to
re-attempt candidates that previously failed. A `partial` status lists failed
candidate counts; inspect their checkpoint errors and retry if appropriate.

## Retrospective refit and predictions

Once every candidate in the selected cohort has a completed or explicitly
failed checkpoint, the following command selects the best successful inner-tail
candidate per auction, refits it on **all eligible annual training rows**, and
writes every usable 2026 target row in the same frozen snapshot:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.refit_full_year_price_proxies --run
```

The default `full_history` cohort compares all 32 subsets of five external
blocks on identical training/inner-tail rows: NORD forecasts, gas, French
observations, Swiss physical/reservoir data, and same-day realized flows. Its
selected subset may or may not contain each block. Some candidate subsets use
same-day actual flows, so their use is retrospective. The more restrictive Swiss
commercial cohort is configured separately and is not chosen by default; its
Italian–Swiss DA spread contains the target Italian DA price. A different cohort
can be chosen in `configs/research/price_full_year_proxy_refit.yaml`, but compare
scores across cohorts only on explicitly matched dates.

The output directory `outputs/price_full_year_retrospective_proxies_2026_cancellations_v1`
contains one model and one Parquet price file per auction, daily coverage CSVs,
plots, and a manifest recording candidates, features, source fingerprint,
failed candidates and whether each predicted row was used in refitting.
Prediction rows only require the selected model's own inputs; training remains
on the common full-cohort rows used for selection. The input snapshot currently
ends on 1 December 2026 and live data may end earlier; the coverage files show
missing observations/predictions rather than fabricating future prices. The
descriptive pooled metrics include in-sample rows and must not be called a
forward-validation score.

IDA models use actual earlier cleared auction prices when predicting at each
IDA gate. Their saved outputs do not provide a forecast of all future IDAs at
the DA gate. They also are not yet a drop-in artifact for the existing
realtime-MPC forecast provider. The later integration must preserve auction
information timing and actual settlement and be validated for all farms; no
simulation framework changes are made here.

## How auction cancellations change the features

Documented GME cancellations are copied from current source snapshot metadata
into `inputs/manifest.json` along with the source snapshot ID and raw checksum.
The flag means **confirmed cancellation**, not simply an absent price. The
builder rejects a cancellation notice that conflicts with observed prices. The
publication time of the historical cancellation notice is not established;
using the notice is therefore a retrospective assumption.

Each IDA model predicts a spread relative to a clearing price for the **same
delivery product**, called its anchor. At the auction's gate, the builder uses
the latest earlier auction that cleared for that product:

| Target | If all earlier auctions cleared | If an earlier auction was cancelled |
| --- | --- | --- |
| IDA1 | DA | DA remains the anchor |
| IDA2 | IDA1 | If IDA1 was cancelled, use DA |
| IDA3 | IDA2 | If IDA2 was cancelled, use IDA1; if both IDA1 and IDA2 were cancelled, use DA |

This assumes the required earlier price is present and available at the
configured cutoff. An unexplained missing earlier price stays missing and can
exclude the row. If the target auction itself was cancelled, there is no target
price or target row for that auction on that day.

The anchor is not the only earlier-auction feature. For each earlier market,
the model also receives its cleared price at the target product and its six
configured neighboring product offsets. If that market was cancelled, those
features are filled from the nearest earlier market's corresponding product
and neighboring products. A binary `earlier_<market>_cancelled` feature is 1
for that cancellation and 0 otherwise. The anchor is selected after these
replacements; the fitted target is `actual IDA price - anchor`, and scoring
and output add the anchor back to the predicted spread.

For example, DA is €100/MWh, IDA1 is cancelled and IDA2 clears at €105/MWh.
The IDA3 inputs include DA = 100, the IDA1 feature filled with DA = 100,
`earlier_IDA1_cancelled = 1`, and IDA2 = 105. The anchor is IDA2 = 105; a
predicted spread of +2 gives an IDA3 prediction of €107/MWh. These earlier
prices are actual values available at the IDA3 gate under the configured
clock assumptions, not prices available to a forecast made at the DA gate.

Two additional groups of binary indicators cover historical features:

- `past_<market>_<lag>h_cancelled` marks a confirmed cancellation on the
  requested previous-auction lag, which is 24 or 48 elapsed hours. If that
  price is missing because of the cancellation, the builder searches the
  corresponding earlier daily product times for a published price, up to
  `cancellation_fallback_days` (14) days. If none is available, it stays
  missing.
- `own_curve_<lag>d_cancelled` marks a confirmed cancellation on the model's
  own D-1, D-2 or D-7 historical curve. The builder searches backward up to
  14 days for a fully available earlier daily curve. It uses one complete
  curve, rather than joining hours from different fallback days; otherwise
  the curve stays missing.

The pre-existing own-price lag features already use their configured historical
fallback when a requested lag is unavailable; that behavior is separate from
these cancellation flags. In every case, the source price's configured
publication clock must precede the decision cutoff.

The cancelled day can still remove rows if another required input is missing,
and a missing source record without a cancellation notice is not classified as
cancelled. In the prepared snapshot, the confirmed IDA1 cancellation on
10 December 2025 retains all 96 IDA2 and 48 IDA3 native targets in the
`full_history` training cohort. The coverage CSV labels confirmed cancellations
separately from unexplained missing or partial days.

The earlier prepared snapshot has a different feature contract and cannot be
reused for this cancellation-aware search.

## Inspect a saved DA winner before the search finishes

The inner-search checkpoints store aggregated scores, not per-day predictions.
To inspect the current best completed DA candidate on one matched cohort and
compare its expanded holiday features with the original holiday flag, run:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.diagnose_price_holiday_winner \
  --search outputs/price_feature_search_full_year_2026_cancellations_v2 \
  --cohort full_history --run
```

It performs two serial one-thread CPU fits using the same selected candidate,
hyperparameters and chronological inner-training rows. One fit uses all 27
expanded holiday indicators; the other removes those indicators while retaining
the original national-holiday flag. It writes `daily_mae.csv`,
`predictions.parquet` and `summary.json` in
`outputs/price_holiday_winner_diagnostic`. The current winner may change while
the full search runs, and the output directory is not overwritten. Use
`--cohort swiss_commercial_2026 --output <new-directory>` to examine that
cohort instead; its matched dates differ from `full_history`.

Inner-validation scores were used to select the candidate and are exploratory.
The default 1 May 2026 diagnostic is scored on rows that were used to train
both fitted models, so its error and holiday-feature comparison are in-sample;
they do not establish performance on a new holiday year. No holiday-specific
hyperparameter retuning is performed. The diagnostic does not modify or lock
the running search's immutable matrices or atomic trial checkpoints.

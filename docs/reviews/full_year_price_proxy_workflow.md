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

Documented GME cancellations are copied from current source snapshot metadata
into the pinned input manifest. A cancelled preceding auction's features and
spread anchor use the nearest earlier cleared auction, with explicit binary
cancellation flags. When lagged other-auction prices or own-auction daily curves
refer to a cancelled auction, the feature builder searches up to the configured
14 earlier days for a published value or complete curve and adds a flag. If no
fallback exists, the input remains missing. An unexplained data gap is never
treated as a cancellation. Cancellation-notice availability is retrospective;
the source does not establish the historical publication clock of the notice.
The old prepared snapshot has a different feature contract and cannot be reused.

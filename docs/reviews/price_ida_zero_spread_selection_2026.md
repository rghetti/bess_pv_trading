# Explicit cleared-price baseline in IDA model selection

This search includes a guarded `zero_spread` candidate: it predicts zero
residual and therefore returns the configured cleared-auction anchor. The
search rejects the candidate if any feature set lacks an anchor. A unit test
checks both the returned price and rejection of unanchored use. The candidate
was ranked alongside price-only ridge, Lasso and Elastic Net candidates with
full, 180-day and 365-day histories, using DA and latest-cleared spread anchors.
Ranking used February–April 2026 MAE, followed by weekly refits on available
September–October products. Both periods had been inspected previously, and
historical publication clocks are partly assumed.

Early ranking selected the latest-cleared-price baseline itself for IDA2 and
a full-history ridge model with the latest-cleared anchor for IDA3. Later
results are:

| Auction | Early-selected method | Products | MAE EUR/MWh | WAPE | Latest-cleared baseline MAE EUR/MWh | Baseline WAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| IDA2 | Prior IDA1 price | 4,704 | 9.939 | 4.640% | 9.939 | 4.640% |
| IDA3 | Ridge spread to IDA2 | 2,400 | 14.964 | 6.997% | 14.998 | 7.013% |

The saved baseline predictions equal the independently verified anchor
features on identical timestamps. All 64 early and six later fits completed
without recorded failures. The IDA3 difference is small and does not establish
an improvement; prior paired day-bootstrap intervals include zero. Exact
candidates, feature selection, coverage and lineage are preserved in
`price_model_search_ida_zero_spread_rank.yaml`,
`price_model_rolling_ida_zero_spread_rank.yaml` and their manifests. This does
not claim an exhaustive model search or an untouched prospective evaluation.

## IDA1 baseline-inclusive search

A separate IDA1 grid ranked the DA-price baseline against ridge, Lasso, Elastic
Net and time-of-day interaction candidates with full, 180-day and 365-day
histories. February–April MAE selected a full-history interaction Elastic Net.
On the later 4,800 September–October products it scored 8.782 EUR/MWh MAE and
4.116% WAPE, versus 8.885 EUR/MWh and 4.164% for the DA-price baseline. The
paired predictions have identical product timestamps and targets. A 5,000-
resample independent-day bootstrap with seed 173 gives a 95% interval of
-0.121 to +0.332 EUR/MWh for the model's MAE gain; it omits serial dependence
and model-selection uncertainty. The small observed improvement is not
conclusive. All 25 early and two later fits completed without recorded failures
or convergence warnings. Both periods were previously examined, publication
clocks are partly assumed, and the missing-day coverage is documented in
`price_price_only_sep_oct_2026.md`. The search and rolling configs are
`price_model_search_ida1_zero_shape_windows.yaml` and
`price_model_rolling_ida1_zero_shape_windows.yaml`.

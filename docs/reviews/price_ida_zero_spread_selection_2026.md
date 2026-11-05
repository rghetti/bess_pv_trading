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

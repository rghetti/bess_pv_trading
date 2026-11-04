# Late-2025 IDA validation coverage

The preflight for `configs/research/price_model_rolling_ida_2025_terna.yaml`
examined delivery days from 3 September through 30 November 2025. It retained
58 complete shared days for IDA1 and IDA2 and 59 for IDA3, across the base
and Terna-load feature sets. Scores from that run describe those retained days,
not the whole interval. Preflight checks inputs; it does not fit or score models.

| Auction | Excluded days | Days with missing target products | Days missing prior-day Terna load |
|---|---:|---:|---:|
| IDA1 | 31 | 2 | 29 |
| IDA2 | 31 | 0 | 30 |
| IDA3 | 30 | 0 | 29 |

The categories can overlap. Of the days missing Terna load, 28 for IDA1 and
29 for each of IDA2 and IDA3 fall in November; the remainder fall in October.
Some IDA2 and IDA3 days also lack earlier-auction price features. No missing
input was replaced by zero. The catalog feature is a historical prior-day
load series with an assumed release lag, so the comparison remains
retrospective rather than proof of real-time data availability.

The configured run records a day-by-day breakdown in `incomplete_days` in
its preflight output and run manifest. The separately queued evaluation
without the Terna feature has its own complete-day sample; comparisons between
the two evaluation runs must first align their actual delivery products.

## Completed matched rolling evaluation

The evaluation completed all 21 selected and reference fits without failures.
For each market, the following row is the top candidate selected on the
June–August matched validation period, before the later evaluation. The
Nine weekly refits use a one-day gap and released prior labels.

| Auction | Selected feature set and candidate | Later WAPE | Later MAE €/MWh | Relative MAE | Products |
|---|---|---:|---:|---:|---:|
| IDA1 | `spread_cleared_fundamentals`, `hist_l1` | 6.54% | 7.19 | 0.510 | 5,568 |
| IDA2 | `cleared_terna_load`, `catboost_shallow` | 6.81% | 7.55 | 0.486 | 5,568 |
| IDA3 | `cleared_fundamentals`, `lasso_1` | 9.95% | 11.19 | 0.525 | 2,832 |

The top selected candidates for IDA1 and IDA3 do not include the Terna
feature. On the same later products, the paired `ridge_20` reference models
with and without Terna load differ in MAE by between -0.009 and +0.001 €/MWh
across the direct and spread feature sets. This reference-model result does
not prove that every model is insensitive to the feature. WAPE uses the sum
of absolute observed prices as its denominator; relative MAE is measured
against the seasonal-lag reference on identical products.

The completed manifest and forecasts are in
`outputs/energy_price_forecasts/nord_ida_2025_terna_selected_rolling_holdout_v1/`.

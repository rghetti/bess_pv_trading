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

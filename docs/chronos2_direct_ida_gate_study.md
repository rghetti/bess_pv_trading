# Direct Chronos-2 IDA forecasts at multiple decision gates

This prepared study predicts each IDA's **price directly**, using the frozen
Chronos-2 checkpoint. No spread target, historical spread covariate, same-day
cleared auction price or predicted DA price is an inference input. Earlier
clearings may appear only as separately labelled scoring references, and only
if published at that decision gate. Target observations are retained solely
for retrospective scoring.

The configured pilot covers delivery dates 1–7 October 2025. Each of these nine
decision/target combinations has its own request bundle and report:

| Decision gate | Direct target auctions |
| --- | --- |
| DA | IDA1, IDA2, IDA3 |
| IDA1 | IDA1, IDA2, IDA3 |
| IDA2 | IDA2, IDA3 |
| IDA3 | IDA3 |

Input price history uses the target auction's native products for 28 days,
shortened to a complete recent suffix of at least seven days if required.
Every historical price must be published at the **decision** gate. IDA3
preserves the existing afternoon-only concatenation and associated compressed
overnight gaps. Input bundles are conditioned on the presence of target-day
labels for scoring, so cancellation days without a target price are excluded
from this research evaluation; this is not yet a live prediction provider.

No explicit weekday or Italian holiday indicators are provided. The D load
forecast is a known-future covariate at every gate. Actual NORD load, solar and onshore wind generation
are used in their historical context up to the assumed availability cutoff;
missing or unavailable historical intervals use the eligible forecast first,
then the configured short-gap interpolation/zero fallback. Forecasts for D−1
are eligible at the D−1 gates under the specified historical assumptions.
Solar, onshore wind and scheduled aggregate generation forecasts for D are
past-only at DA and IDA1; their D horizons are permitted at IDA2 and IDA3
provided their publication eligibility and coverage checks pass. Their
earlier-gate exclusion also applies if an earlier timestamp exists in the source.

Gas, reservoir hydro and run-of-river hydro are past-only actual series. The
unpublished tail after the availability boundary is filled with zeros, including
when it is shorter than two hours. Other missing runs follow the agreed
interpolation/zero fallback; zero is not a measured value. Native missing-value
masking is not yet integrated. Pumped hydro is excluded because production and
consumption directions need a separate interpretation. Offshore wind is
excluded because the pinned snapshot has no records. IDA3's delivery-day
morning actuals are not included in this aligned afternoon-only study.

All auction decisions occur at the configured project cutoff, 15 minutes before
gate closure. Historical publication timestamps and issue vintages in the
ENTSO-E data are not generally verified. Actual data and forecast values use
latest available revisions, so the results are retrospective even though the
eligibility rules follow the cutoff. The [publication research](chronos2_nord_realized_price_inputs.md)
explains the one-hour actual lag and forecast deadlines. Forecast overlap with
pretraining cannot be ruled out for all past data.

The price snapshot and NORD actuals snapshot are specified in
`configs/research/chronos2_direct_ida_gates.yaml`. If the latter is absent, pin
it only after the downloader has stopped:

```bash
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.pin_nord_realized_inputs --output outputs/price_realized_snapshot/chronos2/merged
```

The following command **prepares inputs only**, using CPU processes. Preparation
does not run Chronos or alter the database:

```bash
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_chronos2_direct_ida_gates --prepare-only --prepare-workers 3
```

After reviewing coverage and approving the pilot, replace `--prepare-only` with
`--run` and choose `--devices`, `--batch-size`, `--threads`, and
`--report-workers`. The runner saves independent model checkpoints and reports
for each gate/auction pair. Source checksums, settings, gate definitions,
builder code and pinned model identity are checked before resuming. No inference
was run during implementation.

The no-calendar pilot writes to `outputs/chronos2_direct_ida_gates_pilot_no_calendar`.
The previous calendar-enabled pilot has its own output directory; its checkpoints
cannot be reused for these inputs.

# Direct Chronos-2 auction forecasts at multiple decision gates

This prepared study predicts DA and IDA **prices directly**, using the frozen
Chronos-2 checkpoint. No spread target, historical spread covariate, same-day
cleared auction price or predicted DA price is an inference input. Earlier
clearings may appear only as separately labelled scoring references, and only
if published at that decision gate. Target observations are retained solely
for retrospective scoring.

The pilot covers the delivery dates in the study configuration. Each of these ten
decision/target combinations has its own request bundle and report:

| Decision gate | Direct target auctions |
| --- | --- |
| DA | DA, IDA1, IDA2, IDA3 |
| IDA1 | IDA1, IDA2, IDA3 |
| IDA2 | IDA2, IDA3 |
| IDA3 | IDA3 |

Input price history uses the target auction's products over 28 calendar days.
Cancelled or otherwise unavailable historical IDA prices remain in position
as zero placeholders with `context_observed_mask=0`; observed prices have mask
1, including genuine zero prices. The Chronos worker converts masked positions
to NaN for Chronos-2's native missing-value mask. Load and generation covariates
retain their own values aligned to these timestamps. There is no seven-day
waiting period after cancellation; at least one observed historical price
product is required. The DA price-history adapter retains its original policy.
Pre-October 2025 hourly DA prices are repeated across four quarter-hours;
DA targets from October onward retain their native quarter-hour products.
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
masking is not applied to these gas or hydro inputs. Pumped hydro is excluded because production and
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
`configs/research/chronos2_direct_ida_gates_masked.yaml`. If the latter is absent, pin
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

The masked no-calendar pilot writes to `outputs/chronos2_direct_ida_gates_masked`.
The earlier no-calendar and calendar-enabled pilots retain their own outputs;
their checkpoints cannot be reused for these changed inputs and worker code.

Use `--gate DA --target DA` to prepare or run only the DA-price forecast. The
DA-only filter does not modify the configured list of other gate/target jobs.

Once all configured gates for an IDA target have complete reports, reporting
automatically creates `gate_comparison/<auction>/` with matched-product scores,
overall gate MAE, daily MAE, hourly MAE, and signed-error curves for the first
configured days of each month. Only identical delivery products are compared;
`coverage.csv` records per-gate exclusions. Development and reserved periods
are separate. Target labels, model settings, feature configuration and input
snapshot identities must agree across gates. If gates are missing, automatic
reporting waits; the standalone command can also generate a clearly marked
partial comparison from at least two completed gates.

To regenerate plots from an existing completed study without running inference:

```bash
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.report_chronos_gate_comparison --output outputs/chronos2_direct_ida_gates_pilot
```

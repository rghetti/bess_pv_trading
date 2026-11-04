# Proposed four-cycle reviewed-PV HPO

This is a plan, not a completed or running experiment. The user requested a
48-hour target window starting at each weather cycle's issue time. The source
contains cycles at 00, 06, 12 and 18 UTC, with hourly leads from 0 to 48 hours.
The configured weather release is six hours after issue; this lag is an
assumption, not a verified publication timestamp.

## Training examples and availability

Pair each cycle's forecast weather and the historical measured/interpolated
GHI with the reviewed AC target at 15-minute resolution over the following 48
hours. Validation, test and inference use forecast GHI. The
same physical production quarter can appear with several forecast vintages;
these are correlated examples, not independent observations. Keep all copies
of each target in its original chronological split: 2025 training,
January–April 2026 validation, and May–August 2026 testing, with the existing
availability embargo. Fit all imputation and scaling on training examples
only. Preserve curated measured and approved synthetic targets; replace retained
daylight exclusions with the stored historical-GHI pvlib reference, as the
user requested. Forecasts remain bounded by AC capacity and zero in the
configured monthly nighttime window.

The first six hours of each requested issue-time window precede the assumed
release of that weather cycle. They can be used as retrospective training
examples if requested, but their scores must be labelled hindsight and must not
be presented as available forecasts. Use only intervals at or after cycle
availability for causal model selection. Report issue cycle and lead time;
compare predictions on matching target quarters. Select 3,000 eligible
physical nighttime quarters per farm, spread evenly across the training year,
before expanding them into the available cycle views.

The prior search's 21 extended features each had a missingness indicator, but
all indicators were zero in every cached row for all four farms. They did not
provide observed missingness information in that search. As requested, the new
search has no missingness indicators. Required core weather must be present
for a labeled training or scoring quarter; other missing weather is imputed
from training-only medians. Missingness in sequence context is reported
separately; do not infer that unavailable future features were historically
available.

## Candidate search and reporting

The search uses the five neural families from the preceding study:
MLP, TCN, bidirectional GRU, bidirectional LSTM and Transformer. It tests core
and extended features, 64 sampled configurations per group, widths 128, 256
and 512, and depths two through four, using 42 single-threaded CPU workers.
Batch sizes are 2, 4 or 8 to control memory. Statistical families are excluded.
Neural models use configured MSE or Huber training losses and validation early
stopping (maximum 150 epochs, patience 15); there is no post-hoc calibration. Selection is by
day-balanced validation daylight nRMSE, with train,
validation and test metrics separated by target source. Freeze all selected
configurations before test scoring. Report top three per farm, MAE, RMSE,
bias, cycle/lead and measured/synthetic breakdowns, and simple zero and
training-only constant baselines. Compare family winners to the model selected
on validation. Do not interpret
overlapping forecasts as independent measurements when estimating
uncertainty; use day-based blocks.

The historical 2026 period has already been reviewed, and 910 contains
substantial synthetic targets. The four-cycle training views differ from the
current single-cycle models, so this is a new
experiment rather than a continuation of the previous one. Dataset coverage,
missingness, sample counts, timing masks, runtime, disk use and the exact
nighttime sample counts must be preflighted before presenting a concrete
launch for explicit approval. No training is started by this document.

## Checked data and launch

The no-training preflight found the same counts per farm: 195,264 training
pairs, including 24,000 views of the 3,000 physical nighttime targets;
47,656 validation pairs and 56,826 test pairs. The validation and test counts
cover eligible daylight intervals after the configured six-hour release lag.
Each farm's supervised rows have no missing features. Per farm, 672 training
and 532 test sequence-context rows at dataset boundaries have at least one
missing weather feature; these are imputed from training-only medians without
binary indicators. There are no such missing validation-context rows.

There are 2,560 HPO fits, then 40 family/feature winner refits and up to eight
additional refits to retain the top three configurations per farm. A synthetic
single-process memory check of a width-512, depth-4 Transformer with batch 8
and 192 steps peaked around 1.15 GiB. This is not a full-run memory estimate;
42 workers, optimizer state, data caches and other users add overhead. The
configuration requires at least 64 GiB of available system memory before
launch; this is a conservative configurable guard, not a guarantee against OOM.

Core inputs are historical GHI for training (forecast GHI for validation and
test), forecast direct horizontal radiation, 2 m temperature, 10 m wind u/v,
cyclical local hour, day-of-year and month (sine/cosine pairs), forecast lead
time and UTC cycle hour (sine/cosine). The extended set adds dew point, total,
low, middle and high cloud cover, precipitation, snow water equivalent,
visibility and 100 m wind u/v. Neither set includes binary missingness flags.

```bash
cd /home/rghetti/bess_pv_trading
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_hpo.py --config configs/pv_experiments/reviewed_four_cycle_neural_hpo.yaml --check-only
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_hpo.py --config configs/pv_experiments/reviewed_four_cycle_neural_hpo.yaml --run
```

The second command starts training and should be run by the user in tmux.
Outputs are grouped under `outputs/pv_reviewed_hpo/four_cycle_neural_night3000_v1/`
and `docs/results/pv_reviewed_hpo/four_cycle_neural_night3000_v1/`. Completed
trials can resume only when code, data, configuration and dependencies retain
the same run identity. Older test results are retrospective and include
synthetic labels; the first six hours of each training view were unavailable
at the configured forecast-release time.

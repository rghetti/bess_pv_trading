# Reviewed weather-to-PV model search

This is a run plan and launch guide. The search is started only with `--run`;
`--check-only` reads and validates inputs without fitting predictive models or
creating the run output. The configuration is
`configs/pv_experiments/reviewed_ghi_hpo.yaml`.

## Data and forecast contract

The source is `data/farms/curated/reviewed_ghi_2025_to_2026_08/`, with a
`modeling_views.parquet` for each farm. Targets are `model_target_mw`, including
the reviewed synthetic replacements and zeroed episodes. Prior exclusions stay
excluded. For training, 3,000 eligible nighttime targets per farm are selected
deterministically across the calendar year, with approximately equal counts per
day and evenly spaced quarters within each day. Shortfalls on excluded or
embargoed days are redistributed across days with eligible targets. The
curated dataset itself is unchanged. A row without a power label cannot
supervise a model. At the dataset boundary, missing nighttime GHI is set to
zero if that quarter is selected for training.

The original split is retained: 2025 training, January–April 2026 validation,
and May–August 2026 testing, with the existing boundary embargo. Training uses
historical GHI, interpolated where already provided by the curated dataset.
Validation and test use forecast GHI; no observed validation/test GHI is used as
a model input. All splits use weather from the preceding delivery day's 00 UTC
cycle. The configured six-hour release delay is an assumption; it does not
prove historical availability at an auction cutoff. Forecast hourly mean GHI
and BHI are end-labelled and repeated over the four constituent quarter-hours,
preserving the hourly mean. Instantaneous weather variables are interpolated
to quarter-hour midpoints; hourly precipitation is divided among quarters.

Every prediction is a 15-minute AC-power value. Training includes the selected
nighttime labels; performance is computed only inside the existing monthly
daylight windows. Missing weather is imputed using statistics fitted on the
training split only, with missingness indicators. Forecast GHI must be present
for validation/test scoring. The 2026 period has already been inspected, and
some evaluation targets are synthetic, including the requested farm 910
replacements. Results are retrospective and errors against synthetic labels do
not measure accuracy against independent power observations.

## Models and selection

Core features are GHI, forecast BHI, temperature, 10 m wind components, and
cyclic local-time and season features. Extended features add forecast cloud
layers, dew point, precipitation, snow water equivalent, visibility and 100 m
wind components. The exact lists, search ranges and preprocessors are in the
YAML. Physical-model outputs are not candidate models or input features; the
reviewed targets themselves may be pvlib-derived.

For each farm and feature set, 64 reproducibly sampled configurations are
evaluated for each of 16 families: ridge, elastic net, Huber, spline ridge,
polynomial ridge, random Fourier feature ridge, extra trees, random forest,
histogram gradient boosting, LightGBM, CatBoost, MLP, TCN, bidirectional GRU,
bidirectional LSTM and Transformer. This is 8,192 fits, followed by 128
deterministic refits of the best configuration in each farm/family/feature
group, plus up to eight additional refits so that the three best individual
configurations per farm are all retained. The search is sampled, not an
exhaustive Cartesian product.

Models train on power normalized by farm AC capacity, with equal weight per
training day. Losses include squared, absolute or Huber loss according to
family and configuration. Neural candidates use AdamW, sampled MSE or Huber
loss (Huber delta 0.05 of AC capacity), up to 150 epochs and patience 15; their
early stopping uses validation daylight error. Outputs are bounded between
zero and rated AC power. There is no post-hoc calibration. Configuration
selection uses day-balanced validation RMSE normalized by AC capacity. Test
scores are generated only after every winner has been frozen. Zero and the
training-day-balanced mean are reported as baselines. Reports also separate
measured and synthetic targets, months and GHI interpolation provenance, and
include paired seven-day-block bootstrap comparisons on test days. These
intervals do not remove bias from the previously inspected period or from
synthetic labels.

## Launch and outputs

Run the read-only preflight if desired:

```bash
cd /home/rghetti/bess_pv_trading
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_hpo.py --config configs/pv_experiments/reviewed_ghi_hpo.yaml --check-only
```

To start the search in your tmux session:

```bash
cd /home/rghetti/bess_pv_trading
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_hpo.py --config configs/pv_experiments/reviewed_ghi_hpo.yaml --run
```

The search uses 30 CPU worker processes and one numerical-library thread per
worker. It may require substantial time and memory, particularly for the tree
and neural families; no runtime is assumed. It does not require a GPU or
install packages. Progress appears in the terminal and in
`outputs/pv_reviewed_hpo/historical_ghi_all_farms_night3000_v1/run.log`, with a machine
readable `status.json` alongside it. A lock prevents concurrent runs. Completed
trials and refits can be resumed with the same command if the data, code,
configuration and dependency identity are unchanged. If a process terminates
without releasing its lock, first verify that no run is active before removing
the stale `run.lock`.

Models, preprocessing, checksums, forecasts and daily plots are stored under
`outputs/pv_reviewed_hpo/historical_ghi_all_farms_night3000_v1/`. Tables and a report are
written to `docs/results/pv_reviewed_hpo/historical_ghi_all_farms_night3000_v1/`.
Individual HPO trials retain validation records; selected family/feature models
and any additional top-three configurations are retained as model bundles.
The report shows the top three configurations per farm, ranked by validation,
with their validation and test metrics. `top_three_metrics.csv` includes train,
validation and test results, separated by measured and synthetic targets.
The inference adapter in
`src/forecasting/inference/pv_reviewed_hpo.py` requires the forecast vintage,
availability and decision cutoff explicitly and rejects an unavailable cycle.

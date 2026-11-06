# Four-cycle statistical PV model search

## Scope

This separate study compares classical regression and tree models with the
completed four-cycle neural search. It uses the same curated AC labels,
training/validation/test dates, 15-minute forecast-target pairs, cycle set,
48-hour issue horizon, historical-GHI training inputs, forecast-GHI validation
and test inputs, and monthly nighttime clipping policy. The current selected
neural models are the same-cycle baseline. The previous one-cycle HPO scores
remain contextual because they used different forecast input rows.

Each statistical model treats a forecast-target pair as one row; it does not
consume the 48-hour issue as a sequence. It uses the requested core or extended
weather and calendar features, with no missing-feature indicators. Nighttime
targets remain in training, including the 3,000 physical quarter-hours sampled
per farm before the samples are duplicated across forecast cycles.

## Search space and selection

The search contains 11 families: Ridge, Elastic Net, Huber regression,
spline Ridge, degree-two polynomial Ridge, RBF Ridge, ExtraTrees, Random
Forest, histogram gradient boosting, LightGBM, and CatBoost. It samples 64
configurations per family and feature set for each of four farms:

`11 families × 64 configurations × 2 feature sets × 4 farms = 5,632 HPO fits.`

Validation day-balanced daylight nRMSE selects models. Test scores are
reported only after selection is frozen and are retrospective because this
2026 period has already been inspected in prior work. Reports separate
measured and pvlib-synthetic labels, month, forecast cycle, lead time, bias,
nMAE, and nRMSE. Night predictions are zeroed before scoring.

## Compute and command

CPU estimators and numerical libraries are limited to one thread per fit, with
10 concurrent workers. The preflight found 195,264 train pairs, 47,656
validation pairs and 56,826 test pairs per farm. The preceding one-cycle
statistical search used about 7.5 CPU-hours for 1,408 fits on farm 1 with about
21,000 training rows. Scaling that measured CPU time by the larger four-cycle
training set suggests roughly 24–36 wall-clock hours on ten workers. Tree
methods, memory pressure and shared CPU load drive most of the uncertainty;
this is a planning estimate, not a benchmark of the new dataset.

Run the preflight first; it checks the dataset, trial grid, dependencies and
available memory without fitting models:

```bash
cd /home/rghetti/bess_pv_trading
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_hpo.py --config configs/pv_experiments/reviewed_four_cycle_statistical_hpo.yaml --check-only
```

After reviewing that preflight and approving the experiment, launch:

```bash
cd /home/rghetti/bess_pv_trading
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_hpo.py --config configs/pv_experiments/reviewed_four_cycle_statistical_hpo.yaml --run
```

The separate run and report are stored under
`outputs/pv_reviewed_hpo/four_cycle_statistical_night3000_v1/` and
`docs/results/pv_reviewed_hpo/four_cycle_statistical_night3000_v1/`.

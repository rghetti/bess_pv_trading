# Run the expanded price search on a Slurm cluster

The earlier tree search is complete: **500 inner fits, 25 finalist refits, zero
failures**. It used LightGBM, CatBoost, XGBoost, histogram gradient boosting and
ExtraTrees on five prepared datasets. Its best development WAPEs were 8.60%
for hourly DA, 11.06% for quarter-hour DA, and 4.66/5.35/9.16% for IDA1/2/3.
Those completed experiments retain their original local-weather/fundamental
inputs and are comparisons. Their files under `/tmp/price_tree_hpo` are not
dependencies of the expanded search.

The expanded study has a versioned runner, parameter spaces, DuckDB input
builder and a single-node [64-CPU batch script](../../scripts/slurm/price_search_64.sbatch).
Preparing matrices and testing software do not mean that its large model search
has finished. A research run is finished only when `status.json` says
`status: complete` and all expected candidate/refit outputs exist.

## Files to transfer

Use this branch's committed code and the updated catalog from
`/home/rghetti/bess_pv_trading_features/data/api/market.duckdb`.
Do not substitute the older database in this checkout's own `data/api/`.
Git does not contain the large database or generated results. Fetch this branch
on the cluster after publishing the local commits to the GitHub remote.

No farm Zarr, static farm parameters, PV model bundles, saved price matrices,
Gurobi installation or Gurobi license are needed. Native data, original source
blobs and source snapshot checksums are already inside the updated DuckDB.
The new search selects no farm weather or local Open-Meteo cycles.

## Environment and launch

From the repository root, create the pinned environment once using the
cluster's allowed environment-installation procedure:

```bash
conda env create -f environment-price-search.yml
conda activate bess-price
```

The batch job uses the Python executable supplied by the submission environment.
Set its path explicitly so the compute node uses the intended environment:

```bash
export PRICE_SEARCH_PYTHON="$(command -v python)"
export PRICE_DB=/absolute/path/to/market.duckdb
export PRICE_SEARCH_OUTPUT=/absolute/path/to/results/price_feature_search
sbatch scripts/slurm/price_search_64.sbatch
```

Add `--account=...` and `--partition=...` to `sbatch` when required by your
cluster. The script requests **one node, one task, 64 CPUs, 128 GB RAM and
48 hours**. Slurm command-line options can override the memory and time limits.
No GPU is requested. The Python coordinator launches 64 independent fit
processes; each estimator and BLAS/OpenMP library is limited to one thread.
The script never installs packages or fetches API data during a compute job.

Submit from the repository root. Alternatively set `PRICE_SEARCH_ROOT` to its
absolute path. The configuration can be changed with `PRICE_SEARCH_CONFIG`.
The job reads DuckDB once, prepares immutable source tables and memory-mapped
arrays, releases the database, then starts model fits. Workers do not open
DuckDB independently.

To inspect the plan or build all matrices without training models:

```bash
python -m scripts.search_price_models
python -m scripts.search_price_models \
  --database "$PRICE_DB" --output "$PRICE_SEARCH_OUTPUT" --prepare-only
```

To run directly on the current workstation with 30 workers:

```bash
conda activate bess
python -m scripts.search_price_models \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --output outputs/price_feature_search_full \
  --workers 30 --run
```

This launches the price search as a separate job from any PV HPO already running.

## Scope and selection protocol

The default [study configuration](../../configs/research/price_feature_selection.yaml)
defines eight independent groups: hourly and quarter-hour DA, IDA1, IDA2 and
IDA3. Historical hourly models use their original native labels. Hourly prices
are carried within their original interval only when aligning historical or
earlier-cleared price **features**; they are never manufactured into observed
quarter-hour training labels.

| Product groups | Training window | Development validation | Final holdout |
|---|---|---|---|
| Hourly DA | 2024 | January–June 2025 | July–September 2025 |
| Hourly IDA1/2/3 | January–June 2024 | July–September 2024 | October–December 2024 |
| Quarter-hour DA | October 2025–January 2026 | February–April 2026 | May–August 2026 |
| Quarter-hour IDA1/2/3 | January 2025–January 2026 | February–April 2026 | May–August 2026 |

Dates are exclusive at their configured end. A one-day training embargo and
target-label publication checks apply before the next evaluated period. Native
resolution and auction clocks are declared in YAML; historical publication
clocks are study assumptions where the catalog lacks recorded evidence.

Baseline inputs include recent own-product prices, previous local-day hourly
curves, previous-day other-auction prices, calendar terms, an Italian national
holiday flag including Easter, and available earlier-auction prices and nearby
delivery products. Each auction has its own configured decision gate. IDA
models fit a spread around the latest cleared auction price; predictions and
all evaluation metrics are converted back to absolute prices.
IDA3 uses whole-day own-price lags so its 12:00 and 23:45 delivery products are
not dropped simply because the same auction has no overnight products.

The feature selection enumerates every subset of these blocks:

1. NORD load, total generation, solar and wind forecasts.
2. Previous-day and previous-week PSV gas prices.
3. Previous-day French nuclear/hydro generation, balance and commercial exchange with Italy.
4. Previous-day Swiss generation/consumption and physical flows to Italy/France, plus lagged reservoir levels.
5. Same-delivery-day Swiss physical flows and generation/consumption, and French exchange observations.
6. Same-delivery-day Swiss commercial schedules, intraday NTC and DA spot-price spreads.

The full-history cohort evaluates all **32** subsets of blocks 1–5. A separate
cohort evaluates all **64** subsets of blocks 1–6 on the common observations
covered by the 2026 commercial dataset. Each cohort uses identical complete-case
training, validation and test rows across its subsets. Historical groups without
enough observations for the 2026 cohort are explicitly skipped in `prepared.json`.
Source and historical-price gaps are recorded, not zero filled.

Blocks 5–6 are retrospective. For DA, the same-day Italy–Switzerland DA spread
contains the target Italian clearing price, so those rows are additionally
tagged `outcome_derived_da_price`. Selection and holdout tables retain separate
availability classes. They are explanatory comparisons, not evidence of an
operationally available DA predictor. Previous-day forecasts are assumed usable
at every decision time under the user's requested availability convention.

Nine model families are searched: Ridge, ElasticNet, RandomForest, ExtraTrees,
histogram gradient boosting, LightGBM, CatBoost, XGBoost and a CPU MLP. Every
eligible group/cohort/subset gets 20 fixed-seed parameter samples per family.
Feature subsets are exhaustive; the declared parameter distributions are
sampled, not exhaustively enumerated. The preparation report gives the exact
eligible fit count before training starts. This is a substantial CPU job, so
the 48-hour allocation is a resumable job limit rather than a completion estimate.

The updated catalog passes preparation for all eight groups and 12 eligible
cohorts: **512 feature subsets × 9 families × 20 samples = 92,160 inner fits**,
followed by 512 validation refits and the selected later-period refits.

Hyperparameters and model family are selected using the latest 20% of eligible
training days (at least seven days), minimizing absolute-price WAPE and then
MAE. Ridge/ElasticNet minimize regularized squared error; the MLP minimizes
scaled squared error; tree losses are recorded per candidate. Input and target
scalers are fitted only on the corresponding training rows. Convergence
warnings are retained. This study fits point forecasts and performs no
probabilistic calibration or economic/MPC simulation.

Each subset's inner winner is refitted and scored on development validation.
Validation then freezes feature-subset winners separately by availability class,
along with matched price-only baselines. Only these frozen selections are refit
on the eligible pre-holdout history and scored on the final holdout. No holdout
score selects a model. All model scores report MAE, RMSE, WAPE and relative MAE
to the seasonal/latest-cleared reference.
Earlier exploratory studies have already examined some of these dates. This
run freezes its selections before evaluating the later period.

## Resume, results and reproducibility

Resubmit the same `sbatch` command with the same output directory after a time
limit or interruption. Each finished inner candidate is checkpointed atomically;
an interrupted fit is rerun, and completed compatible fits are skipped. A lock
prevents concurrent jobs from sharing one output directory. Code, scientific
configuration, data or model-library changes require a fresh output directory.
Changing worker count or machine storage paths does not change candidate seeds.

Inspect:

- `prepared.json`: exact source selection, native group coverage, eligible/skipped cohorts, arrays and candidate plan.
- `status.json`: running phase, progress, failures and completion state.
- `trials/`: candidate parameters, seeds, losses, warnings, metrics and fit times.
- `validation_scores.csv`: all feature subsets and changes against matched price-only models.
- `selected_models.json`: selections frozen before holdout evaluation.
- `holdout_scores.csv`, `daily_errors.csv`, `summary.md` and per-group plots.
- `models/` and `predictions/`: selected fitted bundles and interval predictions.

For a transient failure, inspect its checkpoint error, then set
`PRICE_SEARCH_RETRY_FAILED=1` when resubmitting. The default preserves failed
records and reports a partial run. A fresh study can use
`PRICE_SEARCH_INPUT_MANIFEST` to pin the source snapshots from a previous
input manifest. Compatible resume uses frozen prepared arrays rather than a
newer current database revision.

The implementation records source snapshots and row hashes, full parameter
definitions, seeds, library versions and code hashes. Cross-machine numerical
roundoff can still depend on the CPU/BLAS build; changing model-library versions
is rejected for resume. The Slurm resource flags follow the official
[sbatch](https://slurm.schedmd.com/sbatch.html) and
[srun](https://slurm.schedmd.com/srun.html) interfaces.

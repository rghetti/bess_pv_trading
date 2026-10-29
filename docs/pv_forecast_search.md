# Reproducing the farm 901/910 forecast search

The inspectable launcher is `scripts/train_pv_forecasts.py`. Statistical fitting,
data checks, multiprocessing and reporting are in
`src/forecasting/training/pv_forecast_search.py`; neural architectures and their training
loop are in `src/forecasting/models/pv_residual_networks.py`. `src/forecasting/evaluation/pv_forecast_comparison.py` compares the
frozen fits and `src/forecasting/online/pv_online_bias.py` implements causal adaptation. This workflow
uses local cached data, makes no API requests and installs no packages.

## Run the searches

From the repository root, with the existing environment activated:

```bash
conda activate bess
python scripts/train_pv_forecasts.py \
  --config configs/pv_experiments/forecast_search_901_910.yaml \
  --workers 42 --run-id my_weather_search --no-resume
```

The default broad search samples 64 trials per farm: six settings for each of
eight statistical families and four for each of four neural architectures.
The second phase adds 24 temporal-convolution trials per farm:

```bash
python scripts/train_pv_forecasts.py \
  --config configs/pv_experiments/forecast_tcn_901_910.yaml \
  --workers 42 --run-id my_tcn_search --no-resume
```

Run these phases consecutively. Each uses at most 42 single-thread processes,
leaving six of the cluster's 48 CPUs available. BLAS/OpenMP thread counts in
child workers are restricted to one; nested tree parallelism is disabled.
If this conda environment reports a C++ runtime import error, prefix the command
with `LD_LIBRARY_PATH="$CONDA_PREFIX/lib"` so SciPy and PyTorch load the same
conda runtime. No installation is needed.

| Argument | Meaning |
| --- | --- |
| `--config PATH` | Required YAML; experiment files import `configs/standard.yaml` directly. |
| `--plan-only` | Print the deterministic trial list and parameters without fitting. |
| `--workers N` | Limit concurrent single-thread CPU fits; maximum 42, with the configured reserve checked against CPU affinity. |
| `--farms 901 910` | Override the farm list. For example, use `--farms 901` to study only that plant. |
| `--run-id NAME` | Name the output directory below `pv_forecast_search.output_root`. |
| `--resume` | Reuse completed fits only when configuration, inputs, code and artifact checksums match. |
| `--no-resume` | Start a fresh run; an existing directory is replaced according to `experiment_outputs.overwrite_existing`. |

Inspect the exact sampled settings with:

```bash
python scripts/train_pv_forecasts.py \
  --config configs/pv_experiments/forecast_search_901_910.yaml --plan-only
```

Grids, trial counts, features, seeds, maximum epochs, patience, batch size, loss,
regularization and input/output paths are YAML options under
`pv_forecast_search` in `configs/standard.yaml`. The TCN configuration overrides
only the architecture list and budgets. A smaller experiment can override
`statistical_trials_per_family`, `neural_trials_per_architecture` or `farm_ids`.
The deterministic grid sampler does not inspect any scores.

## Inputs, targets and evaluation

The input is the existing frozen corpus under
`outputs/pv_neural_tuning/neural_hpo_2025_to_2026_08_42cpu/farm_<id>/parquet/`.
The driver verifies raw target/key equality against the saved reference curves,
forecast availability and chronological target-availability boundaries.
It preserves the six-hour UTC weather availability lag, buffered Italian auction
cutoffs, DST, native quarter-hours and IDA3's noon-onward delivery suffix.

- Fit: 2025, with the existing availability embargo at the split boundary.
- Select hyperparameters, stopping epoch and ensembles: January–April 2026.
- Report: May–August 2026, a **reused holdout** already inspected in earlier work.
  The selection code does not use it, but it is no longer an untouched test of
  the overall research process.

All new models use weather and cyclic time only. Core inputs are forecast
GHI/horizontal-direct radiation, temperature, 10 m wind components/speed,
cyclic local hour/year/month, forecast-cycle hour and lead time. The extended
set adds dew point, cloud layers, precipitation, snow, visibility and 100 m wind.
Training-only medians, means and scales are persisted with missingness indicators.
There are no coordinates, proxy solar/tracker geometry, PVLib-derived features,
auction/month embeddings, elapsed-age features, satellite GHI or measured-power
inputs. Auction labels route separate forecast curves; they are not regressors.

Separate farm models predict measured AC power normalized by the known AC
rating, then clip their output to `[0, rating]` and apply the configured clock
night window. Capacity is an output scale/limit, not a learned input feature.
Raw AC targets receive no efficiency adjustment. Valid zero production and
operating anomalies remain in the data. All valid training quarters are used;
selection/reporting use the same frozen daylight eligibility mask as earlier
benchmarks. That shared scoring mask is never supplied as a geometry input.
Days and auction views receive equal aggregate scoring weight. Forecast views
overlap: do not sum their energy as if it were delivered production.

## Models and adaptation

| Family | Formulation |
| --- | --- |
| Histogram gradient boosting | Direct AC prediction or correction to a training-only Ridge weather curve. |
| Extra Trees | Direct nonlinear weather regression. |
| Huber regression | Robust linear weather regression. |
| Splines + Ridge | Additive spline expansion, direct or residual prediction. |
| Random Fourier features + Ridge | Approximate RBF nonlinear regression, direct or residual prediction. |
| Residual MLP | Dense correction to the training-only Ridge curve. |
| Mixture of experts | Weather-dependent weighted combination of learned residual experts. |
| Bidirectional GRU | Residual sequence model over the already available weather forecast curve. |
| Transformer | Residual attention over that same available forecast curve. |
| Direct/residual TCN | Symmetric temporal convolutions over the available forecast curve. |

Neural fitting uses deterministic CPU seeds, AdamW, configured MSE/Huber loss,
gradient clipping and validation-only early stopping. Sequence models may see
later **forecast** weather from the same issuance, never later measurements.
The exported convolution contract preserves the training padding length,
including 25-hour DST days, so individual-day inference matches saved predictions.

The comparison also averages the top 2, 3 or 5 new models, ranked on validation.
It tests 48 online-gain settings per farm: six base predictors, four history
windows and two minimum-history requirements. A single gain per issuance is
the clipped zero-intercept least-squares coefficient `sum(x*y)/sum(x*x)` from
completed, observed days in the window. Overlapping forecasts are averaged
once per physical quarter before accumulating these statistics. A day is
unavailable until its following local midnight and all retained measurements
are observed. This is forecast-error adaptation, with no assumption about aging.
Static weights remain trained to 2025; online models explicitly receive extra
past production information during 2026.

After editing `pv_online_bias.source_runs` in a config that imports the standard
to point to your two run directories, compare them with:

```bash
PYTHONPATH=src python -m bess_pv_trading compare-pv-forecast-searches \
  --config configs/pv_experiments/forecast_comparison_901_910.yaml
```

The shipped comparison config points to the completed original searches.

## Completed results

The two searches completed **176 trial fits**, excluding fitted linear reference
curves and the earlier tracking-feature research. The comparison evaluated
**96 online settings**. Every choice below uses January–April validation only.
Errors are balanced daylight RMSE as a percentage of installed AC capacity.

| Farm/model | 2025 train | Jan–Apr validation | May–Aug report |
| --- | ---: | ---: | ---: |
| 901 previous selected TCN | 11.286% | 11.212% | 12.559% |
| 901 new five-model mean | 11.308% | 11.046% | 12.770% |
| 901 calibrated PVLib reference | 13.359% | 12.730% | 14.114% |
| 910 previous selected neural model | 17.494% | 17.920% | 18.193% |
| 910 new static two-model mean | 15.002% | 17.753% | 18.259% |
| 910 five-model mean + causal seven-day gain | 13.915% | 15.899% | 17.057% |
| 910 calibrated PVLib reference | 14.849% | 20.451% | 18.430% |

Farm 901 improves validation but does not establish an improvement on the reused
holdout. Retaining its previous model is supported by these results. Farm 910's
adaptive model reduces report-period RMSE by about 6.25% relative to its previous
selection; the corresponding static ensemble does not improve that period.
The adaptive model uses observed history, so this is not solely an architecture
comparison. Seven-day paired block resampling yields candidate-minus-previous
95% intervals of `[-0.080, +0.532]` capacity percentage points for 901 and
`[-2.206, +0.008]` for 910. Both include zero; these exploratory intervals exclude
hyperparameter-selection uncertainty. No new models were promoted into trading
defaults and no new trading optimizations were run for this search.

### Completed-day local-hour correction

The original TCN for farm 901 is still strong, so a separate study keeps its
weights fixed and learns an additive correction from recently completed
production days. Run:

```bash
PYTHONPATH=src python -m bess_pv_trading study-pv-hourly-bias \
  --config configs/pv_experiments/hourly_bias_901_910.yaml
```

Options under `pv_hourly_bias_study` configure farms, source, output ID, forecast
column, first eligible history date, local-hour bucket width, lookback windows,
pseudocount strengths, minimum completed days and plot saving. The historical
study uses only 2026 production for calibration, because the frozen model was
fitted on 2025 production. Each physical quarter contributes one residual after
averaging its overlapping auction forecasts. A day enters only after its next
Italian midnight and all retained measurements arrive.

For local-hour bucket $h$ and cutoff $c$, the correction is the sum of past
`measured AC - forecast AC` divided by `observed quarters + ridge_pseudo_intervals`
within the configured window. It is added to the frozen forecast; output is
clipped to the farm's AC rating. January–April 2026 selects the setting and
May–August is reported separately.

| Farm | Validation: previous → corrected | May–Aug: previous → corrected | Selected setting |
| --- | ---: | ---: | --- |
| 901 | 11.212% → 11.192% | 12.559% → 12.492% | 7 days, 250 pseudointervals |
| 910 | 17.920% → 16.575% | 18.193% → 17.357% | 14 days, 50 pseudointervals |

The farm 901 gain is about **0.5% relative** in the reused report period.
Its seven-day paired interval for the change is `[-0.142, +0.004]` capacity
percentage points and includes zero. For farm 910 the interval is
`[-1.325, -0.415]` points; the earlier five-model adaptive ensemble still
has the lower report error at 17.057%. The farm 901 result needs further
independent evidence.

The study saves all candidate scores, chosen full predictions, completed-day
statistics, monthly metrics, comparison plots, resolved YAML/JSON, input hashes
and source-code hash under `outputs/pv_hourly_bias/<run_id>/`.

### Expanding-history monthly retraining check

The fixed 2025 fit may miss later behavior. The independent
`study-pv-monthly-retraining` command tests that hypothesis:

```bash
PYTHONPATH=src python -m bess_pv_trading study-pv-monthly-retraining \
  --config configs/pv_experiments/monthly_retraining_901_910.yaml
```

For each May–August 2026 delivery month it refits that farm's previously
validation-selected statistical family using all eligible historical AC and
forecast-weather pairs available at local midnight **one day before** the
first delivery day. The model family, original 2025 preprocessing, feature
set and hyperparameters stay fixed; the underlying regression and, when
needed, its Ridge reference are refitted on the expanding sample. There are eight independent
farm/month cases, run on eight single-thread workers. The experiment saves
each month's issue cutoff, final accepted target-availability time, training
count, prediction checksum and daylight score. No May–August result chooses
a model family or setting.

The expanding fit **does not improve farm 901**: its May, June, July and August
normalized RMSE is 18.34%, 10.77%, 12.21% and 10.08%, versus 17.41%, 10.37%,
12.02% and 8.61% for the fixed previous predictor. Farm 910 is mixed, improving
June and July but worsening August. This check does not justify monthly
replacement of the fixed predictor. Inspect
`outputs/pv_monthly_retraining/expanding_2026_05_to_08/csv/monthly_metrics.csv`
and `json/fit_provenance.json` for every case.
The monthly plot's top panel shows daylight RMSE, and the bottom panel shows
signed forecast bias. Blue is the monthly refit, gray is the existing frozen
model, and orange is calibrated pvlib; dots mark the four delivery months.

### Diagnose remaining errors without changing the training data

```bash
PYTHONPATH=src python -m bess_pv_trading diagnose-pv-forecast-errors \
  --config configs/pv_experiments/error_diagnostics_901_910.yaml
```

This reads the frozen May–August 2026 comparison and keeps the raw measured AC
target and common daylight evaluation mask. It groups each farm's squared PV
error by absolute difference between available forecast GHI and satellite GHI,
local delivery hour, auction, and pre-existing unconfirmed production-condition
flags. Satellite GHI is used only for diagnosis. It is never passed to an
operational forecast or used as the production target. The sensitivity table
shows scores if flagged views or the leading flagged day are *excluded from
reporting only*; all headline scores still include them.

These frozen results preceded the sustained daylight-zero modeling mask now
documented in the README. Their reported scores and candidate shares use the
old, uncurated evaluation population; compare new model fits only on a single
versioned mask and report retained-interval coverage as well as error.

| Farm | Previous predictor RMSE, all views | Largest GHI-error quarter's squared-error share | Flagged suspected-outage view share | Those views' squared-error share |
| --- | ---: | ---: | ---: | ---: |
| 901 | 12.559% | 48.50% | 0.44% | 12.82% |
| 910 | 18.193% | 35.60% | 2.70% | 21.18% |

These are balanced daylight capacity-normalized scores. For farm 901, every
suspected-outage view is on 1 May 2026: 34 unique physical quarters have zero
measured AC while the mean auction-view forecast is 2.67 MW and mean satellite
GHI is 732 W/m². This is a striking mismatch, but the available data do not
identify whether it is plant unavailability, curtailment or a measurement issue.
Removing that day *only for sensitivity reporting* changes the previous
predictor's RMSE from 12.559% to 11.731% and the new five-model average from
12.770% to 11.825%; the previous predictor remains better. After that exclusion,
the largest GHI-disagreement quarter still contributes 53.1% of squared error.
Farm 910 has several similar flagged days, including 1–3 May. Weather disagreement
and anomalous production therefore both merit investigation; their error shares
can overlap and must not be added.

`outputs/pv_error_diagnostics/frozen_2026_05_to_08/` contains resolved settings,
input checksums, per-farm CSV groups and sensitivity scores, JSON summaries and
plots. In `error_diagnostics.png`, blue bars show the squared-error share of
successive absolute GHI-disagreement quantiles; orange points show sampled signed
GHI and AC forecast errors. Each `flagged_day_*.png` has measured AC in black,
auction-specific AC forecasts in color and farm AC rating as a gray dashed line
on top. The lower panel shows satellite GHI in black and corresponding available
forecast GHI curves in color. Both panels use local delivery time and include
the next-midnight boundary point when the source has it.

## Saved artifacts, inference and exact historical reproduction

Searches: `outputs/pv_forecast_search/weather_only_901_910_hpo_v1` and
`outputs/pv_forecast_search/market_free_tcn_901_910_v2`. Each contains resolved
settings, identity/input fingerprints and per-farm `cache`, `models`, `arrays`,
`csv`, `json`, `parquet` and `plots` folders. Neural trial JSONs include parameter
counts, epoch histories and runtime; all trial JSONs record warnings, convergence
indicators, specifications, seeds, software versions and checksums.

Comparisons: `outputs/pv_forecast_comparison/market_free_901_910_comparison`.
Each farm has full predictions, monthly metrics, model definitions, selected
names, paired uncertainty intervals, comparison plots and portable `bundles`.
`selected_static` and `selected_including_online` include all constituent models,
frozen preprocessors, farm identity, checksum manifest and AC rating. Adaptive
bundles also contain a historical-statistics snapshot, filtered by issue cutoff.
Supply refreshed completed-day statistics via `--history` for later operation.

```bash
python scripts/predict_pv_forecast.py \
  --bundle outputs/pv_forecast_comparison/market_free_901_910_comparison/farm_901/bundles/selected_static \
  --farm-id 901 --input weather.parquet --output forecast.parquet
```

Input Parquet needs the trained weather fields and `interval_start_utc`,
`forecast_cycle_utc`, `lead_hours`, `delivery_day`; multiple views also retain
`auction` for routing. Online prediction additionally requires
`decision_cutoff_utc`. Measured targets and proxy geometry are not required.
Inference rejects farm mismatches or altered artifact bytes. First/middle/last
day replay and both DST transitions for all four selected exports match
benchmark forecasts within `7.2e-7 MW`; the report is
`json/inference_validation.json`.

The completed searches retain their exact source under `code/` and canonical
resolved YAML under `config/resolved_config.yaml`. Current shared defaults can
evolve, so use these pinned files to resume the original run:

```bash
python outputs/pv_forecast_search/weather_only_901_910_hpo_v1/code/scripts/train_pv_forecasts.py \
  --config outputs/pv_forecast_search/weather_only_901_910_hpo_v1/config/resolved_config.yaml \
  --workers 42 --resume
```

Use the analogous TCN path for its run. For an independent exact repetition,
use the same pinned source/config with a **new** `--run-id` and `--no-resume`.
Saved JSON is for inspection; pinned YAML is the configuration entry point.
Do not replace the original directory while executing its archived launcher.

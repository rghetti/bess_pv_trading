# Member-wise probabilistic PV studies

The study compares ensemble weather and deterministic weather from matched
00 UTC cycles. It treats the existing farm-specific 2025 physical
parameters as prior farm metadata and uses raw measured AC power after the
approved episode exclusions. The existing six-hour cycle-to-availability
assumption is recorded in the inputs. Ensemble and deterministic forecasts are
scored on the same quarter-hours.

## January–August 2026 study

`probabilistic_january_august_2026.yaml` uses the January and August ensemble
archives and 61 delivery days: January 1–31 and August 2–31. The January 1
first local hour lacks required instantaneous ensemble weather; the August 1
delivery forecast lacks its preceding July 31 cycle. September is omitted
because the existing measured-power and deterministic archives do not cover
complete September delivery days. Forecast weather windows February–July are
not invented or filled.

The chronological split is 49 days for training (January 1–31 and August 2–19)
and 12 days for validation (August 20–31): approximately 80:20 by delivery day.
There is no independent test set. Model selection uses the validation period,
so the selected scores are exploratory and optimistic. The first validation day
uses the completed August 19 training targets retrospectively, after its
historical day-ahead gate. The ensemble archive has one daily 00 UTC cycle, so
the DA and IDA1 weather comparison uses the same cycle even though the original
deterministic source may have later cycles available for IDA1.

A separate `probabilistic_january_august_same_day.yaml` experiment uses each
same-day 00 UTC cycle retrospectively and fills the hours before that UTC cycle
with the preceding-day forecast. This means four quarter-hours in winter and
eight in summer. A same-day cycle is assumed available at 06 UTC, so its use
for earlier intervals is retrospective and the cycle was unavailable at the
day-ahead and IDA1 gates for that delivery day. Member labels are stitched
across cycles for the midnight fill, but member identity across forecast runs is
not established; these are not proven joint trajectories. Both experiments use
the same 61 candidate delivery days and the same 49/12 split for comparison.
For future auction-focused work, use the preceding-day vintage while only the
00 UTC ensemble cycle is available. The same-day experiment is retained as a
retrospective comparison and is not the default for future studies.

## Commands

After the user approves the specific experiment, run from the repository root
in the existing `bess` environment. The launcher
reads shared defaults through its experiment YAML; it does not install packages
or download data.

```bash
PYTHONPATH=src python scripts/study_pv_probabilistic.py \
  --config configs/pv_experiments/probabilistic_january_august_2026.yaml --plan-only

PYTHONPATH=src python scripts/study_pv_probabilistic.py \
  --config configs/pv_experiments/probabilistic_january_august_2026.yaml --stage all

PYTHONPATH=src python scripts/study_pv_probabilistic.py \
  --config configs/pv_experiments/probabilistic_january_august_same_day.yaml --stage all
```

The stages `prepare`, `pvlib`, `search` and `report` can run separately.
`--farms 1 345` selects farms, and `--workers N` sets an upper bound on search
workers. The January–August configurations request 42 workers and disable the
shared-load cap at the user's request; CPU affinity and memory limits still apply.
Completed search trials and their out-of-fold/model checkpoints are reused only
when their source, input, configuration, code and artifact fingerprints match.
Choose a new run ID if those inputs or the design change.

The earlier January-only configuration and outputs remain available as a
separate research result. Changes to study code alter its resume fingerprints;
use a fresh run ID to reproduce that older design with the newer implementation.

## Method and outputs

Each predictor is shared by the 50 weather members of one farm. The measured AC
target is paired with each member; member weights sum to one per observed
interval. Core inputs are irradiance, temperature, 10 m wind and local hour.
The weather feature set additionally includes available cloud, precipitation,
snow, dew-point, visibility and 100 m wind variables. Members and quarters from
a delivery day remain together in the chronological split.

### Training losses and validation selection

Each farm/configuration fits one predictor, not separate predictors optimized
for MAE, RMSE and CRPS. Power targets are divided by farm AC rating for fitting.
Each valid member row receives weight `1 / number_of_valid_members_for_interval`;
the total weight for each measured interval is one. Thus training compares each
member-wise prediction with the same observed power, rather than minimizing
the error of their mean or median directly. Prediction values are converted back
to MW, clipped to `[0, AC rating]` and set to zero during the configured night
window before scoring. The inherited setting is 21:30–05:00 Europe/Rome;
this is a fixed clock-time policy, not a solar-position calculation, and it
does not delete or alter the measured targets.

| Predictor | Training objective used in this study |
| --- | --- |
| Ridge | Weighted squared error with L2 coefficient penalty. |
| Huber | Weighted Huber robust loss with L2 penalty and fitted residual scale. |
| Extra trees | Squared-error split criterion and leaf means; sample weights enter fitting. |
| Histogram gradient boosting | Squared-error loss, with configured tree regularization. |
| LightGBM | Default regression squared-error objective. |
| CatBoost | Default RMSE regression loss; its minimum agrees with MSE for the same data. |
| MLP and TCN | Weighted mean squared member-wise error. Neural early stopping uses weighted MSE on the last two fit-period days; the model is then refitted on the entire fit period for the selected number of epochs. |
| Farm-specific pvlib | Physical baseline with previously supplied farm parameters; no predictor was trained in this experiment. |

Separately, seven linear quantile regressors per predictor use the training
out-of-fold member-PV mean, p10, p50 and p90, plus sine and cosine of local
hour, to predict measured power. Each minimizes its own quantile pinball loss
with the configured L1 coefficient penalty; none is trained directly on CRPS.
The pvlib baseline has a similarly fitted quantile calibrator, using its training
period predictions. The validation report evaluates each fitted candidate under
all applicable point and probabilistic scores, then ranks it independently by
three specific criteria: RMSE of the raw member mean, MAE of the raw member
median, and approximate CRPS of the calibrated quantiles. These are model
selection criteria, not the candidates' training objectives; winners can differ.

### Scoring procedure

For each farm and quarter-hour, retain a target only when the approved target
exclusions permit it and the measured power, deterministic prediction and all
50 member predictions are finite. This ensures matched point and raw ensemble
comparisons. Calibrated scores additionally require finite calibrated quantiles.
The scores include eligible nighttime quarters and are computed separately for
the reporting window or day. Hourly scores first average all four eligible
quarters of each member and the observation; incomplete hours are dropped.

For observations `y_i` and a chosen point forecast `p_i`, the error is
`e_i = p_i - y_i`: MAE is `mean(abs(e_i))`, RMSE is `sqrt(mean(e_i**2))`,
and bias is `mean(e_i)` (positive means overprediction). The point forecast may
be the deterministic-weather prediction, raw ensemble mean or median, or
calibrated mean or p50. RMSE is always at least MAE on the same observations
and forecasts because squaring gives large errors more influence. Values from
different forecasts or different scoring samples need not obey that comparison.
Percent scores equal `100 * score_MW / farm_AC_rating_MW`.

For each eligible interval, with 50 predicted powers `x_1,...,x_50` and
observation `y`, empirical ensemble CRPS is exactly

`(1/50) * sum_j |x_j-y| - (1/(2*50**2)) * sum_j sum_k |x_j-x_k|`.

The implementation sorts members to calculate the pairwise term efficiently;
the formula is unchanged. It includes self-pairs and all ordered pairs and
assumes equal 1/50 weights. The reported raw CRPS is the average of these
interval scores, not the CRPS of the ensemble mean. For calibrated CRPS,
the corrected p05–p95 quantile values are linearly interpolated, with p05 and
p95 extended as constant tails to probabilities 0 and 1. At 201 equally spaced
probabilities, compute pinball loss `max(q*(y-p_q), (q-1)*(y-p_q))`, integrate
it with the trapezoidal rule and multiply by two; average over eligible
intervals. Raw and calibrated CRPS therefore describe different distributions,
and the latter is an approximation.

Pinball loss measures each calibrated quantile separately. For example, p90
penalizes an observation above the predicted p90 nine times as much as the same
distance below it. Coverage is the fraction of observations inclusively between
p25–p75, p10–p90 or p05–p95, respectively nominally 50%, 80% and 90%; interval
width is the mean upper-minus-lower bound. The reliability table separately
reports fractions strictly below and below-or-equal to each quantile, because
zero-power ties are common. Calibrated p50 is the corrected median; calibrated
mean integrates the interpolated quantile curve with constant tails.

Calibration inputs use out-of-fold predictions entirely within training:
January 15–31 from a January 1–14 fit, August 2–10 from a fit through January
31, and August 11–19 from a fit through August 10. The final predictor is fitted
through August 19. Linear quantile regressions use those out-of-fold predictions.
For folds whose fitting period ends on the immediately preceding delivery day,
the first predicted day is also retrospective at its historical DA gate because
the complete prior-day power target was not yet known at that gate. The folds are
still disjoint by target delivery day.
Raw quantiles, crossings, physical-bound adjustments and night-policy changes
are retained. The calibrated mean integrates a piecewise linear quantile curve
with constant tails; it is an approximation.

Results are saved under `outputs/pv_probabilistic/january_august_2026_80_20/`:

- `README_pvlib.md` and `README.md` describe the physical and full-search results.
- `trial_plan.json`, `resolved_config.yaml`, input manifests and per-trial
  `result.json` files record design, checksums, settings and status.
- Each farm folder contains matched member and deterministic predictions,
  quarter-hour and hourly metrics, exclusion and missingness counts, model and
  calibrator artifacts, and daily plots for pvlib and selected model winners.
- `validation_rankings.csv` reports separate rankings for member-mean RMSE,
  member-median MAE and calibrated quantile CRPS. Error, bias, CRPS, pinball and
  interval-width tables include both MW and a percentage of farm AC nameplate.
- `quantile_reliability.csv` and calibration plots record strict and inclusive
  observed fractions, exposing ties at zero production.

Equal empirical member weights do not establish calibrated probabilities.
Post-processed quantiles are marginal per interval; they do not by themselves
form calibrated temporal trajectories for a stochastic optimizer. Hourly raw
ensemble statistics average each member's four quarters; hourly calibrated
quantiles are not inferred by averaging marginal quarter-hour quantiles.

## CRPS and quantile calibration

The continuous ranked probability score (CRPS) compares a forecast distribution
with the single power value subsequently observed. It rewards distributions
that put probability near the observation while penalizing unnecessarily wide
forecasts. Lower is better; zero is ideal. It is measured in MW, and this study
also divides it by installed AC power and multiplies by 100 for a capacity
percentage. A deterministic forecast is a one-point distribution, for which
CRPS equals its absolute error.

For a predictive random variable `X` and observation `y`, CRPS is
`E[|X-y|] - 0.5 E[|X-X'|]`, with `X'` independently drawn from the same
predictive distribution. For instance, equally weighted forecasts 0 and 2 MW
with actual 1 MW have CRPS 0.5 MW. The raw 50-member score uses the exact
empirical-ensemble formula. For calibrated quantiles, CRPS is approximated by
numerically integrating quantile pinball loss, using a piecewise-linear
quantile curve and constant tails. The approximated and empirical scores use
the same MW units but are computed with these different numerical methods.

Calibration fits quantile regressors on historical out-of-fold PV predictions
and measured power. It adjusts the predicted quantiles to better reflect
observed outcomes; p10–p90 should contain about 80% of outcomes if its coverage
is reliable. The report shows coverage, width, pinball loss and CRPS. The raw
linear quantile fits can cross or leave physical bounds: the stored corrected
values are sorted and bounded, and raw predictions plus correction flags remain
available. Calibration on limited, seasonally unbalanced data does not guarantee
future coverage. Nighttime zeros produce ties, so the reliability output shows
both the fraction of observations strictly below each quantile and the fraction
below or equal to it. Validation remains a model-selection period, not a test.

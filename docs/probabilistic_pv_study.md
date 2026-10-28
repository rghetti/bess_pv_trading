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

## Commands

Run from the repository root in the existing `bess` environment. The launcher
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

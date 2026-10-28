# Auction-aware online CRPS PV study

This exploratory comparison uses the saved January–August 2026 models selected
by calibrated validation CRPS: farm 1 `huber_core_03`, farm 345
`tcn_weather_02`, farm 901 `tcn_weather_01`, and farm 910 `tcn_core_04`.
The predictors were fitted on member-wise Huber or squared-error objectives,
not CRPS. The same August 20–31 days selected the models and evaluate this
study: there is no independent test. The saved models include all August 19
targets, which makes August 20 DA/IDA1/IDA2 forecasts retrospective. IDA3's
August 20 cutoff follows those targets; all results still use selected models.

## Timing and forecast sources

The ensemble always comes from the preceding delivery day's 00 UTC cycle. The
deterministic cycle is the latest present in the original weather archive whose
issue plus the configured six-hour lag is no later than auction submission.
Normally this is prior-day 00, 06, 12 UTC for DA, IDA1, IDA2, and same-day
00 UTC for IDA3. IDA3 only scores products from noon local onward. Forecasts
retain their exact cycle, availability, auction cutoff and target-availability
timestamps; unobserved or excluded targets are never used for updates.

The selected models are refitted in their original three chronological
training folds to produce out-of-fold deterministic predictions at each
auction's cycle. Source OOF ensemble predictions and final saved models are
checksum-verified. The fitted per-farm deterministic-error quantile regressions
use deterministic power plus local-hour sine/cosine, p05 through p95 in 5%
steps, L1 penalty 0.01, clipping to AC nameplate, sorting of crossings and the
existing 21:30–05:00 local night policy. Each issue fits only on previous OOF
or validation forecasts with targets available by that issue, excluding
forecasts whose model training targets were unavailable when issued. Seven
earlier delivery days with usable history are required. Raw and corrected
quantiles and correction flags are saved.

## Online learning

For each farm, auction and local three-hour block, separate ML-Poly learners
weight either 50 power-ranked ensemble forecasts or a combined pool of those
50 forecasts, one deterministic forecast and 19 deterministic quantiles.
Uniform initialization gives the groups 50/70, 1/70 and 19/70 of the combined
pool's initial weight; quantiles are related predictions, not independent
weather members. Ties use stable sorting. Equal-weight ensemble and fixed
ensemble quantile calibrator provide comparison baselines.

Warm learners replay earlier eligible forecasts; cold learners start uniformly
on the first validation issue. At each decision, all intervals in a block use
the frozen current weights. When observations become available, the replay
computes CRPS linearized gradients using the values and weights originally
issued, updates cumulative regret and squared regret, and applies the paper's
ML-Poly weight rule. Updates are processed before the next issue. Retrospective
model forecasts never train the online weights or quantile models. The modified
delayed, block-pooled setting does not inherit the paper's regret guarantee
without a separate proof.

Each forecast is a marginal empirical distribution. Weighted member ranks and
quarter-hour quantiles do not constitute calibrated full-day trajectories.
Daytime in the reports means the fixed 06:00–21:30 local clock window; it is
not a solar-position estimate. Nighttime eligible observations remain in the
all-hours comparison.

## Running and results

After user approval, run from the repository root in the existing environment:

```bash
PYTHONPATH=src python scripts/study_pv_online_crps.py --config configs/pv_experiments/online_crps_january_august_2026.yaml --plan-only
PYTHONPATH=src python scripts/study_pv_online_crps.py --config configs/pv_experiments/online_crps_january_august_2026.yaml --stage all --workers 42
```

Stages `prepare`, `evaluate`, and `report` may be run separately. Preparation
uses up to four farm workers; evaluation uses up to sixteen farm/auction jobs.
Output under `outputs/pv_online_crps/january_august_2026/` includes the resolved
study manifest, fold models, prepared forecasts, quantile predictions and flags,
issued weights, delayed-update history, matched metrics, reliability and
spread-skill tables, sample coverage and daily plots. Exact weighted empirical
CRPS is used for member distributions; deterministic and fixed calibrated
quantile curves use numerical integration with constant tails. Weighted member
quantiles use the inverse empirical CDF; for 50 equally weighted members, the
reported median is the lower middle member, whereas the earlier study used the
average of the two middle members. Quantile scores for levels absent from the
earlier seven-level calibrator use its linearly interpolated quantile curve.

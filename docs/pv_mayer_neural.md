# Neural PV post-processing

The neural study is opt-in. It consumes the normalized hourly member predictions and target masks from a completed Mayer-style preparation run. The existing farm-specific neural PV predictors remain frozen; these additional networks learn distributions over their outputs. No historical neural model is trained by loading configuration or using `--check-only`.

The input for one hourly case is the same-cycle deterministic control **proxy** followed by the 50 sorted ensemble PV predictions, all normalized by AC capacity. Each case has one measured hourly target. The study includes:

- A doubly censored Gaussian distributional regression network (CN DRN), trained with differentiable closed-form CRPS. Ten independently seeded networks average location and scale, as in the paper.
- A 51-output quantile regression neural network (QRNN), trained with smooth quantile loss. Its raw outputs are preserved and, when configured, rearranged for scoring.
- A Bernstein quantile network (BQN), using positive coefficient increments to guarantee ordered quantiles.
- A non-crossing quantile network (NCQRNN) using a nonnegative triangular cumulative construction. The precise parameterization is an implementation choice because the Mayer paper summarizes, but does not reproduce, the full source architecture.

All neural models pool forecast leads within each farm, as in Mayer et al. The configured DRN architecture, Adam learning rate, batch size, patience and ensemble size follow the paper. The quantile-model widths, activation and Bernstein degree are configurable starting values, not the paper's plant-specific tuned results. The positive scale floor is a numerical implementation choice. Quantile outputs outside the physical range are flagged and excluded from common distribution scores, not silently clipped. Quantile CRPS is integrated numerically with constant tails; DRN CRPS uses the analytic expression.

For the three quantile-network families, the configured default is 100 seeded random hyperparameter trials per farm and family. Each five-day fit-period block supplies three training days, one early-stopping day and one trial-selection day. The best trial is retrained with every fifth fit-period day held out for early stopping. The separate outer evaluation days never supply gradients or determine the stopping epoch. The paper uses Optuna for its search; this implementation uses reproducible random draws over its reported ranges and does not claim to reproduce Optuna or the paper's four-year sample. The current data are sparse and the selected PV predictors and 2026 evaluation period are retrospective.

Inspect the configured inputs and split without training:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python scripts/train_pv_mayer_neural.py \
  --config configs/pv_experiments/mayer_neural_2026.yaml --check-only
```

After a concrete training experiment has been explained and approved, omit `--check-only`. The run writes separate farm checkpoints, validation histories, input hashes, hourly distributions, matched scores and calibration plots to a new output directory. It does not fit copulas or create MILP scenarios.

The non-crossing design follows the triangular positive-weight principle described by [Song et al.](https://doi.org/10.1007/s00376-023-3184-5). The differentiable censored score is checked against the [scoringRules normal-distribution reference](https://search.r-project.org/CRAN/refmans/scoringRules/html/scores_norm.html).

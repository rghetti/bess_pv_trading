# IDA own-gate zero-shot study preparation

This workflow prepares 27 independent studies: IDA1, IDA2 and IDA3, each with 7-, 28- and 60-day requested contexts, and three input variants: own-auction price history only; price history plus the four NORD forecasts; and those inputs plus historical spread curves. All three frozen foundation models run for each study. No historical inference is launched during preparation.

The spread is each historical target auction's actual price minus the latest earlier auction clearing for the **same delivery day**, using only the curve published by that historical target auction's decision cutoff. A documented cancellation excludes that earlier auction. The next earlier complete published same-day curve is used; no previous-day auction is silently substituted. Each historical anchor and fallback is recorded in request metadata. Future spread values are unknown and never included as future covariates. They enter through the models' native historical-only covariate interface.

IDA1/2 retain their native full-day quarter-hours, including 92/96/100 DST products. IDA3 covers only 12:00–24:00 local, so its sequence concatenates 48 native products per delivery day. This compresses the overnight non-trading interval; it is an explicit native-auction sequence representation rather than a continuous 24-hour signal. Actual product timestamps are retained for every historical observation and target. A 60-day IDA3 sequence therefore has about 2,880 observations, while 60-day IDA1/2 have about 5,760.

Historical inputs are trimmed to the longest complete recent suffix of **whole delivery days**, with at least seven days. Cancelled or missing target-history days break that suffix and are not deleted from its middle. Missing delivery-horizon forecasts still exclude that variant; reducing history cannot replace missing future inputs. Requests and coverage record requested and effective days, trimming reason and target cancellations.

This initial feature-isolation benchmark deliberately does not feed the **current** same-delivery anchor price curve to the model. It is a baseline only. The `nord_spread` variant supplies only historical spread curves. Thus these variants do not yet exploit the complete information available at the own gate; an anchor-conditioned extension would be a separate experiment.

The baseline is the most recent eligible same-delivery earlier clearing, and the weekly-naive curve matches the own auction's local product seven days earlier. Source timestamps are checked against the decision gate. The archive's NORD forecast issue times are not verified; archived forecast values are assumed available at the target gate. Pretrained-data overlap is possible. November 2026 remains a separate reserved report period without selection. TimesFM 3 weights are non-commercial.

Inspect the launch plan (including missing inputs), or prepare missing bundles without inference:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_ida_zero_shot_study
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_ida_zero_shot_study --prepare-only
```

Launch in tmux; missing bundles for the configured windows are prepared automatically before inference:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_ida_zero_shot_study --run --batch-size 8
```

One study runs at a time and uses up to three independent GPU workers, one per model. The shared worker batches compatible context lengths/horizons and checkpoints each delivery day. Synthetic GPU preflight precedes each historical study. Optional `--devices cuda:0 cuda:1 cuda:2` selects visible GPU ordinals; `--preflight-only --run` performs only synthetic compatibility checks.

Inputs, runs and their automatic Markdown/CSV reports are grouped under `outputs/ida_zero_shot_context_windows/<auction>/<window>d/<variant>/`. The final launcher step writes matched comparisons and Markdown summaries under each auction's `comparison/` directory. It requires complete reports and identical target values on the common sample. This stage prepares **own-gate forecasts only**. IDA forecasts made at the earlier DA gate, conditioned on predicted rather than actual DA prices, require separate heads and are not produced by this workflow.

The same optional gas/French/Swiss extension can be prepared against any IDA NORD bundle after the shared external-feature preparer supports native context timestamps, for example:

```bash
python -m scripts.prepare_da_cross_border_covariates \
  --nord-bundle outputs/ida_zero_shot_context_windows/IDA2/60d/nord/inputs \
  --output outputs/ida_zero_shot_context_windows/IDA2/60d/nord_gas_fr_ch/inputs
```

This extension retains the earlier retrospective prior-day quantity assumptions and may have substantially lower coverage. It is not part of the 27-study own-gate launcher by default.

Input preparation is native to the launcher. Adding a window such as 56 or 84 in the study YAML requires no separate preparation command. Existing per-window bundles are reused only when their model specifications, pinned source manifest, semantic study settings and file checksums match. Global window/variant lists do not invalidate completed per-window inputs. New bundles are prepared in temporary sibling directories and installed atomically. Incomplete or incompatible existing directories are rejected; choose a new `--root` rather than overwriting pinned experiments. Plan-only mode does not create inputs or run models. `--prepare-only` prepares inputs but launches no inference.

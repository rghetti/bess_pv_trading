# NORD DA zero-shot history-only comparison

The prepared bundle covers 1 October 2025 through 12 November 2026, based on the latest complete native DA day in the pinned snapshot. Each eligible request contains only the preceding 28 calendar days of published NORD DA prices. Hourly values before the October 2025 transition are repeated across their four quarter-hours. Native quarter-hours are retained thereafter. Missing historical products exclude a day, and cancelled or missing targets are recorded. The assumed fallback DA publication time is 12:57 local on the previous day; the decision cutoff is 11:45 local on the previous day. Source-specific recorded publication timestamps take precedence. DST days use their real 92 or 100 products. Prior-day and weekly references match local product time, repeating the first autumn 02:xx occurrence when no second reference exists.

Three independent frozen models produce nine marginal quantiles and a median forecast. The launcher assigns one model per GPU and batches eight compatible delivery days per model by default. Batches group matching history lengths and product counts; cross-series learning, clipping and symmetric averaging are disabled. FP32, one native thread per worker and a fixed seed are used. Synthetic GPU preflight tests the largest context and all forecast horizons before historical inference. Forecasts, fingerprints and statuses are atomically checkpointed per day. A restart requires the same device allocation, batch size, model weights, environment and source bundle. Keep the same `--devices` or use a new output directory if the available GPU set changes.

This is retrospective development, with possible pretrained-data overlap. November 2026 is reported separately and must not be used for model selection. TimesFM 3 weights and results are subject to their non-commercial terms. No farm data, model fitting or optimization is involved.

On this machine, from the repository root, preview the allocation or launch in your own tmux session:

```bash
bash scripts/launch_da_zero_shot_probe.sh --plan
bash scripts/launch_da_zero_shot_probe.sh
```

To set the batch size or visible GPU ordinals explicitly:

```bash
bash scripts/launch_da_zero_shot_probe.sh --devices cuda:0 cuda:1 cuda:2 --batch-size 4
```

Auto discovery respects `CUDA_VISIBLE_DEVICES`, selects GPUs below 1024 MiB used and 20% utilization, and queues models if fewer than three GPUs qualify. These checks do not reserve a device; coordinate with other users. Set `PROBE_OUTPUT`, `PROBE_BUNDLE`, `MODEL_ENV_ROOT` or `DATA_PYTHON` to change paths. Model logs and individual status files are under the output directory. Repeating the same command resumes completed days. Run only a synthetic preflight with `--preflight-only`; no historical forecasts are produced in that mode.

The report directory contains matched product forecasts, completion and runtime tables, overall/monthly/hourly/daily MAE, WAPE, RMSE, bias, reference-relative MAE, quantile pinball losses, empirical coverages, and daily curve and error plots. The common-product comparison includes a day only when all three model forecasts are available; incomplete runs are labelled. For an interim report:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.da_zero_shot_probe report --bundle outputs/da_zero_shot_inputs --forecast-directory outputs/da_zero_shot_run --output outputs/da_zero_shot_run/report
```

For another machine, transfer the checkout and the entire `outputs/da_zero_shot_inputs` bundle. The latter includes the original source manifest and checksummed DA source rows; the DuckDB and farm data are not needed to run this bundle. Create a separate Python 3.14 data environment with `pip install -r requirements/zero_shot/data.txt`, and create the three separate model environments with:

```bash
python3.14 -m scripts.setup_da_zero_shot_envs --root /path/to/new/model_envs --install
```

Set `DATA_PYTHON` to the data environment's interpreter and `MODEL_ENV_ROOT` to that new root. The model weight cache must be transferred or populated on the new machine. Add `--allow-download` to the launcher to download the exact pinned revisions if network access is available. The setup script never modifies the `bess` environment. GPU architectures, drivers and package wheels differ across machines; the required synthetic preflight runs before historical inference and fails visibly if they are incompatible.

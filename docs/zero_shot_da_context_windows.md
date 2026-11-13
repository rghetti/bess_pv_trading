# DA context-window comparison

The study compares 7, 28 and 60 calendar days of published NORD DA price history with and without the four NORD load/generation forecast profiles. The same three frozen foundation models, quarter-hour resolution, native hourly repetition before October 2025, publication assumptions, DST products, 408-day pinned snapshot, daily forecasts, nine quantiles and benchmarks described in `zero_shot_da_history.md` and `zero_shot_da_nord_covariates.md` apply. Forecast issue vintages remain unverified, so the forecast-covariate runs are retrospective. November 2026 is reported separately and is not used for selection.

On this machine, launch in tmux:

```bash
bash scripts/launch_da_context_windows.sh
```

The script runs 7- and 60-day histories first, then the 7-, 28- and 60-day covariate variants. It reuses the completed 28-day history results and resumes compatible existing checkpoints. Each run uses up to three independently assigned GPUs and defaults to batches of eight days. Pass `--batch-size 4` if the longer contexts exceed GPU memory. The model-specific synthetic preflight tests the longest context and each horizon before historical inference. Use the same device allocation and batch size when resuming a run.

After all six reports are complete, the script writes `outputs/da_zero_shot_context_comparison/scores.csv`, `coverage.csv` and `matched_forecasts.parquet`. Each model's six settings are scored on the same delivery products, with conflicts in actual prices rejected. The original individual reports remain in their own output directories. Re-run only the comparison with `python -m scripts.compare_da_context_windows`.

Prepared input coverage: 7-day history and covariates, and 60-day history, each cover all 408 days. The 60-day covariate bundle covers 398: 1–10 October 2025 lack values in the archived total-generation forecast within their historical context. No values are imputed. The six-way comparison therefore uses the intersection across all settings and records its observation count.

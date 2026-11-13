# Adaptive historical inputs for zero-shot price studies

The study inputs now support the longest complete recent suffix of full local delivery days, capped at the requested 7/28/60 days, with minimum_context_days configurable (default seven). This changes context construction, not model weights: foundation models remain frozen. A missing value in a price, historical forecast or past-only feature trims all inputs together through that local day. No zeros, interpolation or backward fill are introduced. Requested and effective calendar days, timestamps, missing counts and reasons are stored. A missing delivery-horizon covariate cannot be repaired by shortening history and remains an exclusion. This preparation policy does not yet add an operational optimization fallback.

Completed fixed-window bundles and results remain immutable. New DA bundles live under `outputs/da_zero_shot_adaptive/<window>d/{history,nord}/inputs`. Prepare and launch separately:

```bash
python -m scripts.prepare_da_adaptive_study
bash scripts/launch_da_adaptive_study.sh
```

Preparation runs no model inference. The launch script runs the historical studies and final matched comparison; each run automatically writes Markdown metrics and effective-context summaries. The recovered October 1–10, 2025 NORD requests use 50–59 days rather than excluding them. Their price-only counterparts can retain their full60-day context; comparisons must disclose the actual input lengths. For a controlled test of feature value at identical effective context lengths, build additional paired history-only requests sliced to the covariate context; the current model comparison only matches target products.

The gas/France/Swiss preparer respects the same policy when its input bundle declares adaptive contexts. To prepare its 60-day DA extension:

```bash
python -m scripts.prepare_da_cross_border_covariates \
  --nord-bundle outputs/da_zero_shot_adaptive/60d/nord/inputs \
  --output outputs/da_zero_shot_adaptive/60d/nord_gas_fr_ch/inputs
```

Actual publication clocks of the archived regional quantities remain unverified. Previous-day realized quantities at a delivery clock can occur after the DA cutoff; this remains explicitly retrospective. Adaptive contexts cannot recover missing horizon values.

Capacity measurements are in `outputs/price_context_capacity/summary.md` and `results.json`. Tests used synthetic signals, batch1, FP32, a100-product output and one native thread. Chronos passed85 ordinary days (8160 steps) and rejected86 against its8192-step cap. TimesFM passed160 ordinary days (15360 steps) but rejected160 days plus an autumn repeated hour;159 days is a safe calendar cap for that model. T0 passed365 days (35040 steps), including four synthetic covariates; an instrumented transformer shape confirmed full context processing.365 days is the largest tested length, not a theoretical maximum. These tests establish runtime feasibility, not forecast skill or memory sufficiency at batch8/16.

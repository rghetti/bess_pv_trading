# DA gas and cross-border input preparation

The prepared extension starts from the 60-day NORD forecast bundle and adds PSV gas prices lagged one and seven days; French nuclear and hydro generation, French balance and commercial flows with Italy lagged 24 hours; Swiss generation, consumption and physical flows with Italy and France lagged 24 hours; and Swiss reservoirs lagged one week. Definitions come from the pinned price-feature configuration. No same-day price spread or target-derived field is included.

This is retrospective: archived forecast vintages are unverified, and a previous-day quantity at the same delivery clock can be realized after the DA decision on the previous day. It must not be described as fully gate-available. Complete cases are required across the full history and forecast horizon. In the current pinned snapshot only 234 of the 398 NORD-eligible days are complete; none are in November 2026. French data cause 140 exclusions, with Swiss data also incomplete for 42 days. See `inputs/coverage.csv` for per-day and per-feature counts. Comparing this full block against the 60-day NORD model is possible only on the common products; it does not measure out-of-sample November performance.

The prepared bundle is `outputs/da_zero_shot_context_windows/60d/nord_gas_fr_ch/inputs`. This is preparation only: no experiment has been run. The existing three-model launcher and its synthetic GPU preflight can use this bundle. Run the candidate in its own output directory, then compare with the 60-day NORD-only report:

```bash
PROBE_BUNDLE=outputs/da_zero_shot_context_windows/60d/nord_gas_fr_ch/inputs \
PROBE_OUTPUT=outputs/da_zero_shot_context_windows/60d/nord_gas_fr_ch/run \
bash scripts/launch_da_zero_shot_probe.sh

/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.compare_da_nord_covariates \
  --history-report outputs/da_zero_shot_context_windows/60d/nord/run/report \
  --nord-report outputs/da_zero_shot_context_windows/60d/nord_gas_fr_ch/run/report \
  --history-label nord_only --nord-label nord_plus_gas_fr_ch \
  --output outputs/da_zero_shot_context_windows/60d/nord_gas_fr_ch/comparison
```

Each run and paired comparison now writes a Markdown summary automatically. Quantile forecasts remain marginal; TimesFM 3 weights are non-commercial. To prepare a different context window, invoke `python -m scripts.prepare_da_cross_border_covariates --nord-bundle PATH --output NEW_PATH` against the same pinned snapshot.

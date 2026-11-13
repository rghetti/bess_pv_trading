# DA gas and cross-border input preparation

The prepared combined extension starts from the 60-day NORD forecast bundle and adds PSV gas prices lagged one and seven days; French nuclear and hydro generation, French balance and commercial flows with Italy lagged 24 hours; Swiss generation, consumption and physical flows with Italy and France lagged 24 hours; and Swiss reservoirs lagged one week. Definitions come from the pinned price-feature configuration. No same-day price spread or target-derived field is included. The DA study launcher also prepares separate gas, France-only, Switzerland-only, and France-plus-Switzerland bundles so that the two countries' incremental information can be measured independently.

This is retrospective: archived forecast vintages are unverified, and a previous-day quantity at the same delivery clock can be realized after the DA decision on the previous day. It must not be described as fully gate-available. Complete cases are required across the full history and forecast horizon. In the current pinned snapshot only 234 of the 398 NORD-eligible days are complete; none are in November 2026. French data cause 140 exclusions, with Swiss data also incomplete for 42 days. See `inputs/coverage.csv` for per-day and per-feature counts. Comparing this full block against the 60-day NORD model is possible only on the common products; it does not measure out-of-sample November performance.

The combined bundle is `outputs/da_zero_shot_context_windows/60d/nord_gas_fr_ch/inputs` (234 eligible of 398 NORD-eligible days). It is input-prepared, but model inference and paired evaluation remain to be run. Use the shared launcher in [the DA study guide](zero_shot_da_studies.md), which puts all the external variants and six core context-window studies into one GPU queue, automatically reports them, and scores each variant on its matched products against 60-day NORD-only:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_da_zero_shot_studies --run
```

Each run and paired comparison writes a Markdown summary automatically. Quantile forecasts remain marginal; TimesFM 3 weights are non-commercial. To prepare one standalone variant manually, use `scripts.prepare_da_cross_border_covariates --blocks france --output NEW_PATH` (or `gas`, `switzerland`, or any supported combination) against the same pinned snapshot.

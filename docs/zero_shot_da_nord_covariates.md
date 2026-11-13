# DA zero-shot comparison with NORD forecasts

This comparison uses the same 28-day DA price histories and the four archived NORD forecasts: load, total generation, solar generation and onshore wind generation. The past and delivery-day profiles enter as covariates; they are not target prices. Their units are MW. No city-weather forecasts are included. Historical ENTSO-E issue times are not verified; the study assumes the archived forecasts were available at the DA gate on the previous day. The archives may contain retrospectively revised values. Historical covariate profiles correspond to archived forecast values for those periods, not verified real-time vintages. Any missing covariate in a historical context or forecast horizon excludes that day from all models.

The bundled input `outputs/da_zero_shot_nord_inputs` is pinned to the same snapshot as the completed history-only run. It has 408 eligible days. The usual three-GPU worker, batching and model-specific preflight apply; no fitting is performed. November is reported separately and excluded from selection. TimesFM 3 has non-commercial weight terms. All quantiles are marginal.

Run in tmux:

```bash
PROBE_BUNDLE=outputs/da_zero_shot_nord_inputs PROBE_OUTPUT=outputs/da_zero_shot_nord_run bash scripts/launch_da_zero_shot_probe.sh
```

After the run and automatic report finish, compare the same models on common delivery products:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.compare_da_nord_covariates
```

Outputs are `outputs/da_zero_shot_nord_comparison/scores.csv`, `coverage.csv` and `paired_forecasts.parquet`. Each product is joined by delivery timestamp, auction, gate and model, and actual prices must agree. The baseline report uses the previously saved history-only forecasts and does not refit or rerun them. Recreate the bundle from the same pinned snapshot with `python -m scripts.prepare_da_nord_covariates --output NEW_PATH`.

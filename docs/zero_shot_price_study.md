# Zero-shot auction-price study: preparation status

This work is **not ready for historical inference**. The configuration pins four
checkpoint revisions and the requested development period. A synthetic GPU probe,
a pinned-snapshot coverage audit and a matched-product scoring library are
available. The gate-aware zero-shot replay, DA-conditioned IDA inputs, full
checkpointed launcher and comparison with the rolling statistical forecasts are
not implemented yet. Do not use the commands below as a study launch.

No model is fitted by these tools. The historical period is retrospective:
pretraining may overlap with it. November 2026 is reserved and is not part of
model selection. Weather runs are assumed available six hours after initialization.
The existing rolling input builder retains publication times, cancellation flags,
native DA resolution and explicit local-time DST slots. Its input manifest is
checksummed before examples are built.

Once the weather downloader has finished and no other process is writing DuckDB,
prepare a fresh snapshot and audit it:

```bash
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.prepare_rolling_price_inputs \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --output outputs/zero_shot_price_inputs --prepare

PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.audit_zero_shot_price_inputs \
  --snapshot outputs/zero_shot_price_inputs/merged --output outputs/zero_shot_price_coverage
```

The audit writes daily and monthly observed-target and feature-block coverage;
weather revisions are reported separately from current-cycle availability. The
reference builder can use an older complete weather run, and records this choice
in each example. Missing feature values are not evidence of historical
availability. The 2026-06-10 12 UTC request was previously reported unavailable;
the final archive must be audited again after acquisition.

`scripts/probe_zero_shot_price_model.py` loads an exact revision and produces a
synthetic 28-day, 96-product forecast with nine quantiles on one selected GPU.
It records package and hardware details, runtime, memory and any exception;
it does not establish memory sufficiency for longer contexts or covariates.
The three accelerators do not pool memory. GPU kernel support and the precision
actually used must be verified for each combination. Chronos-2's context limit
must be respected and any 90-day truncation identified explicitly.

The open runtime packages and their model weights have distinct licenses.
TimeH2O's gated weights have non-commercial terms and its checkpoint does not use
past-only covariates; it forecasts target series independently. TimesFM 3 weights
also carry non-commercial terms. Neither license is accepted by this workflow;
outputs from restricted weights must be kept in separate research-only locations.

For a future prediction parquet with columns `delivery_day`, `product_start`,
`auction`, `gate`, `variant`, `actual`, `prediction`, optional
`weekly_reference`, `latest_reference` and `q10` through `q90`, the scoring
script can calculate aggregate, monthly and local-hour metrics. Use
`--match-variants history covariates` to restrict scores to products present in
both variants. Reference-relative scores use only products where that reference
is present; pinball loss and empirical marginal coverage are reported separately
for every quantile. These utilities do not generate forecasts or assert that a
baseline was available at a gate.

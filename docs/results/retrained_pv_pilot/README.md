# Retained per-farm pilot and worst-error review

The approved research pilot uses direct TCNs for all four farms: farm 1
`direct_tcn_02` without timing; farm 345 `direct_tcn_12` without timing;
farm 901 `direct_tcn_timing_05`; farm 910 `direct_tcn_timing_paired`.
The last two include lead hours and sine/cosine of forecast-cycle UTC hour.
Checkpoints are in `models/pv/farm_<id>/pilot_2025_train/direct_tcn/`.

An 80-fit timing ablation compared one paired treatment and 15 additional
HPO trials for each of five farm/family candidates. Timing was retained for
901 and 910, where errors improved; it was omitted for 1, where it worsened
report-period error, and 345, where the TCN improvement was inconclusive.
Farm 345 deliberately uses its stronger summer TCN alternative rather than
the winter-validation pvlib winner. These are research choices informed by
May–August 2026; that period is reused, not a fresh unbiased holdout.
Original validation rankings are preserved in the main study.

## Worst-error analysis

All retained day/auction profiles in train, validation and test are ranked
separately. Each model is evaluated unchanged with operational weather and
with satellite GHI substituted. The forecast direct/global fraction and other
weather are retained; this is not perfect weather and is diagnostic only.
Training errors are in-sample. Report-period comparisons are May–August 2026.
All errors below are daylight RMSE divided by installed AC capacity.

| Farm | Example profile | Forecast-input TCN | Satellite-input same TCN | Satellite-input pvlib |
| --- | --- | ---: | ---: | ---: |
| 1 | 2026-05-15 IDA1 | 15.74% | 6.76% | 6.28% |
| 345 | 2026-06-05 IDA1 | 26.00% | 16.34% | 14.83% |
| 901 | 2026-05-06 IDA1 | 26.09% | 8.46% | 8.93% |
| 910 | 2026-08-02 IDA3 | 32.39% | 33.10% | 25.18% |

Farm 1 and farm 901 examples improve strongly with GHI. Farm 345 improves
but retains substantial residual error. Farm 910's 2 August afternoon remains
inconsistent with both satellite-input models despite only 37.3 W/m² GHI MAE.
Between 15:15 and 18:15 Italian time, a persistent retained low-output episode
has mean measured AC around 2.07 MW, versus 4.84 MW for the neural
counterfactual and 4.07 MW for pvlib. Available weather does not explain this
difference; missing operational information, model errors and measurement
issues remain possible. It is not a confirmed fault or curtailment verdict.

The review partition labels a profile GHI-correctable when the input swap
reduces squared error by at least 50%; remaining RMSE of 10% capacity defines
a large residual. Thresholds are configurable diagnostic labels, not cleaning
rules. `situation_error_shares.csv` contains counts and squared-error shares;
days can occur in several categories because auction vintages differ.
`common_noon_auction_metrics.csv` compares all auctions on identical noon-onward
targets, avoiding unfair full-day versus IDA3-suffix comparisons.

Plots show measured AC, forecast-input and satellite-input neural curves,
pvlib references, and GHI, with AC rating. Companion feature plots show changes
in actual weather inputs. Curves use connected lines; satellite interior gaps
are interpolated and marked. Primary scores use observed satellite GHI only.
Gray shading shows previously CSV-excluded quarters as raw context; these
instances have no model prediction or score. No additional exclusions were
created from this analysis. Residual evidence and review candidates are not
automatic evidence of operator action.

## Reproduce

```bash
PYTHONPATH=src python scripts/ablate_pv_forecast_timing.py --config configs/pv_experiments/timing_features_restored.yaml
PYTHONPATH=src python scripts/diagnose_retrained_pv.py --config configs/pv_experiments/retrained_error_diagnostics.yaml
PYTHONPATH=src python scripts/summarize_retrained_pv_errors.py --config configs/pv_experiments/retrained_error_diagnostics.yaml
```

The ablation config controls trials, seed, workers, models and output path;
shared defaults live in `configs/standard.yaml`. Preserve
`data/farms/curated/timing_ablation_2025_2026/`: this fingerprinted preparation
snapshot removes dependencies on disposable old experiment directories.
Diagnostics load published pilot bundles and save full predictions, rankings,
review candidates and plots under
`outputs/pv_error_diagnostics/retrained_timing_all_farms/`.
The shared pilot registry is `pv_forecast_pilot.models_by_farm`. Trading
replacement remains opt-in: copy the chosen registry mapping to
`pv.forecast_replacement.models_by_farm` and enable replacement. Availability
is conservatively September 2026; earlier retrospective use requires the
explicit existing retrospective-artifact flag.

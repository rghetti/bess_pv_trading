# Previous IDA prices in quarter-hour day-ahead forecasts

The native quarter-hour day-ahead search tested 35 candidate models across
four matched feature sets for February–April 2026. The additional features
are IDA1 and IDA2 prices for delivery products 24 and 48 elapsed hours
earlier, joined using their assumed source publication clocks. On the 82
shared validation days, the best model with those features had 9.46% WAPE
and €12.12/MWh MAE; the best without them had 9.64% and €12.35/MWh.
Those two best-model scores were selected on the same validation period.

The paired validation diagnostic reports lower MAE with the IDA features for
30 of 35 fundamentals candidates and 27 of 35 combined-feature candidates,
with median changes of -€0.233/MWh and -€0.254/MWh respectively. Candidate
results and day-block bootstrap intervals are stored in
`outputs/energy_price_forecasts/nord_da_2026_previous_idas_ablation_v1/`.

## Later rolling comparison

The three best earlier-validation candidates were frozen, with their matched
counterparts and ridge references evaluated on 3 May–31 August 2026. All
10 fits completed without failures, with 18 weekly refits and a one-day
training gap. Each model was scored on the same 107 complete days and 10,272
quarter-hour products; the remaining requested days were excluded by the
shared-input coverage rule. Some related 2026 periods had already been
inspected in earlier experiments, so this is not a globally untouched test.

| Model and base features | MAE without IDA €/MWh | MAE with IDA €/MWh | WAPE without IDA | WAPE with IDA |
|---|---:|---:|---:|---:|
| `interaction_ridge_200`, fundamentals | 16.314 | 16.021 | 11.23% | 11.03% |
| `interaction_elastic_0.1`, fundamentals | 15.711 | 15.473 | 10.82% | 10.65% |
| `elastic_05`, combined | 15.230 | 15.056 | 10.49% | 10.37% |
| `ridge_20`, fundamentals | 17.719 | 17.537 | 12.20% | 12.07% |
| `ridge_20`, combined | 17.153 | 16.991 | 11.81% | 11.70% |

The first row is the single top-ranked earlier candidate; choosing another row
after seeing the later scores is exploratory. WAPE divides total absolute
error by total absolute observed price. The earlier 9.46% and later 11.03%
scores use different periods and model refitting schedules. Historical auction
and feature publication clocks are assumptions, and the results are
retrospective. Detailed coverage, source lineage, forecasts and refit
records are in
`outputs/energy_price_forecasts/nord_da_2026_previous_idas_selected_rolling_v1/`.

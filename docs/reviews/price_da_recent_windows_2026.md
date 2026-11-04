# Recent training windows for quarter-hour day-ahead prices

The earlier 2026 search compared 30-, 60- and 90-day windows with full
available native quarter-hour training histories. On the shared 82
February–April validation days, the lowest MAE came from
`interaction_elastic_0.1_last_90d` with `curve_fundamentals`: 9.01% WAPE.
The same estimator and features with the full history scored 9.74% WAPE.
This candidate was selected on that validation period.

The selected candidates and their matching full-history and feature-set
counterparts were then evaluated with weekly refits over the common complete
days from 3 May through 31 August 2026. The run completed eight fits with no
fit failures, using 107 days and 10,272 quarter-hour products per model.
The first row is the top model selected on the earlier period.

| Feature set | Training history | Later MAE €/MWh | Later WAPE |
|---|---|---:|---:|
| `curve_fundamentals` | Last 90 days | 16.092 | 11.08% |
| `curve_fundamentals` | Full | 15.711 | 10.82% |
| `ida_curve_fundamentals` | Last 90 days | 15.894 | 10.94% |
| `ida_curve_fundamentals` | Full | 15.473 | 10.65% |
| `curve_combined` | Last 60 days | 16.381 | 11.28% |
| `curve_combined` | Full | 15.149 | 10.43% |
| `ida_curve_combined` | Last 60 days | 16.052 | 11.05% |
| `ida_curve_combined` | Full | 14.928 | 10.28% |

The 60-day `ida_curve_combined` fit reported an optimizer convergence
warning during its rolling refits. The other seven fits reported none.
Scores use the sum of absolute observed prices as the WAPE denominator.
Some later dates were inspected in earlier work, and publication clocks are
assumed, so this is a retrospective comparison rather than a globally
untouched test. The run was resumed after interruption: input arrays, source
lineage and saved predictions were checked, but legacy implementation hashes
were unavailable at the initial start.

Details, coverage, warnings and refit records are in
`outputs/energy_price_forecasts/nord_da_2026_recent_windows_selected_rolling_v1/`.

# Day-ahead delivery-time harmonics

The quarter-hour day-ahead comparison allowed each lag and exogenous input to
interact with one, two or three daily sine/cosine harmonics. It evaluated
ridge and elastic-net candidates with and without past IDA prices on 82 shared
February–April 2026 validation days. All 12 fits completed without reported
failures or convergence warnings. The best observed model used two harmonics
and prior IDA prices: 9.44% WAPE, compared with 9.46% for the best
one-harmonic model. Both were selected on that same validation period.

The top three validation candidates, their first-harmonic estimator
counterparts and matched feature arms were then evaluated with weekly refits
and a one-day gap from 3 May through 31 August. All six fits completed without
reported failures or convergence warnings, on 108 complete shared days.

| Feature set | Model | Later MAE €/MWh | Later WAPE |
|---|---|---:|---:|
| With prior IDA prices | Elastic net, two harmonics | 15.539 | 10.68% |
| With prior IDA prices | Elastic net, one harmonic | 15.559 | 10.70% |
| Without prior IDA prices | Elastic net, two harmonics | 15.774 | 10.85% |
| Without prior IDA prices | Elastic net, one harmonic | 15.813 | 10.87% |

The first row was the top earlier-validation candidate. Differences are small;
the later dates have been examined in other studies, and publication clocks
remain assumptions. They do not establish a globally untouched test or
general superiority. Forecasts, coverage, selection records and warnings are
in `outputs/energy_price_forecasts/nord_da_2026_harmonics_selected_rolling_v1/`.

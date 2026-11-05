# Weekday-dependent DA price-lag effects

The feature diagnostic on a separate May–August DA replay showed different
average errors on Sundays and Mondays. Motivated by that retrospective finding,
this CPU search adds local-weekday indicator terms and interactions with the
first three or five recent price lags. Ridge and Elastic Net candidates were
compared with the plain and delivery-hour-interaction Elastic Nets across the
four price, prior-IDA and weather feature arms. The weekday signal is decoded
from the existing local calendar encoding; no holiday feature is added.

All 24 fits completed without recorded failures or convergence warnings.
The shared early selection sample has 9,644 training products through January
and 7,868 February–April 2026 products across 82 days. The best weekday
candidate used five lag interactions and the combined prior-IDA/weather arm.

| Candidate and feature arm | MAE EUR/MWh | WAPE |
| --- | ---: | ---: |
| Weekday Elastic Net, prior IDAs and weather | 12.453 | 9.715% |
| Delivery-hour interaction Elastic Net, same inputs | 12.133 | 9.465% |
| Plain Elastic Net, same inputs | 12.327 | 9.617% |
| Weekday Elastic Net, prior IDAs without weather | 13.102 | 10.221% |
| Delivery-hour interaction Elastic Net, prior IDAs without weather | 13.128 | 10.242% |

The paired MAE gain of the delivery-hour model over the best weekday model
was 0.320 EUR/MWh. Its 95% independent-day bootstrap percentile interval,
using 5,000 resamples and seed 173, was -0.012 to 0.658 EUR/MWh. It omits
serial dependence and selection uncertainty. The weekday specification was
motivated by an already examined later period, and the selection period was
also repeatedly examined. These results do not establish a general ranking or
the 4% target. Neither top-ranked model was a weekday candidate, so the
predeclared later weekday replays were skipped. Historical auction clocks and
weather vintage remain assumptions. Full results and source hashes are in
`outputs/energy_price_forecasts/nord_da_2026_weekday_lag_early_v1/`.

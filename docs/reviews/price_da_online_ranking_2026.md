# Chronological DA forecast ranking

This exploratory CPU post-processing study uses eight saved candidates from
the completed earlier-selected regional-weather rolling replay. Each day it
ranks candidates by MAE on preceding products in a configured trailing window,
using only products whose assumed publication and delivery-end clocks precede
that day's earliest decision. It then averages the top-ranked candidates. With
no eligible history, it uses the equal-weight average of all eight candidates.
It does not retrain the underlying models or use future forecast errors.

The grid tests 7/14/28/56-day windows and the best 1/2/4 candidates. Settings
were selected by pooled May–June MAE on 5,376 products; the chosen rule uses a
14-day window and averages the best two candidates. The fixed rule was scored
on the same 4,896 products across 51 July–August days as the early-selected
prior-IDA plus weather interaction Elastic Net reference.

| July–August forecast | MAE EUR/MWh | WAPE |
| --- | ---: | ---: |
| Fixed early-selected reference | 14.896 | 9.106% |
| Equal-weight mean of all eight | 15.537 | 9.498% |
| Online-ranked top two | 15.207 | 9.296% |

The ranked forecast's paired MAE gain over the fixed reference was -0.311
EUR/MWh. A 5,000-resample independent-day bootstrap with seed 173 gives a 95%
percentile interval of -0.695 to -0.065 EUR/MWh. It omits serial dependence and
model/rule-selection uncertainty. Both periods were already examined in other
studies; historical auction clocks and operational-versus-hindcast weather
vintage remain assumptions. The tested ranking rules did not establish an
improvement or the 4% target.

The configuration is `configs/research/price_da_online_ranking_2026.yaml`.
Input hashes, the complete grid, daily model selections and aligned test
predictions are in `outputs/energy_price_forecasts/nord_da_2026_online_ranking_v1/`.

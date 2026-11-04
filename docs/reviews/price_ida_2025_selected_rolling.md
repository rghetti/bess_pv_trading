# Selected IDA price models: late-2025 rolling evaluation

The 2025 IDA search ranked candidates by pooled MAE on June–August 2025.
The table follows the **first-ranked model for each auction**, selected before
the subsequent September–November evaluation. The latter refits weekly with
a one-day gap, using labels whose assumed publication and product end precede
the refit cutoff. IDA targets use native quarter-hour prices; earlier-cleared
auction prices are inputs. The earlier day-ahead source includes its historical
hourly regime and is expanded as an input where appropriate.

| Auction | Selected feature set and candidate | June–August WAPE | Later WAPE | Later MAE €/MWh | Later relative MAE | Later products |
|---|---|---:|---:|---:|---:|---:|
| IDA1 | `spread_cleared_combined`, `lightgbm_l1` | 5.41% | 6.10% | 6.88 | 0.498 | 8,160 |
| IDA2 | `cleared_only`, `lightgbm_l1` | 5.81% | 6.26% | 7.12 | 0.472 | 8,064 |
| IDA3 | `cleared_combined`, `lasso_1` | 10.08% | 9.08% | 10.54 | 0.534 | 4,176 |

WAPE is total absolute error divided by total absolute observed price;
relative MAE divides MAE by that of the same-product seasonal-lag baseline
on the same observations. The later evaluation has 13 refits per model and
uses shared complete days from 3 September through 30 November; it does not
include every requested delivery day. The selection rule in this study was
fixed before that evaluation, but it is not a claim that all related 2025
experiments were globally unseen. Historical feature publication clocks are
assumptions, so results are retrospective.

The complete evaluation includes the top three validation combinations and
ridge references for every feature set, 27 fits in total. Selecting a better
candidate after inspecting this evaluation would be an additional exploratory
comparison and must not be presented as the frozen-selection result above.
Detailed predictions, refit records, coverage and provenance are in
`outputs/energy_price_forecasts/nord_ida_2025_selected_rolling_holdout_v1/`;
the selection search is `nord_ida_cleared_context_2025_v1/`.

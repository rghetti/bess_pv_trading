# Prior-day Swiss border context for IDA spreads

This CPU comparison tests the previous-day Swiss–Italy commercial schedule,
directional intraday capacities and Italian-NORD/Swiss spot spread alongside
price and calendar inputs. Candidates predict the spread from the latest
cleared auction. The zero-spread candidate is exactly the latest-cleared-price
baseline. The commercial schedule is not a physical-flow measurement; the
source-day midnight publication clock is a retrospective assumption, and the
24-hour elapsed shift does not preserve local clock time across every DST
transition.

The 2026 border archive restricts training to January–February and selection
to matched complete March–April days. All 36 fits completed without reported
failures. Within each auction both feature arms share training and evaluation
products. The best candidate in both arms was the zero-spread baseline, which
does not use any features; the best nonzero border model is shown separately.

| Auction | Days | Products | Latest-cleared MAE EUR/MWh | Baseline WAPE | Best nonzero border model MAE EUR/MWh | Model WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IDA1 | 53 | 5,088 | 6.370 | 4.766% | 6.419 | 4.803% |
| IDA2 | 52 | 4,992 | 7.028 | 5.214% | 7.199 | 5.341% |
| IDA3 | 56 | 2,688 | 12.252 | 9.487% | 12.418 | 9.615% |

The best nonzero border model was Lasso in all three groups. The period was
already examined and both models and features were selected on it; these
results neither establish the 4% target nor rule out other border features or
models. Source lineage, exclusion details and forecasts are in
`outputs/energy_price_forecasts/nord_ida_2026_swiss_border_spread_early_v1/`.

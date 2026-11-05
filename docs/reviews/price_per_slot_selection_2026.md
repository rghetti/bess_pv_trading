# Per-delivery-slot price model selection

The completed per-slot searches fit separate ridge, Lasso and Elastic Net
models for each local delivery clock slot, with at least 50 training products
per slot. The DA search contains separate hourly and quarter-hourly regimes;
the IDA targets retain native quarter-hourly products and earlier-cleared
auction prices. Repeated local DST slots share a fitted model. These are
LEAR-inspired candidates, not an exact reproduction of a published method.

The searches completed 40 DA and 72 IDA fits without recorded failures. Their
older manifests do not contain per-fit convergence-warning fields, so warning
status cannot be verified from those manifests. The best quarter-hourly model
in each search was chosen by MAE on previously inspected February–April 2026
selection products. The references below come from later searches on exactly
the same ordered product timestamps and actual prices; model inputs differ,
so these are matched-score comparisons rather than feature ablations.

| Market | Products | Best slot model | Slot MAE EUR/MWh | Slot WAPE | Reference | Reference MAE EUR/MWh | Reference WAPE |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| DA | 8,252 | Ridge, combined features | 13.869 | 10.905% | Price-and-weather Elastic Net | 12.797 | 10.062% |
| IDA1 | 8,444 | Lasso, cleared prices | 6.594 | 5.178% | Latest cleared DA | 5.986 | 4.701% |
| IDA2 | 8,252 | Lasso, cleared prices | 7.577 | 5.906% | Latest cleared IDA1 | 6.910 | 5.386% |
| IDA3 | 4,128 | Lasso, cleared prices | 11.747 | 9.346% | Latest cleared IDA2 | 11.592 | 9.223% |

The best hourly DA slot candidate scored 9.554 EUR/MWh and 7.980% WAPE on
4,247 January–June 2025 hourly products; it is a different regime and period
from the quarter-hourly scores above. These selection-period results do not
establish the 4% target or rule out other per-slot architectures. Historical
publication clocks are partly assumed and some forecast vintages are
operational-or-hindcast-unverified. Full candidates, selected-day coverage,
snapshots and predictions are in `nord_da_per_slot_search_v1` and
`nord_ida_per_slot_search_v1` under `outputs/energy_price_forecasts`.

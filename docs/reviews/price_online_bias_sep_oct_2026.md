# Online bias correction of later price-only forecasts

The underlying weekly rolling models were selected on February–April data.
For this post-processing experiment, correction settings are chosen by MAE in
September, then held fixed for October. The residual estimator updates using
only prior delivery-day products whose configured publication and product-end
clocks precede the forecast decision. Tested choices include no correction,
3/7/14/28-day windows, 1/4/24-hour local blocks and shrinkage 0/24/96. Actual
publication times remain assumptions. The September and October predictions
were previously inspected, so this is an exploratory follow-up.

| Market | October days | Products | MAE before EUR/MWh | MAE after EUR/MWh | WAPE before | WAPE after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DA | 22 | 2,112 | 17.025 | 16.894 | 7.51% | 7.46% |
| IDA1 | 20 | 1,920 | 9.071 | 8.773 | 3.99% | 3.86% |
| IDA2 | 19 | 1,824 | 10.422 | 10.384 | 4.53% | 4.52% |
| IDA3 | 20 | 960 | 15.077 | 15.077 | 6.49% | 6.49% |

IDA1's available October subset is below 4% before and after correction, but
this does not establish that the target is met across months, auctions or an
untouched prospective sample. IDA3 selected no correction. October target and
input gaps and lag fallback behavior are described in
`price_price_only_sep_oct_2026.md`. Complete grids, selected settings, aligned
prediction files and daily-error plots are saved in
`nord_da_2026_price_only_sep_oct_bias_v1` and
`nord_ida_2026_price_only_sep_oct_bias_v1` under `outputs/energy_price_forecasts`.

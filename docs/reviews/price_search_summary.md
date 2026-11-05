# Electricity-price model search: current evidence

This is a retrospective research summary, not a final model-selection or
deployment claim. WAPE is 100 times total absolute error divided by total
absolute observed price; MAE is reported in EUR/MWh. Lower WAPE across
different periods can reflect a higher underlying price level. Historical
publication clocks are partly assumed, weather forecast vintage is classified
as operational-or-hindcast-unverified, and the periods below have been
examined during model development. The search continues.

## Quarter-hourly September–October 2026

The DA row comes from an early-selected price-and-weather Elastic Net refitted
weekly. The IDA rows are the earlier-selected price-only models, compared on
the same products with the latest cleared auction price: DA for IDA1, IDA1 for
IDA2, IDA2 for IDA3. Each market has a different set of complete days.

| Market | Days | Products | Model MAE | Model WAPE | Latest-cleared baseline MAE | Baseline WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DA | 52 | 4,992 | 16.630 | 7.749% | — | — |
| IDA1 | 50 | 4,800 | 8.781 | 4.116% | 8.885 | 4.164% |
| IDA2 | 49 | 4,704 | 10.319 | 4.817% | 9.939 | 4.640% |
| IDA3 | 50 | 2,400 | 14.964 | 6.996% | 14.998 | 7.013% |

Paired day-bootstrap intervals in the source reports do not establish a gain
over the stronger latest-cleared baseline for IDA1 or IDA3; the selected IDA2
tree did worse. A separate October-only exploratory correction reaches 3.864%
for IDA1 over 20 days, versus 3.896% for its latest-cleared baseline, with an
uncertain paired improvement. This short subset does not establish the 4%
target across months or products. October 23–31 lack targets; other market-
specific missing dates and fallback use are detailed in the linked reports.

The plain DA price-only model on the weather study's shared products scored
16.845 EUR/MWh and 7.850% WAPE; the 0.216 EUR/MWh improvement interval
includes zero. DA errors in that sample are highest in the 12:00–17:59 local
block: 10.660% with weather. In a separate 120-day May–August comparison,
the weather model scored 11.357% versus 11.171% for the paired price-only
model; the paired gain interval also includes zero. The later IDA weather
replay did not establish an improvement on the latest-cleared benchmark and
its IDA2 weather model scored worse. See [DA price and weather](price_da_price_weather_2026.md),
[price-only comparison](price_price_only_sep_oct_2026.md),
[October correction](price_online_bias_sep_oct_2026.md), and
[IDA weather replay](price_ida_spread_weather_2026.md).

## Separate historical product regimes

The hourly DA regime was modeled separately: an early-selected Extra Trees
model scored 7.242 EUR/MWh and 6.647% WAPE on 2,112 hourly products in
July–September 2025. This score is not directly comparable to the later
quarter-hourly DA score. See [hourly DA](price_da_hourly_extra_trees_2025.md).
For IDAs, an earlier selection followed by September–November 2025 weekly
replay scored 6.10%, 6.26% and 9.08% WAPE for IDA1, IDA2 and IDA3,
respectively, on their own complete-day samples; see
[2025 IDA replay](price_ida_2025_selected_rolling.md).

## Scope of the search

The explored CPU families include regularized linear and time-interaction
models, exact median regression, Extra Trees, random forests, histogram
boosting, LightGBM, CatBoost, XGBoost with squared and absolute losses, nearest
neighbors, support-vector regression, and regularized neural networks. These
are tested configurations, not an exhaustive enumeration of every possible
model or feature combination. The [XGBoost](price_xgboost_2026.md),
[neural](price_da_regularized_mlp_2026.md), and
[IDA weather](price_ida_spread_weather_2026.md) reports give examples of
matched comparisons and their limitations. The longer IDA weather and later DA
interaction replays are underway; their manifests, rather than partial scores, determine
completion.

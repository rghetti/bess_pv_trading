# Chronos-2 price inputs and publication assumptions

This is a prepared own-gate DA/IDA input pipeline, not a completed inference
study. The frozen Chronos-2 checkpoint and package are pinned in
`configs/research/zero_shot_price_study.yaml`; no model is trained. The active
study configuration is `configs/research/chronos2_nord_realized_study.yaml`.
It uses the existing 28-day price-history bundles and their publication-aware
price and cancellation handling. IDAs also receive the historical spread to
the latest available non-cancelled earlier auction; the contemporaneous anchor
remains a scoring reference, not a new model input. Seven day-of-week
indicators and the Italian national holiday flag are known calendar covariates
for both history and horizon.

## Inputs and missing values

For load, solar, onshore wind and offshore wind, a single covariate consists of
historical actual values followed by the forecast horizon. Historical missing
actuals first use an available forecast of the same quantity, then interpolate
gaps of at most two hours, and finally fill remaining missing values with zero.
The existing gap logic handles a missing run at an edge with the available
adjacent value. If the horizon is incomplete or unavailable at the gate, the
series is supplied as a past-only covariate. Fill and availability counts are
written to `coverage.csv`. Hydro pumped, river, reservoir and gas generation are
past-only, with interpolation and zero fill; A75 production rows are selected
by technology and pumping consumption is excluded. Aggregate generation
forecasts are retained as a separate forecast-history plus forecast-horizon
series, or past-only if the horizon is incomplete. They are scheduled total
generation, not a sum constructed from actual technology records.

Historical price context ends before delivery-day midnight even at IDA3. Thus
available actuals from the morning of IDA3's delivery day are not used yet.
IDA3's native price context consists of the auction's afternoon products; its
covariate history follows those same product timestamps. No earlier daily
auction forecast is supplied as a future covariate by this study.

## What the publication rules say

All clocks below are local Rome/Brussels time, including DST. The project
chooses to forecast 15 minutes before gates: DA D-1 11:45, IDA1 D-1 14:45,
IDA2 D-1 21:45 and IDA3 D 09:45. The actual gates are 12:00, 15:00, 22:00 and
10:00, respectively. The [GME vademecum](https://gme.mercatoelettrico.org/Portals/0/Documents/en-us/20250101VademecumBorsaElettrica_En.pdf)
documents the IDA gates and IDA3's 12:00–24:00 delivery window.

| ENTSO-E data | Published deadline or documentation | Conservative default for target day D |
| --- | --- | --- |
| Day-ahead total load, art. 6.1.b | At least two hours before the DA gate; significant changes updated. | D-1 10:00: usable at all four decision cutoffs, subject to data coverage. |
| Scheduled total generation, art. 14.1.c | By D-1 18:00 Brussels time. | Past-only at DA and IDA1; horizon eligible at IDA2 and IDA3. |
| Solar and onshore/offshore wind, art. 14.1.d | By D-1 18:00; intraday updates include one by D 08:00. | Past-only at DA and IDA1; horizon eligible at IDA2 and IDA3. The dataset does not establish which intraday vintage is archived. |
| Actual total load, art. 6.1.a | Within one hour after the operational period. | Historical interval end + 60 minutes; mask later intervals. |
| Actual generation by technology, art. 16.1.b/c | Within one hour after the operational period; estimated renewables may be updated. | Historical interval end + 60 minutes; mask later intervals. |

The legal deadlines are stated in [Commission Regulation 543/2013, articles 6,
14 and 16](https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32013R0543).
ENTSO-E describes its [load data](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647979768084-Actual-Total-Load-Day-ahead-Per-Bidding-Zone-6-1-A-6-1-B),
[generation actuals](https://transparencyplatform.zendesk.com/hc/en-us/articles/16648290299284-Actual-Generation-per-Production-Type-16-1-B-C)
and [renewable forecast revisions](https://transparencyplatform.zendesk.com/hc/en-us/articles/16648445340180-Generation-Forecasts-for-Wind-and-Solar-14-1-D).
For example, at 11:45 D-1, a quarter-hour actual ending at 10:45 may be used
under the one-hour assumption; one ending at 11:00 is masked. The tail is
filled first from the available D-1 forecast, then by the remaining gap policy
above. The D forecast is appended as the future segment only when eligible at
the decision cutoff; otherwise that variable is past-only. A forecast known to have been published
before 18:00 can be used sooner when its verified publication timestamp is in
the pinned data; 18:00 is a deadline, not a claim that nothing is released
earlier.

Terna's distinct [market load forecast API](https://developer.terna.it/docs/read/apis_catalog/market/Load_Forecast)
describes a forecast prepared by 06:30 D-1 and sent to GME at 07:00. Its
[wind forecast API](https://developer.terna.it/docs/read/apis_catalog/generation/Wind_Production_Forecast)
shows a publication date, a forecast-creation time and a session, but the
example's publication date alone does not establish an intraday publication
clock. Terna's [market load API](https://developer.terna.it/docs/apis_catalog/load/Market_Load)
describes a distinct market-perimeter historical series refreshed at midnight;
it is not automatically interchangeable with the ENTSO-E total-load series.
These Terna endpoints are not substituted into this configured study.

## Retrospective revisions and limits

The user explicitly chose the latest records available in our dataset, including
later revisions. Therefore publication deadlines control *when the kind of
information may be used*, but the numerical values can contain later updates.
These inputs are not an as-published causal archive. ENTSO-E document creation
time is not proof of publication, and currently downloaded records generally
have no verified historical publication timestamp. Where a verified timestamp
exists, it is respected; otherwise the configured deadline is an assumption,
not a guarantee of actual provider compliance or completeness. The precise
historical Terna/ENTSO-E forecast vintage and the D 08:00 update cannot be
reconstructed from current snapshots. Strict `recorded_only` mode is available
in the config but may leave almost everything missing.

## Pin and prepare after the downloader has stopped

Do not open DuckDB while another process is writing it. Pick a fresh output
directory each time; the example matches the configured snapshot path.

```bash
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.pin_nord_realized_inputs --output outputs/price_realized_snapshot/chronos2/merged
PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_timesfm_realized_study --config configs/research/chronos2_nord_realized_study.yaml --prepare-only --prepare-workers 5
```

Preparation verifies source checksums and writes independent request bundles.
The model worker accepts their dynamic past and future covariates using
Chronos-2's native `past_covariates` and `future_covariates` arguments. For an
explicit inference run, replace `--prepare-only` with `--run` and choose
`--devices`, `--batch-size`, `--threads`, and `--report-workers`; this study
has not been launched. Reusing a prepared bundle checks the model specification,
builder, source manifests, publication policy and feature definitions.

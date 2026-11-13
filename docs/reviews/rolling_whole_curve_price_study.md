# Rolling whole-curve auction-price study

This workflow prepares a pinned market snapshot and supplies a user-launched
chronological development replay. Preparation does not fit models. The default
replay covers delivery dates from 1 October 2025 to 31 October 2026 inclusive;
November is reserved and the runner refuses to score it. The snapshot extends
back to April 2025 to support the hourly DA bootstrap and prequential forecasts.

## Acquisition and preparation

The city-weather acquisition uses the sibling features checkout and shared
DuckDB. The public [Single Runs API](https://open-meteo.com/en/docs/single-runs-api)
documents individually archived 00/06/12/18 UTC ECMWF IFS runs. The six-hour release time is an **assumption**, not a
verified historical publication timestamp. Farm data are not used.

The 12 UTC backfill was started separately for initialization dates from
1 July 2025 through 12 November 2026. Each cycle is committed independently;
if interrupted, rerun the following command after the existing process stops:

```bash
cd /home/rghetti/bess_pv_trading
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.download_rolling_city_weather \
  --features-repo /home/rghetti/bess_pv_trading_features \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --start 2025-07-01 --end-exclusive 2026-11-13 \
  --cycles 12 --lag-hours 6 --max-requests 1000 --run
```

Existing 00/06 weather cycles stop in late August 2026. To extend those cycles
through the latest completed day, after the 12 UTC download is finished, use:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.download_rolling_city_weather \
  --features-repo /home/rghetti/bess_pv_trading_features \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --start 2026-08-30 --end-exclusive 2026-11-13 \
  --cycles 0 6 --lag-hours 6 --max-requests 1000 --run
```

The first day is inclusive and the end date exclusive. Calls are bounded,
cache-aware and record gaps. Run only one database writer at a time. Other
market inputs must also be updated through their existing daily or manual GME
workflows. In particular Swissgrid's energy workbook may not yet contain all
October quantities. Missing-weather dates do not become weather predictions.

After the downloader completes, pin an immutable snapshot; choose a new output
directory when refreshing inputs:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.prepare_rolling_price_inputs \
  --database /home/rghetti/bess_pv_trading_features/data/api/market.duckdb \
  --output outputs/rolling_price_inputs_20261113 --prepare
```

Inspect the gate and weather coverage without fitting models:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_rolling_price_study \
  --snapshot outputs/rolling_price_inputs_20261113/merged --check-only
```

Then launch the study in tmux. Use `--workers` to choose the number of CPU
processes; each estimator and native numerical library uses one thread:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_rolling_price_study \
  --snapshot outputs/rolling_price_inputs_20261113/merged \
  --output outputs/rolling_price_study_20261113 --workers 5 --run
```

Omit `--run` to print the plan without reading the snapshot or fitting. The
default includes 84 independent combinations: seven feature variants, three
families, two window sizes and two IDA target representations. `--variants`
and `--families` restrict a run. This is potentially substantial compute;
progress is printed every 25 completed days per job. Day-level Parquet files
and their checksummed JSON records are installed atomically. Repeating the same
command resumes completed days. A manifest rejects changed snapshot, effective
configuration, implementation or software; use a new output for changes.
An interrupted status file may still say `running` until a resumed run finishes.

## What each model sees

There is one separate DA output curve and one curve for each IDA at the DA
gate and at that IDA's own gate. Outputs use all actual delivery product starts:
92/96/100 quarter-hours for full-day auctions and 48 for the afternoon IDA3.
Training uses complete ordinary-day target curves; spring and autumn days are
forecast using local product slots, with the ordinary repeated-hour prediction
used for both autumn occurrences. No artificial target prices are fitted.

Core inputs are the 1-, 2- and 7-day price profiles of DA, IDA1, IDA2 and IDA3;
curves of earlier auctions only when published; cancellation and fallback
indicators; seven weekday flags; an Italian public-holiday flag; and flags for
the preceding and following day being holidays. Profiles retain 100 local-clock
positions so repeated-hour products are distinct in inputs. There are no annual
or seasonal trigonometric time encodings. Historical DA prices before October
2025 are hourly and repeat into quarter-hour input slots; IDA labels remain
native quarter-hourly. A cancelled earlier auction is replaced only when a
documented cancellation supports the fallback.

The selectable external blocks are:

- `nord_forecasts`: forecast NORD load, total generation, solar and wind.
- `gas`: PSV IGI prices lagged one and seven days.
- `france`: nuclear and hydro generation, France–Italy commercial exchange and
  French generation balance, each lagged by 24 hours.
- `switzerland`: Swiss generation, consumption, net physical exports to Italy
  and France lagged 24 hours, and reservoir level lagged 168 hours.
- `weather`: temperature, shortwave radiation, cloud cover, and 10 m and 100 m
  wind for Milan, Turin, Genoa, Bologna, Verona, Venice, Bolzano, Trieste and
  Aosta; each includes a latest-eligible forecast profile and its change from
  the cycle available at the DA gate.

The national forecasts use the previously approved historical availability
assumption, and prior-day realized quantities are retrospective. No same-day
realized cross-border quantity or target-derived price spread is a feature. The
weather model requires complete current and revision profiles over actual
products; it does not fill an entirely absent archive with synthetic weather.
Coverage and cycle timestamps are saved with each prediction. A same-cycle
revision is zero, rather than a newly issued forecast.

At the DA gate, IDA models receive a DA curve predicted out of sample for each
historical training day and the newly predicted DA curve for inference. They
never see the delivery-day DA clearing as an input at that gate. At each IDA's
own gate, the own-gate model can use published earlier clearings. At intermediate
gates, predictions carry forward the most recent available DA-gate forecast or
earlier published clearing, with the method recorded.

## Fitting, outputs and optimization use

For each daily fit, each target uses published eligible labels in a 60- or
90-calendar-day window. The last 14 eligible days select one of two predetermined
configurations for ridge, multi-task Elastic Net or partial least squares;
remaining earlier days supply the inner fit. Refitting then uses all eligible
days in the window. Every delivery day uses a newly fitted model; its coefficients
stay fixed through later gates. DA starts with native hourly labels and switches
to native quarter-hour labels after 30 eligible ordinary quarter-hour days.
The IDAs compare price targets against spreads from the available price anchor.

The `jobs/` directory stores per-day forecasts, actual prices used only for
subsequent evaluation, selected models and provenance; `metrics/` stores
per-job hourly, monthly, holiday and overall scores. `comparisons/` contains
same-product core versus added-block comparisons, explicit low-coverage flags,
own-gate monthly plots and matched October pointwise-model comparisons when
those artifacts exist. MAE, WAPE, RMSE, bias, reference-relative MAE and spread
direction are reported. Check coverage and the recorded assumptions when
interpreting scores; October is development, not a final independent test.
Overall metrics include documented training-median input substitutions. Paired
feature comparisons require complete selected external blocks for both variants
and count the excluded rows.

The opt-in controller path is exposed through
`attach_rolling_price_provider(prepared, engine, examples, snapshot_id=...,
model_identity=..., cache_directory=...)` after normal continuous-input
preparation. Build the engine and examples from the same pinned snapshot and
explicit study settings, and derive `snapshot_id` and `model_identity` from
their immutable hashes. The provider fits once at each DA gate, caches daily
models across farms, and gives the continuous controller gate-aware planning
curves. Its separately verified publication handler books actual cash once.
The ordinary controller and perfect-price paths remain the defaults; the hook
does not by itself launch an optimization or select a winning model.
The opt-in optimization hook currently fails closed for skipped or cancelled
auctions and when precomputed own-gate price features disagree with the replay's
published prices. The existing continuous preflight filters its schedule using
eventual price availability, so a cancellation-aware causal simulation needs a
separate announcement-time schedule before this guard can be relaxed. The
forecast-only study still records cancellations and retains later auction heads.

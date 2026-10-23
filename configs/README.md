# Configuration layout

Every shipped configuration imports **[`standard.yaml`](standard.yaml) directly**. It contains the shared source schema, PV conversion, farm metadata, battery, SOC, throughput economics, calendar, solver settings, execution and settlement policies, optional comparison flags, acquisition defaults and reporting flags. Experiment files hold only overrides; no experiment inherits another experiment.

- `api_experiments/`: downloads, provider diagnostics, workbook imports and DuckDB acquisition.
- `price_experiments/`: price-model training, calibration and evaluation.
- `full_experiments/`: historical/synthetic optimization and replay, participation studies and prepared-input templates.
- Root files: data preparation, PV generation, audit and reporting entry points.

`api_experiments/manual_market_workbooks.yaml` stages current GME/Terna Excel
files into a separate database while API acquisition continues. Its generated
`import.yaml` is used by `merge-market-workbooks` and `check-market-history` only
after the downloader stops. All defaults are in `manual_market_import` in the
standard configuration; the audit includes preexisting January–March IDA history.

Mappings merge recursively. Scalars and lists replace inherited values; explicit `null` is retained (including a free terminal SOC). Includes are relative to the including file. Data/output paths remain repository-relative: run commands from the repository root. Use the shared `load_config()` loader when reading experiment YAML from Python; a raw `yaml.safe_load()` does not apply inheritance.

## Change farms

Use `full_experiments/gme_day.yaml` for a new single-day study. Select `farm_id: 1`, `345`, `901` or `910` in YAML, or override it on the command line:

```bash
python -m bess_pv_trading resolve-config --config configs/full_experiments/gme_day.yaml --farm-id 345
python -m bess_pv_trading experiment-gme-oracle --config configs/full_experiments/gme_day.yaml --farm-id 345
```

`resolve-config` prints the complete effective YAML, without opening data stores, contacting APIs or solving models. Add `> /tmp/my_effective_config.yaml` to save it. The experiment command runs a real offline study; resolving alone does not launch it.

`${farm_id}` references are resolved **after all YAML and CLI overrides** and keep their numeric type when they occupy the whole value. They link `plant.farm_id` and `forecast.farm_id`; output IDs may embed the same reference. Selected farm metadata from Parquet determines capacity, tracking, tilt and azimuth in pvlib. The selected farm's weather and measured-power series are used; shared location/temperature/irradiance assumptions can be overridden normally. Temperature and wind influence the configured module-temperature model.

`plant.poi_import_mw: auto` and `poi_export_mw: auto` use the selected farm's declared AC nameplate in `farms.<id>.ac_capacity_kw`, divided by 1000. This is an illustrative connection-rating proxy, **not a measured POI rating**. PV forecasts use that rating as their maximum output. pvlib's DC reference capacity is derived by dividing the AC rating by the independent `pv.dc.reference_conversion_efficiency` (0.96 in the shared defaults). The explicit or fitted output coefficient converts DC to AC, followed by the transformer factor and an AC rating cap. Measured production is already AC and receives only the configured kW-to-MW unit conversion. `configs/pv_calibration_2025-01_farm910.yaml` demonstrates the offline fit; `pv_calibration` defaults and output location are in `standard.yaml`. Set measured directional POI ratings once per farm:

```yaml
include: ../standard.yaml
farm_id: 345
farms:
  345:
    poi_import_mw: 1.0
    poi_export_mw: 1.2
```

A numeric `plant.poi_import_mw` or `poi_export_mw` overrides the automatic rule for that study. A farm without capacity or an explicit per-farm rating cannot use `auto`.

`battery.capacity_mwh: auto` and `max_power_mw: auto` resolve independently from `farms.<plant.farm_id>.battery` in the shared standard. The farm sizes are hypothetical study choices:

| Farm | PV AC rating (MW) | Battery capacity (MWh) | Symmetric charge/discharge power (MW) |
| --- | ---: | ---: | ---: |
| 1 | 0.95328 | 0.75 | 0.5 |
| 345 | 1.52460 | 1.0 | 0.5 |
| 901 | 4.16000 | 2.0 | 1.0 |
| 910 | 6.63800 | 5.0 | 2.5 |

Shared efficiencies are `sqrt(0.90)` in each direction, giving 90% AC-to-AC round-trip efficiency including conversion losses. The throughput degradation model is retained. SOC bounds remain 7–93% of the selected capacity, so their MWh values are recomputed by the physical asset constructor. The standard initial/terminal targets are 50%; individual studies set their own targets. Explicit numeric `battery.capacity_mwh` or `max_power_mw` overrides that dimension for the experiment, while `auto` on the other dimension still selects its farm value. Explicit zero-capacity/zero-power PV-only studies remain supported. A missing or invalid automatic farm dimension fails clearly. Resolved run configs save numeric dimensions, efficiencies and farm metadata; previously saved numeric sizes retain their values during resume normalization.

## Integrated January 2026 forecast comparison

The paired configs `full_experiments/continuous_optimization_2026-01_forecast_pilot.yaml` and `full_experiments/continuous_optimization_2026-01_pvlib_soc40.yaml` both select the farm battery automatically. Each runs integrated all-auction planning and a perfect-PV reference. They share 90% initial SOC, a 40% planned daily terminal minimum, a 50% final-day target, progressive GME transaction fees and signed imbalance settlement. Both set `resume: false`, so subsequent runs reuse their stable farm-labelled run IDs and replace the previous outputs. No new comparison runs are implied by resolving these configs.

For example, select farm 901 for the two offline runs:

```bash
PYTHONPATH=src python -m bess_pv_trading experiment-continuous --config configs/full_experiments/continuous_optimization_2026-01_forecast_pilot.yaml --farm-id 901
PYTHONPATH=src python -m bess_pv_trading experiment-continuous --config configs/full_experiments/continuous_optimization_2026-01_pvlib_soc40.yaml --farm-id 901
```

The fitted-model config maps the selected farm to these existing artifacts:

| Farm | Pilot forecast model | Held-out daylight NMAE: PVLib / pilot | Held-out daylight NRMSE: PVLib / pilot |
| --- | --- | ---: | ---: |
| 1 | `tcn_weather_05`, temporal convolutional neural network with extended weather | 5.41% / 3.74% | 7.34% / 6.09% |
| 345 | `ols_weather_a0e00`, ordinary least-squares regression with extended weather | 14.09% / 10.05% | 17.74% / 13.01% |
| 901 | `tcn_core_07`, temporal convolutional neural network with core weather | 9.63% / 7.71% | 14.11% / 12.56% |
| 910 | `mlp_core_05`, feedforward neural network with core weather | 12.27% / 14.08% | 18.43% / 18.19% |

These diagnostic scores use the existing May–August 2026 test set, normalized by farm AC rating. Fits use 2025 observations; within-family hyperparameter selection uses January–April 2026 validation. The farm-345 OLS pilot choice reflects its stronger held-out generalization than the selected neural model, so the pilot model choices and January study are retrospective. Farm 910's MLP remains a comparison candidate despite its worse MAE and January financial result; it is not established as superior to PVLib. There is no retraining when changing farm or battery size.

Core inputs include forecast GHI/beam irradiance, temperature and wind, plus deterministic solar geometry, calendar, forecast-cycle and lead-time features. Extended weather adds cloud layers, dew point, precipitation, snow, visibility and higher-altitude wind. Each auction uses the latest cutoff-available forecast with the configured UTC availability lag. Predictions are AC power with nighttime handling and the farm's rating limit. PVLib is the configured fallback for a missing artifact or a cutoff-ineligible artifact; retrospective artifact use is explicitly enabled for this pilot. Resolved configuration and solve provenance record the model/artifact actually used. The companion config always uses the calibrated PVLib path. Actual historical auction prices remain oracle inputs in both studies.

## New studies and existing results

Start with `include: ../standard.yaml`, select a farm and set the command's dates, inputs and output IDs. Copy the overrides from the closest existing example into the new file rather than importing that example. The current historical defaults use exact planned delivery, signed imbalance settlement after execution when enabled, and committed-grid feedback. Legacy synthetic/prepared examples retain their own previous policies explicitly. All feature flags are visible in the standard, even in older examples.

The generic `gme_day.yaml` uses the configured day to set calendar validity, refreshes causally available PV at each auction, settles actual NORD imbalances, and saves the perfect-information and PV-only comparisons. It generates a fresh output timestamp. Auction dataset native resolutions must match the chosen date; the supplied setup is hourly DA and quarter-hour IDA for January 2025. Participation studies choose native durations separately for each delivery date.

`gme_oracle_experiment.auction_planning_mode` defaults to `integrated`. Select `current_auction_only` for a single-day sequential replay that plans only the current auction's trades while retaining the full remaining physical horizon. For matched participation comparisons, profiles retain integrated planning and `participation_study.sequential_all_auctions_baseline: true` adds a separate all-auction sequential row. That switch defaults to false and is enabled in the refreshed January monthly YAML.

`experiment_outputs.overwrite_existing: true` is the shared default for continuous, participation, single-day GME and synthetic experiments. With resume disabled, rerunning the same ID clears its previous run output, including stale plots, CSVs and checkpoints, before saving the new resolved configuration and results. Single-day GME/synthetic runs replace only their matching farm/day scope; other farms/days under the same ID are retained. Change `continuous_optimization.output_root`, `participation_study.output_root`, `experiment_outputs.root` (single-day GME), or `outputs.root` (synthetic) to keep runs in separate output directories. Set `overwrite_existing: false` to reject replacing nonempty results.

An explicit `resume: true` takes precedence over overwriting and preserves checkpoints; it validates configuration and input fingerprints and still rejects changed scientific assumptions. Continuous reruns acquire the run lock before removing artifacts, so active runs cannot be replaced. Saved numeric configurations retain their original physical settings. Changing farms changes the farm-labelled run IDs in supplied historical studies, so another farm does not resume the farm-910 checkpoint. Missing real data remain errors under the selected data-quality policies.

For mixed API downloads, `api.terna_http_403_policy: continue_other_providers` pauses Terna after a quota/access denial and completes the remaining ENTSO-E and other-provider requests. Use `error` for immediate failure. A partial run saves failed/deferred chunks in its manifest and exits with code 2 after finishing the other providers; rerunning the same command retries missing Terna chunks without refreshing completed data.

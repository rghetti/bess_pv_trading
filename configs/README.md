# Configuration layout

Every shipped configuration imports **[`standard.yaml`](standard.yaml) directly**. It contains the shared source schema, PV conversion, farm metadata, battery, SOC, throughput economics, calendar, solver settings, execution and settlement policies, optional comparison flags, acquisition defaults and reporting flags. Experiment files hold only overrides; no experiment inherits another experiment.

- `api_experiments/`: downloads, provider diagnostics, workbook imports and DuckDB acquisition.
- `price_experiments/`: price-model training, calibration and evaluation.
- `full_experiments/`: historical/synthetic optimization and replay, participation studies and prepared-input templates.
- Root files: data preparation, PV generation, audit and reporting entry points.

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

A numeric `plant.poi_import_mw` or `poi_export_mw` overrides the automatic rule for that study. A farm without capacity or an explicit per-farm rating cannot use `auto`. **Battery size stays independent of farm selection**: change `battery.capacity_mwh`, `max_power_mw`, SOC percentages and efficiencies explicitly. Shared defaults are 5 MWh, 2.5 MW, 10–90% SOC, initial/terminal 50%, with 0.97 combined charge/discharge efficiencies; individual historical/synthetic examples can override them.

## New studies and existing results

Start with `include: ../standard.yaml`, select a farm and set the command's dates, inputs and output IDs. Copy the overrides from the closest existing example into the new file rather than importing that example. The current historical defaults use exact planned delivery, signed imbalance settlement after execution when enabled, and committed-grid feedback. Legacy synthetic/prepared examples retain their own previous policies explicitly. All feature flags are visible in the standard, even in older examples.

The generic `gme_day.yaml` uses the configured day to set calendar validity, refreshes causally available PV at each auction, settles actual NORD imbalances, and saves the perfect-information and PV-only comparisons. It generates a fresh output timestamp. Auction dataset native resolutions must match the chosen date; the supplied setup is hourly DA and quarter-hour IDA for January 2025. Participation studies choose native durations separately for each delivery date.

`gme_oracle_experiment.auction_planning_mode` defaults to `integrated`. Select `current_auction_only` for a single-day sequential replay that plans only the current auction's trades while retaining the full remaining physical horizon. For matched participation comparisons, profiles retain integrated planning and `participation_study.sequential_all_auctions_baseline: true` adds a separate all-auction sequential row. That switch defaults to false and is enabled in the refreshed January monthly YAML.

Saved results retain their original effective configuration. Existing monthly checkpoints are compared after filling new standard defaults; a genuine parameter change still rejects resume. Changing farms changes the farm-labelled run IDs in supplied historical studies, so another farm does not resume the farm-910 checkpoint. Use a new run ID or `resume: false` when changing scientific assumptions. Missing real data remain errors under the selected data-quality policies.

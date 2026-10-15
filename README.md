# PV farm data audit

The first implementation audits four anonymous PV plants before any PV model is fitted. It reads the full Zarr time series and Parquet farm metadata, keeps all raw values, and writes reviewable CSV tables and PNG plots. There is no automatic correction, exclusion mask, time shift, or fault verdict.

The source files live in `data/farms data/`. The `data/` directory is also reserved for later price downloads; raw and downloaded data are ignored by Git.

## Run

Use the existing `bess` environment and run from this repository:

```bash
conda activate bess
python -m bess_pv_trading.audit --config configs/audit.yaml
```

The command creates a new `outputs/qa/<UTC run ID>/` directory. Set `outputs.run_id` in the YAML to give a run a fixed name; a run ID cannot overwrite an existing output directory. The CLI prints the destination at completion. Every run includes a copy of its effective YAML config.

The config chooses input paths, the inclusive UTC historical interval-start window, Milan proxy coordinates, the assumed forecast publication lag, diagnostic bins and thresholds, and plot limits. Paths in the config are resolved from this repository. The bins are *count sensitivities*: changing a bin does not change any source value. The episode thresholds only select review candidates and are also adjustable.

`plots.pinned_events` can name a farm, category, and UTC instant to keep specific observed episodes in the bounded plot set. The supplied config pins the four previously identified examples; pins outside a selected date window are skipped.

## Input conventions

| Coordinate | Meaning |
| --- | --- |
| `time` | Start of a historical 15-minute interval. `power` is average DC kW and `ghi` is mean satellite irradiance in W/m² over `[time, time + 15 minutes)`. |
| `cycle_time` | Timestamp identifying one forecast issue or vintage, usually six hours after the preceding cycle. It is not a delivery timestamp. |
| `lead_time` | Offset from `cycle_time` to a forecast target, at hourly steps from zero to 48 hours in the current store. |
| `valid_time` | `cycle_time + lead_time`. For radiation, this marks the *end* of the forecast hour: an hourly value at 11:00 UTC represents `[10:00, 11:00)`. |

The current store has one forecast per cycle and **no ensemble-member dimension**. Later ensemble data can add members while retaining the vintage and valid-time structure. A cycle is treated as usable from `cycle_time + availability_lag_hours` (four hours in the default config). For each historical quarter-hour the audit attaches the latest forecast whose cycle was available before its forecast hour started. Forecast accuracy tables separately score every forecast vintage against the mean of its four matching historical GHI quarters, where all four quarters are present. No future-issued cycle is used as event context.

The Parquet file supplies the farm-level metadata by `farm_id`, including `total_power`, which is **installed DC capacity in kW**. Solar elevation is calculated at the midpoint of each historical interval with Milan as a proxy; farm coordinates are anonymous, so twilight classifications are approximate. All timestamps in outputs are UTC, including those on daylight-saving transition days.

The new Zarr labels `shortwave_radiation` and `direct_radiation` as hourly interval-average W/m² without identifying the source parameters. If they originated from ECMWF `ssrd` and `fdir`, respectively, their physical counterparts are total shortwave and direct-beam irradiance on a **horizontal** surface. ECMWF defines [ssrd](https://codes.ecmwf.int/grib/param-db/169) as direct plus diffuse surface radiation and [fdir](https://codes.ecmwf.int/grib/param-db/228021) as its direct component on a horizontal plane. That source mapping remains an inference, not a verified field-level provenance claim. The audit compares shortwave to historical GHI as a diagnostic; it does not call either forecast ground truth.

## Review files

| File | Use |
| --- | --- |
| `summary.md` | Coverage, grid and missing checks, event counts, reporting-bin sensitivity, forecast errors, lag correlations, and examples to inspect. |
| `events.csv` | One row per adjacent suspicious episode, with a stable ID, UTC interval, raw extrema, neighboring values, solar elevation, and available forecast context. A simultaneous daytime GHI-zero episode has a portfolio row as well as individual-farm rows. |
| `historical_context.csv` | One row per farm and 15-minute historical interval, with raw power/GHI and time-aligned forecast context; open or plot directly in VS Code or pandas. |
| `forecast_checks.csv` | Forecast-vs-GHI error summaries by farm and lead hour. |
| `review_template.csv` | Event IDs with blank `decision` and `notes` columns for a human review. |
| `plots/<farm_id>/<date>/` | PNG overview and bounded event-window plots. |

An event is a prompt to inspect a period, not a judgement that the underlying GHI or power is wrong. The candidate families include deep-night positive power, daylight zero GHI, unexpectedly low power under substantial GHI, abrupt power or GHI moves, constant positive power, missing measurements, and values outside configurable physical ranges. Event windows include surrounding points to show whether the series recovers. Shared GHI events help distinguish a common irradiance data issue from one plant's power issue.

The CSV uses short decimal representations of the source `float32` values. Reading those columns back as `float32` exactly reproduces the Zarr values; reading them as default `float64` can display tiny decimal differences because of the change in numeric type.

The next step is to review and label these events. Cleaning decisions, pvlib calibration, tracker-axis assumptions, AC conversion, forecast scenarios, and market optimization follow that review.

## Cloud correlation study

Run the supplementary study against a completed audit:

```bash
python -m bess_pv_trading.cloud_check --config configs/audit.yaml \
  --audit-dir outputs/qa/<run_id>
```

It compares shared GHI zeros and sudden DC power drops with available total, low, middle and high cloud forecasts. The main comparisons select intervals with recorded/forecast irradiance or production evidence; they do not infer actual daylight from Milan coordinates. An irradiance-only comparison shows the effect of possible faulty power readings. Milan solar elevation is retained as a reference and selection sensitivity, and never determines a fault verdict. Instantaneous hourly cloud cover is interpolated within one forecast cycle to each measurement midpoint. Cloud tendencies over 15 and 60 minutes also use one cycle, which prevents forecast revisions from appearing as passing clouds. The study writes CSVs, a report, and plots below that audit's `cloud_cover_check/<study_run_id>/` directory.

The comparisons include cloud-bin event rates, month/UTC-hour matched controls for shared zeros, selection and hourly aggregation sensitivity checks, and power-drop rates conditioned on enough preceding production to allow the drop. Cloud cover is a forecast of coverage, and does not measure cloud optical thickness or verify what happened directly over the farm. `cloud_analysis` in the YAML sets the comparison thresholds and plot limits. These define reporting cohorts, not cleaning rules.

The supplementary study checks that the audit's date window, solar proxy, publication lag and event-selection thresholds still match the supplied YAML. Rerun the main audit first if those choices change.

# Reviewed PV replay input and provenance

The continuous research configurations use `data/farms/curated/reviewed_ghi_2025_to_2026_08`, whose manifest SHA-256 is `755b2647feb8a55f06aab63b921ca579d5cf3e71e84460308717c3d4b2cc3583`. Its parent is the `historical_ghi_20261103` corpus. The parent retains unresolved exclusions and is not the target dataset selected for these replays.

The reviewed Parquet retains raw AC observations and exclusion flags for audit. Its `model_target_mw` includes user-reviewed synthetic replacements. At replay, `load_curated_realized_pv` additionally replaces remaining daylight `retained_prior_exclusion` rows with the existing `pvlib_historical_ghi_mw` reference and sets the configured night window to zero. Every quarter is tagged by source and the manifest, corpus and modeling-view SHA-256 hashes are saved in replay provenance. Synthetic or night-policy values are retrospective counterfactuals rather than independent meter observations. Missing daylight references or a mismatch against the manifest cause preflight failure; the loader does not silently substitute a forecast or zero.

A full modeled-view audit found 58,360 intervals per farm, no nonfinite or negative replay target after the declared rules, and the following source counts. `monthly_night_zero` includes intervals whose raw measurement may already have been zero.

| Farm | Metered AC | Configured night zero | Reviewed historical-GHI pvlib | Retained exclusion filled with pvlib | Explicit requested zero |
|---|---:|---:|---:|---:|---:|
| 1 | 35,659 | 21,915 | 732 | 54 | 0 |
| 345 | 34,803 | 21,915 | 799 | 841 | 2 |
| 901 | 36,111 | 21,915 | 245 | 80 | 9 |
| 910 | 30,805 | 21,915 | 5,165 | 475 | 0 |

On 2–3 August 2026 farm 910 has 132 historical-GHI pvlib daylight values and 60 night-policy values; neither the controller replay nor its forecast scores on those two days provide an independent meter-based validation. The raw stored AC values remain untouched and are retained in provenance.

The January 2026 affine ensemble calibration was refitted on the reviewed target, and evaluates the same frozen fit in the January and August development windows. Its output and per-interval source audits are under `outputs/research_inputs/pv_calibration_reviewed`. It uses interval completion as a retrospective hypothetical target-availability assumption; the reviewed reconstruction was actually created later. On the selected August dates calibrated CRPS worsened for all four farms. No evaluation date was used to select a better model or refit the calibration.

The loader caches the checked corpus and view once per unchanged farm/source signature in each worker. This avoids rereading the complete dataset for every delivery day while keeping source hashes and checking for changed files. Forecast weather, auction prices and settlement prices retain their separately recorded sources; selecting the reviewed execution target does not turn historical GHI into an operational forecast.

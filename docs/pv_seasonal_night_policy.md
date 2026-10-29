# Seasonal night policy for physical and ensemble PV studies

The realized-irradiance physical study uses the monthly local-time windows in
`pv_realized_irradiance_search.night_policy`. The intervals are half-open and
refer to quarter-hour starts in `Europe/Rome`. The configured window is applied
both to the physical prediction and, when `night_target_treatment: zero`, to
the fitting target. The untouched target is saved as `measured_ac_raw_mw`, and
`night_target_zeroed` marks every nonzero value changed for that study. Missing
targets stay missing; the original Zarr and frozen curated corpus are unchanged.

The initial windows were derived by sampling apparent solar elevation at
15-minute intervals for 2024 through 2027 at the Cartesian product of
latitudes 44, 45.5 and 47.2 degrees north and longitudes 6.5, 10 and 13.8
degrees east. They enclose elevation above -6 degrees (civil twilight) and add
a 15-minute margin. Reproduce the base table with:

```bash
python scripts/derive_pv_night_windows.py --latitudes 44 45.5 47.2 \
  --longitudes 6.5 10 13.8 --start-year 2024 --end-year-exclusive 2028 \
  --timezone Europe/Rome --elevation-threshold -6 --margin-minutes 15 \
  --sampling-minutes 15
```

May, June, July and August are explicit user-selected overrides of
**21:30–05:00**. Other months use the geometry-derived windows. The exact farm coordinates
are not known, and a sampled geographic envelope and satellite-derived GHI do
not prove that no physical production can occur in any treated interval.
Review source values rather than interpreting a treated zero as a measurement.

New probabilistic ensemble studies use the same monthly policy for predictions,
quantile calibration and treated fitting targets. Their inputs retain
`measured_ac_raw_mw` and `night_target_zeroed`, and the input manifest records
the policy and treatment. Older cached inputs without this treatment cannot be
reused as if they followed the new policy; prepare a fresh study run. Existing
saved predictions and model results are not rewritten.

The independent general `policy.night` remains unchanged for other workflows;
this study policy does not silently modify existing simulation settings. The
seasonal night window also defines which intervals are included in the physical
search's non-night selection score.

An audit of the user-selected windows against all original Zarr quarter-hours
for 2025 through August 2026 is saved under
`outputs/pv_seasonal_night_policy/may_override_raw_2025_2026/`. Reproduce it
with `python scripts/audit_pv_night_ghi.py --config configs/standard.yaml
--output OUTPUT_DIRECTORY`. The audit does not apply reviewed exclusions. Its
GHI values are raw satellite-derived observations; a zero is not proof of zero
irradiance, and missing GHI is counted separately.

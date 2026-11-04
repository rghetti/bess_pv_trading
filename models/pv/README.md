# Selected PV forecast models

The four validation-selected reviewed-GHI model bundles are in
[`reviewed_ghi_2026_selected/`](reviewed_ghi_2026_selected/README.md), one
per farm. Their manifest records the original search and checksums.

The bundles use the reviewed-GHI inference schema. They are not compatible
with the older continuous-optimization `weather_bundle` loader without an
integration change. Older pilot configurations and the disabled-by-default
registry in `configs/standard.yaml` may still refer to removed legacy bundles;
they must not be treated as configured to use these new models.

For all uses, clip PV predictions to zero in the configured local monthly
nighttime windows. The current reviewed-GHI adapter returns raw predictions,
so its caller must perform that clipping.

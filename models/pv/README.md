# Selected PV forecast models

The four validation-selected reviewed-GHI model bundles are in
[`reviewed_ghi_2026_selected/`](reviewed_ghi_2026_selected/README.md), one
per farm. Their manifest records the original search and checksums.

The bundles use the reviewed-GHI inference schema, supported by the continuous
optimization forecast-replacement registry in `configs/standard.yaml`.
Replacement remains disabled by default. Older study-specific pilot
configurations may still refer to removed legacy bundles.

The reviewed-GHI adapter applies the configured local monthly nighttime zero
policy. Forecast cycles must be available at the decision cutoff; using later
cycles differs from the weather vintage used to train the models.

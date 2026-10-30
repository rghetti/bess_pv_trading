# Mayer-style hourly PV post-processing

This study converts each explicit ensemble weather member with the existing farm-specific neural bundle, averages its quarter-hour AC predictions to hours, and fits three non-neural marginal models. The deterministic forecast for the same cycle supplies a control **proxy**; it is not an identified ensemble control member.

The methods are per-lead censored Gaussian EMOS fitted by CRPS, pooled-lead boosted censored Gaussian EMOS fitted by likelihood, and pooled-lead unpenalized linear quantile regression using the 50 sorted member values and the proxy. The R subprocess uses `crch` and `scoringRules`; the R and package versions and the boosted-model package defaults are saved with the output. This is not a trajectory generator or an optimizer input.

The configured dates assume a future completed ensemble and measurement record through November 2026. The currently declared ensemble windows do not cover the full period, and the measured-power store ends in August. The market calendar is also currently bounded through August. Configure verified archive windows, the measurement source and an appropriately supported calendar before a run. Historical publication of a cycle remains a six-hour availability **assumption**. Saved neural pilots were selected retrospectively, so results are retrospective even with chronological post-processor fitting.

Use a read-only coverage check before requesting approval for a concrete study:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python scripts/study_pv_mayer.py \
  --config configs/pv_experiments/mayer_postprocessing_2026.yaml --check-only
```

After the data and run have been approved, omit `--check-only`. A study requires a new output directory and, by default, a complete declared period. The optional `allow_declared_gaps` setting evaluates only explicitly listed windows and records all uncovered days; it never bridges or fills them. The first evaluation cutoff determines which training labels are actually available, including the preceding day. Input measurements and CSV-excluded quarters are retained, and only complete four-quarter hours with valid, in-range measurements and complete forecast paths enter fit or scoring. Missing required weather leaves that quarter's forecast missing; the model processes the remaining contiguous subsequences independently without weather-gap interpolation. Existing configured nonnegative weather treatment still applies and is reflected in the source fingerprint.

Daylight uses the solar zenith at the hour midpoint calculated from configured farm coordinates, which are documented proxies, not verified farm positions. Valid daytime zeros are retained. Zero ensemble variance is not floored: the affected hour cannot be scored by basic log-variance EMOS. An unsuccessful fit is recorded per lead. Crossed linear quantiles are sorted when `rearrange_quantiles` is enabled; raw quantiles and crossing flags are retained. Quantiles outside physical bounds are not clipped, and their hours are excluded from the matched distribution comparison. Reports include the number of excluded or unmatched hours so this common subset is visible.

Each farm directory contains original quarter-hour member predictions and hourly forecasts in MW, normalized inputs and target masks, fitted models and R diagnostics, evaluation distributions, monthly and overall score tables, and a reliability plot. The manifest identifies forecast/target fingerprints, model and exclusions hashes, the exact configuration, forecast-cycle assumption and retrospective limitation. Empirical ensemble and R censored-normal CRPS are exact; linear-quantile CRPS uses numerical integration with constant tails. Interval scores use interpolation on the documented quantile grid. The reliability plot uses `P(Y <= q)` and is descriptive in the presence of boundary masses.

For a completed run created before quantile rearrangement was enabled, run `scripts/report_pv_mayer_rearranged.py --run-dir <run-directory>` to make a separate report using the saved predictions. This never refits a model or modifies the original results.

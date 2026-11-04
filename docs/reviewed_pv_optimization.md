# Reviewed PV forecasts and counterfactual optimization replay

`configs/full_experiments/reviewed_pv_optimization.yaml` imports the shared
configuration and enables the selected reviewed-GHI model and reviewed PV
execution targets. A concrete experiment must also define its dates, market,
battery and solver settings. This configuration explicitly allows use of the
November 2026 model artifacts in earlier historical periods; such a replay is
retrospective. It is not evidence that the model existed at those earlier
decision times. The default shared configuration leaves model replacement off
and does not allow retrospective artifacts.

Each 15-minute execution target is taken from `model_target_mw` in the reviewed
curated dataset. This preserves measured labels and the previously approved
synthetic replacements. Where a prior daylight exclusion remains, the replay
uses the stored historical-GHI pvlib reference for that interval. The local
monthly night window is set to zero. An unrepresented interval, an unavailable
daylight reference or another missing daylight target stops the replay rather
than using a forecast or inventing a zero. The reader checks the curated corpus
and its manifest and records the source of each interval.

The execution profile is a simulated counterfactual wherever a target is
synthetic. PV energy, imbalance and profit computed from those intervals are
counterfactual outcomes, not observed plant or market performance. The same
execution profile must be shared by strategies being compared. Forecasts are
formed from weather available at each decision cutoff; the future execution
profile is not a forecast input. In simulated operation a reconstructed value
may stand in for a completed-quarter disclosure, which is a counterfactual
telemetry assumption rather than historical evidence.

The selected models were trained on historical GHI and weather from the
preceding delivery day's 00 UTC cycle. Deterministic optimization forecasts use
the latest causally available weather cycle at each cutoff, including later
cycles as requested. The cycle, assumed weather availability and model
artifact availability are recorded. This changes the weather distribution
relative to training and has not yet been evaluated. Ensemble forecasts use
the ensemble cycles present in their configured archive; members are processed
independently. The model adapter requires an enabled monthly night policy and
returns zero during its night window. Sequence models use the forecast rows in
the requested horizon, so changing the remaining horizon can change their
context relative to the full-day historical evaluation.

## Proposed later-cycle evaluation

Use the already selected four models and the May–August 2026 test split. At
each available DA, IDA1, IDA2 and IDA3 decision cutoff, select the latest
weather cycle causally available using the configured publication lag, align
its hourly fields to 15-minute products, and score eligible daylight targets
from the reviewed corpus. Report cycle and lead-time coverage, missingness,
model and weather availability assumptions, per-farm and per-auction nRMSE,
nMAE and bias, plus measured-versus-synthetic target breakdowns and matched
preceding-day 00 UTC comparisons on the same products. Keep 910's synthetic
August results separate. Compare only products eligible for each auction;
selection remains frozen and no model is retrained.

The 2026 period was already reviewed, and model selection used earlier 2026
validation data. Thus it is a retrospective comparison, not a historically
available live forecast. A concrete run plan, resource estimate and explicit
approval are required before running this evaluation or an optimization.

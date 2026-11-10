# Same-day auction planning: January–August 2026 SOC sweep

Run 76 continuous greedy grid-tracking trajectories: farms 1, 345, 901 and
910, each with a hard planning target of 7%, 10% to 90% in five-point steps,
or 93% at each delivery-day midnight. Initial SOC is 50% for every trajectory.
Each auction, including IDA3, plans only the relevant part of its delivery
day. Submitted quantities remain fixed, and a valid prior planned order is
used as the existing fallback when an auction optimization is infeasible.
The greedy tracker does not enforce the planned target on actual SOC.

Score Italian-local delivery days from January 1 through August 29, 2026;
start with the December 31 day-ahead auction for January 1. The reviewed PV
data extends through August 31, but the selected models require complete
weather forecast contexts that are unavailable for the final two days. The
preflight must verify all required forecast, PV and market products; it must
fail rather than silently shorten the replay.

Use the same reviewed AC PV data, night policy and labeled synthetic or
reconstructed replacements as the prior sweeps. Use the latest eligible
deterministic weather issue with the established six-hour availability lag
and 48-hour context; use the reviewed four-cycle BiGRU models for farms 1 and
345 and TCN models for farms 901 and 910. Model training, loss functions,
hyperparameters and selection split are recorded in
[the original study plan](ida3_next_day_january_2026_plan.md). No training,
calibration, new model selection or downloading is part of this run. Model
selection used 2026 validation data, so comparisons are retrospective and
subject to selection leakage. The tracker uses realized interval-average PV,
and the planner uses actual auction prices as an oracle; actual settlement
prices are used for accounting. These are not causal live-market results.

After all four farms pass preflight, schedule eight independent trajectories
at a time and refill workers as they finish. Each uses two Gurobi threads,
with a relative MIP gap of 0.0001; up to 16 solver threads can oversubscribe
the 10 logical CPUs reported on this machine. Forecast and market inputs are
prepared once per farm and shared across targets; successful batches release
the temporary preflight cache. Daily resumable checkpoints are disabled for
this study, and solver-attempt histories are written at completion or on a
handled failure rather than rewritten after every auction. An interrupted
trajectory must therefore restart from the beginning. Final analysis outputs
and input provenance are still retained. The prior 31-day sweep occupied
approximately 7 GiB including checkpoints; actual storage for this longer
period will depend on the retained outputs and solver behavior.

Report each target's operating profit, auction and imbalance cash flows,
modeled degradation, absolute imbalance, curtailment, battery usage, actual
SOC and fallback counts in per-farm Markdown tables and plots, with a combined
index and machine-readable results. Each farm's targets share its input data;
record provenance and failures. Profit includes modeled degradation but does
not value battery energy remaining at the end of the replay, which can bias
comparisons between targets with different final SOC. No target is selected
as a winner automatically.

# January extended IDA3 sweep with only the final midnight target

Use the same farms, 19 SOC targets, retrospective forecasts and realized PV,
auction-price oracle, scoring period, worker count and solver settings as
[the two-midnight January sweep](ida3_soc_sweep_january_2026_plan.md). This is
an adjacent 76-trajectory experiment, not a new model training or calibration.
The first auction is December 31 DA for January 1; February 1 is unscored
planning context.

At each IDA3 optimization, remove the SOC constraint at the midnight between
the current and following delivery days, but impose the selected target as a
hard equality at the following day's midnight. DA, IDA1 and IDA2 continue to
target that delivery day's final midnight. Reoptimizing at the next IDA3 moves
the final target forward, and the greedy real-time tracker does not enforce
planned SOC, so no actual midnight equality is guaranteed. The battery's
physical energy and SOC bounds remain active at all times.

After all four farms pass preflight, run up to seven trajectories concurrently
with two Gurobi threads per worker, refilling workers as jobs finish. Cache a
shared input snapshot per farm within the batch; each target retains its own
configuration and provenance. Retain caches on failure and remove them on
success. At most 9,652 auction submissions are expected. The current machine
reports 10 logical CPUs, so 14 solver threads may oversubscribe it.

Generate an independent Markdown report and compact plot for each farm,
including auction revenue, profit, imbalance, curtailment, degradation,
actual SOC and fallback counts across targets. Compare each completed target
with the same farm and target in the two-midnight sweep, matching prepared
input fingerprints and checking realized PV and auction/settlement prices.
Report missing, failed or incompatible pairs instead of hiding them. The
comparison can be regenerated from completed results if the original sweep is
still running when the adjacent sweep finishes. Profit does not account for
residual battery inventory, and the actual-price oracle and model-selection
leakage make this a retrospective comparison.

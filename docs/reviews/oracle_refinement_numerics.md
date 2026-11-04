# Oracle refinement numerical diagnostic

The provisional January audit found seven rejected second-stage refinements in
each of the farm 1 `ensemble_price_oracle` and `ensemble_oracle_gap_fallback`
cases. Each retained its independently validated first-stage control, without
falling back to the tracking controller. Six rejections per case involved a
grid-balance residual around `8.56e-7` MW; the other exceeded the reconstructed
imbalance cap. These are rejected candidates, not violations in the executed
first-stage plans.

For one of the six, the 2026-01-16 09:00 UTC decision, a diagnostic reconstructed
the 51 saved PV paths, their weights, the state, the then-submitted commitments
and the actual settlement curve. The horizon came from the immutable
`reviewed_january_matrix_v1` farm 1 `ensemble_price_oracle` output. The configured
study has no hard terminal target, uses throughput degradation, a common auction
price curve and one Gurobi thread. It uses independent ten-second stage limits.
This is a replay of a recorded optimization input, not a new closed-loop study.

| Variant | Result | Reconstructed candidate MWh | Declared cap MWh |
|---|---|---:|---:|
| Original `FeasibilityTol=1e-8` | Rejected physical grid-balance residual `8.56e-7` MW | Not reached | 0.142675519779 |
| Recompute grid from extracted controls, same tolerance | Rejected reconstructed cap | 0.142675556215 | 0.142675519779 |
| Original extraction, `FeasibilityTol=1e-9` | Economic refinement passed the existing checks | 0.142675343502 | 0.142675519779 |

The configured MIP gap is `1e-4`; an `optimal` solver status here is relative to
that tolerance, not a claim of an exact mathematical minimum. Reconstructing
grid quantities alone did not make this candidate acceptable under the declared
cap and existing numerical allowance. Tightening the feasibility tolerance
worked on this one instance, but its performance and effect on other horizons
have not been established. No study configuration was changed on this evidence,
and the ongoing matrix remains pinned to its original solver settings.

Subsequently, all seven rejected horizons from the farm 1 `ensemble_price_oracle`
case were reconstructed independently and solved at both tolerances. All seven
reproduced rejection at `1e-8`; all seven used the economic refinement and passed
the existing validation at `1e-9`. These horizons were selected because they had
already failed, so the result is not an estimate of rejection rates or runtime
across the complete study. The captured results are in
`oracle_rejected_horizons_tolerance_comparison.json`. The second, gap-fallback
case had the same seven rejection times, but was not independently replayed in
this comparison. The running study remains unchanged.

An additional recorded counterexample was found on farm 345 in the deterministic
oracle case at 2026-01-15 15:45 UTC. At `1e-8` its second stage exceeded the
reconstructed cap; replaying it at `1e-9` still rejected the second stage, with
the same reconstructed `1.928068159e-6` MWh against a cap of `9.640340796e-7`
MWh. The first-stage action remained the accepted plan. Thus the tighter setting
is not a general fix, even among the already observed rejection cases. See
`oracle_farm345_tighter_tolerance.json` for the replay diagnostic.

For that same farm 345 horizon at `1e-9`, individual diagnostic re-solves with
Gurobi `Presolve=0`, `NumericFocus=3`, and both settings together each passed the
unchanged reconstructed cap check. The baseline at the same tolerance again
failed it. Each option was tried once on this small 29-interval deterministic
horizon; these observations neither identify a unique cause nor establish the
runtime or reliability of these settings on other instances. No solver-default
change was made. The full results are in
`oracle_farm345_solver_options.json`.

At the original `1e-8` feasibility tolerance, a separate `NumericFocus=3`
replay of the seven selected farm 1 horizons and the farm 345 horizon accepted
four refinements and still rejected four. This does not support treating
numerical focus alone as a guaranteed recovery. The selected-input results and
diagnostics are in `oracle_rejections_numeric_focus.json`.

An opt-in budgeted retry was then added: after a physical or cap rejection it
can reset the invalid solution and re-solve with configured numerical focus and
a tighter feasibility tolerance, using only remaining economic-stage time.
The first-stage action remains available if retry fails or time has expired.
With focus 3 and tolerance `1e-9`, it recovered all eight selected recorded
horizons while preserving the cap checks. Those cases were selected because
the initial refinement failed; the result is not a general success rate. The
two diagnostic files `oracle_rejections_budgeted_retry.json` and
`oracle_rejections_numeric_focus_tight.json` retain the individual outcomes.
The existing January matrix did not enable the retry.

The adjacent `oracle_rejection_*.json` files contain the individual diagnostic
outputs, including solver stages, reconstructed components and failure reasons.
The original-extraction comparison can be repeated with
`scripts/replay_oracle_horizon.py BATCH JOB_ID INTERVAL_ID --output PATH`, and
`--feasibility-tolerance 1e-9` selects the tighter-tolerance diagnostic. The
script reads checksummed saved scenario history and permits only throughput
degradation with no hard terminal target or floor; it refuses to reconstruct
unsupported battery segment state. A working Gurobi license is required.

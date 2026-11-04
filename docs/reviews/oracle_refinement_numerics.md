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

The adjacent `oracle_rejection_*.json` files contain the individual diagnostic
outputs, including solver stages, reconstructed components and failure reasons.
The original-extraction comparison can be repeated with
`scripts/replay_oracle_horizon.py BATCH JOB_ID INTERVAL_ID --output PATH`, and
`--feasibility-tolerance 1e-9` selects the tighter-tolerance diagnostic. The
script reads checksummed saved scenario history and permits only throughput
degradation with no hard terminal target or floor; it refuses to reconstruct
unsupported battery segment state. A working Gurobi license is required.

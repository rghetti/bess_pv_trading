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

The adjacent `oracle_rejection_*.json` files contain the individual diagnostic
outputs, including solver stages, reconstructed components and failure reasons.

# Integrated Naumann resolution comparison

Status: prepared, not approved or run. Run experiments only after explicit approval;
the user starts them. This is an initial comparison, not a claim that the model is
physically calibrated or that further structural simplification is unnecessary.

## Scope

Farm 1, 1–3 January 2026, integrated auction planning and grid tracking. Three
otherwise matched configurations differ only in nominal C-rate bin width:
0.125C, 0.25C and 0.5C. Their upper-edge stresses also affect executed aging;
therefore their different SOH trajectories mix dispatch response and rate-bin
bias. Replay every saved trajectory independently under the same exact-rate
and depth assumptions before interpreting differences. Keep the original 0.25C
case as a within-study reference, not as physical ground truth.

The PV input uses the latest cutoff-eligible deterministic weather forecast with
six-hour availability lag and the existing retrospective farm-specific PV model;
the latter was selected after the study period. Curated realized PV, including
reviewed substitutions, is the execution reference. Auction prices are realized
curves used as perfect foresight: these are retrospective diagnostics, not a
prospective forecast test. Missing market products follow the workflow's existing
skip-if-absent/fail-if-partial policy. No training, fitting, train/test split or
hyperparameter selection occurs here; the models are pre-existing.

All cases retain four depth segments, the current calendar and square-root error
budgets, 25 C, fresh battery, zero self-discharge, 50% daily planned terminal SOC,
no extended IDA3, 80% EOL, and the shared 2025 stationary-pack replacement proxy.
They use 8 Gurobi threads, a 240-second per-solve limit, 1e-4 relative MIP gap,
and separate output paths. Solver feasibility and integrality tolerances are
1e-4. The validator uses max(physical tolerance, twice the solver feasibility
tolerance), giving 2e-4 for its ordinary residual checks; integrality uses 1e-4.
Aging exposure residuals are in scaled micro-units, while other checks include
MWh and MW. This relaxed numerical policy can admit larger physical and aging
residuals and does not guarantee the configured aging-approximation budgets;
report solver violations and independent replay errors. Stop on validation failure.
Detailed Gurobi output is disabled. No run should overlap a database downloader.

Collect each auction's solver status, incumbent, bound, gap, runtime, model sizes,
and numerical violations; report failures and time-limited solves instead of
averaging successful solves alone. Compare committed daily SOH and capacity,
calendar and cycle loss, actual throughput and market cash, estimated noncash
wear, imbalance, and total wall time. Independent rainflow on the full executed
path with carried residues must compare piecewise and analytic depth stress at a
common reference C-rate; it does not validate cell calibration or the variable-rate
model. The report also reweights recorded interval exposure by the actual
interval rate to isolate the upper-bin approximation on the same path. This is a
local-rate surrogate, not completed-cycle mean-rate rainflow aging; it neither
reoptimizes dispatch nor updates the original capacity trajectory. Check whether
any fixed-capacity baseline trajectory violates faded bounds.

Selection criterion: prefer the least computationally expensive setting that finds
validated solutions and has acceptable health and economic deviations against a
common replay benchmark; no acceptable error threshold is presumed. Solver gaps,
three winter days, model calibration uncertainty, and historical look-ahead limit
generalization. All cases use the structural simplification that evaluates
square-root graphs only at endpoints and local midnights. This comparison does
not isolate its runtime effect against the earlier formulation. Fixed-path and
midnight software checks do not replace an economic-dispatch comparison.

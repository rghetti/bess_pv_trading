# Fixed-rate depth-resolution comparison

Prepared for explicit approval before execution. Farm 1, January 1–3, 2026,
uses integrated auction planning, current-auction commitments, grid-tracking
execution and a 50% daily planned terminal SOC. The three cases use 4, 8 or 16
nominal cycle-depth segments and a fixed 0.5C aging factor in both optimization
and execution. Actual power is still constrained; its effect on aging is omitted.
At 25 C, calendar stress continues to depend on the full SOC path. The battery
starts fresh and commits SOH at Europe/Rome midnight; there is no self-discharge.

Inputs and limitations match the former three-day study: eligible deterministic
weather vintages with an assumed six-hour lag; preselected retrospective PV model;
curated realized PV including reviewed substitutions; actual auction clearing
prices as a perfect-foresight oracle; 15-minute dispatch and settlement. Missing
auction policy skips absent products and rejects partial products. No training,
model fitting or split occurs. These are retrospective experiments, not a
prospective forecast evaluation. No automatic replacement is modeled.

All cases use the 2025 stationary pack cost proxy, 80% reporting EOL, same
calendar and square-root resolution, eight Gurobi threads, a 240-second per-solve
limit, 1e-4 relative MIP gap, 1e-5 solver feasibility and integrality tolerances,
and 1e-4 ordinary validation tolerance. The fixed reference-rate model is not a
bound on actual-rate aging. Validation tolerances have mixed physical and scaled
exposure units; inspect violations, replay bounds, and solver feasibility.

Run cases sequentially in separate directories. Report every solve status, gap,
model size, runtime, numerical residual and any failure. On completed cases,
report daily SOH, calendar and cycle loss, capacities, market cash and noncash
wear, throughput and imbalance. Compare executed paths using independent
rainflow with analytical and segmented depth stress at fixed 0.5C, retaining
half-cycle residues across midnight. Reweight interval cycle exposures by actual
rate only as a sensitivity, not as an independent cycle-mean benchmark. Rank
runtime and agreement without treating the finest case as empirical truth; high
gaps, calibration uncertainty and three winter days limit all conclusions.

The analytic worst absolute interpolation errors of the unscaled depth-dose
function G(D), over D in [0, 1], are 3.23059e-5, 1.20269e-5 and 4.03827e-6
for 4, 8 and 16 segments, respectively. These are per-depth-function errors
before the squared C-rate factor and half-cycle weights; they are not annual
SOH-error bounds. The measured square-root chord bound is 0.000195313 per
component, and the annual calendar-interpolation bound is 0.000481687 SOH.
Neither includes physical calibration uncertainty or solver feasibility errors.

# Daily SOH and depth-dependent LFP aging

`degradation.model: naumann_lfp` is an opt-in, Naumann-calibrated reduced aging
model. Existing throughput and discharge-only Xu modes retain their behavior.
The example [configuration](../configs/full_experiments/naumann_lfp.yaml) imports
`configs/standard.yaml` directly and declares every study assumption. It is not
a Huawei/BYD calibration and has not been used to start a dispatch experiment.

## Physical meaning and calibration

Original internal nameplate capacity `E0` never changes. Stored energy `e` is in
internal MWh, charge/discharge commands are AC MW, and available capacity is
`E0 * committed_soh`. Conversion efficiencies are applied exactly once.

The reference is the Sony/Murata US26650FTC1 cylindrical LFP/graphite cell:
2.85 Ah rated, approximately 3 Ah measured. Pack scaling assumes identical,
balanced cells, representative constant voltage and capacity loss in MWh
proportional to measured Ah loss. Temperature is held at 25 C; there is no
thermal simulation, resistance growth or changing conversion efficiency.

Sources:

- [Naumann et al., calendar aging, 2018](https://doi.org/10.1016/j.est.2018.01.019).
- [Naumann et al., cycle aging, 2020](https://doi.org/10.1016/j.jpowsour.2019.227666).
- [Author thesis with calibration tables](https://mediatum.ub.tum.de/doc/1434981/1434981.pdf).
- [Xu's cycle-memory construction](https://arxiv.org/html/1707.04567v2).

The persistent exposures `A` and `B` determine capacity losses as fractions of
original capacity:

```text
q_calendar = sqrt(A)
q_cycle = sqrt(B)
health_readout = 1 - q_calendar - q_cycle
```

Independent component exposures are our explicit variable-condition extension.
They do not reproduce the ambiguous combined-loss virtual-FEC expression in the
2020 paper verbatim, and the paper's dynamic-validation errors do not validate
this derived implementation. Shallow-cycle reversible capacity recovery,
cycling mean-SOC interactions, aging knees and cell imbalance are omitted.

## Calendar exposure includes charging and discharging

At 25 C, with SOC `z` relative to the day's available capacity:

```text
K(z) = 1.2571e-5 * [2.8575 * (z - 0.5)^3 + 0.60225]
dA = K(z)^2 * dt_seconds
```

The coefficient is in fractional capacity per square-root second, converted
from the published percentage coefficient. Calendar damage accrues during all
elapsed operation, including idle intervals. It is not restricted to periods
with zero battery power.

For a constant-power transition, SOC is affine. Four-point Gauss-Legendre
quadrature integrates `K(z)^2` exactly because it is a degree-six polynomial.
This integrates the entire transition, not simply its initial, mean or final
SOC. Real command changes define separate physical paths; bookkeeping splits
leave the exposure unchanged up to floating-point roundoff.

The optimizer evaluates a certified upper PWL stress curve at those same four
points. Each segment's error is bounded at all stationary points of the
polynomial-minus-chord difference. A global upward shift covers concave parts.
The grid is refined until its configured annual calendar-SOH budget is met.
Execution uses the analytical stress directly. The interpolation error bound
does not cover empirical calibration uncertainty.

## Cycle exposure and rate

The published constant-condition fractional loss is

```text
k_C(c) = 0.063*c + 0.0971
k_D(D) = 4.0253*(D - 0.6)^3 + 1.0923
q_cycle = k_C(c) * k_D(D) / 100 * sqrt(FEC)
```

A corresponding full-cycle squared-loss exposure is

```text
F(D,c) = D * [k_C(c) * k_D(D) / 100]^2
```

Repeating `N` identical full cycles gives `B = N*F`, reproducing the constant
condition equation with `FEC=N*D`. Depth and fitted C-rate use original nominal
capacity, consistent with the experimental convention. `P_internal/E0` is the
energy-based proxy for current/Ah C-rate. The effective rate relative to
remaining capacity is reported separately and has a configurable power cap.

Two independent depth-memory banks track `e` and `E0-e`. Physical discharge
produces counted outflow in the first; physical charge produces counted outflow
in the second. Each contributes half-dose. Both banks fill and empty the
shallowest available segment first. They are history states, not absolute SOC
buckets, and must never be reset or resized when capacity declines.

Nonuniform depth knots resolve shallow excursions. Positive depth-dose
increments are required; nonconvex marginal slopes are allowed because binaries
enforce the canonical ordering. Finite bank segmentation remains an
approximation to rainflow aging; independent rainflow comparisons and its
resolution error are separate from calendar interpolation.

Since `F(D,c)=G(D)*k_C(c)^2`, the MILP disaggregates one scalar depth dose by
rate bin. It does not need a segment-by-rate flow array. Upper-edge rate bins
are conservative for this local attribution and use a deterministic lower-bin
tie convention. They do not reproduce arbitrary completed-cycle mean-rate
aging, particularly asymmetric charge/discharge cycles. Rates below 0.2C
extrapolate the fitted rate range. Although a 1% test was performed, the published
depth-factor fit uses 5--100% endpoints; depths below 5% extrapolate that fit,
and depths below 1% also fall below the experiment's characterized depths.

The optional `fixed_cycle_rate_c` replaces rate bins with one fixed stress factor
in both planning and execution. It removes rate-dependent cycling damage and its
binary rate choices, while preserving the physical nominal and effective power
limits. This changes the aging model; a fixed value is neither an upper nor a
lower bound on variable-rate aging. The original bin mode remains the default.

## Chronological capacity updates and economics

Executed commands accumulate exposures immediately. **Committed physical SOH
and available capacity change only at Europe/Rome calendar midnight.** This
clock is independent of auctions, settlements, optimizer cutoffs and horizon
ends. DST days contribute their actual 23 or 25 elapsed hours. Partial days
retain uncommitted exposure in checkpoints. At midnight the previous interval
finishes before the new day's control or auction actions.

The selected study does not extend IDA3 through the following delivery day. Each
auction plan targets 50% SOC at the end of its delivery day, using that day's
forecast available capacity. This is the single terminal constraint for that
delivery day, separate from the chronological health handoff. The grid-tracking
executor reacts to actual PV and may finish at a different realized SOC.

The optimizer predicts future loss without mutating actual health. Its current
day capacity is a parameter. Future midnight capacity bounds depend on forecast
loss; the cost forecast's calendar SOC normalization remains frozen at the
capacity known at the optimization cutoff, including in multi-day horizons. A
separate capacity-guard exposure uses worst-case calendar stress at the configured
maximum SOC after the first future midnight. This prevents the frozen denominator
from understating subsequent-day physical fade; it does not alter the SOC-aware
cost forecast. Causal prefix projection
provides future starting states; subsequently realized health is never supplied
to earlier auctions. Outputs distinguish that normalization approximation from
PWL and rate-bin error bounds.

Square-root equality graphs are built at horizon endpoints and local midnights,
where they affect wear or capacity. Other reporting values are reconstructed
after solving from the same piecewise graph; exposures remain interval states.
Square-root equality graphs use configurable chord-error bounds. Forecast
capacity guards subtract both maximum loss-underestimation errors; current
physical state uses exact square roots. Internal solver exposures and stress
are scaled to avoid losing small increments to solver feasibility tolerance.
All persisted state and reported arrays retain unscaled physical units.

Physical execution protects energy and directional power limits. At midnight,
it also checks the capacity after the candidate command, reducing charging or
increasing discharge when needed and feasible. Energy and memory are never
silently clipped to accommodate fade. An impossible physical handoff raises a
diagnostic error. End of life stops after the completed chronological day,
retaining the accepted market ledger; there is no automatic replacement.

Future noncash wear is priced once:

```text
wear_value = net_replacement_eur / (1 - end_of_life_soh)
forecast_wear = wear_value * [future_calendar_loss + future_cycle_loss]
```

Each residual solve subtracts its starting exposure readout. Past damage stays
sunk. The previous throughput/Xu euro charge is not added. Forecast PWL wear,
executed analytical wear and realized market cash flow remain distinct.

## Configuration, persistence and verification

`PhysicalState.battery_health` contains both exposures, both banks, committed
daily SOH, active local date and a fingerprint of calibration/resolution
settings. A fresh study declares zero exposure. A used battery requires explicit
consistent history; aggregate SOH alone cannot reconstruct it. Incompatible
checkpoints and changes of calibration cannot silently resume. Nonzero
self-discharge is rejected until its bank-memory treatment is defined.

Available-capacity percentage targets are represented separately from absolute
MWh targets, including boundaries inside multi-day horizons. Explicit MWh
configuration entries take precedence, including an explicit null that disables
the corresponding target. The synthetic example uses a percentage-only target,
so its 50% target follows the forecast available capacity.

The optional energy-only continuation-value approximation conditions its energy
range and power cap on health known at cutoff. It freezes that health and omits
future Naumann memory and aging costs beyond the modeled horizon; this limitation
is recorded in its provenance. It adds no separate terminal SOH charge.

Outputs include committed SOH/capacity, separate losses and wear, directional
nominal/effective rates, projected health and approximation diagnostics. Depth,
calendar, square-root and rate errors are reported separately. Changing
replacement economics changes valuation, not physical exposure equations.

Routine software verification in the existing `bess` environment:

```bash
python -m pytest tests/unit/test_naumann_aging.py tests/unit/test_naumann_solver.py tests/unit/test_naumann_integration.py tests/unit/test_naumann_replay.py -q
```

The solver uses relative MIP gap `1e-4`, with no absolute gap. Creating/loading
the configuration does not approve a calibration fit, yearly simulation or
financial experiment; those require their own run plan and explicit approval.

Verification on 11 November 2026: 74 focused aging/reporting tests passed. The
broader unit run reported 1,379 passed, 3 skipped and 23 failures. All 23 failures
were reproduced on unchanged baseline `f699a73`: 15 DuckDB errors, 5 unavailable
`numpy.trapezoid` cases, an RTE indexing case, a missing historical-output fixture
and an incomplete checkpoint test stub. No packages were changed to address
those unrelated failures. Licensed aging fixtures explicitly assert zero
quadratic constraints and zero quadratic objective terms.

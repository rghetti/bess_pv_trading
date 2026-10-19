# Depth-dependent battery cycle aging

`xu_segmented` implements the segmented-energy approximation in Xu, Zhao, Zheng,
Litvinov and Kirschen, *Factoring the Cycle Aging Cost of Batteries Participating
in Electricity Markets*. The author preprint is available at
[arXiv](https://arxiv.org/html/1707.04567v2), with the published paper at
[DOI 10.1109/TPWRS.2017.2733339](https://doi.org/10.1109/TPWRS.2017.2733339).
The existing `throughput` YAML option remains the AC charge-plus-discharge
throughput baseline.

## Equations and electrical basis

Use nameplate internal energy `E = battery.capacity_mwh`, total battery-pack
replacement cost `R = degradation.replacement_cost_eur`, and the configurable
convex depth stress curve

```text
Phi(depth) = stress_coefficient * depth ** stress_exponent
```

Depth is a fraction of nameplate energy, including energy outside the operational
SoC limits. The usable SoC window does not replace the nameplate denominator.
`stress_exponent >= 1` is required. The defaults `0.000524` and `2.03` reproduce
the illustrative NMC stress curve in the paper's Eq. (24). They are not a
calibration of batteries at the supplied anonymous farms.

For `J = segment_count`, the equal internal-energy segment capacities are
`E / J`. Eq. (5) gives the marginal cost of each segment:

```text
c[j] = R * J / (eta_discharge * E)
       * (Phi((j+1)/J) - Phi(j/J))       # j = 0,...,J-1
```

These coefficients have units EUR per AC discharged MWh and are nondecreasing.
Each scenario's cost follows Eq. (7):

```text
aging = sum_t sum_j duration_hours[t] * c[j] * segment_discharge_mw[t,j]
```

Charging carries no aging charge. Under the paper's accounting convention, a
discharging half cycle incurs the same cost as a full cycle of that depth;
charging half cycles incur zero. There is no factor of one half. The AC discharge
efficiency in the coefficient cancels the AC discharge efficiency when evaluating
a given internal-energy depth. For example, discharging an entire segment at an
AC efficiency of 0.8 sells `0.8 * E/J` MWh and incurs exactly
`R * (Phi((j+1)/J) - Phi(j/J))` EUR.

The existing combined converter-and-cell efficiency is used once in each energy
transition. Eqs. (8), (9), (12)--(15) become:

```text
charge_mw[t]    = sum_j segment_charge_mw[t,j]
discharge_mw[t] = sum_j segment_discharge_mw[t,j]
segment_energy[t+1,j] = segment_energy[t,j]
    + duration_hours[t] * (eta_charge * segment_charge_mw[t,j]
                          - segment_discharge_mw[t,j] / eta_discharge)
0 <= segment_energy[t,j] <= E/J
energy_mwh[t] = sum_j segment_energy[t,j]
```

The physical model separately enforces the configured SoC box, AC power limits,
POI limits, exclusive charge/discharge mode and terminal-energy equality.

## Segment order and initialization

The segments represent cycle-depth memory. They are not fixed absolute-SoC
buckets. Appendix Eqs. (37) and (38) prioritize the shallowest available segment
for **both charging and discharging**. Charging fills its available headroom;
discharging empties its stored energy before using deeper segments. For a full
1 MWh battery with four segments, a 0.5 MWh discharge leaves segment energies
`[0, 0, 0.25, 0.25]`. A subsequent 0.25 MWh charge produces
`[0.25, 0, 0.25, 0.25]`. Enforcing a monotonic absolute-SoC occupancy would erase
this memory and implement a different model.

The original convex dispatch formulation admits the shallow-first policy as a
minimizer. This implementation explicitly selects that representative using
priority binaries: using any deeper segment requires every preceding segment to
finish full during charging, or empty during discharging. This also fixes
otherwise indifferent charging allocations near the horizon. It reproduces the
appendix's numerical policy and makes state propagation deterministic. This adds
`2 * (J-1)` binary variables per physical-control node and interval.

All segment charge/discharge commands and priorities belong to the same
information node as the battery command. Scenarios sharing observable history
therefore share aging actions. Segment-energy transitions are checked on every
leaf, including leaves with zero probability.

At the explicitly assumed study start only, Eq. (39) initializes known energy
into shallow segments first. Thereafter `PhysicalState.cycle_segment_energy_mwh`
must carry each segment's actual internal energy. A state with previous
throughput or aging cannot silently initialize from aggregate energy. A nondefault
state origin also requires explicit segment history. Provided segment energies
must sum to aggregate energy and respect the configured capacities and count.

The executor advances memory using executed commands after its POI safety
adjustments. Intrainterval auction cutoffs project the already committed battery
command with the same accounting. Reoptimization accepts this vector unchanged;
it never reconstructs it from SoC. Cumulative degradation is sunk, while the new
residual solve includes only future degradation in its objective. Dispatch CSVs
and final-state JSON expose the carried segment vector for inspection.

## Configuration and commands

An illustrative configuration is supplied at `configs/full_experiments/synthetic_xu.yaml`:

```yaml
degradation:
  model: xu_segmented
  cost_eur_per_mwh_throughput: 0.0
  replacement_cost_eur: 300000.0
  stress_coefficient: 0.000524
  stress_exponent: 2.03
  segment_count: 8
  assumptions: illustrative stress shape and assumed total EUR pack replacement cost
```

The replacement cost is total EUR for the pack, not EUR/MWh. The EUR 300,000
example is an explicit study assumption; no currency conversion or historical
pack quote is implied. Both cost and depth-stress parameters can be changed in
YAML. Combining a nonzero throughput coefficient with Xu aging is rejected to
prevent double charging. Nonzero self discharge is currently rejected in Xu mode,
because neither a stress attribution nor a memory rule for that loss is specified
by the implemented paper equations.

From the repository using the existing `bess` environment:

```bash
python -m bess_pv_trading optimize --config configs/full_experiments/synthetic_xu.yaml
python -m bess_pv_trading run-synthetic --config configs/full_experiments/synthetic_xu.yaml
python -m pytest tests/unit/test_xu_degradation.py tests/unit/test_xu_solver.py -q
```

The solver retains relative MIP gap `1e-4` (0.01%) and configures no absolute gap.

## Validation and limits

`evaluate_xu_dispatch` is a numerical implementation of the appendix's greedy
policy with no Gurobi dependency. The independent validator rebuilds its segment
trajectories from aggregate commands and compares energy, charge and discharge
arrays, component costs, objective reconciliation and information sharing. It
also checks the extracted segment bounds and balances.

`rainflow_discharge_depths` provides a separate ex-post aggregate-energy benchmark
following Section II-C's four-point cycle extraction and the discharge-residue
convention of Eq. (43). Tests cover known nested cycle depths, full and half
cycles, non-knot convergence, efficiency cancellation, repeated shallow versus
deep cycles with equal throughput, and arbitrary replay splits. Licensed MILP
tests compare fixed trajectories against independent rainflow costs at segment
knots, test economic depth selection, verify shared two-leaf history and exercise
four-auction replay with carried and projected state. Tampering with segment
history or aging cost is detected by the validator.

Finite segmentation is a piecewise-linear approximation to the configured depth
stress curve. Refining `segment_count` improves depth resolution and increases
MILP size. Historical cycle memory is unavailable before the declared study
start unless the caller supplies the segment vector. The ex-post rainflow helper
also neglects pre-history and must not be treated as a fresh-history benchmark
for an arbitrary residual window. The model does not estimate calendar aging,
temperature, current-rate stress, capacity fade feedback or a measured service
life. Charging-only terminal trajectories carry memory without booking their
future discharge cost.

## Executed synthetic example

The eight-segment, one-scenario configuration was executed on 16 October 2026
for the 96 products of delivery day 2 October 2025. The licensed optimization
returned optimal profit EUR 542.6887395775245, with EUR 2.307702752756917 aging.
It built 4,953 variables, including 1,536 binaries, and 5,195 constraints in
0.188 seconds, then solved in 0.161 seconds. The achieved relative gap was zero;
the configured relative tolerance remained `1e-4`, with no absolute gap setting.
Evidence is saved in
[the optimization result](../outputs/trading/optimize_20261016T102111253219Z/result.json)
and [extracted arrays](../outputs/trading/optimize_20261016T102111253219Z/solution.npz).

The corresponding full-day auction replay completed 100 residual solves in
15.790 seconds, with the same realized profit and aging cost, zero terminal SoC
error and maximum independent validation error below `1.94e-12`. Final segment
memory was saved alongside aggregate energy. The
[run manifest](../outputs/trading/20261016T102115868137Z/farm_1/2025-10-02/manifest.json)
links the configuration and policy results; the
[policy summary](../outputs/trading/20261016T102115868137Z/farm_1/2025-10-02/deterministic_integrated/summary.json)
and [dispatch plot](../outputs/trading/20261016T102115868137Z/farm_1/2025-10-02/deterministic_integrated/dispatch.png)
provide the numerical and visual evidence. These are synthetic Xu-model results,
separate from the constant-throughput benchmark and any historical backtest.

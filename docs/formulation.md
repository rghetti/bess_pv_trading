# Implemented optimization formulation

The implementation is `src/bess_pv_trading/domain.py` and
`src/bess_pv_trading/optimization/{model,validation}.py`. One farm is optimized
per model. The same builder serves deterministic and stochastic runs; stochastic
leaves live in one deterministic equivalent, without Gurobi's multiple-scenario
feature. All optimization powers are MW, energies MWh, prices EUR/MWh and cash
flows EUR. Every interval carries its elapsed duration `delta_hours`.

## Information ownership and validation

An `InformationTree` records node parents, information cutoffs, conditional
probabilities, observations and optional observation publication clocks. Each
scenario maps to a unique outcome leaf. Child probabilities sum to one, and
the product along a root-to-leaf path equals that scenario's probability.
Bid ownership is `bid_nodes[auction][scenario]`; physical ownership is
`control_nodes[scenario][interval]`. A decision's owner must lie on its leaf's
ancestry. Physical decisions use information available by interval start;
successive controls cannot discard history.

Battery charge, discharge and mode variables have keys `(interval, node)`.
An auction adjustment has keys `(auction, node, product)`. This construction
shares decisions, including binary modes, without equality constraints between
independently optimized paths. Future coefficients affect expected value but
cannot reveal the realized future path to a decision. A finite tree must be
constructed from observable history before it reaches this builder.

Published prices use observation tokens `auction:DA`, `auction:IDA1`, etc.
All descendants of such an observation must agree on its realized auction price
vector; a bid owner cannot already observe its own auction price. Tokens
`pv:<interval_id>` cannot be observed before interval end. Publication times
supplied to a node cannot exceed its cutoff. These checks complement, and do
not replace, the scenario generator's obligation to use causal histories.

By default curtailment shares the pre-interval information owner of battery
controls. A separate `curtailment_nodes` table explicitly permits response
through interval end; this is an interval-average execution approximation when
final average production is used. It is not evidence of demonstrated
sub-interval hardware performance. The tree's execution convention labels the
selected abstraction.

## AC bus and battery

Let `c`, `d`, `u`, `x`, `e` be AC charge MW, AC discharge MW, curtailed AC MW,
signed grid export MW, and internal battery MWh, respectively. Let `g` be
available PV AC MW, `p_pv_poi` be PV injection at the POI, and `h` interval
duration. Each leaf satisfies

\[
p^{PV,POI}_{ts}+u_{ts}=g_{ts},
\]
\[
x_{ts}=p^{PV,POI}_{ts}+d_{t,n(t,s)}-c_{t,n(t,s)},
\]
\[
e_{t+1,s}=r_t e_{ts}+\eta_c c_{t,n(t,s)}h_t
                    -d_{t,n(t,s)}h_t/\eta_d.
\]

Bounds apply to every state including the final boundary:
`energy_min <= e <= energy_max <= capacity`,
`-poi_import <= x <= poi_export`, and `0 <= u <= g`.
Curtailment appears only in the PV plant balance; it has no term in the POI
or stored-energy balance. With exact delivery and a net import `q_ts < 0`, the
POI equation implies `c = p_pv_poi + d + abs(q_ts)/h_t`: imported power must
be absorbed by charging.
For a binary charging mode `z`,
`0 <= c <= max_charge*z` and `0 <= d <= max_discharge*(1-z)`.
Configured AC inverter ratings must be consistent with these AC power limits.
Hourly self-discharge `loss` yields `r_t = (1-loss)**h_t`.

Under `converter_and_cell`, `eta_c = converter_charge * cell_charge` and
`eta_d = converter_discharge * cell_discharge`. The converter's explicitly
assumed default is 0.97 in each direction. Under `combined`, supplied combined
efficiencies are used directly and cell losses cannot be specified a second
time. A converter-only charge/discharge round trip returns 0.9409 of charged
AC energy. Direct PV export and financial adjustments do not pass through this
converter. Already measured farm AC production is not multiplied by 0.97.

The input state timestamp must equal the start of the residual delivery grid.
Pre-delivery callers must explicitly supply a known delivery-start assumption;
replay must carry the executed state. A configured terminal target is an
equality in every leaf. An input check rejects targets unreachable under
battery power and energy limits; remaining POI/PV/scenario restrictions can
still make a model infeasible. Infeasibility is reported, without relaxation.
No salvage-value option is currently implemented.

## Financial positions and settlement

Accepted trades form an immutable ledger with submission/publication clocks,
signed quantity, price, product, and contractual cash flow. Opposite new
adjustments add to existing positions; they never replace past trades.
For each leaf and product,

\[
q_{ts}=q^{\rm accepted}_t+\sum_a v_{a,n(a,s),t},\qquad
b_{ts}=x_{ts}h_t-q_{ts}.
\]

Only eligible open auction/node/products receive trade variables. No separate
commercial trade or position cap is imposed. Cumulative positions after each
modeled auction, including accepted and pending commitments, obey
\(-P^{POI,import}h_t\le q^{(a)}_{ts}\le P^{POI,export}h_t\).
Each individual auction trade also obeys
\(-P^{POI,import}h_t\le v_{a,t,s}\le P^{POI,export}h_t\).
These POI-derived energy bounds also apply to final net positions; physical grid
power obeys the same import/export ratings. An auction adjustment may close and
reverse an earlier position, subject to its own direction-specific POI rating
and the cumulative position bound.
Hourly product allocations are equal across their quarter-hour energy shares.
Before 1 October 2025 this applies to DA orders only. IDA1/2/3 prices and orders
are native quarter-hour products in the supported 2025 history. Physical battery
commands are independently free on every quarter-hour, including in the first
optimization that anticipates later IDAs. An hourly DA price does not impose an
hourly battery trajectory. `decision_resolution_minutes: 60` can explicitly
request coarse battery controls; the corrected GME studies use `15`, and an
unspecified control resolution follows the delivery grid.
The historical distinction is described in [GME Technical Rule 25, effective
January 2025](https://www.mercatoelettrico.org/Portals/0/Documents/en-US/20250101DTF25MPE_en.pdf),
which explicitly treats hourly MGP and quarter-hour MI products separately.

Settlement uses the exact positive and negative parts `b+`, `b-` of imbalance.
They obey `b=b+ - b-`, nonnegative bounds, and a binary direction constraint
that prevents simultaneous positive and negative settlement. Tight bounds are
derived from `(poi_import_mw + poi_export_mw) * interval_hours` in either direction;
an optional `imbalance_limit_mwh` tightens them. This remains exact for negative
prices and either ordering of directional prices.

\[
R^{\rm imbalance}_s=\sum_t
 [\pi^+_{ts}b^+_{ts}-\pi^-_{ts}b^-_{ts}],\qquad
C^{\rm imbalance}_s=\rho\sum_t(b^+_{ts}+b^-_{ts}).
\]

Single-price settlement sets both prices equal. The configured fixed
absolute-imbalance penalty `rho` is a study deterrent in addition to market
settlement; it does not modify or claim to establish historical settlement
rules. Choosing its size does not itself prove legal compliance. Additional
grid import fees, if configured, are nonnegative and use an exact signed-grid
direction split. The synthetic default excludes this fee as a study assumption.

New auction revenue is
`sum(price[auction,scenario,product] * trade[auction,node,product])`, with no
second multiplication by product duration. Accepted contractual cash flows
are reported separately as sunk constants and are not booked again in the
residual objective.

`OptimizationInput.pending_position_mwh` separately records quantities already
submitted but awaiting results. These fixed quantities participate in every
physical settlement position before prices are published; they are never
temporarily forgotten between a gate and its results. Accepted plus pending
positions must fit the POI-derived interval energy bounds. The unpublished-price contractual
cash-flow constant is explicitly excluded from the residual objective and
identified in diagnostics. This exclusion does not change optimizing decisions
because the submitted quantities are already fixed; expected-profit levels at
those intermediate cutoffs exclude that stated constant. Acceptance later moves
the quantities into the ledger, with no overlap.

## Degradation and objective

The retained `throughput` baseline costs **AC charge-plus-discharge throughput**:

\[
C^{\rm aging}_s=k\sum_t(c_{t,n(t,s)}+d_{t,n(t,s)})h_t.
\]

Its coefficient is nonnegative and is an illustrative user configuration;
no battery lifetime information is present in the farm dataset. This penalty
is not a cycle-depth model. `PhysicalState` carries cumulative throughput and
cumulative degradation cost through reoptimization. The numerical evaluator
adds new throughput/cost to those carried totals; only new cost enters the
residual objective.

The alternative `xu_segmented` model implements Xu et al.'s depth-dependent
segmented-energy formulation. With nameplate internal energy \(E\), total
replacement cost \(R\), \(J\) equal segments and convex stress
\(\Phi(\delta)=a\delta^b\), Eq. (5) gives the AC discharged-energy coefficient:

\[
k_j=\frac{RJ}{\eta^{\rm dis}E}
 \left[\Phi(j/J)-\Phi((j-1)/J)\right],\qquad j=1,\ldots,J.
\]

Following Eq. (7), the scenario aging cost is
\(C_s^{\rm aging}=\sum_{t,j}h_t k_jd_{t,j,n(t,s)}\).
Charging incurs zero aging; a discharge half cycle incurs the same cost as a
full cycle of that depth. The energy balance applies the same AC-to-internal
efficiencies once per direction. Eqs. (8)--(15) split charge/discharge over the
segments, constrain each internal-energy segment to \(E/J\), and sum segment
energy to physical SoC. Operational SoC limits do not rescale the depth
denominator.

Appendix Eqs. (37)--(40) prioritize the shallowest available segment for both
charging and discharging. Explicit priority binaries select this greedy
representative: using any deeper segment requires each preceding segment to
finish full when charging, or empty when discharging. These are cycle-memory
segments, not fixed absolute-SoC buckets. Their flows and priorities belong to
the physical control's information node. Initial occupancy follows Eq. (39)
once at the assumed study start; carried `cycle_segment_energy_mwh` is required
thereafter and is propagated through execution and committed-command boundary
projection without resetting.

`replacement_cost_eur`, the stress coefficient/exponent and segment count are
configurable. The default stress shape follows the paper's illustrative Eq. (24);
neither it nor the example EUR pack cost is calibrated from farm data. Nonzero
self discharge and a simultaneous nonzero throughput coefficient are rejected
in Xu mode. Independent numerical greedy replay and aggregate rainflow reference
tests verify depth dependence, efficiency accounting and history continuity.
See [the full equation mapping and validation limits](xu_degradation.md).

For leaf probabilities `p_s`, the implemented linear expected-profit objective
is

\[
\max\sum_s p_s\left(R^{\rm auction}_s+R^{\rm imbalance}_s
 -C^{\rm imbalance}_s-C^{\rm aging}_s-C^{\rm other}_s\right).
\]

Named components are retained for extraction and independent accounting.

## Solving and numerical validation

`build_model(inputs)` accepts validated in-memory objects. `solve_model(built,
settings)` configures threads, time limit, relative MIP gap (default 1e-4, or 0.01%), seed and
logging. It extracts values only when Gurobi reports a positive solution
count. No absolute MIP gap parameter is configured. The solver distinguishes
time limits with/without an incumbent, and records model
sizes, build/solve times, objective bounds, MIP gaps and available violations.
Ambiguous infeasible/unbounded results are re-solved with dual reductions
disabled. Optional LP/MPS export and infeasible-model IIS output are supported.
`optimize` reports license/environment failures without fabricating a solution.

`validate_solution` uses extracted arrays and the ledger, independently of
Gurobi expressions, to recompute power/energy balances, bounds, mode
integrality/exclusivity, accepted and new positions, intermediate position
limits, signed settlement, node sharing and probability-weighted objective.
It detects altered results even if solver expressions remain unchanged.

Executable checks are in `tests/unit/test_optimization_contracts.py` and
`tests/unit/test_optimization_solver.py`. The latter skips when opening a
licensed Gurobi model fails. Licensing success and execution counts belong in
the generated implementation report; sandbox HostID limitations must not be
described as proof that a licensed external execution is unavailable.

## Consulted primary references

- Gurobi, [Multiple Scenarios](https://docs.gurobi.com/projects/optimizer/en/current/features/multiscenario.html), accessed 2026-10-16: related-model scenario facility, distinct from explicitly shared stochastic decisions.
- Gurobi, [Optimization Status Codes](https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html), accessed 2026-10-16: termination classifications and ambiguous infeasible/unbounded diagnostics.
- Xu, Zhao, Zheng, Litvinov and Kirschen, [Factoring the Cycle Aging Cost of Batteries Participating in Electricity Markets](https://arxiv.org/html/1707.04567v2), original author preprint, equations (5), (7)--(16), (24), (37)--(40) and (43), accessed 2026-10-16; published DOI [10.1109/TPWRS.2017.2733339](https://doi.org/10.1109/TPWRS.2017.2733339).

The current Gurobi model-attributes page returned HTTP 429 during consultation;
the implementation queries installed `gurobipy` model attributes defensively.

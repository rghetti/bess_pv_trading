# Battery lifetime modeling for annual operations

Initial research and implementation proposal, 10 November 2026. The opt-in
[Naumann LFP implementation](naumann_lfp_aging.md) now specifies the selected
model, its chronological execution and its numerical verification. The broader
alternatives below are research context, not descriptions of that implementation.
No annual simulation or model experiment has been run for this research.

## Findings and recommendation

Use an LFP-oriented, explicitly calibrated approximation with (1) a carried
capacity state of health (SOH), (2) calendar damage depending on SOC, cell
temperature and age, (3) the existing Xu depth-dependent cycle-memory model,
and (4) separate stored-energy loss and auxiliary consumption. Update realized
health from executed operation, never from a forecast dispatch. Use one
consistent incremental damage estimate both to update SOH and, with an explicit
future-value coefficient, to influence dispatch. Keep the monetary charge as a
modeled opportunity cost, separate from actual market cash flows.

An additive calendar-plus-cycle capacity model is a useful, testable approximation,
not a universal physical law. Interactions, resistance growth, capacity recovery,
cell-to-pack imbalance, thermal controls and the aging knee can matter. A single
cycle-life guarantee cannot identify the two components or their stress curves.
An independently calibrated model and sensitivity studies are necessary before
interpreting a year-long result as a battery-specific forecast.

## What the repository already does

* `configs/standard.yaml` currently selects constant AC charge-plus-discharge
  throughput wear. Its cycle-life reference is an illustrative Victron LFP
  product, not a Huawei or BYD utility system. The end-of-life capacity fraction
  records an assumption; it does not drive physical capacity fade.
* `src/optimization/degradation.py` already implements Xu's segmented
  depth-dependent objective, greedy cycle-history state and an independent
  rainflow comparison. `docs/xu_degradation.md` explains the convention. The
  configured Xu stress curve was taken from an illustrative NMC cell and is not
  calibrated for these LFP assets.
* `src/optimization/physical.py` already applies exponential standing energy
  loss; its configured hourly rate defaults to zero. Nonzero self-discharge is
  explicitly rejected in Xu mode because its interaction with segment memory
  has not been defined. It is not safe just to delete the rejection.
* Battery energy bounds and terminal targets use fixed capacity; the persisted
  physical state carries energy and Xu history, not health. The actual daily
  executor and MPC, stochastic optimizer, replay and checkpoint paths must agree
  before health can be used in annual simulation.

The configured farm battery sizes are illustrative: 0.75 MWh / 0.5 MW,
1 MWh / 0.5 MW, 2 MWh / 1 MW, and 5 MWh / 2.5 MW. Matching a product's scale
or duration does not make its lifetime data a calibration of these assets.

## Published models and evidence

1. [Xu et al., 2017](https://arxiv.org/html/1707.04567v2): a convex cycle-depth
   stress curve is approximated with segment slopes; segment state carries
   cycle-depth memory and approximates rainflow. The proposed accounting charges
   the full cycle cost to a discharge half-cycle and zero to a charging half-cycle.
   These segments are not absolute-SOC bands. Carry memory across day boundaries
   and auction solves. The existing implementation already does this.
2. [Analytical Model of Rainflow-Based Cyclic Aging, 2024](https://ieeexplore.ieee.org/abstract/document/10552058):
   a MILP-compatible rainflow model considers both depth and average SOC of a
   cycle. Its benchmark uses two assumed aging-stress shapes rather than two new
   physical cell-aging experiments. Its compact `hh` approximation is additive
   across absolute SOC bands and cannot represent an arbitrary depth-only stress
   curve; its numerical errors are not physical lifetime-validation errors.
3. [Optimal V2G Scheduling with Calendar and Cycle Aging, 2024](https://ieeexplore.ieee.org/document/10488452/):
   a MILP includes calendar effects from SOC, age and temperature and cycle
   effects from temperature, C-rate and throughput. Its EV application supports
   feasibility of the approach, not parameter transfer to these stationary packs.
4. [Development and analysis of a battery degradation model for MILP, 2026](https://www.sciencedirect.com/science/article/pii/S2352152X26001751):
   piecewise-linear calendar and cyclic approximations are investigated; model
   variants have substantially different solve times. A detailed formulation
   must be checked on this project's scenario trees and rolling-solve workload.
5. [Naumann et al., 2018](https://portal.fis.tum.de/en/publications/analysis-and-modeling-of-calendar-aging-of-a-commercial-lifeposub/)
   report 29 months of calendar aging measurements for a commercial Sony/Murata
   LFP/graphite cell, including different temperatures and SOCs and a dynamic
   validation study. Their [published dataset](https://data.mendeley.com/datasets/kxh42bfgtj/1)
   offers a traceable generic LFP research calibration, not Huawei/BYD cell data.
   The companion [cycle-aging study](https://doi.org/10.1016/j.jpowsour.2019.227666)
   considers depth, SOC, temperature and C-rate. [Schimpe et al.](https://www.nrel.gov/docs/fy18osti/70616.pdf)
   model multiple LFP aging mechanisms. Calendar and cycle aging can interact;
   adding independently fitted curves should be validated against combined use.
6. [BLAST-Lite](https://github.com/NatLabRockies/BLAST-Lite/blob/main/README.md)
   provides open laboratory-based aging models as potential ex-post comparisons.
   Its authors caution that years-long test data extrapolated to decade-scale
   operation are qualitative and do not capture all pack-level failure modes.
   It is not a ready-made optimization formulation or a manufacturer warranty.

High SOC can therefore affect health even when charge and discharge are zero.
The magnitude, and whether a simple monotonic curve is adequate, depend on cell
chemistry, temperature, age and testing. Do not assume that a generic high-SOC
penalty is measured for our unnamed packs.

## Products and calibration limits

| Product | Evidence relevant to sizing | Lifetime evidence and limits |
| --- | --- | --- |
| Huawei LUNA2000-4472-2S | [Huawei's user manual](https://www.sig.energy/wp-content/uploads/2025/02/luna2000-4472-2s-manual-en.pdf) lists 4,472 kWh nominal energy, 2,236 kW nominal power and liquid cooling. This is close in scale and nominal two-hour duration to the illustrative 5 MWh / 2.5 MW farm. Check DC versus AC equipment boundaries before copying ratings. | No cell-specific calendar/SOC stress curve or self-discharge rate was established from the source. |
| Huawei LUNA2000-2.0MWH-2H1 | [Huawei-authored 2 MWh warranty terms](https://www.sig.energy/wp-content/uploads/2024/12/luna2000-2.0mwh-warranty.pdf), hosted by a third party, list a 0.5C use case, 100% DoD, 60% EOL capacity, and 6,570 cycles outside China; the performance warranty expires after ten years or the cycle limit, whichever comes first, subject to the document's other conditions. | This is a **different product** from the 4,472 kWh system. A contractual limit under specified conditions is neither an observed lifetime nor a decomposition of cycle and calendar degradation. Do not transplant it to the other product or fit two stress curves from it. |
| BYD MC Cube-T | [BYD's 2024 annual report](https://www1.hkexnews.hk/listedco/listconews/sehk/2025/0324/2025032401244.pdf) identifies the MC Cube-T as a utility energy storage product with a 6.432 MWh container. | Public material located does not establish a matching cell-aging curve, cycle-life test conditions, SOC stress function or self-discharge calibration. Do not label a generic LFP fit as a BYD lifetime model. |

A [distributor-hosted Huawei 2H1 SOH chart](https://kennisbank.wattkraft.nl/hc/nl/article_attachments/20890831089042)
shows a combined aging trajectory for 0.5C and one cycle per day. The chart does
not state enough temperature or test details to isolate calendar and cycle
aging or establish its predictive and warranty status. It can be used as an
illustrative combined-operation check, not a unique calibration.

Before manufacturer-specific calibration obtain the exact product and battery
cell, capacity/energy convention, SOC window, rated AC and DC power, temperature
control and expected cell temperature, cycle-life test curves at multiple depths
and SOCs, calendar test curves, test duration and EOL threshold. Request applicable
warranty throughput, time and temperature conditions separately. Grid-facing
auxiliary consumption and whether it is included in quoted round-trip efficiency
must be established independently.

Even if a reference test reaches health `h_EOL` after `N` cycles at depth `D`,
`Phi(D) = 1/N` only calibrates a **life-fraction** stress at one depth. The
illustrative conversion `capacity_loss_per_cycle = (1-h_EOL)/N` assumes linear
capacity fade and attributes all loss to cycling. If the measured cycle test also
contains calendar aging, adding a separate calendar term on top overestimates
loss; estimate and remove the calendar contribution using test duration and
conditions, or label the split explicitly as uncertain. One point cannot identify
the depth exponent, high-SOC response, aging knee, or temperature response.

## Proposed linear formulation

### State and units

Let `E0` be original nominal internal energy in MWh; `h` be remaining capacity
relative to `E0`, not percentage life consumed; and `e` be stored internal MWh.
At a known update boundary, `Q = E0*h` is the estimated available capacity.
Persist `h`, calendar and cycle capacity losses, operating age and the Xu segment
memory alongside energy. If no actual initial health is known, explicitly mark the
configured initial value as an assumption, and do not silently infer it from age.

For each short solver horizon use the last known `h` or a declared exogenous
forecast for future dates. With frozen `Q`, the following are linear:

```text
z_t = e_t / Q
z_min * Q <= e_t <= z_max * Q
E_end_target = z_target * Q
```

A terminal target and every controller, clipping rule and SOC report must use
the same capacity basis. Nameplate `E0` stays fixed; do not overwrite it with
`Q`, because existing Xu segment geometry, stress calibration and checkpoints
use nameplate energy. Daily freezing is an approximation; it does not model the
small within-day reduction of the physical bound. A future formulation can use
scenario-specific health variables in bounds (`e_t <= z_max * E0 * h_t` is
linear), but SOC normalization `e_t/(E0*h_t)` and SOC-stress coupling then need
an explicit MILP approximation.

### Calendar aging

At each interval approximate a calibrated capacity-loss rate by a piecewise
linear function `r_cal(z; T_t, age)` in fraction of original capacity per hour.
`T_t` and age-related coefficients must be known inputs to each solve; forecast
cell temperature if available, otherwise use an explicitly declared thermal
assumption. Weather temperature is ambient, not necessarily cell temperature.
For example, with SOC knots `z_k` and rates `r_k,t`:

```text
sum_k w[t,k] = 1,  w[t,k] >= 0,  w[t,*] SOS2
e[t] = Q * sum_k z_k * w[t,k]
d_cal[t] = delta_hours[t] * sum_k r_k,t * w[t,k]
```

SOS2 enforces interpolation between adjacent knots even for nonconvex curves;
when the function is known convex and the minimizing objective prices damage,
an epigraph may be cheaper. The calibrated time law may be sublinear: compute
age-dependent increments consistently with that particular model rather than
restart a square-root clock each day. A static reference-temperature constant
is a sensitivity assumption, not thermal simulation. Rate resolution, SOC
sampling convention and exposure to transitions need explicit validation.

### Cycle aging and SOH update

For Xu, let `ell_cyc[t]` be the segment-derived life fraction, with identical
segment/history convention in the MILP and realized replay. It is not already
capacity loss. Only when a selected model explicitly assumes a linear mapping
between life consumption and capacity loss, set

```text
d_cyc[t] = (1 - h_EOL) * ell_cyc[t]
h_next = h_current - sum_t d_cal[t] - sum_t d_cyc[t]
```

Both terms are fractions of original nominal capacity. Calibrate the curve and
conversion together to avoid counting the calendar contribution of the cycle
test twice. Other cell models may require a different, nonlinear mapping.
Accounting all cycle damage at discharge leaves charge-half-cycle damage
unbooked until a later discharge; persist segment memory and document this
accounting approximation. Report calendar and cycle damage separately.

A next-day health update can make carried energy exceed the newly computed SOC
maximum. Define and validate a policy before implementation: for example, retain
physical energy and impose a scheduled pre-boundary constraint based on an
available conservative capacity forecast, or record inaccessible energy via an
explicit, physically justified state transition. Do not silently clip stored
energy. Keep the Xu memory consistent with the chosen transition; rescaling or
reinitializing segment history would change the modeled cycle depths. Treat
EOL/replacement, continuation below EOL and augmentation as explicit policies,
not automatic consequences of a chosen threshold.

### Self-discharge and parasitic demand

For a fixed configured hourly standing-loss fraction `sigma`, the existing
linear stored-energy equation is

```text
e[t+1] = (1-sigma)**delta_hours[t] * e[t]
         + eta_charge * p_charge[t] * delta_hours[t]
         - p_discharge[t] * delta_hours[t] / eta_discharge
```

This is a loss of stored energy; it is not itself permanent capacity loss.
Calendar aging is capacity loss even while `p_charge = p_discharge = 0`.
In addition, HVAC, BMS and conversion equipment can draw auxiliary power from the
AC bus. Model this as a separate known or piecewise-linear load with POI and
cash-flow consequences, if it is not already included in efficiency data. A
manufacturer's peak cooling rating is not its hourly consumption.

Xu requires an explicit rule for how standing loss changes segment memory and
whether it represents a cycling event. Proportional leakage, shallow-first
removal and another rainflow-compatible accounting convention give different
answers. Select and test a physically justified convention, including idle and
mid-cycle examples; do not count standing loss as discharged energy sold or
charge cycle wear to it by accident. Until then the current rejection is correct.

## Economics and avoiding double counting

The physical state update and an objective penalty have different roles:
shrinking capacity constrains future operation, while an incremental aging
charge values damage extending beyond a short optimization horizon. Both can
coexist; physical SOH tracking is not by itself a second cash payment.

A simple short-horizon proxy is `C_wear = v_h * (d_cal + d_cyc)`, where `v_h`
is EUR per unit of SOH fraction. Under a deliberately simplified straight-line
replacement-cost allocation `v_h = R_net / (1-h_EOL)`. For a life-fraction Xu
curve and the linear mapping above, this recovers `R_net * ell_cyc`; do not add
that Xu cost again. Use one rate for each damage contribution, unless separately
justified by an explicit continuation-value model. A constant unavoidable calendar
component changes reported cost but not dispatch; the SOC-dependent part can
change dispatch.

Actual replacement spending, cash revenues, and physical fade belong in a
lifetime cash-flow ledger. Aging charges are noncash decision proxies, so do not
subtract their full sum **and** the same future replacement cost or terminal
health loss again when reporting total realized lifetime profit. Likewise, a
terminal SOH value or dynamic continuation value must be coordinated with the
running damage penalty. A replacement-cost proxy is only an approximation of
the true marginal value of health: future prices, discounting, EOL, augmentation,
calendar aging, and future opportunities change that value.

## Time and information consistency

Delivery-day auctions are submitted before the start of that day, while prior
operation and therefore delivery-start health may still be uncertain. Taking the
realized future SOH as a parameter in a prior auction leaks information. Forecast
or bound it using only information available at the decision; specify whether
health is common or scenario-dependent and impose physical feasibility for every
scenario. At the actual midnight boundary update the state only from executed
operation, preserve already submitted market commitments, and reoptimize when
new information arrives. The daily update must follow the replay's actual event
ordering, including intrainterval projection and daylight-saving days.

## Implementation and verification roadmap

1. Define typed configuration and persistence for capacity versus life fractions,
   initial health, calendar rates, temperature and economic value, all with
   provenance. Keep study assumptions in study YAML; retain backward-compatible
   fixed-capacity behavior and checkpoint versioning.
2. Implement one shared physical health/damage evaluator and equivalent solver
   expressions. Extend actual execution and projected states, daily handoff,
   stochastic dispatch, MPC, adaptive MPC, validation, terminal targets and
   reports together. Explicitly handle cross-boundary energy and Xu memory.
3. Verify idle high-SOC versus low-SOC exposure, constant-SOC calendar time,
   shallow/deep cycle stress and rainflow consistency, energy conservation,
   boundary cases, no damage/cost duplication, per-scenario feasibility, actual
   versus projected state, checkpoint/resume, and the Xu/self-discharge interaction.
   These are software checks, not an authorized market experiment.
4. Prepare a separate experiment plan for the user's approval before any annual
   run: calibrated ranges and uncertainty, vintage availability, baselines,
   ex-post reference model, metrics, solver workload and outputs. Compare
   physical fade and cash flow as well as modeled objective value; generic
   cell results must not be advertised as manufacturer-specific predictions.

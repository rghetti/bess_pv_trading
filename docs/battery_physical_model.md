# Battery energy, power and loss conventions

The current optimizer represents an AC-coupled PV plant and BESS. The PV engine
provides AC generation; the battery has a bidirectional conversion stage on the
same AC bus. Both share the configured grid point of interconnection (POI).
In conversation, “inverter” and “transformer” refer to this conversion stage.
Using both words does not introduce two conversion losses.

## SOC transition

Let $E_t$ be internal stored energy (MWh), $\Delta t$ interval duration (hours),
$P^{ch}_t\ge0$ AC power entering the battery conversion stage, and
$P^{dis}_t\ge0$ AC power delivered from that stage to the AC bus (MW).
The hourly fractional self-discharge is $\sigma$.

$$
E_{t+1}=(1-\sigma)^{\Delta t} E_t
       +\eta_{ch} P^{ch}_t\Delta t
       -\frac{P^{dis}_t\Delta t}{\eta_{dis}}.
$$

SOC is $E_t/E_{nom}$. YAML accepts only `soc_min_percent` and `soc_max_percent`,
so resizing the battery preserves the SOC box. No MWh limits are entered by the
user. The optimizer keeps an internal energy state for dimensional consistency.
Equivalently, for percentage SOC $z_t=100E_t/E_{nom}$:

$$
z_{t+1}=(1-\sigma)^{\Delta t}z_t+
 \frac{100\Delta t}{E_{nom}}\left(\eta_{ch}P^{ch}_t-
 \frac{P^{dis}_t}{\eta_{dis}}\right),\qquad
z_{min}\le z_t\le z_{max}.
$$

The efficiencies are in opposite positions because a charge receives only
$\eta_{ch}$ of the AC input, while an AC discharge consumes more stored energy
than its AC output. Powers must not be multiplied by these efficiencies again
in the grid balance:

$$
P^{grid}_t=P^{PV,AC}_t-P^{curt}_t+P^{dis}_t-P^{ch}_t,
\qquad -P^{POI,import}\le P^{grid}_t\le P^{POI,export}.
$$

Positive grid power is export. Charging may use PV, grid imports, or both.
There is no local consumption. A binary forbids simultaneous battery charging
and discharging. `battery.max_power_mw` is the single symmetric AC rating:
$0\le P^{ch}_t\le P_{max}u_t$ and
$0\le P^{dis}_t\le P_{max}(1-u_t)$. There are no separate battery inverter
power limits; the conversion stage is assumed capable of the full BESS rating.
There is no extra constraint limiting the sum of PV and battery power at a
shared inverter; only their net AC power shares the POI constraint.

## Execution feedback and realized SOC

The current study selects `execution.dispatch_policy: track_committed_grid`.
The earlier `follow_battery_plan` policy holds battery commands fixed and lets
PV errors change grid exchange. Grid imports are physically allowed, so under
that policy equal planned/executed SOC is possible even with PV forecast error.

The tracking controller freezes the target at each interval start as the sum
of submitted quantities divided by the interval duration. Submitted offers count
even before their prices are published. Actual PV is taken from realized `power`
for physical execution, with the explicitly configured missing-value fallback.
The controller releases scheduled curtailment first when net export falls short.
It then requests signed battery power

$$
b^{req}_t=P^{grid,target}_t-P^{PV,actual}_t+P^{curt,adjusted}_t,
\quad
b^{actual}_t=\mathrm{clip}(b^{req}_t,-\bar P^{ch}(E^{actual}_t),\bar P^{dis}(E^{actual}_t)).
$$

Positive signed power is discharge. The exact interval controller is as follows.
Write $[x]_+=\max(x,0)$ and
$\operatorname{clip}(x,l,u)=\min(u,\max(l,x))$. All power variables below are on
the AC side. For actual starting energy $E_t$, retained energy $E_t^-$ and the
configured percentage bounds,

$$
E_t^-=(1-\sigma)^{\Delta t}E_t,\qquad
E_{min}=E_{nom}\frac{z_{min}}{100},\qquad
E_{max}=E_{nom}\frac{z_{max}}{100},
$$
$$
\bar c_t=\min\left(P_{max},\frac{[E_{max}-E_t^-]_+}{\eta_{ch}\Delta t}\right),
\qquad
\bar d_t=\min\left(P_{max},\frac{\eta_{dis}[E_t^--E_{min}]_+}{\Delta t}\right).
$$

The grid target is $g_t^*=\sum_{a\in\mathcal A_t^{submitted}}q_{a,t}/\Delta t$.
Here $q_{a,t}$ is the signed energy assigned to this quarter-hour, including
submitted commitments whose clearing prices are not yet published. Let $p_t$
be actual PV and $(\hat c_t,\hat d_t,\hat k_t)$ the active auction's planned
charging, discharging and curtailment. First bound these commands and release
curtailment against a shortfall:

$$
c_t^0=\operatorname{clip}(\hat c_t,0,\bar c_t),\quad
d_t^0=\operatorname{clip}(\hat d_t,0,\bar d_t),\quad
k_t^0=\operatorname{clip}(\hat k_t,0,p_t),
$$
$$
g_t^0=p_t-k_t^0+d_t^0-c_t^0,\qquad
k_t^1=k_t^0-\min(k_t^0,[g_t^*-g_t^0]_+).
$$

Then adjust battery power and curtail any remaining PV surplus:

$$
b_t=\operatorname{clip}(g_t^*-p_t+k_t^1,-\bar c_t,\bar d_t),\qquad
c_t=[-b_t]_+,\quad d_t=[b_t]_+,
$$
$$
k_t=k_t^1+\min\left(p_t-k_t^1,[p_t-k_t^1+b_t-g_t^*]_+\right).
$$

Thus $c_td_t=0$ without a new binary decision or optimization. POI protection
then reduces charging if the import limit would be exceeded, or reduces
discharging and adds curtailment if the export limit would be exceeded. The
final executed powers determine both SOC and throughput wear:

$$
E_{t+1}=E_t^-+\eta_{ch}c_t\Delta t-\frac{d_t\Delta t}{\eta_{dis}},\qquad
z_{t+1}=100\frac{E_{t+1}}{E_{nom}},
$$
$$
g_t=p_t-k_t+d_t-c_t,\qquad
\delta_t=(g_t-g_t^*)\Delta t,\qquad
C_{deg,t}=c_{thr}(c_t+d_t)\Delta t.
$$

When $E_t^-\in[E_{min},E_{max}]$, the state-dependent power caps and mutually
exclusive charge/discharge guarantee $E_{t+1}\in[E_{min},E_{max}]$ before POI
protection; reducing either active power preserves those bounds. This applies
at every step of the current study, which configures $\sigma=0$. If nonzero
self-discharge drops retained energy below the minimum and feedback cannot
restore it, execution raises a physical-infeasibility error. It does not silently
raise the energy state. An exhausted battery leaves an actual grid shortfall
that is priced ex post; it cannot discharge energy that was only forecast to
have arrived. The implementation also checks the final energy against the box
before saving the interval, allowing only numerical roundoff.

The controller is deterministic physical feedback, not an additional market
optimization: four active auctions still give four MILP solves. The historical
hourly equality applies to planned battery commands and native auction products;
physical feedback can adjust each quarter-hour. Actual settlement prices are
never used by this controller.

At an auction cutoff inside an unfinished quarter-hour, its next boundary is
estimated by passing forecast PV through the same controller from the last
observed SOC. The unfinished interval's final measured PV is not used. This
forecast projection is labeled `forecast_projected_boundary_under_feedback`;
the subsequent physical interval always starts from the actual executed state.
Persistent production errors can make previously fixed commitments infeasible
under strict planning delivery. The optimizer reports that condition rather than
resetting SOC or silently restoring imbalance decision variables.

This is an interval-average feedback simulation. It does not reconstruct
sub-quarter-hour fluctuations from unavailable high-frequency telemetry. The
run saves requested and executed powers, active-plan SOC, grid target and error,
feedback adjustments, and the `plots/execution_tracking.png` comparison plot.

The energy balance and AC/DC power conventions agree with the distinction made
in [SAM's AC-connected battery model](https://samrepo.nlr.gov/help/battery_storage_btm.html).
The present model does not include voltage, current, nonlinear resistance,
temperature-dependent battery efficiency, auxiliaries, or a DC-coupled shared
PV/BESS converter. These would require their own topology and measured parameters.

## Efficiency configuration

For the one-day GME examples the configuration now uses:

```yaml
battery:
  capacity_mwh: 5.0
  max_power_mw: 2.5
  soc_min_percent: 10.0
  soc_max_percent: 90.0
  efficiency_convention: combined
  combined_charge_efficiency: 0.97
  combined_discharge_efficiency: 0.97
```

These are explicit study assumptions for the complete AC-to-stored-energy and
stored-energy-to-AC paths, giving 94.09% round-trip efficiency. They preserve
the previous numerical behavior while avoiding a misleading separately listed
cell factor of 1.0. In the alternative `converter_and_cell` convention,
`eta_charge = converter_charge_efficiency * cell_charge_efficiency` and likewise
for discharge. A cell factor of 1.0 assumed no *additional* internal loss;
it never made the effective efficiency 100%. The two conventions must not be
applied together. A measured whole-system round-trip efficiency must not be
multiplied by the same converter efficiency a second time.

At 2.5 MW AC over 15 minutes, charging adds 0.60625 MWh and discharging removes
0.644330 MWh. For a 5 MWh battery at 50% SOC this means 62.125% after charging
or about 37.1134% after discharging. The 10–90% box leaves 4 MWh usable.
Within an hour starting at 50%, those bounds allow at most 2.06186 MW constant
charging or 1.94 MW constant discharging. Hourly equality constraints before
October 2025 remain the user's configured dispatch assumption.

`state.terminal_soc_percent: null` in a GME experiment, or
`state.terminal_target_mwh: null` in prepared/synthetic optimization, omits the
terminal SOC constraint entirely. Initial energy and all intermediate SOC/power
bounds remain applicable. The pipeline reports a null terminal error in that case.

## Replacement cost and throughput loss

The farm-910 sensitivity uses the [BNEF December 2024 generic lithium-ion pack
benchmark](https://about.bnef.com/insights/commodities/lithium-ion-battery-pack-prices-see-largest-drop-since-2017-falling-to-115-per-kilowatt-hour-bloombergnef/)
of USD 115/kWh. The [ECB 2024-12-31 rate](https://www.ecb.europa.eu/stats/exchange/eurofxref/shared/pdf/2024/12/20241231.pdf)
is USD 1.0389 per EUR. This is a published pack-price proxy for replacement,
not a contractor's installed BESS replacement quote. Labor, inverter, other
balance of system and salvage are excluded, with configurable additions.

The lifetime reference is 2500 cycles at 80% DoD to at least 80% residual capacity
from [Victron's Lithium NG technical data](https://www.victronenergy.com/media/pg/Lithium_NG_battery_51%2C2_V/en/technical-data.html).
Applying this reference to a generic pack price is an explicit LFP study
assumption, not identification of the actual farm's battery. Capacity is treated
as constant when amortizing the reference lifetime; calendar aging is omitted.

For nominal battery energy $E_{nom}$, reference depth $d$ and cycle count $N$:

$$
C_{rep}=1000E_{nom}\left(\frac{p_{USD/kWh}}{FX_{USD/EUR}}
  +p_{additional,EUR/kWh}\right)(1-f_{salvage}),
$$
$$
c_{thr}=\frac{C_{rep}}{N d E_{nom}(1/\eta_{ch}+\eta_{dis})},\qquad
C_{deg}=c_{thr}\sum_t(P^{ch}_t+P^{dis}_t)\Delta t.
$$

Thus a reference charge/discharge cycle costs $C_{rep}/N$. The objective
subtracts this amortized wear cost, not the whole replacement expense in one day.
Both direct coefficients and source-based calculation are supported; specify
only one. Input sources and resolved EUR coefficients are retained in the KPIs.

## Optional imbalance decisions and ex-post settlement

The current GME experiment sets `trading.optimize_imbalances: false`. In this
mode there are no positive/negative imbalance variables or imbalance sign
binaries, and no imbalance cash flow or deterrence penalty in the objective.
Each scenario and optimization interval must satisfy

$$
q^{fixed}_t+\sum_{a\in\mathcal A_{open}}q_{a,t}
=\Delta t(P^{PV,AC}_t-P^{curt}_t+P^{dis}_t-P^{ch}_t).
$$

This equality applies per quarter hour even when native auction products are
hourly. An hourly surplus cannot cancel a planned quarter-hour shortfall. Native
auction-product and historical hourly battery-dispatch equalities still apply.
Curtailment or feasible battery actions must reconcile the forecast PV profile
with that position. A changed forecast or fixed commitments can make strict
delivery infeasible; the runner reports it instead of silently enabling slack.
With multiple forecast scenarios this constraint holds in every scenario with
the configured nonanticipativity, so it can be more conservative than stochastic
imbalance recourse.

`true` retains the original optimized-settlement formulation for explicit
comparisons. Omission retains `true` for compatibility with older configurations;
both real-data GME study YAMLs explicitly select `false`. The optional
`planned_delivery_balance_resolution_minutes` only adds aggregate delivery
constraints; setting it to `null` never disables strict interval equality when
`optimize_imbalances` is false.

Execution remains distinct from planning. After physical protection and PV
realization, `simulation/settlement.py` recomputes

$$
\delta_t=P^{grid,executed}_t\Delta t-q^{committed}_t,\qquad
CF^{imb}=\sum_t[\max(\delta_t,0)\lambda_t^+
                    +\min(\delta_t,0)\lambda_t^-].
$$

For single pricing, both price curves are identical; negative prices are valid.
These actual settlement curves enter only post-processing, never planning.
`profit_before_imbalance_settlement_eur` subtracts executed wear and grid import
fees from auction revenue. `realized_imbalance_cashflow_eur` and
`realized_profit_after_imbalance_eur` remain `null` while prices are unavailable.
Model deterrence penalties are not actual cash settlement charges. The replay
retains legacy assumed-settlement components and `realized_operating_profit_eur`
for compatibility, with an explicit provisional status; use the new fields for
actual-price accounting. `import-terna-workbooks` imports the configured monthly
XLSX exports into the same DuckDB catalog. The GME runner loads and validates
the actual-price curve only after all optimizations and execution have finished.
When prices are supplied, its dispatch cash flows and summary operating profit
are updated to actual settlement; the assumed cash flow remains separately named.

Historical electricity imbalance data are published by [Terna, Corrispettivi →
Sbilanciamento Aggregato Zonale → Riepilogo Mensile](https://dati.terna.it/corrispettivi).
For the current study request quarter-hour data for macrozone NORD, January
2025 initially (January–August 2025 for the full imported auction-price period).
Save the original export and publication/revision metadata under
`data/manual/terna/imbalance_prices/2025/` before catalog import. Preserve the
full imbalance price, base price, incentive component, macrozone, interval and
timezone. Terna's [Monthly Summary documentation](https://developer.terna.it/docs/read/apis_catalog/fees/Monthly_Summary)
also links manual SunSet download. GME's page named Prezzo Sbilanciamento is a
gas-market dataset, not the electricity price needed here. The original manual
export is an input artifact; reusable normalized data must use our DuckDB catalog.
The January–August 2025 exports currently reside at the explicitly listed paths
under `data/`; they are retained in place and their exact bytes are archived in
the checksummed monthly DuckDB snapshots.

### Expected versus realized accounting

Every replay saves the initial auction solve's objective as
`initial_expected_profit_eur` and its probability-weighted gross auction cash
flow as `initial_expected_auction_revenue_eur`. Each solve also saves
`expected_auction_revenue_eur`; later solves cover their remaining modeled
opportunities and must not be summed to form a daily expectation.

The summary (`json/<strategy>/summary.json`) and GME `json/kpis.json` /
`csv/kpis.csv` save:

- `realized_minus_expected_profit_eur` and
  `realized_minus_expected_profit_percent`;
- `realized_minus_expected_auction_revenue_eur` and
  `realized_minus_expected_auction_revenue_percent`;
- both initial expectations and `expectation_baseline: initial_auction_solve`.

For either metric $X$, the signed mismatch is
$\Delta X=X^{realized}-X^{initial,expected}$ and its percentage is
$100\Delta X/|X^{initial,expected}|$. Positive means realized performance exceeds
the initial expectation, including when the expectation was a loss. The
percentage is `null` for a zero expectation. Final net profit and its mismatch
remain `null` while actual imbalance settlement is unavailable; gross auction
revenue can already be compared once all auction cash flows are booked. Actual
settlement recomputes the comparison after pricing execution, using executed
wear and import fees. Gross auction revenue excludes these costs and imbalance
cash flows. These differences include the effects of subsequent auction
replanning and execution; they are not an isolated price-forecast error. The
profit baseline is the configured model objective, which can include model
penalties if enabled, and covers the initial strategy's modeled opportunities.

### Earlier zero-cost and hourly-balance sensitivities

Auction revenue is based on accepted contracts; physical energy is linked to
those contracts through imbalance settlement, penalties or a hard imbalance cap.
When price and penalty are both zero and no cap is configured, energy not
delivered against a contract is free. Battery discharge then brings no extra
auction revenue, while throughput degradation costs money. Zero dispatch is an
economic consequence even with a free terminal SOC. The recorded contract
cash flow under that sensitivity is not a deliverable-market-profit estimate.

The saved earlier backed-delivery farm-910 run instead set
`trading.planned_delivery_balance_resolution_minutes: 60`. For each scenario
and native market hour $h$ it requires:

$$
\sum_{t\in h}\left(q^{fixed}_t+\sum_{a\in\mathcal A_{open}}q_{a,t}\right)
=\sum_{t\in h}\Delta t\left(P^{PV,AC}_t-P^{curt}_t+P^{dis}_t-P^{ch}_t\right).
$$

This is a planning constraint, separate from imbalance pricing. Quarter-hour
PV can vary within an hourly product; it is not forced flat. Realized output
may differ from that forecast after execution, and those errors remain recorded
without an assumed settlement charge in this sensitivity. Use `null` to disable
the planning balance explicitly. This policy does not constitute a verified
historical imbalance settlement model.

The earlier saved studies used separate trading-volume caps: 0.25 MWh per native
hourly product per auction and 0.25 MWh cumulative position per quarter hour
(1 MW). These caps have been removed from the model and YAML configurations.
Future runs bound cumulative contracted positions after each auction by
`[-plant.poi_import_mw * interval_hours, plant.poi_export_mw * interval_hours]`.
The farm-910 configuration therefore permits contracted import/export power
up to 6.638 MW, matching its configured physical POI. Individual adjustments
have no separate cap and may close and reverse an earlier position. Existing
saved results still reflect their recorded earlier settings and require a new
run for an economic comparison under the corrected limits.

## Four auction decisions

The auction-based GME replay solves once immediately before each active auction.
Only the current auction's quantity is submitted. Future auction quantities in
that optimization are plans and remain free at subsequent decisions. Previously
submitted quantities are fixed, including the interval before prices are known.
At publication the replay books quantity times actual clearing price exactly
once and updates its known information. It does not optimize a closed market.
The next auction decision conditions future price forecasts on those known
results. The oracle uses actual future curves throughout as an explicit benchmark.

With zero settlement price/penalty and import fees, the residual objective is:

$$
\max\;\mathbb E_s\left[\sum_{a\in\mathcal A_{open},t}
\pi^s_{a,t}q^s_{a,t}-c_{thr}\sum_t(P^{ch,s}_t+P^{dis,s}_t)\Delta t\right].
$$

Accepted past cash flow belongs to the external ledger and is not counted
twice in this incremental objective. The free variables by auction are:

| Decision | Free new market quantities | Fixed quantities |
| --- | --- | --- |
| MGP | MGP, MI-A1, MI-A2, MI-A3 plans | None |
| MI-A1 | MI-A1, MI-A2, MI-A3 plans | Submitted MGP |
| MI-A2 | MI-A2, MI-A3 plans | Submitted MGP and MI-A1 |
| MI-A3 | MI-A3 eligible products | Submitted MGP, MI-A1 and MI-A2 |

Remaining-horizon BESS controls, curtailment and states are optimized at each
decision, subject to the observed/projected boundary state and configured
information constraints. Missing auctions are omitted. Publication events are
saved separately in `oracle/price_publications.json`; four active auctions give
four solves in this auction-only replay. Optional synthetic physical-control
refreshes are a separate policy and are counted separately.

## Configured battery tests

Battery reference assets, analytical input cases and independently tabulated
expected results are in `tests/fixtures/configs/battery_cases.yaml`. Production
configuration checks read the actual full-experiment YAML, so changing study
capacity does not require editing fixed MWh bounds or test battery ratings.

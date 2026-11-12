# Illustrative linear LFP health calibration

The study uses an operational proxy, not a fitted model of the simulated farm's
unknown battery. A full nominal internal-energy equivalent cycle (EFC) is
`(internal charge + internal discharge) / (2 * nominal energy)`. Calendar and
cycling losses are additive fractions of original capacity. Health and capacity
are committed at Europe/Rome midnight after executed dispatch, not at a market
gate or the end of an optimization horizon. The economic penalty values the
same cycling loss once; forecast calendar wear is an exogenous time cost and
must not be counted again as a throughput charge.

The initial study assumptions are:

| Parameter | Value | Basis and limitation |
| --- | --- | --- |
| Cycle reference | 6,000 full 100% DoD EFC | A 280 Ah LFP container brochure reports at least 6,000 cycles at 100% DoD and 80% SOH. It is a manufacturer specification, not a measured warranty or a description of our farm; the cycle test temperature and calendar/cycle split are not established. [Container brochure](https://www.lvwo-energy.com/Content/upload/PDF/2020543126/Container-energy-storage-system.pdf). |
| Cycling loss at 6,000 EFC | 17 percentage points | An explicit allocation of the specification's 20-point total loss, not an independently fitted cycling coefficient. The remaining 3 points are attributed to calendar loss over an illustrative roughly three-year accelerated test. Actual duration and allocation are unknown. |
| Calendar loss | 1 percentage point per year | A fixed-rate illustrative allocation, not a measured universal LFP value. Temperature, SOC, nonlinear time dependence and storage conditions are omitted. |
| End-of-life threshold | 80% SOH | Reporting and stopping assumption aligned with the cited cycle endpoint, not automatic replacement. |

The allocated reference is internally consistent only under the assumed test
duration: `17% cycling + 3 years * 1%/year = 20% total`. A cycle lasting four
hours at constant 0.5C in both directions would make 6,000 cycles about 2.74
years before rests; this is a plausibility illustration, not an identified test
protocol. At exactly one full EFC/day and constant 1% annual calendar loss, the
proxy reaches 80% SOH in roughly 9.8 years. This is arithmetic, not a promise
of service life. Higher cycling or temperature can shorten life; reduced depth
effects, temperature, recovery, rate dependence, resistance and augmentation
are not represented.

An independent DNV evaluation of utility-storage cells reports roughly 86%
capacity after 4,000 cycles for a 280 Ah LFP cell at 25°C and 0.5C, with the
test still in progress. Its published summary does not establish the depth of
discharge or calendar exposure, so this is a plausibility cross-check, not a
fit to either coefficient. [DNV/CLOU evaluation, pp. 28–33](https://www.clouess.com/storage/file/20240805/1722841497213663.pdf).

The NREL utility-scale baseline assumes about one cycle per day over a 15-year
system life, but its operating costs include augmentation to maintain rated
capacity; its 15 years should not be treated as unaugmented cell life or an
independent 80%-SOH observation. [NREL utility-scale storage assumptions](https://atb.nrel.gov/electricity/2024/2023/utility-scale_battery_storage).
The older laboratory Naumann LFP papers used in the detailed model provide a
contrasting measured nonlinear aging model, not verification of this linear
curve or of the commercial cell. [Calendar study](https://doi.org/10.1016/j.est.2018.01.019),
[cycling study](https://doi.org/10.1016/j.jpowsour.2019.227666).
In particular, its square-root loss can be steep immediately after a fresh
initialization. A 1% loss in the first ten simulated days must not be linearly
extrapolated to claim 36.5% annual loss or a specific end-of-life date.
The cycle-depth dependency visible in the published [Victron LFP NG data](https://www.victronenergy.com/media/pg/Lithium_NG_battery_51%2C2_V/en/technical-data.html)
also shows why this fixed-throughput approximation should be checked against
rainflow models. That smaller product is not the selected utility-cell calibration.

The standard inherited pack replacement-price proxy and 80% threshold price
each fraction of capacity loss once. This straight-line valuation assumes the
same economic value for every unit of lost SOH and does not model augmentation,
discounted replacement timing or full installed-system replacement costs. Do
not use the cycle count independently as an additional financial penalty.

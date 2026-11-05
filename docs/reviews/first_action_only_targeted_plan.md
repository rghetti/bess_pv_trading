# Shared-first-action ensemble MPC: targeted comparison plan

This plan was approved in the session and completed. See `first_action_only_targeted_results.md` and `first_action_only_targeted_audit.md` for results and independent checks.

The proposed comparison uses 15–17 January 2026 and farms 1/345/901/910,
matching the previous provisional-IDA3 window. It compares four continuous
closed-loop cases, each with its own grid-tracking baseline:

| Case | Future controls | Objective | Ordinary-midnight inventory term |
| --- | --- | --- | --- |
| common_costs | Common across every member | EUR 500/MWh expected absolute imbalance plus wear/imports | Soft 50% target at EUR 50/MWh |
| first_action_costs | Independent per member after a shared first command | Same monetary objective | Same soft target |
| common_imbalance_only | Common across every member | Expected absolute imbalance in MWh only | None |
| first_action_imbalance_only | Independent per member after a shared first command | Expected absolute imbalance in MWh only | None |

All cases start at 50% SOC, carry actual state across midnight, and impose a
50% planning terminal SOC on the final study day. Every scenario obeys the
same physical bounds, efficiencies and exclusivity restrictions. Pure
imbalance cases retain physical degradation accounting but do not minimize
its financial cost. Final executed energy and terminal misses are reported.

The first-action-only formulation deliberately allows different future
commands even when past/current PV observations have not distinguished
members. It therefore assumes scenario-path revelation after the first
action inside the optimization. Only the shared current command executes;
the next decision replans. This is an optimistic two-stage approximation
rather than the previously tested observed-history tree.

This structure corresponds to the two-stage stochastic MPC formulation in
[Groß, *Stochastic Model Predictive Control for Smart Grid Applications*,
section 2.3](https://publications.syscop.de/Gross2023a.pdf). That PV and
battery work uses an instantaneous current measurement; the current-interval
average in this retrospective comparison is a stronger information assumption.
[van der Meer, Wang and Munkhammar (2021), *Applied Energy*](https://www.diva-portal.org/smash/get/diva2%3A1502295/FULLTEXT01.pdf)
also use a common first action across scenarios in a PV-and-battery application.

The fifty weather members remain frozen from the preceding delivery day's
00 UTC cycle; the deterministic member updates to the latest eligible cycle
at each cutoff. Historical publication lag is six hours. All 51 trajectories
retain raw uniform weights 1/51, without probability calibration, dependence
postprocessing or scenario reduction. The existing reviewed-GHI selected
model per farm converts every weather member. Its documented training used
historical GHI and validation/test used preceding-day weather; the updating
weather inference is an input/vintage shift. All model outputs obey the
configured monthly nighttime-zero rule. No models are fitted or selected.

Execution uses reviewed curated production, with retained source flags for
metered AC, reviewed synthetic replacements, historical-GHI reference fills
and nighttime zeros. These counterfactual intervals and post-period model
selection make the run retrospective. Both MPCs have explicitly disclosed
current-interval average PV; later realized PV remains hidden. The average
of the entire controlled interval is additional retrospective information.

Every case anticipates the IDA3 order saved in the completed IDA2 plan until
actual IDA3 submission, then uses the binding quantity without double-counting.
The previous targeted comparison reduced portfolio imbalance 12.03% but
worsened farm 910, so this is a planning input choice rather than an assurance
of a controller improvement. Actual trading/settlement uses submitted orders
only. All four auctions reoptimize continuously with the same existing
actual-auction-price oracle benchmark. The physical MPC uses no future
settlement prices. Auction SOC projections use the stored deterministic path
and the executor, with no projection re-solves.

The proposed matrix contains sixteen jobs, 4,608 nominal MPC decisions and
384 nominal auction solves. Four independent CPU workers use one Gurobi and
native thread each, with no GPU work. Shared-first models are larger than
common-control models. Each decision keeps its ten-second total build/solve
budget; auctions retain 120-second limits, relative gap 1e-4 and no absolute
gap. Report time-limited incumbents, rejected plans, protection and tracking
fallbacks explicitly. Solver-time caps total at most 12.8 hours for MPC and
12.8 hours for auctions across all jobs, excluding preparation, validation,
reporting and retries; these are upper limits rather than runtime predictions.

Primary evaluation is realized absolute imbalance, split into shortage and
excess. Attribute profit differences to signed settlement, auction revenue,
executed degradation, imports and fees. Include curtailment, throughput,
noon/end SOC, projection error, runtime and solver status. Pair recourse
variants within each objective to isolate schedule sharing; pair objectives
to expose the wear/inventory tradeoff. Trackers should be identical across
cases with matched inputs. Three development days support no annual or
statistical superiority claim. The full-year benchmark remains deferred.

Configuration: `configs/research/first_action_only_targeted.yaml`.

```bash
conda activate bess
PYTHONPATH=src python scripts/run_energy_research_study.py --config configs/research/first_action_only_targeted.yaml --check-only
PYTHONPATH=src python scripts/run_energy_research_study.py --config configs/research/first_action_only_targeted.yaml --run
```

Historical optimization requires approval of this concrete matrix under the
project experiment-control instructions. Software verification and check-only
preflight do not start the historical controller comparison.

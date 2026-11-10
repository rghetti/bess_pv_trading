# January 2026 extended IDA3 planning study

This is a proposed retrospective comparison, not a launched experiment. Run a
full input preflight and obtain explicit approval before starting the study.

## Purpose and design

Run one continuous greedy grid-tracking trajectory for each of farms 1, 345,
901 and 910, scoring 1–31 January 2026 in Italian local time. At each IDA3,
jointly plan the eligible remainder of that day and the complete next delivery
day, linked by battery state, with a minimum SOC of 40% at each modeled
midnight. Only current-auction orders are submitted; later orders are
provisional. Reoptimize DA, IDA1 and IDA2 at their gates. February 1 is
planning and auction context, not a scored delivery day. Initialize the
battery at 50%; never reset actual SOC at midnight. Greedy tracking does not
enforce the floor in execution. The first event is the December 31 DA gate for
January 1. There is no December 31 IDA3 solve; the first extended IDA3 is on
January 1 and plans January 1–2.

Compare January against the previously completed same-day grid-tracking
trajectories for the four farms. Those used a 50% planning equality; changes
therefore combine horizon and SOC-policy effects. Both use independent rolling
auction decisions, so commitments can differ. No MPC or K=25 run is included.

## Inputs and information

Use the reviewed curated AC production dataset, including metered production,
monthly night zeros, reviewed synthetic/historical-GHI replacements and
reconstruction of retained exclusions. These replacements are counterfactual,
not independent measurements. The tracker responds to the realized interval
average retrospectively; future realized PV is not supplied to the planner.
Actual auction and imbalance prices come from the local DuckDB catalog. The
auction planner uses future actual clearing curves as an explicit price oracle,
including next-day curves at IDA3; this is not a causal price forecast.

Frozen four-cycle neural models consume the latest eligible deterministic
weather issue with a full 48-hour context and a six-hour release assumption.
Forecast cycles are updated at eligible auction cutoffs, and the configured
monthly night-zero policy applies. Full context coverage, price products and
artifact checksums must pass preflight, including the final-day context.

The models use weather and calendar inputs; the extended feature set adds
weather variables. No model is trained or calibrated during this study. The
frozen models were trained on 2025 historical GHI inputs, with forecast GHI for
validation/test; they were selected using January–April 2026 validation. Thus
this January comparison is retrospective and validation-selected, with possible
model-selection leakage. No claim of an untouched out-of-sample January result
is justified. Training loss and best validation epoch are distinct from this
study's financial and physical evaluation metrics:

| Farm | Model/features | Width/depth | Dropout | Learning rate | Weight decay | Batch | Training loss | Best epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | BiGRU/extended | 128/3 | 0.20 | 0.000181046 | 0.000113364 | 16 | Huber | 4 |
| 345 | BiGRU/core | 512/3 | 0.00 | 0.001296122 | 0.000116038 | 48 | MSE | 1 |
| 901 | TCN/extended | 512/3 | 0.05 | 0.000121069 | 0.000172323 | 16 | Huber | 2 |
| 910 | TCN/core | 512/3 | 0.10 | 0.000750208 | 0.002005052 | 48 | Huber | 1 |

The frozen training selected epochs using validation normalized RMSE; no
selection or hyperparameter search is performed in this study. The baseline
and new runs use the same pinned model identities and checksums. Their
historical provenance includes different inference thread counts; numerical
bitwise equality of predictions is not assumed.

## Compute, checks and outputs

Use four processes, one per farm, and six Gurobi threads per process by
default (up to 24 solver threads), CPU-only, with a 120-second solve limit and
relative MIP gap 1e-4. There are up to 127 auction submissions per farm:
four for each January delivery day plus February 1 DA, IDA1 and IDA2 before
the replay boundary. Missing/cancelled auctions and fallback affect the count.
The preflight performs no optimization and compares January realized PV,
auction-price fingerprints, settlement values and model checksums with the
historical results. Preserve failure diagnostics and do not start if it fails.
This machine reports 10 logical CPUs, so 24 solver threads can oversubscribe
it; the launcher allows the worker and solver counts to be changed.

Report per-farm and combined revenue, settlement, degradation, profit,
imbalance, curtailment, battery throughput and SOC, all modeled versus actual
midnight SOC, floor violations, auction fallback and solver time. Profit does
not value final inventory. Record provenance, configuration, cutoff times,
future plans, orders and price publication. Check historical and new actual
inputs on matching January products; fail clearly if they differ. The
comparison is a descriptive historical evaluation, not model selection. At
successful completion, generate hourly tables and January quarter-hour profiles
for absolute imbalance, curtailment, final signed net position, actual SOC and
auction/settlement prices for both frameworks.

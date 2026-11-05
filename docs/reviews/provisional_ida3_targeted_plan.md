# Provisional IDA3 targets: targeted continuous comparison

Approved in this session and completed. See
`provisional_ida3_targeted_results.md` and the independent audit beside it.

The change uses the IDA3 quantities forecast by the already completed IDA2
auction optimization as provisional afternoon MPC targets. Actual submitted
positions stay in their own array and are the only settlement ledger. At IDA3
submission, the forecast contribution is removed and the newly submitted
quantity enters the binding position once, before publication.

Configuration: `configs/research/provisional_ida3_targeted.yaml`.

The proposed targeted comparison covers 15–17 January 2026 for farms 1, 345,
901 and 910. It compares `submitted_only` and `ida2_plan` using deterministic
MPC, each with its own continuous grid-tracking baseline. Auctions reoptimize
from each controller's own state; later actual auction positions may therefore
differ. Eight jobs contain 2,304 MPC decisions and run with four CPU workers,
one solver/native thread per worker, no GPUs and no fitting.

Both cases use the frozen reviewed-GHI validation-selected models already
present in this checkout. Their README records historical GHI for training
and preceding-day 00 UTC forecasts for validation/test. This replay uses updating
eligible forecast cycles at each cutoff with the six-hour publication assumption.
The irradiance-input and forecast-vintage shifts, together with post-period
model selection make this a retrospective development experiment. The adapter
sets every predicted PV value to zero in the configured monthly local night
window. There is no retraining, loss optimization, hyperparameter search,
recalibration or model selection in this comparison.

Actual execution uses the reviewed curated targets, retaining interval source
labels. Reviewed synthetic daylight replacements and remaining excluded
daylight values filled from the historical-GHI pvlib reference are
counterfactual production. Other unresolved daytime gaps must fail preflight.
Both controllers explicitly receive current-interval average PV; future
realized PV remains hidden. Access to the complete controlled interval average
is an additional retrospective information assumption.

Both cases retain the EUR 500/MWh absolute-imbalance planning weight,
throughput wear and soft 50% midnight inventory target at EUR 50/MWh internal
energy deviation. Initial SOC is 50%; final-study planning SOC is fixed at 50%
and any execution miss remains reported. All four auctions participate with
the same actual-price oracle assumption. MPC receives no future settlement
prices. Relative MIP gap is 1e-4, with no configured absolute gap. Auction
solves retain their 120-second cap; each MPC decision retains its ten-second
cap. These are upper limits rather than predicted runtime.

Primary evaluation is the paired difference in realized absolute imbalance
MWh, with separate shortage/excess and pre-/post-IDA3 results. Report signed
settlement, realized profit, degradation, curtailment, noon/final inventory,
projection errors, fallbacks and solver runtime alongside it. Planning
penalties never become financial charges. Final inventory and changed auction
revenues must remain visible. No significance or full-year strategy claim can
be drawn from three development days.

Commands, from this checkout:

```bash
conda activate bess
PYTHONPATH=src python scripts/run_energy_research_study.py --config configs/research/provisional_ida3_targeted.yaml --check-only
PYTHONPATH=src python scripts/run_energy_research_study.py --config configs/research/provisional_ida3_targeted.yaml --run
```

The historical run requires approval of this concrete plan under the project
experiment instructions. The full-year comparisons remain deferred until the
ongoing GPU training and final model selection finish.

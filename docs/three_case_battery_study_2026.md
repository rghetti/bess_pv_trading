# Three-case matched battery study

The optional `--include-no-wear` study covers farms 1, 345, 901 and 910 from
1 January through 29 August 2026. It runs 12 trajectories: fixed capacity with
zero wear, fixed capacity with the matched cycling-wear coefficient, and daily
linear SOH with the same cycling coefficient plus calendar aging. All three
use integrated auctions, grid tracking, a planned 50% terminal SOC and a
two-EFC-per-day operational cap. The cap uses AC throughput and the day's
committed available capacity; therefore its MWh allowance shrinks in the SOH
case. The no-wear comparator is not an estimate of zero physical aging.

The study uses retrospective, actual auction-price curves as an oracle and
previously selected PV models. Latest eligible weather vintages follow the
configured six-hour availability lag; realized PV comes from the curated dataset.
There is no training, fitting or model selection in this study. Wholly missing
intraday auctions are skipped and partially missing products fail preflight.
Every case independently records the input snapshots, model checksums and
forecast vintages; comparisons require matching provenance. Consult
`linear_soh_all_farms_2026_study.md` for the calibration and known data gaps.

The queue runs up to eight trajectories concurrently and the remaining jobs
start as workers finish. All three cases use the no-wear configuration's
Gurobi thread count by default; `--threads-per-solve` can override it for all
cases. The per-solve limit is 60 seconds and the relative gap is 1e-4. The
launcher previews the resolved thread counts. Solver time limits can leave
material optimality gaps. Results include
the market cash, imbalance cash and absolute energy, curtailment, throughput,
own-model and common-proxy wear, SOH, auction-specific trade cash and volumes,
intraday profiles, summary tables and comparison plots. The common proxy
revalues recorded paths but does not rerun fixed-capacity dispatch with fading.

```sh
conda activate bess
python -u scripts/run_linear_soh_farm_comparison.py \
  --run-id three_case_battery_2026 --include-no-wear --run
```

Resume an interrupted batch using the same run ID and add `--resume`. Input and
configuration identity must remain unchanged; unfinished cases are resumed from
their checkpoints and completed cases are preserved. No automatic battery
replacement occurs at end of life.

# Matched 2026 linear health and throughput comparison

This plan compares eight independent continuous trajectories: four farms (1,
345, 901, 910), each under the new linear throughput SOH model and the legacy
fixed-capacity linear throughput model. Both cases use the same daily limit of
2 full-equivalent cycles in AC-side charge-plus-discharge throughput divided
by twice that day's committed available capacity, the
same 2025 BNEF stationary-pack price proxy, 50% planned terminal SOC, integrated
auction planning and grid tracking. The SOH case uses the separate provisional
calendar and cycling calibration in `linear_soh_calibration.md`; the throughput
baseline uses the standard 2,500 cycles at 80% DoD reference. These distinct
wear coefficients and aging assumptions are intentional. The baseline charges
throughput wear but does not shrink capacity. The SOH model uses internal energy
after efficiency losses for physical aging; the common cap uses AC throughput.
The baseline's denominator stays at nominal capacity, while the SOH case's
denominator shrinks daily. The same two-cycle limit therefore permits different
MWh budgets once health declines.

Delivery is 1 January–29 August 2026 inclusive, leaving the full 48-hour model
forecast context inside the available weather data. The independent jobs use up to eight workers at once,
three Gurobi threads per solve and a 60-second per-solve limit. Native thread
settings start at one per worker; PV inference configures PyTorch to use three
threads per worker. A free worker takes the next queued trajectory until
fewer than eight remain. Failed cases are recorded and do not cancel the other
cases. Gurobi logs are disabled; chronological progress is printed with a farm
and case prefix and saved in the batch's `logs/` folder.

The previous August 31 endpoint failed in preflight: the August 30 06:00 UTC
weather vintage has missing values from September 1 00:00 UTC, while the neural
model requires its complete 48-hour issue context. August 29 is the requested
conservative endpoint; missing daytime data are not filled with zeros.

No training, tuning or calibration is performed. PV is the latest eligible
deterministic weather vintage with the configured six-hour availability lag and
retrospectively selected neural models; the model artifacts postdate delivery,
so this is a retrospective comparison. Actual PV comes from the reviewed GHI
dataset and includes its documented substitutions, exclusions and night policy.
Auction prices are actual historical clearing curves used as a perfect-price
oracle; results do not constitute a live forecast. Imbalance uses historical
settlement prices. The cached market catalog is opened read-only; no data are
downloaded. The independent per-job preflight checks all inputs before solving
and records market snapshot IDs and checksums. Full dataset availability is not
assumed from the aggregate checks below.

A read-only catalog inspection found all 241 selected days for GME DA and TERNA NORD
imbalance. Entirely absent intraday-auction days are skipped according to the
study's `skip_if_no_products_fail_if_partial` policy: IDA1 is absent 16 April,
18 May, 16 July, 4, 5, 7 and 19 August; IDA2 is absent 16 and 22 April and
27 August; IDA3 is absent 1 and 22 April and 13 July. Catalog changes or
partial products are still checked by preflight. Both cases must use the same
input snapshots, forecast-model checksums, weather vintages, delivered PV and
intervals; the post-run comparison rejects mismatches.

From the repository root, show the run plan without starting a simulation:

```sh
conda activate bess
python scripts/run_linear_soh_farm_comparison.py --run-id lfp_comparison_2026_aug29
```

Start the study from a terminal with:

```sh
python -u scripts/run_linear_soh_farm_comparison.py --run-id lfp_comparison_2026_aug29 --run
```

If interrupted, continue unfinished jobs from their checkpoints with the same
run ID and unchanged configuration and inputs:

```sh
python -u scripts/run_linear_soh_farm_comparison.py --run-id lfp_comparison_2026_aug29 --run --resume
```

Existing completed and EOL-stopped jobs are preserved and excluded from the
queue; failed or unrecorded jobs are resumed under the continuous pipeline's identity checks.
Logs append on resume, and prior failures remain in `manifest.json`. If an EOL
stop or an input inconsistency prevents a full run, it is reported as a failure
and no full-period paired result is claimed. The batch produces `comparison.csv`
for complete matching farm pairs, showing realized auction and imbalance cash,
market cash, each model's noncash wear, cash minus model wear, throughput and
SOH when present. It also evaluates both executed trajectories using one
common provisional linear-SOH proxy: elapsed calendar seconds and efficiency-
adjusted internal throughput, the same calibration, EOL and replacement cost.
The `common_proxy_*` columns report estimated calendar/cycling losses, final SOH
and noncash wear; `market_cash_minus_common_proxy_wear_eur` subtracts that wear
once. These are post-processing counterfactuals, not measured health: the
baseline actually operated with fixed capacity and has not been rerun with the
proxy's shrinking capacity. The proxy requires fresh batteries at the start.

The `figures/` folder plots matched market cash, market cash minus the common
proxy wear, and throughput by farm, plus the daily committed SOH trajectories
for completed SOH cases. Because the models use different aging coefficients,
their own wear charges are not a common ground-truth valuation; interpret this
comparison alongside physical aging benchmarks and sensitivity analysis.

After the queue finishes, `summary.json` and `comparison.md` provide a
machine-readable summary and a readable comparison. Incomplete farm pairs are
excluded and listed explicitly; matching-input errors are recorded in the
manifest rather than reported as a valid comparison.
The Markdown, JSON and `auction_comparison.csv` also show booked DA, IDA1,
IDA2 and IDA3 cash flows, gross purchased and sold MWh and transaction fees
for each completed farm and case. Signed auction cash excludes fees and is
reconciled to delivered-interval revenue. It describes booked trades rather
than causally attributing profit; individual trades remain in each run's
`accepted_trades.csv`.

For complete matched farm pairs, the batch also runs the standard intraday
analysis and monthly plots under `intraday_analysis/`, including hourly and
quarter-hour profiles, prices, imbalance and final commitments. SOC in that
analysis uses nominal capacity, as documented by the intraday report.
Transient raw-vintage, PV-conversion and issue-context caches are cleared after
preflight; the pinned forecasts needed for replay remain until the run finishes.
The worker then collects unreachable objects before taking another job. Python
and native allocators may retain freed memory for reuse.

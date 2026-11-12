# Integrated fixed-rate LFP coarse-to-fine study

Farm 1 is replayed from 2026-01-01 through 2026-01-21 using integrated auction
planning, the grid-tracking controller, one deterministic PV forecast, and a
fresh battery. Prior orders remain fixed. Delivery-day planning retains the
50% terminal SOC target; realized tracking can finish elsewhere. IDA3 does not
extend into the next day. Both the forecast model and prices are retrospective:
actual auction prices are an oracle, so the results do not measure live profit.
The latest eligible weather vintage uses the configured six-hour availability
lag. Curated realized PV is used in execution. No training or fitting is run.

The 2-, 3- and 4-segment depth cases progressively tighten the calendar and
square-root approximation budgets as documented in
`naumann_sequential_coarse_depth_study.md`. All use fixed 0.5C cycle stress,
fixed 25°C, daily SOH updates, seven Gurobi threads, a 300-second per-solve
limit and relative MIP gap 1e-4. There may be many solves per day, so 21 days
can take a long time. Coarse graphs may bias decisions and health forecasts.

Run the three configurations sequentially from the repository root:

```sh
conda activate bess
for depth in 2 3 4; do
  python -u -m bess_pv_trading experiment-continuous \
    --config "configs/research/naumann_integrated_depth_${depth}.yaml" \
    --farm-id 1 || break
done
```

Each case writes under
`outputs/energy_research/naumann_integrated_coarse_to_fine_21days_2026/farm_1/`.
Existing results are not overwritten. Once complete, compare matched dispatch,
SOH and independent analytical rainflow estimates with
`scripts/report_naumann_resolution.py`, and compare realized market cash and a
common aging proxy against the historical throughput run with
`scripts/compare_aging_baseline.py`. Historical and new configurations still
differ in replacement economics and capacity fading; compare provenance before
attributing differences to a single modeling choice. Use held-out battery
measurements, not rainflow agreement alone, to assess physical accuracy.

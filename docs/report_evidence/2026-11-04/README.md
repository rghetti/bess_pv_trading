# Research report evidence snapshot

This directory accompanies `docs/mathematical_model.tex` and the shorter
`docs/research_overview.tex`. It contains exports of existing results, not new
model runs. The main status snapshot was captured on 4 November 2026 at
15:52:57 UTC (16:52 CET). Historical sensitivity artifacts added in the final
review have separate addition timestamps in `snapshot.json`.

`snapshot.json` records the three checkout revisions, their working-tree state,
saved search statuses, and SHA-256 hashes of the evidence files. A saved running
status does not establish that a process is alive. Completed results do not by
themselves establish convergence, historical availability, or an untouched test.

The CSV files retain numerical precision; LaTeX tables round for readability:

- `reviewed_pv_selected.csv`: selected PV models, split and target-source scores.
- `monthly_auction_comparisons.csv`: the named mixed-model monthly comparison
  family, kept distinct from earlier physical-model comparisons.
- `continuous_participation.csv`: all 17 saved profiles for each of four farms.
- `auction_sensitivities.csv`: 220 earlier penalty, battery and 10/20-day
  participation records, with original source paths and heterogeneous columns.
- `january_mpc_by_farm.csv`: 104 completed farm/case jobs, their own tracking
  baselines, differences, and solver diagnostics.
- `january_mpc_aggregate.csv`: sums across the four farms over two days; these
  are not daily averages. Divide by two for mean daily portfolio differences.
- `january_case_settings.json`: the relevant resolved settings for all 26
  farm-1 cases, with source paths and hashes. Other farms' asset ratings are in
  the source manifest's job plan. These were copied during the final audit.
- `feature_benchmark.csv`: the saved 90-day feature-workflow benchmark.
- `price_run_inventory.csv`: 40 saved price experiment manifests and statuses.
- `price_candidate_results.csv`: saved candidate result entries; nested fields
  preserve the exported representation. Entries are not independent trials or
  a uniform evaluation population.

Cash profit excludes artificial planning penalties and terminal salvage values.
Final stored energy must be inspected separately. Measured, reconstructed and
policy-zero targets retain different evidential meanings. Do not join studies
by farm and date alone: forecast vintage, target policy, objective, chronology
and initial/terminal conditions also matter.

Build from the main project root with an external auxiliary directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/bess_report_build docs/mathematical_model.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/bess_overview_build docs/research_overview.tex
```

The report source includes the frozen table fragments here. No training,
optimization, database access or data acquisition is required to rebuild it.

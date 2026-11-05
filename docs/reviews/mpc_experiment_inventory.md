# MPC and feedback-controller experiment inventory

Snapshot: 2026-11-05T16:49:01.114048+00:00; repository revision `ce8999ee0a8e`. This is a record of completed or attempted controller experiments, not authorization to run new ones.

The machine-readable archive contains 207 research jobs across 12 completed batches, and 6 main-checkout comparison modes including unsuccessful modes. The accompanying CSV has one row per research job. The compressed JSON preserves exact resolved configuration snapshots, source and input fingerprints, controller metrics, provenance and original as well as preserved output paths.

## Completed continuous research batches

| Batch | Delivery dates | Farms | Jobs / cases | What was tested | Evidence |
| --- | --- | --- | ---: | --- | --- |
| `curtailment_recovery_jan15_21_2026_v1` | 2026-01-15 to 2026-01-21 | 1, 345, 901, 910 | 8 / 2 | Original deterministic MPC vs optional grid-neutral curtailed-PV recovery. | [curtailment_recovery_one_week_results.md](../../docs/reviews/curtailment_recovery_one_week_results.md) |
| `economic_continuous_development_v1` | 2026-01-15 to 2026-01-16 | 1 | 5 / 5 | Economic, settlement forecast, adaptive recourse and reduced-scenario pilot. | [continuous_mpc_validation_economic_v1.md](../../docs/reviews/continuous_mpc_validation_economic_v1.md) |
| `ensemble_oracle_corrected_v3` | 2026-01-15 to 2026-01-16<br>2026-08-02 to 2026-08-03 | 1, 345, 901, 910 | 8 / 1 | Corrected ensemble price-oracle validation in January and August. | [continuous_mpc_validation_ensemble_oracle_v3.md](../../docs/reviews/continuous_mpc_validation_ensemble_oracle_v3.md) |
| `first_action_only_jan15_17_2026_v1` | 2026-01-15 to 2026-01-17 | 1, 345, 901, 910 | 16 / 4 | Ensemble common vs shared-first controls, cost and pure-imbalance objectives. | [first_action_only_targeted_results.md](../../docs/reviews/first_action_only_targeted_results.md) |
| `first_interval_priority_jan15_21_2026_v1` | 2026-01-15 to 2026-01-21 | 1, 345, 901, 910 | 12 / 3 | Original MPC vs weight-10 and lexicographic current-imbalance priority. | [first_interval_priority_one_week_results.md](../../docs/reviews/first_interval_priority_one_week_results.md) |
| `legacy_continuous_validation_v1` | 2026-01-15 to 2026-01-16<br>2026-08-02 to 2026-08-03 | 1, 345, 901, 910 | 32 / 4 | Earlier deterministic/ensemble MPC and price-oracle tests; legacy oracle-cap audit issues. | [continuous_mpc_validation_legacy_v1.md](../../docs/reviews/continuous_mpc_validation_legacy_v1.md) |
| `new_research_pilot_v3` | 2026-01-15 to 2026-01-16 | 1 | 4 / 4 | Calibrated ECC/shuffle, forecast auction prices and continuation-value pilot. | [continuous_mpc_validation_pilot_v3.md](../../docs/reviews/continuous_mpc_validation_pilot_v3.md) |
| `provisional_ida3_jan15_17_2026_v1` | 2026-01-15 to 2026-01-17 | 1, 345, 901, 910 | 8 / 2 | Deterministic MPC submitted-only vs planned IDA3 commitments. | [provisional_ida3_targeted_results.md](../../docs/reviews/provisional_ida3_targeted_results.md) |
| `reviewed_farm910_august_v1` | 2026-08-02 to 2026-08-03 | 910 | 4 / 4 | Reviewed-target August farm 910 ensemble, adaptive and oracle cases. | [continuous_mpc_reviewed_farm910_august_v1.md](../../docs/reviews/continuous_mpc_reviewed_farm910_august_v1.md) |
| `reviewed_january_matrix_v1` | 2026-01-15 to 2026-01-16 | 1, 345, 901, 910 | 104 / 26 | Reviewed-target 26-case, four-farm January research matrix. | [reviewed_january_matrix_audit.md](../../outputs/energy_research/reviewed_january_matrix_audit.md) |
| `reviewed_october_deterministic_v1` | 2025-10-01 to 2025-10-07 | 1, 345, 901, 910 | 4 / 1 | Reviewed-target October seven-day deterministic MPC vs tracking. | [continuous_mpc_reviewed_october_v1.md](../../docs/reviews/continuous_mpc_reviewed_october_v1.md) |
| `two_stage_gap_comparison_v4` | 2026-08-02 to 2026-08-03 | 1 | 2 / 2 | Two-stage price-oracle and accepted-gap fallback comparison. | [mpc_stage1_performance.md](../../docs/reviews/mpc_stage1_performance.md) |

All above batch statuses and job statuses are `complete`; this does not mean every historical method passed its independent audit. In particular, the legacy run had oracle imbalance-cap violations and has corrected follow-up evidence. Some older runs were copied from `/tmp` snapshots: use the preserved repository paths in the archive, not the original temporary paths. Old batches are not interchangeable with the newer reviewed-target/model studies.

## Main checkout: historical modes and status

| Study | Dates | Mode | Status | Controllers completed |
| --- | --- | --- | --- | --- |
| `farm1_2026_01_15_16_mpc` | 2026-01-15 to 2026-01-16 | frozen_commitments | complete | track_committed_grid, deterministic_mpc, deterministic_mpc_price_oracle, ensemble_mpc, ensemble_mpc_price_oracle |
| `farm1_2026_01_15_16_mpc_tcn_current_pv` | 2026-01-15 to 2026-01-16 | continuous | complete | track_committed_grid, deterministic_mpc |
| `farm1_2026_01_15_16_mpc_tcn_current_pv` | 2026-01-15 to 2026-01-16 | frozen_commitments | complete | track_committed_grid, deterministic_mpc, deterministic_mpc_price_oracle, ensemble_mpc, ensemble_mpc_price_oracle |
| `farm1_january_2026_tcn_current_pv_all_auctions` | 2026-01-01 to 2026-01-31 | continuous | complete | track_committed_grid, deterministic_mpc |
| `farm1_january_2026_tcn_current_pv_all_auctions` | 2026-01-01 to 2026-01-31 | frozen_commitments | complete | track_committed_grid, deterministic_mpc |
| `farm1_may_2026_tcn_current_pv_all_auctions` | 2026-05-01 to 2026-05-31 | frozen_commitments | failed | none |

The full-January Farm 1 TCN/current-PV frozen and continuous modes have complete manifests, even though an older study-plan document still describes them as unexecuted. The May frozen Farm 1 run failed and the all-farm May summary is interrupted with no completed comparison; do not treat these as completed results. Frozen-ledger runs are recorded for historical completeness; the user currently prefers continuous comparisons and has not requested a frozen-ledger rerun.

## The 26-case January development matrix

Each case was run on farms 1, 345, 901 and 910 for 15–16 January 2026. The table identifies the declared comparison; all numerical settings and per-farm results are in the archive and CSV.

| Case | Controller | Family | Declared reference | PV information |
| --- | --- | --- | --- | --- |
| `adaptive_continuation` | `ensemble_mpc_adaptive` | continuation_value | `adaptive_raw` | forecast |
| `adaptive_ecc` | `ensemble_mpc_adaptive` | temporal_dependence | `adaptive_raw` | forecast |
| `adaptive_monetary_risk` | `ensemble_mpc_adaptive` | monetary_risk | `adaptive_raw` | forecast |
| `adaptive_raw` | `ensemble_mpc_adaptive` | adaptive_recourse | `ensemble_economic` | forecast |
| `adaptive_reduced` | `ensemble_mpc_adaptive` | decision_aware_reduction | `adaptive_raw` | forecast |
| `adaptive_shuffled` | `ensemble_mpc_adaptive` | temporal_dependence | `adaptive_raw` | forecast |
| `calibrated_double_energy` | `ensemble_mpc_adaptive` | forecast_flexibility | `double_battery_energy` | forecast |
| `calibrated_half_poi` | `ensemble_mpc_adaptive` | forecast_flexibility | `half_poi_capacity` | forecast |
| `calibrated_half_power` | `ensemble_mpc_adaptive` | forecast_flexibility | `half_battery_power` | forecast |
| `calibrated_shuffled` | `ensemble_mpc_adaptive` | temporal_dependence | `adaptive_ecc` | forecast |
| `causal_auctions` | `ensemble_mpc_adaptive` | price_information | `adaptive_raw` | forecast |
| `causal_fundamentals` | `ensemble_mpc_adaptive` | price_information | `causal_auctions` | forecast |
| `da_only_frozen` | `ensemble_mpc_adaptive` | information_auction_factorial | `da_only_latest` | forecast |
| `da_only_latest` | `ensemble_mpc_adaptive` | information_auction_factorial | `adaptive_raw` | forecast |
| `deterministic_economic` | `deterministic_mpc_economic` | price_information | `deterministic_penalty` | forecast |
| `deterministic_oracle_gap_fallback` | `deterministic_mpc_price_oracle` | oracle_gap_fallback | `deterministic_price_oracle` | forecast |
| `deterministic_penalty` | `deterministic_mpc` | controller | `tracking` | forecast |
| `deterministic_price_oracle` | `deterministic_mpc_price_oracle` | price_information | `deterministic_penalty` | forecast |
| `double_battery_energy` | `ensemble_mpc_adaptive` | forecast_flexibility | `adaptive_raw` | forecast |
| `ensemble_economic` | `ensemble_mpc_economic` | price_information | `ensemble_penalty` | forecast |
| `ensemble_oracle_gap_fallback` | `ensemble_mpc_price_oracle` | oracle_gap_fallback | `ensemble_price_oracle` | forecast |
| `ensemble_penalty` | `ensemble_mpc` | controller | `deterministic_penalty` | forecast |
| `ensemble_price_oracle` | `ensemble_mpc_price_oracle` | price_information | `ensemble_penalty` | forecast |
| `frozen_information` | `ensemble_mpc_adaptive` | information_auction_factorial | `adaptive_raw` | forecast |
| `half_battery_power` | `ensemble_mpc_adaptive` | forecast_flexibility | `adaptive_raw` | forecast |
| `half_poi_capacity` | `ensemble_mpc_adaptive` | forecast_flexibility | `adaptive_raw` | forecast |

The matrix includes deterministic and ensemble imbalance penalties; settlement-price oracle and gap fallback; forecast settlement economics; adaptive observed-history recourse; scenario reduction, temporal shuffling, ECC calibration, monetary CVaR and continuation value; causal auction price forecasts; frozen versus updated information; DA-only variants; and battery energy, power and POI sensitivities. Its ensemble scenario weights are raw unless an explicit calibration is configured. The exact flags and original prices and data sources are in the per-job snapshots; do not infer semantics from the case label alone.

## Recent rerun targets

- **Planned IDA3:** submitted-only vs `ida2_plan`, deterministic MPC, 15–17 January, four farms; current interval average PV disclosed; reference is each case’s grid tracker.
- **Shared-first ensemble:** common vs scenario-specific later controls, under imbalance-and-cost and pure absolute-imbalance objectives; 15–17 January, four farms. The first command is shared; future path revelation is an optimistic assumption. Both variants use provisional IDA3.
- **Curtailment recovery:** original MPC vs grid-neutral recovery after protection, 15–21 January, four farms. Recovery respects SOC and power headroom and the next solve uses the actual state.
- **First-interval priority:** original, weight-10 and lexicographic deterministic MPC, 15–21 January, four farms, including the saved refinement rejection diagnostics.
- **January month:** Farm 1 tracker vs deterministic MPC, 1–31 January; an older TCN and current realized interval-average PV. A completed frozen comparison also exists but is not part of the current continuous-only preference.
- **October and August:** the seven-day October four-farm deterministic test and two-day August farm-910 ensemble/oracle test provide additional historical windows.

## How to prepare a rerun after training

1. Identify the actual newly trained model bundle for each farm from its finished manifest; record artifact and preprocessing checksums, training and selection dates, architecture, weather-feature contract, night clipping and retrospective availability. Do not assume the future model path or that deterministic and ensemble adapters support it.
2. For each selected experiment, create a new study YAML that includes `configs/standard.yaml` directly, pins the new per-farm bundle and explicitly reproduces the archived comparison settings. Use new run IDs and outputs; never overwrite prior evidence or resume old checkpoints with new models. The three recent YAMLs are still exact checksum matches to their original batch configurations. Several older study YAML hashes no longer match current files, so reconstruct them from the saved resolved configurations and historical source checksums before claiming an exact rerun.
3. Verify data and forecast-vintage eligibility, curated target and synthetic provenance, first-day initialization, current-PV information, settlement and auction-price modes, terminal constraints, and whether IDA3 and recovery are enabled. Use two farms/workers and bounded solver threads, no GPU or training unless separately approved. Keep daily plots to at most three dates.
4. Run the plan/check-only preflight and software tests, then present the concrete experiment plan for explicit approval before launching any historical optimization. This inventory itself does not approve or schedule a run.

The recent study runner accepts `--config`, repeated `--farm`, `--case`, `--window`, `--run-id`, `--workers 2`, `--solver-threads 1`, and either `--check-only` or `--run`. The new-model config must first be created and verified; do not run the old YAML unchanged and call it a new-model result.

The study archives under `outputs/` are git-ignored; keep backups. The checked-in `mpc_experiment_inventory_records.json.gz` preserves resolved configs and compact results even if raw outputs are moved. To inspect it with Python:

```python
import gzip, json
with gzip.open('docs/reviews/mpc_experiment_inventory_records.json.gz', 'rt') as stream:
    inventory = json.load(stream)
print(len(inventory["research_jobs"]))
```

Excluded from this inventory: unrelated PV training searches and stand-alone price-model searches. Solver formulation/timing diagnostics are recorded separately in `docs/reviews/mpc_stage1_performance.md`; they are not full historical controller replays.

## Downloadable portable package

The companion files are [research-job index](mpc_experiment_inventory_jobs.csv),
[main-checkout mode index](mpc_experiment_inventory_main_modes.csv), and
[compressed configuration and provenance records](mpc_experiment_inventory_records.json.gz).
The `bundle_config_path` in each index points to that run's individual JSON
snapshot **inside** the portable ZIP. Snapshots retain original paths for
historical evidence; use the preparer to relocate them before running.

The portable ZIP also includes a snapshot of the current source and
configuration code from both repositories, the individual original JSON
configs, and a Linux environment package list. It does not contain the large
weather, curated PV and market datasets, model weights, price forecast
artifacts, historical run outputs, a Gurobi license or Git credentials. Those
must be supplied separately. Historic runs used several code revisions and
occasionally uncommitted source snapshots; the bundled current code is suitable
for the next comparison but is not proof of bitwise reproduction of every old
run. Input snapshot IDs, source checksums and recorded revisions are in the
archive for verification.

After transferring the package and supplying inputs, use the included
`realtime/scripts/prepare_inventory_replay.py --list` to see run IDs. The
preparer can create one configuration with `--id` or all completed continuous
jobs in a research batch with `--batch`; it requires a new run ID and either a
new per-farm `--model-map` or an explicit `--keep-original-model`. It does not
run an experiment. The package README gives the exact commands and explains
model adapter compatibility and retrospective options. A relocated three-day
configuration passed a complete check-only preflight; no controller solve was
started for that test.

Portable preparation was checked against 209 completed continuous
configurations (207 research jobs and two main-checkout modes). The earlier 197
were checked on the previous code snapshot; the 12 first-priority configurations
were checked separately on the current snapshot. Their original
forecast, physical, solver, objective, price and execution settings survived
YAML resolution in the sampled code snapshot; output identity was deliberately
changed. This is a configuration check, not 209 data preflights or solver
runs. The preparer has focused relocation, new-model and numeric-tolerance
regression tests, and one relocated three-day data preflight passed.

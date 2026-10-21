"""Runtime samples, retry accounting and comparison totals must remain distinct."""
from types import SimpleNamespace

import numpy as np
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.reporting.optimization_timing import (
    load_experiment_timing, optimization_timing_report, participation_timing_report,
    report_saved_optimization_timing, timing_statistics,
)
from bess_pv_trading.simulation.synthetic import write_json


@pytest.fixture
def timing_config():
    return load_config("tests/fixtures/configs/optimization_timing_cases.yaml")


def test_native_runtime_statistics_include_failed_solves_and_separate_benchmarks(timing_config):
    rows = timing_config["timing_cases"]["solves"]
    report = optimization_timing_report(rows)
    samples = [row["runtime_seconds"] for row in rows]
    assert report["gurobi_runtime_seconds"]["total"] == pytest.approx(sum(samples))
    assert report["gurobi_runtime_seconds"]["mean"] == pytest.approx(np.mean(samples))
    assert report["gurobi_runtime_seconds"]["std"] == pytest.approx(np.std(samples))
    replay = [row["runtime_seconds"] for row in rows if row["category"] == "replay"]
    assert report["by_category"]["replay"]["gurobi_runtime_seconds"]["mean"] == pytest.approx(np.mean(replay))
    assert report["by_auction"][rows[1]["auction"]]["solve_count"] == 1
    assert timing_statistics([None, samples[0]])["mean"] == samples[0]
    assert timing_statistics([samples[0]])["std"] == 0
    assert timing_statistics([])["mean"] is None
    missing = optimization_timing_report([{**rows[0], "runtime_seconds": None}])
    assert missing["missing_runtime_count"] == 1
    assert missing["gurobi_runtime_seconds"]["count"] == 0


def test_study_statistics_use_case_totals_and_pool_raw_samples_once(timing_config, tmp_path):
    case = timing_config["timing_cases"]
    experiments, raw = [], []
    for index, spec in enumerate(case["experiments"]):
        output = tmp_path / str(index)
        solves = [{"category": "replay", "runtime_seconds": value} for value in spec["replay"]]
        solves.append({"category": "perfect_pv_counterfactual", "runtime_seconds": spec["benchmark"]})
        raw.extend(row["runtime_seconds"] for row in solves)
        write_json(output / "json/optimization_timing.json", optimization_timing_report(solves))
        experiments.append({"profile": case["profiles"][spec["profile_index"]],
            "study_date": case["days"][spec["day_index"]], "output_dir": str(output)})
    report = participation_timing_report(experiments)
    totals = [sum(spec["replay"]) + spec["benchmark"] for spec in case["experiments"]]
    replay_totals = [sum(spec["replay"]) for spec in case["experiments"]]
    assert report["experiment_total_runtime_seconds"]["mean"] == pytest.approx(np.mean(totals))
    assert report["experiment_total_runtime_seconds"]["std"] == pytest.approx(np.std(totals))
    assert report["experiment_replay_runtime_seconds"]["min"] == min(replay_totals)
    assert report["experiment_replay_runtime_seconds"]["max"] == max(replay_totals)
    assert report["gurobi_runtime_seconds"]["mean"] == pytest.approx(np.mean(raw))
    assert report["gurobi_runtime_seconds"]["count"] == len(raw)
    assert report["by_profile"][case["profiles"][0]]["experiment_count"] == sum(spec["profile_index"] == 0 for spec in case["experiments"])
    assert participation_timing_report(experiments) == report
    with pytest.raises(ValueError, match="Duplicate profile/day"):
        participation_timing_report(experiments + [experiments[0]])
    absent = participation_timing_report([{**experiments[0], "output_dir": str(tmp_path / "missing")}])
    assert absent["missing_experiment_runtime_count"] == 1
    assert absent["experiment_total_runtime_seconds"]["mean"] is None
    # Refreshing an existing study reads its completed checkpoint; it never
    # invokes the replay or duplicates samples when run again after completion.
    study = timing_config["participation_study"]
    study.update(output_root=str(tmp_path), run_id="timing_report")
    manifest = {"status": "running", "experiments": experiments, "requested_experiments": len(experiments)}
    manifest_path = tmp_path / study["run_id"] / "json/manifest.json"
    write_json(manifest_path, manifest)
    refreshed = report_saved_optimization_timing(timing_config)
    assert refreshed["status"] == "partial"
    assert refreshed["experiment_count"] == len(experiments)
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    finished = report_saved_optimization_timing(timing_config)
    assert finished["status"] == "complete"
    assert finished["gurobi_runtime_seconds"] == refreshed["gurobi_runtime_seconds"]


def test_existing_raw_diagnostics_can_supply_timings_for_resumed_cases(timing_config, tmp_path):
    rows = timing_config["timing_cases"]["solves"]
    for row, filename in zip(rows[:2], ("solves", "infeasible_auction_decisions")):
        write_json(tmp_path / f"json/oracle/{filename}.json", [{**row, "current_auction": row["auction"]}])
    write_json(tmp_path / "json/perfect_pv_counterfactual.json", rows[-1])
    report = load_experiment_timing(tmp_path)
    assert report["recovered_from_legacy_diagnostics"]
    assert report["gurobi_runtime_seconds"] == optimization_timing_report(rows)["gurobi_runtime_seconds"]


def test_gurobi_retry_preserves_both_native_runtimes(timing_config):
    from bess_pv_trading.domain import MarketState, SolverSettings
    from bess_pv_trading.optimization.model import solve_model
    from bess_pv_trading.optimization.degradation import degradation_settings_from_config
    from bess_pv_trading.transaction_fees import transaction_fees_from_config
    case = timing_config["timing_cases"]["retry"]

    class Model:
        SolCount = NumVars = NumBinVars = NumConstrs = NumGenConstrs = 0
        Params = SimpleNamespace(FeasibilityTol=timing_config["solver"]["feasibility_tolerance"],
            IntFeasTol=timing_config["solver"]["integrality_tolerance"])

        def __init__(self):
            self.calls = 0

        def setParam(self, name, value):
            pass

        def optimize(self):
            self.Status = case["statuses"][self.calls]
            self.Runtime = case["runtimes"][self.calls]
            self.calls += 1

    inputs = SimpleNamespace(tree=SimpleNamespace(nodes=()), scenarios=SimpleNamespace(scenario_ids=()),
        market=MarketState(), degradation=degradation_settings_from_config(timing_config),
        transaction_fees=transaction_fees_from_config(timing_config), optimize_imbalances=False)
    built = SimpleNamespace(model=Model(), inputs=inputs, build_seconds=case["build_seconds"],
        variables={"transaction_fee_formulation": timing_config["transaction_fees"]["mode"]})
    result = solve_model(built, SolverSettings(**timing_config["solver"]))
    assert result.status == "infeasible"
    assert result.diagnostics["runtime_seconds"] == sum(case["runtimes"])
    assert result.diagnostics["optimize_call_runtimes_seconds"] == case["runtimes"]

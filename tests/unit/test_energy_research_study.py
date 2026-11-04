"""Plan arithmetic, fresh farm resolution, non-overwrite and no-solve preflight."""

import json
import os
from pathlib import Path

import pytest
import pandas as pd
import yaml

from bess_pv_trading.simulation import energy_research_study as study


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/research/energy_market_extensions.yaml"


def local_config(tmp_path, change=None):
    """Copy only declarative settings; these tests never read historical datasets."""
    value = yaml.safe_load(CONFIG.read_text())
    value["include"] = os.path.relpath(ROOT / "configs/standard.yaml", tmp_path)
    value["energy_research_study"]["output_root"] = str(tmp_path / "outputs")
    value["energy_research_study"]["farms"] = [1]
    value["energy_research_study"]["workers"] = 1
    value["energy_research_study"]["windows"] = value["energy_research_study"]["windows"][:1]
    value["energy_research_study"]["cases"] = value["energy_research_study"]["cases"][:1]
    value["energy_research_study"].pop("factorial_cases", None)
    value["energy_research_study"].pop("forecast_flexibility_factorials", None)
    if change:
        change(value)
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def test_default_matrix_counts_and_continuous_mode_are_exact():
    plan = study.research_plan(CONFIG)
    assert plan["job_count"] == 224
    assert plan["case_count"] == 28
    assert plan["farm_count"] == 4
    assert plan["window_count"] == 2
    assert plan["mpc_decisions"] == 43008
    assert plan["auction_submission_solves"] == 3392
    assert plan["physical_controller_trajectories"] == 448
    assert plan["solves_started"] == plan["models_fitted"] == 0
    assert all(job["mode"] == "continuous" for job in plan["jobs"])
    assert plan["requested_workers"] == plan["workers"] == plan["configured_cpu_threads"] == 42


@pytest.mark.parametrize("kind", ["deterministic", "ensemble"])
def test_numeric_retry_case_only_changes_retry_settings(kind):
    """The closed-loop comparison isolates the opt-in numerical recovery."""
    baseline_id = f"{kind}_price_oracle"
    retry_id = f"{kind}_oracle_numeric_retry"
    _, _, jobs = study.expand_research_jobs(CONFIG, case_ids=[baseline_id, retry_id],
        farms=[1], window_ids=["january_development"])
    configs = {job["case_id"]: job["config"] for job in jobs}
    baseline = configs[baseline_id]
    retry = configs[retry_id]
    assert retry["realtime_mpc"].pop("oracle_retry_numeric_focus") == 3
    assert retry["realtime_mpc"].pop("oracle_retry_feasibility_tolerance") == 1e-9
    # Run identifiers differ by case even though numerical and information inputs match.
    retry["continuous_optimization"]["run_id"] = baseline["continuous_optimization"]["run_id"]
    assert retry == baseline


def test_fresh_raw_config_resolves_each_farm_before_asset_scaling():
    _, _, jobs = study.expand_research_jobs(CONFIG,
        case_ids=["adaptive_raw", "double_battery_energy"], farms=[1, 910], window_ids=["january_development"])
    configs = {(job["farm_id"], job["case_id"]): job["config"] for job in jobs}
    assert configs["1", "adaptive_raw"]["battery"]["capacity_mwh"] == 0.75
    assert configs["910", "adaptive_raw"]["battery"]["capacity_mwh"] == 5
    assert configs["1", "double_battery_energy"]["battery"]["capacity_mwh"] == 1.5
    assert configs["910", "double_battery_energy"]["battery"]["capacity_mwh"] == 10
    assert configs["910", "adaptive_raw"]["forecast"]["farm_id"] == 910
    assert configs["910", "adaptive_raw"]["plant"]["poi_export_mw"] == pytest.approx(6.638)
    assert all(config["experiment_outputs"]["overwrite_existing"] is False for config in configs.values())


def test_expansion_reads_one_snapshot_and_reloads_it_on_next_call(tmp_path, monkeypatch):
    """Jobs share no resolved state and later calls observe configuration edits."""
    path = local_config(tmp_path)
    original = study._read_mapping
    reads = []

    def counted(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(study, "_read_mapping", counted)
    _, _, first_jobs = study.expand_research_jobs(path)
    value = yaml.safe_load(path.read_text())
    value["state"]["initial_soc_percent"] = 37.0
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    _, _, second_jobs = study.expand_research_jobs(path)
    assert len(reads) == 2
    assert first_jobs[0]["config"]["state"]["initial_soc_percent"] != 37.0
    assert second_jobs[0]["config"]["state"]["initial_soc_percent"] == 37.0


def test_fundamental_price_models_use_each_window_and_2024_ida_history():
    """The August feature contrast must never accidentally load the January fit."""
    _, _, jobs = study.expand_research_jobs(CONFIG, case_ids=["causal_fundamentals"], farms=[1])
    for job in jobs:
        fold = "january" if job["window_id"].startswith("january") else "august"
        settings = job["config"]["energy_price_forecast"]
        assert settings["auction_enabled"]
        assert f"/{fold}/ridge_fundamentals/" in settings["artifacts"]["DA"]
        for market in ("IDA1", "IDA2", "IDA3"):
            assert "extended_2024_training_v2" in settings["artifacts"][market]
            assert f"/{fold}/ridge_fundamentals/" in settings["artifacts"][market]
            assert "extended_2024_training_v2" in settings["history_paths"][market]


def test_neural_study_requires_and_hashes_each_selected_model_bundle():
    """A requested neural run must never silently downgrade to the PV baseline."""
    path = ROOT / "configs/research/neural_pv_research.yaml"
    _, _, jobs = study.expand_research_jobs(path, case_ids=["deterministic_penalty"],
                                             farms=[1, 345, 901, 910], window_ids=["january_development"])
    for job in jobs:
        requirements = {entry["role"]: entry for entry in study._artifact_requirements(job["config"])}
        for role in ("pv_model_record", "pv_model_weights", "pv_model_preprocessor"):
            assert requirements[role]["present"]
            assert requirements[role]["sha256"]
        assert job["config"]["pv"]["forecast_replacement"]["enabled"]
        assert job["config"]["continuous_optimization"]["pv_ensemble"]["cycle_updates"] == "latest_available"


def test_matched_neural_and_pvlib_cases_differ_only_by_forecast_model_and_run_id():
    """Every requested controller comparison uses the same cycles and inputs."""
    from copy import deepcopy

    path = ROOT / "configs/research/matched_neural_pvlib_research.yaml"
    definition, _, jobs = study.expand_research_jobs(path, farms=[1], window_ids=["january_development"])
    by_case = {job["case_id"]: job for job in jobs}
    neural_cases = [case for case in definition["cases"] if not case["id"].startswith("pvlib_")]
    assert len(neural_cases) == 9
    for case in neural_cases:
        neural = deepcopy(by_case[case["id"]]["config"])
        pvlib = deepcopy(by_case[case["baseline_case"]]["config"])
        assert neural["pv"]["forecast_replacement"]["enabled"]
        assert not pvlib["pv"]["forecast_replacement"]["enabled"]
        assert neural["continuous_optimization"]["pv_ensemble"]["cycle_updates"] == "latest_available"
        neural["pv"]["forecast_replacement"]["enabled"] = False
        neural["_pv_shared_settings"]["forecast_replacement"]["enabled"] = False
        neural["continuous_optimization"].pop("run_id")
        pvlib["continuous_optimization"].pop("run_id")
        assert neural == pvlib


def test_requirement_inventory_checks_calibrated_shuffle_and_all_ensemble_sources(tmp_path):
    """These paths are required even when ECC is followed by a dependence ablation."""
    _, _, jobs = study.expand_research_jobs(CONFIG, case_ids=["calibrated_shuffled"],
        farms=[1], window_ids=["january_development"])
    config = jobs[0]["config"]
    config["continuous_optimization"]["pv_ensemble"]["sources"] = [
        str(tmp_path / "winter.zarr"), str(tmp_path / "summer.zarr")]
    config["realtime_scenarios"]["calibrated_marginals_path"] = str(tmp_path / "missing.npz")
    requirements = {row["role"]: row for row in study._artifact_requirements(config)}
    assert {"weather_ensemble_0", "weather_ensemble_1", "calibrated_pv_marginals"} <= requirements.keys()
    assert "weather_ensemble" not in requirements
    assert not requirements["calibrated_pv_marginals"]["present"]


@pytest.mark.parametrize("date, intervals", [("2026-03-29", 92), ("2025-10-26", 100)])
def test_plan_counts_dst_products_instead_of_assuming_96(tmp_path, date, intervals):
    def change(value):
        window = value["energy_research_study"]["windows"][0]
        window.update(start_date=date, end_date=date)
    plan = study.research_plan(local_config(tmp_path, change))
    assert plan["mpc_decisions"] == intervals


def test_missing_models_and_data_block_check_only_without_starting_workers(tmp_path, monkeypatch):
    path = local_config(tmp_path, lambda config: config["energy_price_forecast"].update(auction_enabled=True))
    monkeypatch.setattr(study, "_run_jobs", lambda *args: pytest.fail("Missing inputs must not start a worker"))
    result = study.run_research_study(path, check_only=True, project_root=tmp_path)
    assert result["status"] == "missing_inputs"
    missing = result["plan"]["jobs"][0]["missing_requirements"]
    assert "price_model_DA" in missing
    assert "pv_measurements_and_weather" in missing
    assert not (tmp_path / "outputs").exists()


def test_check_only_does_not_create_study_outputs(tmp_path, monkeypatch):
    path = local_config(tmp_path)
    monkeypatch.setattr(study, "_artifact_requirements", lambda config, checksums=None: [])
    def fake_jobs(jobs, workers, native_threads, check_only, record):
        assert check_only
        assert workers == 1 and native_threads == 1
        return [{"status": "preflight_passed", "job_id": job["job_id"]} for job in jobs]
    monkeypatch.setattr(study, "_run_jobs", fake_jobs)
    result = study.run_research_study(path, check_only=True)
    assert result["status"] == "preflight_passed"
    assert result["solves_started"] == 0
    assert not (tmp_path / "outputs").exists()


def test_run_manifest_preserves_failure_and_refuses_same_destination(tmp_path, monkeypatch):
    path = local_config(tmp_path)
    monkeypatch.setattr(study, "_artifact_requirements", lambda config, checksums=None: [])
    def fake_jobs(jobs, workers, native_threads, check_only, record):
        result = {"job_id": jobs[0]["job_id"], "status": "failed", "error": "fixture failure"}
        record(result)
        return [result]
    monkeypatch.setattr(study, "_run_jobs", fake_jobs)
    monkeypatch.setattr(study, "_reports", lambda *args: {"status": "no_completed_cases"})
    result = study.run_research_study(path)
    destination = Path(result["output_dir"])
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["results"][0]["error"] == "fixture failure"
    assert "optimization/mpc.py" in manifest["source_checksums"]
    assert manifest["config_sha256"]
    with pytest.raises(FileExistsError):
        study.run_research_study(path)


def test_worker_reports_errors_without_swallowing_interrupts(monkeypatch):
    job = {"job_id": "fixture", "config": {"realtime_mpc_comparison": {}}}
    def failed(config):
        raise ValueError("invalid forecast fixture")
    monkeypatch.setattr(study, "run_realtime_mpc_experiment", failed)
    result = study._worker(job, True)
    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"
    assert job["config"]["realtime_mpc_comparison"] == {}
    def interrupted(config):
        raise KeyboardInterrupt()
    monkeypatch.setattr(study, "run_realtime_mpc_experiment", interrupted)
    with pytest.raises(KeyboardInterrupt):
        study._worker(job, False)


@pytest.mark.parametrize("field,value", [("workers", 43), ("workers", 0), ("native_threads", True),
    ("mode", "frozen_commitments"), ("run_id", "../escape")])
def test_invalid_execution_contract_fails_before_input_reads(tmp_path, field, value):
    path = local_config(tmp_path, lambda config: config["energy_research_study"].update({field: value}))
    with pytest.raises(ValueError):
        study.research_plan(path)


def test_selection_typos_fail_instead_of_silently_omitting_cases():
    with pytest.raises(ValueError, match="Unknown"):
        study.research_plan(CONFIG, case_ids=["adptive_raw"])


def test_runtime_batch_id_and_worker_overrides_preserve_the_yaml_defaults():
    study_settings, destination, _ = study.expand_research_jobs(CONFIG,
        case_ids=["ensemble_penalty"], farms=[1], window_ids=["january_development"],
        run_id="legacy_validation_v2", workers=2)
    assert destination.name == "legacy_validation_v2"
    assert study_settings["workers"] == 2
    assert study._study_definition(CONFIG)["run_id"] == "continuous_energy_extensions_v1"
    with pytest.raises(ValueError):
        study.expand_research_jobs(CONFIG, workers=43)
    with pytest.raises(ValueError, match="48-core"):
        study.expand_research_jobs(CONFIG, workers=42, solver_threads=2)


def test_deterministic_dates_need_no_ensemble_or_fitted_price_model():
    path = ROOT / "configs/research/deterministic_continuous_validation.yaml"
    plan = study.research_plan(path)
    assert plan["job_count"] == 16
    assert plan["workers"] == 16
    assert plan["mpc_decisions"] == 29184
    assert plan["auction_submission_solves"] == 2432
    assert all(not row["role"].startswith("price_model") and row["role"] != "weather_ensemble"
        for job in plan["jobs"] for row in job["requirements"])


def test_ecc_declares_required_calibration_artifact_without_fabricating_it():
    plan = study.research_plan(CONFIG, case_ids=["adaptive_ecc"], farms=[910], window_ids=["august_development"])
    requirement = next(row for row in plan["jobs"][0]["requirements"] if row["role"] == "calibrated_pv_marginals")
    assert requirement["path"].endswith("farm_910_2026-08-02.npz")


def test_completed_job_reports_preserve_tracking_baselines_and_factorial(tmp_path):
    cases = ["adaptive_raw", "frozen_information", "da_only_latest", "da_only_frozen"]
    definition, _, jobs = study.expand_research_jobs(CONFIG, case_ids=cases, farms=[1],
        window_ids=["january_development"])
    definition["reporting"] = {"plots": False, "samples": 0,
        "metrics": ["realized_operating_profit_eur"]}
    results = []
    for index, job in enumerate(jobs):
        output = tmp_path / job["job_id"]
        output.mkdir()
        rows = [{"delivery_day": day, "controller": controller,
            "realized_operating_profit_eur": 100 + (index + 1 if controller != "track_committed_grid" else 0)}
            for day in ["2026-01-15", "2026-01-16"]
            for controller in ["track_committed_grid", "ensemble_mpc_adaptive"]]
        pd.DataFrame(rows).to_csv(output / "paired_daily_metrics.csv", index=False)
        results.append({key: value for key, value in job.items() if key != "config"} | {
            "status": "complete", "result": {"output_dir": str(output)}})
    summary = study._reports(tmp_path, results, jobs, definition)
    assert len(summary["comparison_dirs"]) == 4
    assert summary["factorial_status"] == "complete"
    assert (tmp_path / "information_auction_factorial.json").is_file()
    report = json.loads((tmp_path / "comparisons/adaptive_raw/research_comparison.json").read_text())
    comparison = report["reports"][0]["comparisons"][0]
    assert comparison["baseline_controller"] == "track_committed_grid"
    assert comparison["controller"] == "ensemble_mpc_adaptive"


def test_executor_crash_is_recorded_for_the_case(monkeypatch):
    class FailedFuture:
        def result(self):
            raise RuntimeError("Worker exited unexpectedly")
    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["mp_context"].get_start_method() == "spawn"
        def submit(self, function, job, check_only):
            return FailedFuture()
        def shutdown(self, **kwargs):
            assert kwargs["wait"] is True
    monkeypatch.setattr(study, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(study, "as_completed", lambda handles: iter(handles))
    recorded = []
    results = study._run_jobs([{"job_id": "fixture", "config": {}}], 1, 1, False, recorded.append)
    assert results[0]["status"] == "failed"
    assert "Worker exited" in recorded[0]["error"]

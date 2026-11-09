"""Verify controller-level scheduling without running historical optimizations."""

from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def queue_module():
    """Load the standalone launcher from the repository scripts directory."""
    path = Path(__file__).resolve().parents[2] / "scripts/run_k25_controller_queue.py"
    spec = importlib.util.spec_from_file_location("k25_controller_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_queue_fills_slots_and_limits_each_farm(queue_module, monkeypatch):
    """A free worker receives another eligible trajectory without exceeding the farm limit."""
    batches = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, task):
            future = Future()
            future.set_result({"job_id": task["job"]["job_id"], "farm_id": task["job"]["farm_id"],
                               "controller": task["controller"], "status": "complete"})
            return future

    def one_done(active, **kwargs):
        farms = [task["job"]["farm_id"] for task in active.values()]
        batches.append(farms)
        assert all(farms.count(farm) <= 2 for farm in set(farms))
        first = next(iter(active))
        return {first}, set(active) - {first}

    monkeypatch.setattr(queue_module, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(queue_module, "wait", one_done)
    tasks = [{"job": {"farm_id": farm, "job_id": farm + str(number), "case_id": "case"},
              "controller": "track_committed_grid" if number % 2 == 0 else "deterministic_mpc"}
             for farm in ("1", "345") for number in range(4)]
    updates = []
    results = queue_module.run_queue(tasks, 4, 2, 1, lambda records: updates.append(len(records)))
    assert len(results) == 8
    assert batches[0] == ["1", "1", "345", "345"]
    assert len(batches[1]) == 4
    assert updates == list(range(1, 9))


def test_identical_trackers_are_run_once_and_reused(queue_module, tmp_path):
    """K25 changes MPC alone; retain its task and reuse the matched baseline."""
    base = {"continuous_optimization": {"output_root": str(tmp_path), "run_id": "original"},
            "farm_id": 1, "realtime_mpc": {"first_interval_priority": "none", "first_interval_weight": 1}}
    weighted = {**base, "continuous_optimization": {**base["continuous_optimization"], "run_id": "weighted"},
                "realtime_mpc": {"first_interval_priority": "weighted", "first_interval_weight": 25}}
    jobs = [{"farm_id": "1", "job_id": name, "config": settings}
            for name, settings in (("original", base), ("weighted", weighted))]
    tasks = [{"job": job, "controller": controller} for job in jobs
             for controller in ("track_committed_grid", "deterministic_mpc")]
    unique, aliases = queue_module.deduplicate_tracking(tasks)
    assert len(unique) == 3
    assert len(aliases) == 1
    source, target = [queue_module.trajectory_destination(job) for job in jobs]
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    for filename in ("summary__track_committed_grid.json", "input_provenance__track_committed_grid.json"):
        (source / filename).write_text('{}')
    results = [{"job_id": "original", "controller": "track_committed_grid", "status": "complete",
                "input_fingerprint": "same"}]
    queue_module.reuse_tracking_results(results, aliases)
    assert results[-1]["job_id"] == "weighted"
    assert results[-1]["reused_from_job_id"] == "original"
    assert (target / "summary__track_committed_grid.json").is_file()
    assert (target / "input_provenance__track_committed_grid.json").samefile(
        source / "input_provenance__track_committed_grid.json")


def test_each_trajectory_uses_its_controller_and_own_state(queue_module, monkeypatch, tmp_path):
    """A tracker worker selects tracking while leaving the paired MPC config intact."""
    from bess_pv_trading.simulation import continuous
    from bess_pv_trading.reporting import realtime_mpc

    prepared = {"imbalance_prices_eur_per_mwh": [1.0], "input_fingerprint": "same-snapshot",
                "provenance": {"source": "fixture"}}
    monkeypatch.setattr(queue_module, "prepare_continuous_inputs", lambda config, progress: prepared)
    seen = []

    class Replay:
        def __init__(self, config, actual, strategy, destination, progress):
            seen.append((config["execution"]["dispatch_policy"], actual, strategy))
            self.destination = destination
            self.projections = []

        def run(self):
            return {"output_dir": str(self.destination)}

        def _final_dispatch_and_ledger(self):
            return None, None, None

    monkeypatch.setattr(continuous, "ContinuousReplay", Replay)
    monkeypatch.setattr(realtime_mpc, "save_mpc_outputs", lambda *args: {"mpc_metrics": {"mpc_solves": 0}})
    job = {"farm_id": "1", "case_id": "original", "job_id": "sample", "config": {
        "continuous_optimization": {"output_root": str(tmp_path / "runs"), "run_id": "sample"},
        "execution": {"dispatch_policy": "deterministic_mpc"}}}
    destination = queue_module.trajectory_destination(job)
    (destination / "logs").mkdir(parents=True)
    result = queue_module.run_trajectory({"job": job, "controller": "track_committed_grid", "check_only": False,
                                          "preflight_cache": False})
    assert result["status"] == "complete"
    assert result["input_fingerprint"] == "same-snapshot"
    assert seen == [("track_committed_grid", prepared, "integrated")]
    assert job["config"]["execution"]["dispatch_policy"] == "deterministic_mpc"
    assert (destination / "summary__track_committed_grid.json").is_file()
    record = tmp_path / result["shared_input_provenance"]
    assert (destination / "input_provenance__track_committed_grid.json").samefile(record)
    assert json.loads(record.read_text()) == prepared["provenance"]


@pytest.mark.parametrize("matching", [True, False])
def test_batch_pairs_only_identical_input_snapshots(queue_module, monkeypatch, tmp_path, matching):
    """A completed pair produces a comparison only after fingerprint verification."""
    from bess_pv_trading.reporting import realtime_mpc

    monkeypatch.setattr(queue_module, "ROOT", tmp_path)
    (tmp_path / "src").mkdir()
    batch = tmp_path / "batch"
    config = {"continuous_optimization": {"output_root": str(batch / "runs"), "run_id": "job"},
              "realtime_mpc_comparison": {"controllers": ["track_committed_grid", "deterministic_mpc"]}}
    job = {"farm_id": "1", "case_id": "original", "job_id": "job", "config": config}
    study = {"solver_threads": 5, "native_threads": 1}
    monkeypatch.setattr(queue_module, "research_plan", lambda *args, **kwargs: {
        "missing_input_jobs": 0, "jobs": [], "solves_started": 0})
    monkeypatch.setattr(queue_module, "expand_research_jobs", lambda *args, **kwargs: (study, batch, [job]))
    monkeypatch.setattr(queue_module.subprocess, "check_output", lambda *args, **kwargs: "revision\n")
    monkeypatch.setattr(queue_module, "_reports", lambda *args: {"status": "complete"})
    calls = []
    monkeypatch.setattr(realtime_mpc, "save_controller_comparison",
        lambda config, path, summaries, mode: calls.append(set(summaries)) or [])

    def fake_queue(tasks, workers, per_farm, native_threads, completed):
        destination = queue_module.trajectory_destination(job)
        cache = batch / "preflight_cache"
        cache.mkdir()
        (cache / "temporary.pickle").write_bytes(b"temporary")
        results = []
        for task in tasks:
            controller = task["controller"]
            (destination / f"summary__{controller}.json").write_text(json.dumps({"mpc_metrics": {}}))
            (destination / f"input_provenance__{controller}.json").write_text("{}")
            fingerprint = "snapshot" if matching or controller == "track_committed_grid" else "different"
            results.append({"job_id": "job", "farm_id": "1", "controller": controller,
                            "status": "complete", "input_fingerprint": fingerprint})
            completed(results)
        return results

    monkeypatch.setattr(queue_module, "run_queue", fake_queue)
    status = queue_module.main(["--study", "january-august-deterministic", "--run-id", "batch", "--run"])
    manifest = json.loads((queue_module.trajectory_destination(job) / "manifest.json").read_text())
    if matching:
        assert status == 0
        assert calls == [{"track_committed_grid", "deterministic_mpc"}]
        assert manifest["status"] == "complete"
        assert manifest["input_fingerprint"] == "snapshot"
        assert (queue_module.trajectory_destination(job) / "input_provenance.json").is_file()
        assert (queue_module.trajectory_destination(job) / "input_provenance.json").samefile(
            queue_module.trajectory_destination(job) / "input_provenance__track_committed_grid.json")
        assert not (batch / "preflight_cache").exists()
        assert json.loads((batch / "manifest.json").read_text())["preflight_cache_cleanup"]["status"] == "removed"
    else:
        assert status == 1
        assert not calls
        assert manifest["status"] == "failed"
        assert (batch / "preflight_cache/temporary.pickle").is_file()
        assert json.loads((batch / "manifest.json").read_text())["preflight_cache_cleanup"]["status"] == "retained"

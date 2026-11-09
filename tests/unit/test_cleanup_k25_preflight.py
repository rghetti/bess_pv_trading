"""Ensure preflight cleanup cannot remove cache from unfinished studies."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def cleanup_module():
    """Load the cleanup command and its sibling cache helper."""
    directory = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("cleanup_k25_preflight", directory / "cleanup_k25_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_batch(path, status="complete", result_status="complete", reports_status="complete"):
    """Create a minimal completed-batch manifest and a separate result file."""
    path.mkdir()
    (path / "preflight_cache").mkdir()
    (path / "preflight_cache/snapshot.pickle").write_bytes(b"cache")
    (path / "summary.md").write_text("result")
    manifest = {"status": status, "plan": {"scheduler": "controller_trajectories",
        "jobs": [{"job_id": "job", "controllers": ["track_committed_grid", "deterministic_mpc"]}]},
        "results": [{"job_id": "job", "controller": controller, "status": result_status}
                    for controller in ("track_committed_grid", "deterministic_mpc")],
        "reports": {"status": reports_status}}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("status", ["running", "failed", "interrupted"])
def test_unfinished_batch_keeps_cache(cleanup_module, tmp_path, status):
    """An active, failed or interrupted trajectory cannot be cleaned."""
    batch = tmp_path / "batch"
    make_batch(batch, status=status)
    with pytest.raises(ValueError, match="not successfully finished"):
        cleanup_module.completed_cache_cleanup(batch)
    assert (batch / "preflight_cache/snapshot.pickle").is_file()


def test_incomplete_results_or_reports_keep_cache(cleanup_module, tmp_path):
    """A misleading complete status alone is insufficient for deletion."""
    batch = tmp_path / "batch"
    make_batch(batch, result_status="failed")
    with pytest.raises(ValueError, match="incomplete trajectories"):
        cleanup_module.completed_cache_cleanup(batch)
    manifest_path = batch / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["results"]:
        item["status"] = "complete"
    manifest["reports"]["status"] = "running"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reports are not complete"):
        cleanup_module.completed_cache_cleanup(batch)
    assert (batch / "preflight_cache/snapshot.pickle").is_file()


def test_completed_batch_removes_cache_and_records_cleanup(cleanup_module, tmp_path):
    """Only the temporary cache is removed and the manifest records its size."""
    batch = tmp_path / "batch"
    make_batch(batch)
    result = cleanup_module.completed_cache_cleanup(batch)
    assert result == {"status": "removed", "logical_bytes": 5}
    assert not (batch / "preflight_cache").exists()
    assert (batch / "summary.md").read_text() == "result"
    assert json.loads((batch / "manifest.json").read_text())["preflight_cache_cleanup"] == result

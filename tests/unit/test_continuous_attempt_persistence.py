"""Check solver-diagnostic persistence with and without restart checkpoints."""

import json

import pytest

from bess_pv_trading.simulation import continuous


def _replay(tmp_path, *, checkpoint_enabled):
    """Create the minimal replay state needed for solver-diagnostic saving."""
    replay = continuous.ContinuousReplay.__new__(continuous.ContinuousReplay)
    replay.checkpoint_enabled = checkpoint_enabled
    replay.destination = tmp_path
    (tmp_path / "json").mkdir()
    replay.perfect = False
    replay.strategy = "integrated"
    replay.attempts = [{"delivery_day": "2026-01-01", "current_auction": "DA",
                        "cutoff": "2025-12-31T10:55:00+00:00", "status": "infeasible",
                        "runtime_seconds": 0.5}]
    return replay


def test_attempts_written_once_on_completion_without_checkpoint(tmp_path):
    """Repeated per-auction calls are no-ops; a final write keeps all records."""
    replay = _replay(tmp_path, checkpoint_enabled=False)
    replay._save_attempts()
    replay._save_attempts()
    assert not (tmp_path / "json/optimization_attempts.json").exists()
    replay._save_attempts(force=True)
    saved = json.loads((tmp_path / "json/optimization_attempts.json").read_text())
    timing = json.loads((tmp_path / "json/optimization_timing.json").read_text())
    assert saved == replay.attempts
    assert timing["solve_count"] == 1
    assert timing["solves"][0]["status"] == "infeasible"


def test_attempts_remain_incremental_with_checkpoint(tmp_path):
    """Checkpoint-enabled runs keep the historical per-auction diagnostics."""
    replay = _replay(tmp_path, checkpoint_enabled=True)
    replay._save_attempts()
    assert json.loads((tmp_path / "json/optimization_attempts.json").read_text()) == replay.attempts


@pytest.mark.parametrize("flush_fails", [False, True])
def test_interrupted_run_flushes_attempts_without_masking_original(tmp_path, monkeypatch, flush_fails):
    """Flush on interruption, but preserve the actual failure if the disk write fails."""
    replay = _replay(tmp_path, checkpoint_enabled=False)
    replay.events = []
    replay.cursor = None

    class FailingEngine:
        def __init__(self, events):
            pass

        def run(self, handlers):
            raise KeyboardInterrupt("stopped by user")

    monkeypatch.setattr(continuous, "EventEngine", FailingEngine)
    if flush_fails:
        def fail_write(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(continuous, "_atomic_json", fail_write)
    with pytest.raises(KeyboardInterrupt, match="stopped by user") as captured:
        replay.run()
    if flush_fails:
        assert "disk full" in captured.value.__notes__[0]
    else:
        assert json.loads((tmp_path / "json/optimization_attempts.json").read_text()) == replay.attempts

"""The saved-horizon diagnostic rejects state it cannot faithfully reconstruct."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("replay_oracle", ROOT / "scripts/replay_oracle_horizon.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


@pytest.mark.parametrize(
    ("degradation", "state", "reason"),
    [
        ("xu", {}, "Xu segment state"),
        ("throughput", {"terminal_soc_percent": 50}, "hard terminal"),
        ("throughput", {"terminal_soc_min_percent": 10}, "hard terminal"),
    ],
)
def test_rejects_unreconstructable_state(tmp_path, degradation, state, reason):
    """Never substitute a simpler state model for the archived optimization."""
    config = {"degradation": {"model": degradation}, "state": state}
    (tmp_path / "config__case.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=reason):
        replay.recorded_input(tmp_path, "case", "2026-01-16T09:00:00+00:00")

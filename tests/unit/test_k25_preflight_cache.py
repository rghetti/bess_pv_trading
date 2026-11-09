"""Check batch-local preflight sharing without running a historical study."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from k25_preflight_cache import preparation_key, prepare_shared


def config():
    """Provide representative preparation and controller settings."""
    return {"farm": {"id": 1}, "model": "bigru", "dates": ["2026-01-01", "2026-08-29"],
            "continuous_optimization": {"run_id": "a", "output_root": "/tmp/a"},
            "realtime_mpc": {"first_interval_priority": "original", "first_interval_weight": 1}}


def test_cache_key_respects_input_settings():
    """Only output identity and the two objective settings can differ."""
    original = config()
    variant = deepcopy(original)
    variant["continuous_optimization"]["run_id"] = "b"
    variant["realtime_mpc"]["first_interval_weight"] = 25
    assert preparation_key(original) == preparation_key(variant)
    variant["farm"]["id"] = 345
    assert preparation_key(original) != preparation_key(variant)
    variant = deepcopy(original)
    variant["model"] = "different"
    assert preparation_key(original) != preparation_key(variant)
    variant = deepcopy(original)
    variant["dates"][1] = "2026-08-28"
    assert preparation_key(original) != preparation_key(variant)


def test_concurrent_consumers_share_and_get_independent_copies(tmp_path):
    """A locked producer serves independent objects to concurrent consumers."""
    calls = []

    def prepare(settings, progress):
        calls.append(1)
        time.sleep(0.05)
        return {"input_fingerprint": "pinned", "values": [1]}

    def consume(_):
        return prepare_shared(config(), tmp_path, prepare, lambda message: None)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(consume, range(4)))
    assert len(calls) == 1
    results[0]["values"].append(2)
    assert all(result["values"] == [1] for result in results[1:])


def test_corrupt_snapshot_is_rejected(tmp_path):
    """A damaged cache cannot silently become a study input."""
    prepare = lambda settings, progress: {"input_fingerprint": "pinned"}
    prepare_shared(config(), tmp_path, prepare, lambda message: None)
    next((tmp_path / "preflight_cache").glob("*.pickle")).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        prepare_shared(config(), tmp_path, prepare, lambda message: None)


def test_forecast_provider_round_trip(tmp_path):
    """The actual preflight provider class survives serialization."""
    from bess_pv_trading.simulation.continuous_inputs import _PreparedForecasts

    provider = _PreparedForecasts()
    prepare = lambda settings, progress: {"input_fingerprint": "pinned", "forecast_provider": provider}
    prepare_shared(config(), tmp_path, prepare, lambda message: None)
    loaded = prepare_shared(config(), tmp_path, prepare, lambda message: None)
    assert isinstance(loaded["forecast_provider"], _PreparedForecasts)
    assert loaded["forecast_provider"] is not provider

"""Exercise shared provenance publication and concurrent controller access."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def provenance_module():
    """Load the launcher helper without starting an experiment."""
    path = Path(__file__).resolve().parents[2] / "scripts/k25_provenance.py"
    spec = importlib.util.spec_from_file_location("k25_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concurrent_workers_publish_one_shared_readable_record(provenance_module, tmp_path):
    """Concurrent cases preserve legacy paths while storing one physical file."""
    batch = tmp_path / "batch"
    batch.mkdir()
    destinations = [batch / f"controller_{number}.json" for number in range(8)]
    provenance = {"days": {"2026-01-01": {"forecast": "same vintage"}}}
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda destination: provenance_module.publish_provenance(
            batch, "1", "fingerprint", provenance, destination), destinations))
    record = records[0]
    assert all(item == record for item in records)
    assert all(path.samefile(record) for path in destinations)
    assert all(json.loads(path.read_text()) == provenance for path in destinations)
    assert record.stat().st_nlink == len(destinations) + 1


def test_different_farms_and_fingerprints_do_not_share(provenance_module, tmp_path):
    """Separate input identities cannot overwrite or alias one another."""
    first = provenance_module.publish_provenance(tmp_path, "1", "first", {"farm": 1}, tmp_path / "one.json")
    second = provenance_module.publish_provenance(tmp_path, "1", "second", {"farm": 1}, tmp_path / "two.json")
    third = provenance_module.publish_provenance(tmp_path, "345", "first", {"farm": 345}, tmp_path / "three.json")
    assert not first.samefile(second)
    assert not first.samefile(third)
    assert json.loads(third.read_text()) == {"farm": 345}
    with pytest.raises(ValueError, match="different data"):
        provenance_module.link_provenance(third, tmp_path / "one.json")


def test_failed_publication_does_not_leave_partial_record(provenance_module, tmp_path):
    """Invalid data cannot be mistaken for a successfully published snapshot."""
    with pytest.raises(ValueError):
        provenance_module.publish_provenance(tmp_path, "1", "bad", {"value": float("nan")}, tmp_path / "alias.json")
    directory = tmp_path / "input_provenance/farm_1"
    assert not (directory / "bad.json").exists()
    assert not list(directory.glob("*.tmp"))
    assert not (tmp_path / "alias.json").exists()

"""Verify the transfer pilot's chronological separation."""

import numpy as np
import pytest

from scripts.evaluate_hourly_transfer_pilot import evaluate


def test_hourly_transfer_excludes_labels_after_cutoff(tmp_path):
    """A future hourly label cannot change any validation prediction score."""
    rng = np.random.default_rng(4)
    hourly = tmp_path / "hourly.npz"
    quarter = tmp_path / "quarter.npz"
    train_x = rng.normal(size=(20, 8))
    validation_x = rng.normal(size=(3, 8))
    common = {"train_x": train_x, "train_y": rng.normal(size=20),
              "train_starts": np.array(["2025-01-01T00:00:00Z"] * 20),
              "validation_x": validation_x,
              "starts": np.array(["2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z", "2025-01-05T00:00:00Z"]),
              "available_ns": np.array([1735776000000000000, 1735862400000000000, 1736035200000000000]),
              "product_end_ns": np.array([1735776000000000000, 1735862400000000000, 1736035200000000000])}
    np.savez(quarter, train_x=rng.normal(size=(20, 8)), train_y=rng.normal(size=20),
             train_starts=np.array(["2025-02-01T00:00:00Z"] * 20),
             validation_x=rng.normal(size=(5, 8)), validation_y=rng.normal(size=5),
             starts=np.array(["2025-03-01T00:00:00Z"] * 5))
    np.savez(hourly, **common, validation_y=np.array([1.0, 2.0, 3.0]))
    first = evaluate(hourly, quarter, "2025-01-04T00:00:00Z", 1, [20], [20], [0, 1])
    np.savez(hourly, **common, validation_y=np.array([1.0, 2.0, 999999.0]))
    second = evaluate(hourly, quarter, "2025-01-04T00:00:00Z", 1, [20], [20], [0, 1])
    assert first["hourly_rows"] == 22
    assert first["results"] == second["results"]
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate(hourly, quarter, "2025-01-04", 1, [20], [20], [0, 1])
    with np.load(quarter, allow_pickle=False) as data:
        earlier_quarter = {name: data[name] for name in data.files}
    earlier_quarter["train_starts"] = np.array(["2025-01-01T00:00:00Z"] * 20)
    np.savez(quarter, **earlier_quarter)
    with pytest.raises(ValueError, match="precedes the hourly training cutoff"):
        evaluate(hourly, quarter, "2025-01-04T00:00:00Z", 1, [20], [20], [0, 1])

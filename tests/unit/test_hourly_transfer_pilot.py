"""Verify the transfer pilot's chronological separation."""

import numpy as np
import pytest

from scripts.evaluate_hourly_transfer_pilot import evaluate, evaluate_trees


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
    # Tree evaluation uses the same boundary guard and must require release clocks for rolling.
    tree_specs = [{"n_estimators": 5, "min_samples_leaf": 2}]
    tree_result = evaluate_trees(hourly, quarter, "2025-01-04T00:00:00Z", 1,
                                 tree_specs, [20], [0, 1], 17, 1, None)
    assert len(tree_result["results"]) == 2
    assert all(np.isfinite(row["mae"]) for row in tree_result["results"])
    with pytest.raises(ValueError, match="requires product and publication clocks"):
        evaluate_trees(hourly, quarter, "2025-01-04T00:00:00Z", 1, tree_specs, [20], [0, 1], 17, 1,
                       {"timezone": "UTC", "frequency_days": 1, "gap_days": 1})
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate(hourly, quarter, "2025-01-04", 1, [20], [20], [0, 1])
    with np.load(quarter, allow_pickle=False) as data:
        earlier_quarter = {name: data[name] for name in data.files}
    earlier_quarter["train_starts"] = np.array(["2025-01-01T00:00:00Z"] * 20)
    np.savez(quarter, **earlier_quarter)
    with pytest.raises(ValueError, match="precedes the hourly training cutoff"):
        evaluate(hourly, quarter, "2025-01-04T00:00:00Z", 1, [20], [20], [0, 1])


def test_rolling_tree_transfer_does_not_train_on_future_labels(tmp_path):
    """A later target change cannot alter earlier rolling forecasts."""
    import pandas as pd

    rng = np.random.default_rng(7)
    hourly = tmp_path / "hourly.npz"
    quarter = tmp_path / "quarter.npz"
    np.savez(hourly, train_x=rng.normal(size=(20, 8)), train_y=rng.normal(size=20),
             train_starts=np.array(["2025-01-01T00:00:00Z"] * 20),
             validation_x=rng.normal(size=(1, 8)), validation_y=np.array([1.0]),
             starts=np.array(["2025-01-02T00:00:00Z"]),
             available_ns=np.array([1735776000000000000]),
             product_end_ns=np.array([1735776000000000000]))
    starts = pd.date_range("2025-03-01T00:00:00Z", periods=5, freq="D")
    starts_ns = starts.as_unit("ns").asi8
    hour_ns = 3600 * 10**9
    payload = {"train_x": rng.normal(size=(20, 8)), "train_y": rng.normal(size=20),
               "train_starts": np.array(["2025-02-01T00:00:00Z"] * 20),
               "validation_x": rng.normal(size=(5, 8)), "validation_y": rng.normal(size=5),
               "starts": np.asarray(starts.astype(str), dtype=str),
               "decisions_ns": starts_ns - 12 * hour_ns,
               "available_ns": starts_ns + hour_ns, "product_end_ns": starts_ns + hour_ns}
    np.savez(quarter, **payload)
    rolling = {"timezone": "UTC", "frequency_days": 1, "gap_days": 1}
    args = (hourly, quarter, "2025-01-04T00:00:00Z", 1,
            [{"n_estimators": 5, "min_samples_leaf": 2}], [20], [1], 17, 1, rolling)
    first = evaluate_trees(*args, return_predictions=True)
    payload["validation_y"] = payload["validation_y"].copy()
    payload["validation_y"][-1] += 1000000
    np.savez(quarter, **payload)
    second = evaluate_trees(*args, return_predictions=True)
    assert np.allclose(first["results"][0]["predictions"][:4],
                       second["results"][0]["predictions"][:4])

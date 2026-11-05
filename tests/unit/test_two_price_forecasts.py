"""Verify chronological weighting of two saved forecast streams."""

import json

import pandas as pd

from scripts.evaluate_two_price_forecasts import run


def test_later_targets_do_not_choose_blend_weight(tmp_path):
    """Changing later labels cannot change a weight selected on earlier dates."""
    starts = pd.to_datetime(["2026-05-01T00:00Z", "2026-05-02T00:00Z", "2026-07-01T00:00Z"])
    for name, points in [("first", [10, 10, 10]), ("second", [0, 0, 0])]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps({"status": "complete", "failures": []}))
        pd.DataFrame({"product_start": starts, "actual": [10, 10, 10], "point": points}).to_csv(
            folder / "predictions.csv", index=False)
    settings = {"run_id": "one", "output_root": "out", "timezone": "UTC", "tune_start": "2026-05-01",
        "tune_end_exclusive": "2026-06-01", "test_start": "2026-07-01", "test_end_exclusive": "2026-08-01",
        "first_model_weights": [0, 1], "models": [{"run": name, "prediction_file": "predictions.csv"}
                                                      for name in ("first", "second")]}
    first = json.loads((run(settings, tmp_path) / "manifest.json").read_text())
    for name in ("first", "second"):
        path = tmp_path / name / "predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[2, "actual"] = 1
        frame.to_csv(path, index=False)
    second = json.loads((run({**settings, "run_id": "two"}, tmp_path) / "manifest.json").read_text())
    assert first["first_model_weight"] == second["first_model_weight"] == 1

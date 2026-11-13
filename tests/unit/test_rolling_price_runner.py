"""Exercise replay gate ordering without fitting models or reading DuckDB."""

import json
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

from scripts import run_rolling_price_study as runner


def test_replay_hides_target_and_delays_actual_clearing(tmp_path, monkeypatch):
    """DA-gate IDA uses predicted DA; later gates use only published prices."""
    day = "2026-01-02"
    product = "2026-01-02T12:00:00+00:00"
    decisions = {"DA": "2026-01-01T10:45:00+00:00", "IDA1": "2026-01-01T13:45:00+00:00",
                 "IDA2": "2026-01-01T20:45:00+00:00", "IDA3": "2026-01-02T08:45:00+00:00"}
    publications = {"DA": "2026-01-01T12:00:00+00:00", "IDA1": "2026-01-01T14:30:00+00:00",
                    "IDA2": "2026-01-01T21:30:00+00:00", "IDA3": "2026-01-02T09:30:00+00:00"}
    examples = {}
    for number, market in enumerate(runner.MARKETS):
        for gate in (["DA"] if market == "DA" else ["DA", market]):
            examples[(market, gate)] = {"delivery_day": day, "market": market, "gate": gate,
                "decision_at": decisions[gate], "target_publication_at": publications[market],
                "product_starts_utc": [product], "target_values": [100 + number],
                "price_reference": [90.0], "feature_blocks": {"core": [1]}, "metadata": {}}
    class Engine:
        def __init__(self, source, options):
            self.blocks = ("core",)
        def fit_delivery_day(self, day, cutoff):
            return SimpleNamespace(unavailable={})
        def predict(self, fitted, example, da_prediction=None):
            assert "target_values" not in example
            if example["gate"] == "DA" and example["market"] != "DA":
                np.testing.assert_array_equal(da_prediction, [50.0])
            value = 50.0 if example["market"] == "DA" else 55.0
            return {"product_starts_utc": [product], "prediction": [value],
                "forecast_issue_at": decisions[example["gate"]], "model": {}, "metadata": {}}
    monkeypatch.setattr(runner, "RollingModelEngine", Engine)
    monkeypatch.setattr(runner, "EXAMPLES", list(examples.values()))
    monkeypatch.setattr(runner, "BY_DAY", {day: examples})
    monkeypatch.setattr(runner, "DESTINATION", tmp_path)
    monkeypatch.setattr(runner, "STUDY", {"start": day, "end_exclusive": "2026-01-03", "timezone": "Europe/Rome"})
    outcome = runner._job(("core", "ridge", 60, "spread"))
    root = tmp_path / "jobs" / outcome["job"]
    frame = pd.read_parquet(root / f"{day}.parquet")
    initial = frame[(frame.gate == "DA") & (frame.auction == "IDA2")].iloc[0]
    later = frame[(frame.gate == "IDA1") & (frame.auction == "IDA2")].iloc[0]
    assert initial.reference == 50
    assert later.reference == 100
    assert later.prediction == 100
    assert later.method == "carry_published_clearing"
    assert json.loads((root / f"{day}.json").read_text())["rows"] == len(frame)


def test_check_only_summarizes_without_fitting(tmp_path, monkeypatch, capsys):
    """Preflight reports source coverage and does not construct an estimator."""
    (tmp_path / "manifest.json").write_text("{}")
    examples = [{"delivery_day": "2026-01-02", "market": "DA", "gate": "DA",
                 "target_values": [10, 11], "metadata": {"weather_usable": False}}]
    monkeypatch.setattr(runner, "build_examples", lambda *args: examples)
    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not initialize models")
    monkeypatch.setattr(runner, "RollingModelEngine", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_rolling_price_study", "--snapshot", str(tmp_path), "--check-only"])
    runner.main()
    result = json.loads(capsys.readouterr().out)
    assert result["model_fits"] == 0
    assert result["coverage"][0]["complete_target_days"] == 1
    assert result["coverage"][0]["complete_weather_days"] == 0

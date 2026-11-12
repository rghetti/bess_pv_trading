"""Exercise staged proxy export contracts without fitting experimental models."""

import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.studies.price_feature_data import digest
from scripts import refit_staged_price_proxies as staged


def _descriptor(root, name, values):
    """Save a synthetic prepared array and return its checksum descriptor."""
    path = root / f"{name}.npy"
    np.save(path, values)
    return {"path": path.name, "sha256": digest(path)}


def test_refit_uses_selected_features_and_exports_annual_training_flags(tmp_path, monkeypatch):
    """Refit may recover rows omitted by unselected features; IDA targets remain spreads."""
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    starts = pd.DatetimeIndex(["2025-12-01T12:00:00Z", "2026-01-01T12:00:00Z",
                               "2026-09-30T12:00:00Z", "2026-10-01T12:00:00Z"])
    publications = pd.DatetimeIndex(["2025-12-01T12:00:00Z", "2026-01-01T12:00:00Z",
                                     "2026-10-01T12:00:00Z", "2026-10-01T12:00:00Z"])
    data = {
        "x": np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]]),
        "y": np.array([70.0, 100.0, 105.0, 110.0]),
        "anchor": np.array([50.0, 80.0, 85.0, 90.0]),
        "reference": np.array([50.0, 80.0, 85.0, 90.0]),
        "starts": starts.as_unit("ns").asi8,
        "decisions": starts.as_unit("ns").asi8,
        "publication": publications.as_unit("ns").asi8,
        "days": np.array(["2025-12-01", "2026-01-01", "2026-09-30", "2026-10-01"]),
    }
    files = {name: _descriptor(source, name, value) for name, value in data.items()}
    train = _descriptor(source, "train", np.array([0]))
    group = {
        "market": "IDA1",
        "feature_names": ["core", "external"],
        "files": files,
        "cohorts": {"stage_main": {"indices": {"train": train}}},
    }
    winner = {
        "arm": {"columns": [0], "cohort": "stage_main", "availability_class": "assumed_previous_day"},
        "candidate": {"family": "ridge", "parameters": {}, "seed": 1},
    }
    called = {}

    def fake_fit(family, candidate, training_x, training_y, prediction_x):
        called["training_x"] = training_x.tolist()
        called["training_y"] = training_y.tolist()
        return object(), np.full(len(prediction_x), 10.0), []

    monkeypatch.setattr(staged, "_fit", fake_fit)
    monkeypatch.setattr(staged.joblib, "dump", lambda model, path, compress: path.write_bytes(b"fake"))
    monkeypatch.setattr(staged, "plot_predictions", lambda *args: [])
    settings = {"prediction_start": "2026-01-01", "prediction_end_exclusive": "2027-01-01",
                "timezone": "Europe/Rome"}
    split = {"train_start": "2025-10-01", "train_end": "2026-10-01", "validation_start": "2026-10-01"}
    auction = {"decision_day_offset": -1, "decision_time": "14:45"}
    result = staged.refit_winner(source, output, "ida1", group, winner, "main", settings, split, auction)
    frame = pd.read_parquet(result["prediction_path"])
    assert called["training_x"] == [[1.0], [2.0]]
    assert called["training_y"] == [20.0, 20.0]
    assert frame.forecast_eur_per_mwh.tolist() == [90.0, 95.0, 100.0]
    assert frame.used_for_training.tolist() == [True, False, False]
    assert result["prediction_rows_used_for_training"] == 1
    assert result["selection_cohort_train_rows"] == 1


def test_winner_validation_rejects_target_spread_and_retrospective_main(tmp_path, monkeypatch):
    """A selected feature contract cannot smuggle in outcome or realized values."""
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps({"source": "frozen"}))
    feature_blocks = {"retro": {"availability": "retrospective_realized"}}
    group = {"feature_names": ["core", "it_ch_da_spread_same_day"],
             "block_columns": {"retro": [1]}, "cohorts": {"stage_main": {}}}
    prepared = {"scientific_config_sha256": "config", "implementation_sha256": {}, "software": {},
                "scientific_settings": {"feature_blocks": feature_blocks}, "groups": {"da": group}}
    winner = {"arm": {"group": "da", "cohort": "stage_main", "columns": [0, 1]}}
    winners = {"status": "complete", "prepared_sha256": digest(prepared_path),
               "scientific_config_sha256": "config", "groups": {"da": {"main": winner}}}
    monkeypatch.setattr(staged, "_implementation", lambda: {})
    monkeypatch.setattr(staged, "_software", lambda settings: {})
    with pytest.raises(ValueError, match="forbidden same-day price spread"):
        staged.validate_winners(tmp_path, prepared, winners)
    group["feature_names"][1] = "ch_generation_same_day"
    with pytest.raises(ValueError, match="main winner contains retrospective"):
        staged.validate_winners(tmp_path, prepared, winners)
    winner["arm"]["columns"] = [0]
    staged.validate_winners(tmp_path, prepared, winners)


def test_retrospective_matched_baseline_uses_identical_train_and_prediction_rows(tmp_path, monkeypatch):
    """The added realized block cannot change the sample used in its paired comparison."""
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    starts = pd.DatetimeIndex(["2025-12-01T12:00:00Z", "2026-01-01T12:00:00Z", "2026-02-01T12:00:00Z"])
    data = {"x": np.array([[1.0, 5.0], [2.0, 6.0], [3.0, np.nan]]),
            "y": np.array([70.0, 100.0, 110.0]), "anchor": np.array([50.0, 80.0, 90.0]),
            "reference": np.array([50.0, 80.0, 90.0]), "starts": starts.as_unit("ns").asi8,
            "decisions": starts.as_unit("ns").asi8, "publication": starts.as_unit("ns").asi8,
            "days": np.array(["2025-12-01", "2026-01-01", "2026-02-01"])}
    files = {name: _descriptor(source, name, value) for name, value in data.items()}
    train = _descriptor(source, "retro_train", np.array([0, 1]))
    group = {"market": "IDA1", "feature_names": ["core", "ch_generation_same_day"], "files": files,
             "cohorts": {"stage_retrospective": {"indices": {"train": train}}}}
    candidate = {"family": "ridge", "parameters": {}, "seed": 1}
    baseline = {"arm": {"columns": [0], "cohort": "stage_retrospective", "availability_class": "assumed_previous_day"},
                "candidate": candidate}
    retro = {"arm": {"columns": [0, 1], "cohort": "stage_retrospective", "availability_class": "retrospective_realized"},
             "candidate": candidate}
    monkeypatch.setattr(staged, "_fit", lambda family, spec, train_x, train_y, pred_x:
                        (object(), np.zeros(len(pred_x)), []))
    monkeypatch.setattr(staged.joblib, "dump", lambda model, path, compress: path.write_bytes(b"fake"))
    monkeypatch.setattr(staged, "plot_predictions", lambda *args: [])
    settings = {"prediction_start": "2026-01-01", "prediction_end_exclusive": "2027-01-01",
                "timezone": "Europe/Rome"}
    split = {"train_start": "2025-10-01", "train_end": "2026-10-01", "validation_start": "2026-10-01"}
    auction = {"decision_day_offset": -1, "decision_time": "14:45"}
    main_result = staged.refit_winner(source, output, "ida1", group, baseline, "main_matched_retrospective",
                                      settings, split, auction, eligibility_columns=[0, 1])
    retro_result = staged.refit_winner(source, output, "ida1", group, retro, "retrospective",
                                       settings, split, auction)
    assert main_result["train_index_sha256"] == retro_result["train_index_sha256"]
    assert main_result["prediction_index_sha256"] == retro_result["prediction_index_sha256"]
    assert main_result["prediction_rows"] == retro_result["prediction_rows"] == 1

"""Verify retrospective proxy selection and refitting without model experiments."""

import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.studies.price_feature_data import digest
from scripts import refit_full_year_price_proxies as proxies


def test_coverage_counts_native_products_on_daylight_saving_changes():
    """Keep the 23-hour and 25-hour days distinct in the missing-price report."""
    settings = {"prediction_start": "2026-03-29", "prediction_end_exclusive": "2026-10-26"}
    coverage = proxies.coverage_table(np.array([], dtype="U10"), np.array([], dtype="U10"),
                                      "DA", settings, "Europe/Rome").set_index("local_day")
    assert coverage.loc["2026-03-29", "expected_products"] == 92
    assert coverage.loc["2026-10-25", "expected_products"] == 100


def test_selection_requires_complete_tasks_and_uses_matched_cohort(tmp_path):
    """Select the lowest inner WAPE, and reject a missing search checkpoint."""
    trials = tmp_path / "trials"
    trials.mkdir()
    arm = {"group": "da", "cohort": "full_history", "id": "da_full", "columns": [0]}
    task = {"id": "ridge", "arm": arm, "signature": "abc", "candidates": [{"trial": 0}, {"trial": 1}]}
    prepared = {"groups": {"da": {}}, "tasks": [task]}
    checkpoint = {"signature": "abc", "results": [
        {"trial": 0, "status": "complete", "family": "ridge", "metrics": {"wape_percent": 10, "mae_eur_per_mwh": 5}},
        {"trial": 1, "status": "complete", "family": "ridge", "metrics": {"wape_percent": 8, "mae_eur_per_mwh": 6}}]}
    path = trials / "ridge.json"
    path.write_text(json.dumps(checkpoint))
    selected, failures = proxies.select_candidates(tmp_path, prepared, "full_history")
    assert selected["da"]["candidate"]["trial"] == 1
    assert failures == []
    checkpoint["results"].pop()
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="unfinished candidates"):
        proxies.select_candidates(tmp_path, prepared, "full_history")


def test_refit_uses_annual_training_indices_and_marks_in_sample_rows(tmp_path, monkeypatch):
    """Fit only the annual cohort but predict all rows with selected inputs."""
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    starts = pd.DatetimeIndex(["2025-12-01T12:00:00Z", "2026-01-01T12:00:00Z", "2026-10-01T12:00:00Z"])
    raw = {"x": np.array([[1., 99.], [2., np.nan], [3., 99.]]), "y": np.array([70., 100., 110.]),
        "anchor": np.array([50., 80., 90.]), "reference": np.array([50., 80., 90.]),
        "starts": starts.as_unit("ns").asi8, "decisions": starts.as_unit("ns").asi8,
        "days": np.array(["2025-12-01", "2026-01-01", "2026-10-01"]), "train": np.array([0, 1])}
    descriptors = {}
    for name, array in raw.items():
        path = source / f"{name}.npy"
        np.save(path, array)
        descriptors[name] = {"path": path.name, "sha256": digest(path)}
    group = {"market": "IDA1", "feature_names": ["used", "unused"], "files": descriptors,
        "cohorts": {"full_history": {"indices": {"train": descriptors["train"]}}}}
    selected = {"arm": {"columns": [0], "availability_class": "assumed_previous_day"},
                "candidate": {"family": "ridge"}}
    captured = {}

    def fake_fit(family, candidate, x, y, test_x):
        captured["target"] = y.tolist()
        return object(), np.full(len(test_x), 10.), []

    monkeypatch.setattr(proxies, "_fit", fake_fit)
    monkeypatch.setattr(proxies.joblib, "dump", lambda model, path, compress: path.write_bytes(b"fake"))
    settings = {"cohort": "full_history", "prediction_start": "2026-01-01", "prediction_end_exclusive": "2027-01-01",
                "timezone": "Europe/Rome"}
    result = proxies.refit_group(source, output, "ida1", group, selected, settings)
    frame = pd.read_parquet(result["prediction_path"])
    assert captured["target"] == [20., 20.]
    assert frame.forecast_eur_per_mwh.tolist() == [90., 100.]
    assert frame.used_for_training.tolist() == [True, False]

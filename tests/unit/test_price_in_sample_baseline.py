"""Check the explicitly retrospective baseline without running a model experiment."""

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.price_feature_data import digest
from scripts import run_price_in_sample_baseline as baseline


def test_retrospective_fit_includes_prediction_period_and_restores_anchor(tmp_path, monkeypatch):
    """Use an inert fake estimator to check dates, prices and output provenance."""
    root = tmp_path / "source"
    root.mkdir()
    starts = pd.DatetimeIndex(["2025-12-01T12:00:00Z", "2026-01-01T12:00:00Z"])
    raw = {"x": np.array([[1.0], [2.0]]), "y": np.array([70.0, 100.0]),
           "anchor": np.array([50.0, 80.0]), "reference": np.array([50.0, 80.0]),
           "starts": starts.as_unit("ns").asi8, "decisions": starts.as_unit("ns").asi8,
           "days": np.array(["2025-12-01", "2026-01-01"])}
    files = {}
    for name, array in raw.items():
        path = root / f"{name}.npy"
        np.save(path, array)
        files[name] = {"path": path.name, "sha256": digest(path)}

    class FakeModel:
        """Record fit inputs and return a fixed spread without model training."""
        def fit(self, features, target):
            self.target = target.copy()
            return self

        def predict(self, features):
            return np.full(len(features), 10.0)

    model = FakeModel()
    monkeypatch.setattr(baseline, "make_model", lambda *args: model)
    monkeypatch.setattr(baseline.joblib, "dump", lambda instance, path, compress: path.write_bytes(b"fake"))
    monkeypatch.setattr(baseline, "build_holiday_features", lambda *args: (np.zeros((2, 1)), ["holiday"]))
    group = {"market": "IDA1", "base_width": 1, "feature_names": ["base"], "files": files}
    settings = {"train_start": "2025-01-01", "train_end_exclusive": "2027-01-01",
        "prediction_start": "2026-01-01", "prediction_end_exclusive": "2027-01-01",
        "model_family": "ridge", "model_parameters": {"alpha": 1000.0}, "seed": 17}
    output = tmp_path / "output"
    result = baseline.fit_group(root, output, "ida1", group, settings, {}, {})
    prediction = pd.read_parquet(output / "ida1.parquet")
    assert model.target.tolist() == [20.0, 20.0]
    assert result["training_rows"] == 2
    assert result["training_last_day"] == "2026-01-01"
    assert prediction.forecast_eur_per_mwh.tolist() == [90.0]
    assert prediction.provenance.iloc[0].startswith("in_sample")

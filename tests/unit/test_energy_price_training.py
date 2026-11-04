"""Offline training orchestration and projected feature-source semantics."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.forecasting.energy_price_features import load_catalog_features


ROOT = Path(__file__).resolve().parents[2]


def training_script():
    spec = importlib.util.spec_from_file_location("energy_price_training_script", ROOT / "scripts/fit_energy_price_forecast.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_cli_is_a_no_data_read_training_plan(monkeypatch, capsys):
    module = training_script()
    monkeypatch.setattr(module, "_load_inputs", lambda *args: pytest.fail("Plan-only must not open DuckDB"))
    assert module.main(["--config", str(ROOT / "configs/research/price_forecast_training.yaml")]) == 0
    assert '"fit_count": 30' in capsys.readouterr().out


def write_snapshot(path, frame, provider="entsoe", dataset="load_forecast"):
    store = DuckDBCache(path, threads=1)
    identity = {"provider": provider, "dataset": dataset, "zone": "NORD",
                "start": "2026-01-01T00:00:00Z", "end": "2026-01-04T00:00:00Z", "options": {}}
    frame = frame.assign(provider=provider, dataset=dataset, zone="NORD")
    return store.save(identity, b"synthetic fixture", frame, {"retrieved_at": "2026-01-05T00:00:00Z"})


def test_projected_interval_features_retain_hashes_and_explicit_assumptions(tmp_path):
    path = tmp_path / "market.duckdb"
    frame = pd.DataFrame({"product_start": pd.to_datetime(["2026-01-02T00:00:00Z"]),
        "product_end": pd.to_datetime(["2026-01-02T01:00:00Z"]), "value": [100.0], "unit": ["MW"]})
    manifest = write_snapshot(path, frame)
    selections = [{"name": "load", "provider": "entsoe", "dataset": "load_forecast", "zone": "NORD",
                   "expected_unit": "MW", "publication_local_time": "00:00", "publication_day_offset": -1}]
    values, evidence = load_catalog_features(path, selections, start="2026-01-02T00:00:00Z", end="2026-01-02T01:00:00Z")
    assert len(values) == 4
    assert (values.value == 100).all()
    assert (values.available_at == pd.Timestamp("2025-12-31T23:00:00Z")).all()
    assert evidence["features"][0]["snapshot_sources"][0]["snapshot_id"] == manifest["snapshot_id"]
    assert evidence["features"][0]["availability"].startswith("assumed")


def test_weather_interpolation_never_crosses_missing_bracket(tmp_path):
    path = tmp_path / "market.duckdb"
    times = pd.date_range("2026-01-02T00:00:00Z", periods=4, freq="h")
    frame = pd.DataFrame({"product_start": pd.to_datetime([None] * 4, utc=True),
        "product_end": pd.to_datetime([None] * 4, utc=True), "valid_time": times,
        "cycle_time": pd.Timestamp("2026-01-01T00:00:00Z"),
        "scheduled_available_at": pd.Timestamp("2026-01-01T06:00:00Z"),
        "value": [0.0, 4.0, np.nan, 12.0], "unit": "C", "temporal_semantics": "instantaneous",
        "source_kind": "forecast_archive", "location_id": "milan", "variable": "temperature_2m"})
    write_snapshot(path, frame, "open_meteo", "weather_forecast")
    selections = [{"name": "milan_temperature", "provider": "open_meteo", "dataset": "weather_forecast",
        "zone": "NORD", "kind": "weather", "selectors": {"location_id": "milan", "variable": "temperature_2m"},
        "availability_column": "scheduled_available_at", "availability_kind": "inferred_cycle_latency"}]
    values, evidence = load_catalog_features(path, selections, start="2026-01-02T00:00:00Z", end="2026-01-02T04:00:00Z")
    np.testing.assert_array_equal(values.value, [0, 1, 2, 3])
    assert values.product_start.max() == pd.Timestamp("2026-01-02T00:45:00Z")


def test_incompatible_feature_units_fail(tmp_path):
    path = tmp_path / "market.duckdb"
    frame = pd.DataFrame({"product_start": pd.to_datetime(["2026-01-02T00:00:00Z"]),
        "product_end": pd.to_datetime(["2026-01-02T01:00:00Z"]), "value": [100.0], "unit": ["MW"]})
    write_snapshot(path, frame)
    selection = [{"name": "load", "provider": "entsoe", "dataset": "load_forecast", "zone": "NORD",
        "expected_unit": "MWh", "publication_lag_hours": 1}]
    with pytest.raises(ValueError, match="declared unit"):
        load_catalog_features(path, selection, start="2026-01-02T00:00:00Z", end="2026-01-02T01:00:00Z")


def test_summary_compares_only_common_dates(tmp_path):
    module = training_script()
    manifest = {"plan": {"models": [{"id": "a"}, {"id": "b"}], "folds": {"fold": {}}, "targets": {"DA": {}}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for model, days, errors in (("a", ["2026-01-01", "2026-01-02"], [100.0, 2.0]),
                                ("b", ["2026-01-02", "2026-01-03"], [3.0, 200.0])):
        path = tmp_path / "fold" / model / "DA"
        path.mkdir(parents=True)
        pd.DataFrame({"day": days, "point_mae_eur_per_mwh": errors,
                      "crps_eur_per_mwh": errors}).to_csv(path / "daily_scores.csv", index=False)
    scores = module.write_training_report(tmp_path)
    assert [row["paired_days"] for row in scores] == [1, 1]
    assert [row["point_mae_eur_per_mwh"] for row in scores] == [2, 3]
    assert "partial-year" in (tmp_path / "summary.md").read_text()


def test_spawned_price_workers_fit_isolated_artifacts_without_database_reads(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("scripts.fit_energy_price_forecast")
    starts = pd.date_range("2025-12-01T00:00:00Z", "2026-01-04T00:00:00Z", freq="h", inclusive="left")
    history = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(hours=1),
        "available_at": starts + pd.Timedelta(hours=1),
        "price_eur_per_mwh": 40 + np.sin(starts.hour.to_numpy())})
    settings = {"database": "must_not_open.duckdb", "output_root": str(tmp_path), "run_id": "spawned",
        "timezone": "UTC", "interval_minutes": 60, "history_end_exclusive": "2026-01-04T00:00:00Z",
        "native_threads": 1, "workers": 2, "scenario_count": 3, "plot_days": 0,
        "fitting": {"lag_hours": [24], "min_training_rows": 24, "calibration_days": 1,
                    "interval_minutes": 60, "timezone": "UTC", "decision_local_time": "00:00", "decision_day_offset": -1},
        "folds": {"fold": {"fit_cutoff": "2026-01-01T00:00:00Z", "evaluation_start": "2026-01-02", "evaluation_end_exclusive": "2026-01-03"}},
        "models": [{"id": "ridge", "method": "ridge", "feature_set": "none"},
                   {"id": "persistence", "method": "seasonal_persistence", "feature_set": "none"}],
        "feature_sets": {"none": []},
        "targets": {"DA": {"history_start": "2025-12-01T00:00:00Z", "source": {}}}}
    monkeypatch.setattr(module, "_load_inputs", lambda *args: ({"DA": history}, {"DA": {}}, None, None))
    result = module.train_from_config({"energy_price_training": settings})
    assert result["status"] == "complete"
    assert len(result["models"]) == 2
    assert all(row["evaluated_days"] == 1 for row in result["models"])
    assert (tmp_path / "spawned" / "paired_scores.csv").exists()

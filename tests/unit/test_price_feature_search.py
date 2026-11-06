"""Gate-aware features, holiday flags and resumable search software behavior."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data.duckdb_store import DuckDBCache
from bess_pv_trading.forecasting.studies.price_feature_data import _base_matrix, _lookup, _profiles, interval_values, italian_holidays
from bess_pv_trading.forecasting.studies.price_feature_models import point_metrics, sample_parameters, stable_seed
from bess_pv_trading.forecasting.studies.price_feature_search import _fit, run_search


ROOT = Path(__file__).resolve().parents[2]


def test_native_interval_means_preserve_source_gaps_and_gate_clocks():
    """An hourly mean is constant within its interval and a later publication remains hidden."""
    starts = pd.DatetimeIndex(pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T02:00Z"]))
    frame = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(hours=1),
                          "price_eur_per_mwh": [100.0, -10.0], "available_at": starts - pd.Timedelta(hours=1)})
    grid = pd.date_range(starts[0], periods=12, freq="15min")
    values = interval_values(frame, grid, frame.price_eur_per_mwh.to_numpy())
    assert np.array_equal(values[:4], np.full(4, 100))
    assert np.isnan(values[4:8]).all()
    assert np.array_equal(values[8:], np.full(4, -10))
    hidden = _lookup(frame, starts, starts - pd.Timedelta(hours=2))
    visible = _lookup(frame, starts, starts - pd.Timedelta(hours=1))
    assert np.isnan(hidden).all()
    assert visible.tolist() == [100, -10]


def test_italian_holidays_include_april_25_may_1_and_easter():
    """The binary calendar feature includes the requested dates and movable holidays."""
    settings = load_config(ROOT / "configs/research/price_feature_selection.yaml")["price_feature_search"]
    dates = pd.DatetimeIndex(pd.to_datetime(["2025-04-20T12:00Z", "2025-04-21T12:00Z", "2025-04-22T12:00Z",
                                           "2025-04-25T12:00Z", "2025-05-01T12:00Z", "2025-05-02T12:00Z"]))
    assert italian_holidays(dates, settings).tolist() == [1, 1, 0, 1, 1, 0]


def test_sampling_and_scores_are_stable_with_negative_prices():
    """Worker order cannot change candidates, and zero prices do not break WAPE."""
    space = {"alpha": {"log_uniform": [0.1, 100]}, "count": {"choices": [10, 20]}}
    seed = stable_seed(17, "DA", "ridge")
    assert sample_parameters(space, seed, 5) == sample_parameters(space, seed, 5)
    scores = point_metrics([-10, 0, 10], [-9, 1, 9], [0, 0, 0])
    assert scores["wape_percent"] == 15
    assert scores["mae_eur_per_mwh"] == 1
    assert scores["relative_mae_to_reference"] == 0.15


def test_daily_curves_copy_arrays_before_applying_publication_masks():
    """Pandas copy-on-write buffers must not prevent causal masking of a daily profile."""
    starts = pd.date_range("2026-01-01", periods=24, freq="1h", tz="Europe/Rome").tz_convert("UTC").as_unit("ns")
    future = starts + pd.Timedelta(days=1)
    history = pd.DataFrame({"product_start": starts, "price_eur_per_mwh": np.arange(24, dtype=float),
                            "available_at": pd.Timestamp("2026-01-01T13:00Z")})
    cutoffs = pd.DatetimeIndex([pd.Timestamp("2026-01-01T12:00Z")] * 24)
    hidden, names = _profiles(history, future, cutoffs, [1], 60, list(range(24)), "Europe/Rome")
    assert hidden.shape == (24, 24)
    assert np.isnan(hidden).all()
    history["available_at"] = pd.Timestamp("2025-12-31T12:00Z")
    visible, _ = _profiles(history, future, cutoffs, [1], 60, list(range(24)), "Europe/Rome")
    assert visible[0].tolist() == list(range(24))
    assert len(names) == 24


def test_ida3_daily_lags_keep_both_ends_of_the_half_day_window():
    """Non-day lags must not remove the first/last IDA3 delivery hour permanently."""
    settings = load_config(ROOT / "configs/research/price_feature_selection.yaml")["price_feature_search"]
    settings["daily_curve_lag_days"] = []
    settings["auctions"]["IDA3"].update(earlier_markets=[], past_other_markets=[])
    rows = []
    for day in pd.date_range("2026-01-01", "2026-01-20"):
        first = pd.Timestamp(f"{day.date()} 12:00", tz="Europe/Rome")
        starts = pd.date_range(first, first + pd.Timedelta(hours=12), freq="15min", inclusive="left").tz_convert("UTC")
        publication = pd.Timestamp(f"{day.date()} 10:30", tz="Europe/Rome").tz_convert("UTC")
        rows.append(pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(minutes=15),
                                  "available_at": publication, "price_eur_per_mwh": 100.0}))
    history = pd.concat(rows, ignore_index=True)
    target = pd.DatetimeIndex(rows[-1].product_start).as_unit("ns")
    matrix, *_ = _base_matrix(history, {"IDA3": history}, target, settings, {"market": "IDA3", "native_minutes": 15})
    assert len(matrix) == 48
    assert np.isfinite(matrix).all()


@pytest.mark.parametrize("family", ["ridge", "elastic_net", "random_forest", "extra_trees", "hist_gradient_boosting",
                                    "lightgbm", "catboost", "xgboost", "mlp"])
def test_configured_model_family_cpu_api(family):
    """Exercise installed estimator APIs on a small software fixture, including negative targets."""
    settings = load_config(ROOT / "configs/research/price_feature_selection.yaml")["price_feature_search"]
    options = sample_parameters(settings["parameter_spaces"][family], 17, 1)[0]
    for parameter in ["n_estimators", "iterations", "max_iter"]:
        if parameter in options:
            options[parameter] = 2
    x = np.arange(256, dtype=float).reshape(64, 4) / 100
    y = 2 * x[:, 0] - 3
    _, predictions, _ = _fit(family, {"parameters": options, "seed": 17}, x, y, x[:3])
    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()


def test_search_freezes_native_labels_and_resumes_completed_fits(tmp_path, monkeypatch):
    """A tiny synthetic software fixture exercises preparation, refits and exact checkpoint reuse."""
    for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        monkeypatch.setenv(variable, "1")
    database = tmp_path / "market.duckdb"
    store = DuckDBCache(database, threads=1)
    starts = pd.date_range("2024-01-01", "2024-02-21", freq="1h", inclusive="left", tz="Europe/Rome").tz_convert("UTC")
    for provider, dataset, values in [("gme", "day_ahead_prices", 100 + np.arange(len(starts)) * 0.01),
                                       ("entsoe", "generation_forecast", 10 + np.arange(len(starts)) % 24)]:
        frame = pd.DataFrame({"provider": provider, "dataset": dataset, "zone": "NORD", "product_start": starts,
                              "product_end": starts + pd.Timedelta(hours=1), "value": values})
        identity = {"provider": provider, "dataset": dataset, "zone": "NORD", "start": "2024-01-01", "end": "2024-02-21", "options": {}}
        store.save(identity, b"deterministic software fixture", frame, {"source_kind": "test_fixture"})
    settings = load_config(ROOT / "configs/research/price_feature_selection.yaml")["price_feature_search"]
    settings.update(workers=1, trials_per_family=1, lag_hours=[24], daily_curve_lag_days=[], max_lag_age_hours=48,
                    inner_min_days=2, minimum_training_rows=48, minimum_inner_rows=12, minimum_evaluation_rows=12,
                    database_path=str(database), plot_cache_directory=str(tmp_path / "matplotlib_cache"))
    settings["auctions"] = {"DA": settings["auctions"]["DA"]}
    settings["auctions"]["DA"]["past_other_markets"] = []
    settings["product_groups"] = {"da_hourly": {"market": "DA", "native_minutes": 60, "train_start": "2024-01-01",
        "train_end": "2024-02-01", "validation_start": "2024-02-01", "validation_end": "2024-02-11",
        "test_start": "2024-02-11", "test_end": "2024-02-21"}}
    settings["feature_blocks"] = {"forecast": {"availability": "assumed_previous_day", "features": [
        {"name": "nord_generation", "source": "signals", "column": "value"}]}}
    settings["cohorts"] = {"matched": ["forecast"]}
    settings["parameter_spaces"] = {"ridge": {"alpha": 1.0}}
    columns = ["provider", "dataset", "zone", "product_start", "product_end", "value"]
    inputs = {"database_path": str(database), "database_threads": 1, "output_directory": "unused",
              "start": "2024-01-01", "end_exclusive": "2024-02-21", "timezone": "Europe/Rome", "sources": [
                  {"name": "nord_auctions", "provider": "gme", "datasets": ["day_ahead_prices"], "zone": "NORD", "columns": columns},
                  {"name": "signals", "provider": "entsoe", "datasets": ["generation_forecast"], "zone": "NORD", "columns": columns}]}
    output = tmp_path / "search"
    first = run_search(settings, inputs, output)
    assert first["status"] == "complete"
    checkpoints = {path: path.stat().st_mtime_ns for directory in ["trials", "refits"] for path in (output / directory).glob("*.json")}
    second = run_search(settings, inputs, output)
    assert second["status"] == "complete"
    assert checkpoints == {path: path.stat().st_mtime_ns for path in checkpoints}
    prepared = __import__("json").loads((output / "prepared.json").read_text())
    assert prepared["farm_files_required"] is False
    assert prepared["groups"]["da_hourly"]["rows"] == len(starts)
    assert len(prepared["arms"]) == 2
    assert (output / "holdout_scores.csv").exists()
    assert len(list((output / "models").glob("*.joblib"))) >= 1
    # A changed scientific configuration cannot silently reuse an old study's completed fits.
    settings["seed"] += 1
    with pytest.raises(ValueError, match="different settings"):
        run_search(settings, inputs, output, prepare_only=True)
    # A configured product cannot disappear from a supposedly complete multi-product search.
    settings["minimum_evaluation_rows"] = 48
    settings["product_groups"]["uncovered_product"] = dict(settings["product_groups"]["da_hourly"],
        validation_start="2024-02-19", validation_end="2024-02-20", test_start="2024-02-20", test_end="2024-02-21")
    with pytest.raises(ValueError, match="uncovered_product has no eligible cohort"):
        run_search(settings, inputs, tmp_path / "incomplete", prepare_only=True)

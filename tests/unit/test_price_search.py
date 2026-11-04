"""Exercise search chronology, paired coverage and relative-error definitions."""

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.energy_prices import PriceForecastSettings
from bess_pv_trading.forecasting.price_search import complete_shared_days, point_metrics, prepare_search_matrix


def test_search_uses_only_labels_released_before_cutoff_and_shared_complete_days():
    starts = pd.date_range("2024-12-01T00:00Z", "2025-01-04T00:00Z", freq="h", inclusive="left")
    history = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(hours=1),
        "available_at": starts + pd.Timedelta(hours=1), "price_eur_per_mwh": np.arange(len(starts), dtype=float)})
    settings = PriceForecastSettings(lag_hours=(24,), interval_minutes=60, timezone="UTC",
                                     decision_local_time="00:00", decision_day_offset=-1)
    cutoff = pd.Timestamp("2024-12-31T00:00Z")
    result = prepare_search_matrix(history, settings, None, cutoff=cutoff,
                                   validation_start="2025-01-01", validation_end="2025-01-04")
    assert (history.loc[result["training"], "product_end"] <= cutoff).all()
    assert complete_shared_days([result], start="2025-01-01", end="2025-01-04", settings=settings) == [
        "2025-01-01", "2025-01-02", "2025-01-03"]
    altered = dict(result)
    altered["validation"] = result["validation"].copy()
    altered["validation"][np.flatnonzero(result["days"] == "2025-01-02")[0]] = False
    assert complete_shared_days([result, altered], start="2025-01-01", end="2025-01-04", settings=settings) == [
        "2025-01-01", "2025-01-03"]


def test_point_metrics_preserve_zero_negative_prices_and_naive_scale():
    metrics = point_metrics([0, -10], [1, -7], [0, -5])
    assert metrics["wape_percent"] == 40
    assert metrics["relative_mae_to_naive"] == pytest.approx(0.8)
    assert point_metrics([0], [1], [0])["wape_percent"] is None


def test_search_rejects_cutoff_after_first_decision():
    starts = pd.date_range("2024-12-01", "2025-01-02", freq="h", tz="UTC")
    history = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(hours=1),
        "available_at": starts + pd.Timedelta(hours=1), "price_eur_per_mwh": 1.0})
    settings = PriceForecastSettings(lag_hours=(24,), interval_minutes=60, timezone="UTC",
                                     decision_local_time="00:00", decision_day_offset=-1)
    with pytest.raises(ValueError, match="first validation decision"):
        prepare_search_matrix(history, settings, None, cutoff="2024-12-31T12:00Z",
                              validation_start="2025-01-01", validation_end="2025-01-02")


def test_worker_saves_predictions_and_scores(tmp_path):
    """The isolated worker trains and scores only the provided split arrays."""
    from scripts.search_energy_price_models import _fit_candidate

    x = np.arange(60, dtype=float).reshape(30, 2)
    dataset = tmp_path / "data.npz"
    output = tmp_path / "predictions.csv"
    np.savez(dataset, train_x=x[:20], train_y=x[:20, 0], validation_x=x[20:],
             validation_y=x[20:, 0], reference=x[20:, 0] + 1,
             starts=np.array([f"2025-01-01T{hour:02}:00:00Z" for hour in range(10)]))
    result = _fit_candidate({"dataset": str(dataset), "output": str(output), "threads": 1,
        "seed": 4, "group": "hourly", "feature_set": "price_only",
        "candidate": {"id": "ridge", "family": "ridge", "parameters": {"alpha": 1}}})
    assert result["training_rows"] == 20
    assert result["observations"] == 10
    assert len(pd.read_csv(output)) == 10

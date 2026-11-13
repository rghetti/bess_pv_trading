"""Exercise chronology, DA bootstrap and daylight-saving curve projection."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.studies.rolling_price_models import (
    RollingModelEngine,
    _da_context,
    _project,
)


def _history():
    """Build synthetic native hourly DA and quarter-hour intraday observations."""
    examples = []
    for day in pd.date_range("2025-08-01", "2025-10-15"):
        day_string = day.strftime("%Y-%m-%d")
        for market, gate in (("DA", "DA"), ("IDA1", "DA"), ("IDA1", "IDA1")):
            minutes = 60 if market == "DA" and day_string < "2025-10-01" else 15
            starts = pd.date_range(day.tz_localize("Europe/Rome"), periods=24 * 60 // minutes,
                                   freq=f"{minutes}min").tz_convert("UTC")
            clock = (day - timedelta(days=1)).tz_localize("Europe/Rome")
            decision = (clock + pd.Timedelta(hours=11 if gate == "DA" else 14)).tz_convert("UTC")
            publication = (clock + pd.Timedelta(hours=12 if market == "DA" else 15)).tz_convert("UTC")
            values = np.full(len(starts), 100.0 + day.day / 10 + (market != "DA"))
            examples.append({"delivery_day": day_string, "market": market, "gate": gate,
                "decision_at": decision.isoformat(), "target_publication_at": publication.isoformat(),
                "product_starts_utc": starts.strftime("%Y-%m-%dT%H:%M:%S%z").tolist(),
                "target_values": values.tolist(), "price_reference": np.full(len(starts), 100.0).tolist(),
                "feature_blocks": {"core": [float(day.dayofweek), float(day.day)]},
                "metadata": {"native_minutes": minutes}})
    return examples


def _settings():
    """Use short synthetic splits to exercise the identical production logic."""
    return {"window_days": 60, "variant": "core", "feature_blocks": {"core": ["core"]},
        "family": "ridge", "parameters": {"ridge": [100.0]}, "ida_target": "spread",
        "minimum_fit_days": 4, "validation_days": 2, "minimum_native_da_days": 6}


def test_hourly_bootstrap_and_out_of_sample_da_conditioning():
    examples = _history()
    engine = RollingModelEngine(examples, _settings())
    first = next(e for e in examples if e["market"] == "DA" and e["delivery_day"] == "2025-10-01")
    models = engine.fit_delivery_day("2025-10-01", first["decision_at"])
    assert models.models[("DA", "DA")].resolution == 60
    assert models.models[("IDA1", "DA")].metadata["da_conditioning"] == "rolling_out_of_sample"
    da = engine.predict(models, {k: v for k, v in first.items() if k != "target_values"})
    target = next(e for e in examples if e["market"] == "IDA1" and e["gate"] == "DA"
                  and e["delivery_day"] == "2025-10-01")
    with pytest.raises(ValueError, match="predicted DA prices"):
        engine.predict(models, target)
    prediction = engine.predict(models, {k: v for k, v in target.items() if k != "target_values"}, da["prediction"])
    assert len(prediction["prediction"]) == 96
    assert max(models.models[("DA", "DA")].train_days) < "2025-10-01"
    later = next(e for e in examples if e["market"] == "DA" and e["delivery_day"] == "2025-10-08")
    later_models = engine.fit_delivery_day("2025-10-08", later["decision_at"])
    assert later_models.models[("DA", "DA")].resolution == 15


def test_dst_projection_preserves_native_products_without_inventing_labels():
    start = pd.Timestamp("2026-10-25", tz="Europe/Rome")
    end = pd.Timestamp("2026-10-26", tz="Europe/Rome")
    starts = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")
    starts = [value.isoformat() for value in starts]
    assert len(starts) == 100
    predicted = _project(np.arange(96), starts, "IDA1", 15, "Europe/Rome")
    assert predicted[8:12].tolist() == predicted[12:16].tolist()
    context = _da_context({"market": "IDA1", "product_starts_utc": starts}, predicted, "Europe/Rome")
    assert len(context) == 96
    assert np.array_equal(context, np.arange(96))


def test_weather_variant_does_not_impute_an_unavailable_archive():
    examples = _history()
    settings = _settings()
    settings["variant"] = "weather"
    settings["feature_blocks"]["weather"] = ["core", "weather"]
    for example in examples:
        example["feature_blocks"]["weather"] = [float("nan")]
        example["metadata"]["weather_usable"] = False
    engine = RollingModelEngine(examples, settings)
    current = next(e for e in examples if e["market"] == "DA" and e["delivery_day"] == "2025-10-01")
    models = engine.fit_delivery_day("2025-10-01", current["decision_at"])
    assert ("DA", "DA") not in models.models
    assert "need" in models.unavailable["DA:DA"]


def test_future_target_changes_do_not_change_current_forecast():
    """A future realized price must not affect today's selected fit or forecast."""
    examples = _history()
    current = next(e for e in examples if e["market"] == "DA" and e["delivery_day"] == "2025-10-01")
    before = RollingModelEngine(examples, _settings())
    fitted_before = before.fit_delivery_day("2025-10-01", current["decision_at"])
    safe = {key: value for key, value in current.items() if key != "target_values"}
    prediction_before = before.predict(fitted_before, safe)["prediction"]
    for example in examples:
        if example["delivery_day"] >= "2025-10-01":
            example["target_values"] = [1e9] * len(example["target_values"])
    after = RollingModelEngine(examples, _settings())
    fitted_after = after.fit_delivery_day("2025-10-01", current["decision_at"])
    prediction_after = after.predict(fitted_after, safe)["prediction"]
    np.testing.assert_allclose(prediction_before, prediction_after)

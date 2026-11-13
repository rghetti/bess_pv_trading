"""Check rolling-provider chronology, fallback and durable daily fitting."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.domain import DeliveryGrid, MarketState
from bess_pv_trading.scenarios import synthetic_price_tree
from bess_pv_trading.simulation.continuous import ContinuousReplay
from bess_pv_trading.simulation.rolling_price_provider import (
    PublishedAuctionCurve, RollingPriceProvider, forecast_auction_book, validate_replay_price_schedule,
)
from bess_pv_trading.forecasting.studies.rolling_price_data import pad_curve


class FakeEngine:
    """Small deterministic engine with inspectable fit and prediction calls."""

    def __init__(self):
        self.fits = 0
        self.da_prediction = None

    def fit_delivery_day(self, day, cutoff):
        self.fits += 1
        return {"day": day, "cutoff": cutoff.isoformat()}

    def predict(self, fitted, example, da_prediction=None):
        assert "target_values" not in example
        self.da_prediction = da_prediction
        prices = da_prediction if da_prediction is not None else [40.0] * len(example["product_starts_utc"])
        return {"product_starts_utc": example["product_starts_utc"], "prediction": prices,
                "model": "fake", "metadata": {}}


def sample(market, gate, starts):
    core = np.full(1513, np.nan)
    for index in range(3):
        core[1200 + index * 101 + 100] = 0
    return {"delivery_day": "2025-10-01", "market": market, "gate": gate,
        "decision_at": "2025-09-30T09:45:00+00:00", "product_starts_utc": starts,
        "target_values": [999.0] * len(starts), "feature_blocks": {"core": core}}


def test_daily_fit_restores_and_da_ida_uses_forecast_only(tmp_path: Path):
    engine = FakeEngine()
    starts = ["2025-09-30T22:00:00+00:00", "2025-09-30T22:15:00+00:00"]
    provider = RollingPriceProvider(engine, snapshot_id="snapshot", model_identity="model", cache_directory=tmp_path)
    provider.begin_day("2025-10-01", "2025-09-30T09:45:00+00:00")
    provider.begin_day("2025-10-01", "2025-09-30T09:45:00+00:00")
    assert engine.fits == 1
    provider.forecast("2025-10-01", "DA", "DA", "2025-09-30T09:45:00+00:00", starts,
        example=sample("DA", "DA", starts))
    result = provider.forecast("2025-10-01", "DA", "IDA3", "2025-09-30T09:45:00+00:00", starts[1:],
        example=sample("IDA3", "DA", starts[1:]))
    assert result["prediction"] == [40.0]
    assert engine.da_prediction == [40.0]
    restarted = FakeEngine()
    other = RollingPriceProvider(restarted, snapshot_id="snapshot", model_identity="model", cache_directory=tmp_path)
    other.begin_day("2025-10-01", "2025-09-30T09:45:00+00:00")
    assert restarted.fits == 0
    carried = other.forecast("2025-10-01", "IDA1", "IDA3", "2025-09-30T12:45:00+00:00", starts[1:])
    assert carried["prediction"] == [40.0]
    assert carried["method"] == "carry_da_gate_forecast"


def test_fallback_uses_latest_published_curve_and_excludes_future(tmp_path: Path):
    starts = ["2025-09-30T22:00:00+00:00"]
    provider = RollingPriceProvider(FakeEngine(), snapshot_id="s", model_identity="m", cache_directory=tmp_path)
    provider.begin_day("2025-10-01", "2025-09-30T09:45:00+00:00")
    provider.forecast("2025-10-01", "DA", "DA", "2025-09-30T09:45:00+00:00", starts,
        example=sample("DA", "DA", starts))
    provider.forecast("2025-10-01", "DA", "IDA3", "2025-09-30T09:45:00+00:00", starts,
        example=sample("IDA3", "DA", starts))
    da = PublishedAuctionCurve("DA", starts, [55.0], "2025-09-30T10:57:00+00:00")
    ida1_future = PublishedAuctionCurve("IDA1", starts, [900.0], "2025-09-30T14:00:00+00:00")
    result = provider.forecast("2025-10-01", "IDA2", "IDA3", "2025-09-30T12:45:00+00:00", starts,
        published_curves=[da, ida1_future])
    assert result["prediction"] == [55.0]
    assert result["method"] == "carry_published_clearing"
    assert result["source_auction"] == "DA"


def test_rejects_unavailable_da_and_invalid_model_inputs(tmp_path: Path):
    starts = ["2025-09-30T22:00:00+00:00"]
    provider = RollingPriceProvider(FakeEngine(), snapshot_id="s", model_identity="m", cache_directory=tmp_path)
    provider.begin_day("2025-10-01", "2025-09-30T09:45:00+00:00")
    with pytest.raises(ValueError, match="Forecast DA before"):
        provider.forecast("2025-10-01", "DA", "IDA1", "2025-09-30T09:45:00+00:00", starts,
            example=sample("IDA1", "DA", starts))
    with pytest.raises(ValueError, match="precedes"):
        provider.forecast("2025-10-01", "DA", "DA", "2025-09-30T09:44:00+00:00", starts,
            example=sample("DA", "DA", starts))
    with pytest.raises(ValueError, match="cutoff differs"):
        provider.begin_day("2025-10-01", "2025-09-30T09:44:00+00:00")


def test_detects_modified_cached_fit(tmp_path: Path):
    provider = RollingPriceProvider(FakeEngine(), snapshot_id="s", model_identity="m", cache_directory=tmp_path)
    provider.begin_day("2025-10-01", "2025-09-30T09:45:00+00:00")
    (provider.root / "2025-10-01" / "daily_models.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        provider.begin_day("2025-10-01", "2025-09-30T09:45:00+00:00")


def test_auction_book_uses_only_published_prices_and_covers_future_auctions(tmp_path: Path):
    starts = ["2025-09-30T22:00:00+00:00", "2025-09-30T22:15:00+00:00"]
    events = []
    for name, submit, result, eligible in (
        ("DA", "09:45", "10:57", (0, 1)),
        ("IDA1", "12:45", "13:30", (0, 1)),
        ("IDA2", "19:45", "20:30", (0, 1)),
        ("IDA3", "21:45", "22:30", (1,)),
    ):
        events.append(SimpleNamespace(auction=name, submission_at=pd.Timestamp(f"2025-09-30T{submit}:00Z"),
            results_at=pd.Timestamp(f"2025-09-30T{result}:00Z"), eligible_intervals=eligible, available=True))
    examples = {}
    for event in events:
        products = [starts[i] for i in event.eligible_intervals]
        examples[("2025-10-01", event.auction, "DA")] = sample(event.auction, "DA", products)
        own = sample(event.auction, event.auction, products)
        own["decision_at"] = event.submission_at.isoformat()
        if event.auction == "IDA1":
            own["feature_blocks"]["core"][1200:1300] = pad_curve(pd.DatetimeIndex(starts),
                [55.0, 56.0], "Europe/Rome")
            own["price_reference"] = [55.0, 56.0]
        examples[("2025-10-01", event.auction, event.auction)] = own
    provider = RollingPriceProvider(FakeEngine(), snapshot_id="s", model_identity="m", cache_directory=tmp_path)
    curves, trace = forecast_auction_book(provider, "2025-10-01", "DA", events[0].submission_at,
        starts, events, {}, examples)
    assert set(curves) == {"DA", "IDA1", "IDA2", "IDA3"}
    assert curves["IDA3"].tolist() == [0.0, 40.0]
    assert trace["IDA2"]["method"] == "rolling_model"
    # The actual optimizer scenario is constructed entirely from the book;
    # unpublished realized curves were never passed into this calculation.
    grid_starts = tuple(pd.Timestamp(item).to_pydatetime() for item in starts)
    grid_ends = tuple((pd.Timestamp(item) + pd.Timedelta(minutes=15)).to_pydatetime() for item in starts)
    grid = DeliveryGrid(("p0", "p1"), grid_starts, grid_ends)
    names = [item.auction for item in events]
    spreads = {current: curves[current] - curves[previous] for previous, current in zip(names, names[1:])}
    bundle, _ = synthetic_price_tree(grid, np.array([1.0, 1.0]), events,
        base_prices=curves["DA"], stage_spreads=spreads, residual_scale=0.0,
        branch_values=(0.0,), branch_probabilities=(1.0,), decision_cutoff=events[0].submission_at)
    for auction in ("DA", "IDA1", "IDA2"):
        assert np.allclose(bundle.auction_prices_eur_per_mwh[auction], [[40.0, 40.0]])
    assert bundle.auction_prices_eur_per_mwh["IDA3"][0, 1] == 40.0
    curves, trace = forecast_auction_book(provider, "2025-10-01", "IDA1", events[1].submission_at,
        starts, events, {"DA": [55.0, 56.0]}, examples)
    assert curves["DA"].tolist() == [55.0, 56.0]
    assert curves["IDA2"].tolist() == [55.0, 56.0]
    assert trace["IDA2"]["method"] == "carry_published_clearing"
    current, _ = forecast_auction_book(provider, "2025-10-01", "IDA1", events[1].submission_at,
        starts, events, {"DA": [55.0, 56.0]}, examples, targets=("IDA1",))
    assert set(current) == {"DA", "IDA1"}
    with pytest.raises(ValueError, match="has not been published"):
        forecast_auction_book(provider, "2025-10-01", "IDA1", events[1].submission_at,
            starts, events, {"DA": [55.0, 56.0], "IDA1": [999.0, 999.0]}, examples)


def test_cash_is_booked_from_actual_publication_not_forecast():
    replay = ContinuousReplay.__new__(ContinuousReplay)
    replay.submitted = {"2025-10-01": {"DA": {"product": 2.0}}}
    replay.known = {"2025-10-01": []}
    replay.market = MarketState()
    replay.fee_records = {"2025-10-01:DA:product": {"transaction_fee_eur": 0.0}}
    replay.daily = {"2025-10-01": {"grid": SimpleNamespace(interval_ids=("product",)),
        "prices": {"DA": np.array([55.0])}}}
    replay.publications = []
    auction = SimpleNamespace(auction="DA", submission_at=pd.Timestamp("2025-09-30T09:45:00Z"))
    event = SimpleNamespace(payload=SimpleNamespace(delivery_day="2025-10-01", event=auction),
        at=pd.Timestamp("2025-09-30T10:57:00Z"))
    assert replay.market.cashflow_eur == 0
    replay._publish(event)
    assert replay.market.cashflow_eur == 110.0
    assert replay.known["2025-10-01"] == ["DA"]
    with pytest.raises(RuntimeError, match="Invalid publication"):
        replay._publish(event)


def test_replay_refuses_skipped_auctions_without_announcement_times():
    event = SimpleNamespace(auction="DA", available=True, results_at=pd.Timestamp("2025-09-30T10:57:00Z"))
    validate_replay_price_schedule({"2025-10-01": {"schedule": [event], "skipped": []}}, ["DA"])
    with pytest.raises(ValueError, match="hindsight-filtered"):
        validate_replay_price_schedule({"2025-10-01": {"schedule": [event], "skipped": [
            {"auction": "IDA1", "reason": "documented_cancellation"}]}}, ["DA", "IDA1"])
    with pytest.raises(ValueError, match="complete configured schedule"):
        validate_replay_price_schedule({"2025-10-01": {"schedule": [event], "skipped": []}}, ["DA", "IDA1"])


def test_own_gate_rejects_delayed_or_mismatched_published_curve(tmp_path: Path):
    starts = ["2025-09-30T22:00:00+00:00", "2025-09-30T22:15:00+00:00"]
    provider = RollingPriceProvider(FakeEngine(), snapshot_id="s", model_identity="m", cache_directory=tmp_path)
    provider.begin_day("2025-10-01", "2025-09-30T09:45:00+00:00")
    example = sample("IDA1", "IDA1", starts)
    example["decision_at"] = "2025-09-30T12:45:00+00:00"
    example["feature_blocks"]["core"][1200:1300] = pad_curve(pd.DatetimeIndex(starts),
        [55.0, 56.0], "Europe/Rome")
    example["price_reference"] = [55.0, 56.0]
    delayed = PublishedAuctionCurve("DA", starts, [55.0, 56.0], "2025-09-30T13:00:00+00:00")
    with pytest.raises(ValueError, match="until DA is actually published"):
        provider.forecast("2025-10-01", "IDA1", "IDA1", "2025-09-30T12:45:00+00:00", starts,
            example=example, published_curves=[delayed])
    mismatch = PublishedAuctionCurve("DA", starts, [55.0, 60.0], "2025-09-30T10:57:00+00:00")
    with pytest.raises(ValueError, match="differs from the replay"):
        provider.forecast("2025-10-01", "IDA1", "IDA1", "2025-09-30T12:45:00+00:00", starts,
            example=example, published_curves=[mismatch])
    published = PublishedAuctionCurve("DA", starts, [55.0, 56.0], "2025-09-30T10:57:00+00:00")
    result = provider.forecast("2025-10-01", "IDA1", "IDA1", "2025-09-30T12:45:00+00:00", starts,
        example=example, published_curves=[published])
    assert result["prediction"] == [40.0, 40.0]


def test_da_gate_rejects_delivery_day_clearing_in_features(tmp_path: Path):
    starts = ["2025-09-30T22:00:00+00:00"]
    provider = RollingPriceProvider(FakeEngine(), snapshot_id="s", model_identity="m", cache_directory=tmp_path)
    provider.begin_day("2025-10-01", "2025-09-30T09:45:00+00:00")
    example = sample("DA", "DA", starts)
    example["feature_blocks"]["core"][1200] = 55.0
    with pytest.raises(ValueError, match="Unpublished DA clearing"):
        provider.forecast("2025-10-01", "DA", "DA", "2025-09-30T09:45:00+00:00", starts, example=example)

"""Causal ridge, OOS residual calibration, and conditional information tree checks."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.domain import DeliveryGrid
from bess_pv_trading.forecasting.prices import (
    ResidualDistribution, build_price_features, fit_ridge, rolling_origin_residuals,
    train_price_from_config,
)
from bess_pv_trading.data.market_prices import MarketDataError, validate_market_records
from bess_pv_trading.markets import AuctionCalendar
from bess_pv_trading.scenarios import synthetic_price_tree


def tiny_grid():
    start = datetime(2025, 10, 2, 10, 0, tzinfo=timezone.utc)
    return DeliveryGrid(("t0", "t1"), (start, start + timedelta(minutes=15)),
                        (start + timedelta(minutes=15), start + timedelta(minutes=30)))


def test_ridge_training_cutoff_and_scaler():
    x = pd.DataFrame({"lag": [1, 2, 3]})
    publication = pd.date_range("2025-10-01", periods=3, tz="UTC")
    model = fit_ridge(x, [2, 4, 6], feature_available_at=publication,
                      target_available_at=publication, fit_cutoff="2025-10-04T00:00:00Z", alpha=0)
    assert model.center[0] == 2
    assert np.allclose(model.predict(pd.DataFrame({"lag": [10]})), [20])
    with pytest.raises(ValueError, match="future features"):
        fit_ridge(x, [2, 4, 6], feature_available_at=publication,
                  target_available_at=publication, fit_cutoff="2025-10-02T00:00:00Z")


def test_oos_validation_targets_do_not_affect_own_predictions():
    rows = pd.DataFrame({"x": np.arange(6.), "target": np.arange(6.) * 2,
        "delivery_day": ["2025-10-01"] * 2 + ["2025-10-02"] * 2 + ["2025-10-03"] * 2,
        "auction": ["DA"] * 6,
        "product_start": pd.date_range("2025-10-01", periods=6, freq="h", tz="UTC")})
    rows["product_end"] = rows.product_start + pd.Timedelta(hours=1)
    rows["decision_cutoff"] = pd.to_datetime(rows.delivery_day, utc=True)
    rows["feature_available_at"] = rows.decision_cutoff
    rows["target_available_at"] = rows.decision_cutoff + pd.Timedelta(hours=1)
    errors = rolling_origin_residuals(rows, feature_names=["x"], min_training_rows=2)
    changed = rows.copy()
    changed.loc[changed.delivery_day == "2025-10-02", "target"] += 100
    altered = rolling_origin_residuals(changed, feature_names=["x"], min_training_rows=2)
    mask = errors.delivery_day == "2025-10-02"
    assert np.allclose(errors.loc[mask, "prediction"], altered.loc[mask, "prediction"])
    assert not np.allclose(errors.loc[mask, "residual"], altered.loc[mask, "residual"])


def test_bootstrap_samples_whole_vectors_and_gaussian_is_seeded():
    vectors = np.array([[1., 2., 3.], [-1., -2., -3.], [4., 5., 6.]])
    model = ResidualDistribution(vectors, "2025-10-01T00:00:00Z")
    samples = model.sample(20, method="bootstrap", seed=42)
    assert all(any(np.array_equal(sample, source) for source in vectors) for sample in samples)
    assert np.array_equal(model.sample(3, seed=4), model.sample(3, seed=4))
    with pytest.raises(ValueError, match="In-sample"):
        ResidualDistribution(vectors, "2025-10-01T00:00:00Z", provenance="in_sample")


def test_conditional_tree_preserves_unresolved_future_outcomes():
    grid = tiny_grid()
    events = AuctionCalendar().schedule("2025-10-02")
    scenarios, tree = synthetic_price_tree(grid, np.array([1., 2.]), events,
        branch_probabilities=(.2, .8), branch_values=(-1, 1))
    assert len(scenarios.scenario_ids) == 16
    assert np.isclose(scenarios.probabilities.sum(), 1)
    assert len(set(tree.bid_nodes["DA"])) == 1
    assert len(set(tree.bid_nodes["IDA1"])) == 2
    assert len(set(tree.bid_nodes["IDA2"])) == 4
    assert len(set(tree.bid_nodes["IDA3"])) == 8
    # Equal observed DA prices retain eight distinct descendants, not one path.
    low = scenarios.auction_prices_eur_per_mwh["DA"][:, 0] == 65
    assert low.sum() == 8
    assert len(np.unique(scenarios.auction_prices_eur_per_mwh["IDA3"][low, 0])) > 1
    histories = scenarios.provenance["branch_histories"]
    expected = [.2 ** sum(v == 0 for v in history.values()) * .8 ** sum(v == 1 for v in history.values()) for history in histories]
    assert np.allclose(scenarios.probabilities, expected)
    tree.paths()


def test_tree_known_results_fixed_and_future_known_rejected():
    grid = tiny_grid()
    events = AuctionCalendar().schedule("2025-10-02")
    cutoff = pd.Timestamp("2025-10-01T12:00:00Z")
    scenario, tree = synthetic_price_tree(grid, np.ones(2), events,
        known_prices={"DA": np.array([100., 110.])}, decision_cutoff=cutoff)
    assert len(scenario.scenario_ids) == 8
    assert "DA" not in tree.bid_nodes
    assert np.all(scenario.auction_prices_eur_per_mwh["DA"] == [100, 110])
    with pytest.raises(ValueError, match="not published"):
        synthetic_price_tree(grid, np.ones(2), events, known_prices={"DA": np.ones(2)})


def test_real_price_training_blocker_is_actionable(tmp_path):
    with pytest.raises(MarketDataError, match="import actual"):
        train_price_from_config({"price_forecasting": {"input_path": str(tmp_path / "missing.csv")}})


def _price_record(auction, start, publication, price):
    return {"provider": "fixture", "dataset": "prices", "zone": "NORD", "auction": auction,
        "product_start": start.isoformat(), "product_end": (start + pd.Timedelta(minutes=15)).isoformat(),
        "price_eur_mwh": price, "published_at": publication.isoformat(),
        "publication_provenance": "synthetic fixture", "revision": "1"}


def test_same_clock_lags_preserve_dst_fold_and_previous_results_cutoff():
    calendar = AuctionCalendar()
    rows = []
    # One clock product per day; repeated autumn hours have a distinct fold.
    for day in pd.date_range("2025-10-18", "2025-10-26"):
        local = day.strftime("%Y-%m-%d")
        grid = calendar.delivery_grid(local)
        for start in grid.starts:
            if start.tz_convert("Europe/Rome").hour == 2 and start.minute == 15:
                rows.append(_price_record("DA", start, calendar.schedule(local)[0].results_at, day.day))
    records = validate_market_records(pd.DataFrame(rows))
    features = build_price_features(records, "DA", calendar)
    last = features.loc[features.delivery_day == "2025-10-26"].sort_values("product_start")
    assert len(last) == 2
    assert last.lag_1d.iloc[0] == 25
    assert pd.isna(last.lag_1d.iloc[1])
    # Delayed DA results are not available to the IDA1 feature row.
    grid = calendar.delivery_grid("2025-10-26")
    start = grid.starts[20]
    events = calendar.schedule("2025-10-26")
    prices = validate_market_records(pd.DataFrame([
        _price_record("DA", start, events[1].results_at + pd.Timedelta(hours=1), 100),
        _price_record("IDA1", start, events[1].results_at, 110)]))
    ida = build_price_features(prices, "IDA1", calendar)
    assert pd.isna(ida.previous_auction_price.iloc[0])
    assert pd.isna(ida.target.iloc[0])


def test_manual_fixture_trains_real_cli_boundary_and_saves_oos(tmp_path):
    calendar = AuctionCalendar()
    rows = []
    for day in pd.date_range("2025-10-01", "2025-10-15"):
        day_string = day.strftime("%Y-%m-%d")
        start = calendar.delivery_grid(day_string).starts[12]
        rows.append(_price_record("DA", start, calendar.schedule(day_string)[0].results_at, 50 + day.day))
    path = tmp_path / "actual_fixture.csv"
    validate_market_records(pd.DataFrame(rows)).to_csv(path, index=False)
    result = train_price_from_config({"price_forecasting": {"input_path": str(path),
        "output_directory": str(tmp_path / "models"), "fit_cutoff": "2025-10-16T00:00:00Z",
        "minimum_training_rows": 2, "auctions": ["DA"]}})
    assert result["models"]["DA"]["status"] == "trained"
    assert result["models"]["DA"]["oos_residual_rows"] > 0
    assert (tmp_path / "models" / "DA_model.json").is_file()
    assert (tmp_path / "models" / "DA_oos_residuals.csv").is_file()


def test_fitted_models_generate_conditional_future_tree():
    from bess_pv_trading.scenarios import fitted_price_tree
    grid = tiny_grid()
    events = AuctionCalendar().schedule("2025-10-02")
    available = pd.date_range("2025-09-20", periods=3, tz="UTC")
    cutoff = events[0].submission_at
    da = fit_ridge(pd.DataFrame({"x": [0., 1., 2.]}), [70., 75., 80.],
        feature_available_at=available, target_available_at=available, fit_cutoff=cutoff, alpha=0)
    spread = fit_ridge(pd.DataFrame({"previous_auction_price": [80., 100., 120.]}), [8., 10., 12.],
        feature_available_at=available, target_available_at=available, fit_cutoff=cutoff, alpha=0, target="IDA_spread")
    models = {"DA": da, "IDA1": spread, "IDA2": spread, "IDA3": spread}
    features = {"DA": pd.DataFrame({"x": [1., 1.]})}
    features.update({name: pd.DataFrame({"previous_auction_price": [0., 0.]}) for name in models if name != "DA"})
    distributions = {name: ResidualDistribution(np.array([[-1., -1.], [1., 1.]]), cutoff.isoformat()) for name in models}
    scenarios, tree = fitted_price_tree(grid, np.ones(2), events, models=models,
        future_features=features, feature_available_at={name: cutoff for name in models},
        residual_distributions=distributions, decision_cutoff=cutoff, residual_method="bootstrap", seed=6)
    assert 1 <= len(scenarios.scenario_ids) <= 16
    assert np.allclose(scenarios.auction_prices_eur_per_mwh["IDA1"] - 1.1 * scenarios.auction_prices_eur_per_mwh["DA"],
                       np.round(scenarios.auction_prices_eur_per_mwh["IDA1"] - 1.1 * scenarios.auction_prices_eur_per_mwh["DA"]))
    assert len(set(tree.bid_nodes["DA"])) == 1
    with pytest.raises(ValueError, match="unpublished"):
        fitted_price_tree(grid, np.ones(2), events, models=models, future_features=features,
            feature_available_at={name: cutoff + pd.Timedelta(hours=1) for name in models},
            residual_distributions=distributions, decision_cutoff=cutoff)


def test_identical_observable_price_outcomes_merge_before_descendants():
    grid = tiny_grid()
    events = AuctionCalendar().schedule("2025-10-02")
    scenarios, tree = synthetic_price_tree(grid, np.ones(2), events, residual_scale=0,
                                           branch_probabilities=(.2, .8))
    assert len(scenarios.scenario_ids) == 1
    assert scenarios.probabilities.tolist() == [1.]
    assert all(len(set(owners)) == 1 for owners in tree.bid_nodes.values())
    assert not any("branch" in observation for node in tree.nodes for observation in node.observations)
    # A deterministic observed DA result keeps independent future uncertainty.
    scenarios, tree = synthetic_price_tree(grid, np.ones(2), events,
        residual_scale={"DA": 0, "IDA1": 10, "IDA2": 10, "IDA3": 10})
    assert len(scenarios.scenario_ids) == 8
    assert len(set(tree.bid_nodes["IDA1"])) == 1
    assert len(set(tree.bid_nodes["IDA2"])) == 2
    assert np.isclose(scenarios.probabilities.sum(), 1.)

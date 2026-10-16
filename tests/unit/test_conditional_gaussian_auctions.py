"""Auction-scoped Gaussian products and branch-conditional absolute price means."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.domain import (
    BatteryParameters, MarketState, OptimizationInput, PhysicalState, PlantParameters, SolverSettings,
)
from bess_pv_trading.forecasting.gaussian_prices import (
    GaussianPriceCalibration, calibrate_gaussian_prices, evaluate_gaussian_prices, gaussian_price_tree,
)
from bess_pv_trading.markets import AuctionCalendar
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.scenarios import conditional_price_tree
from test_gaussian_prices import prediction_rows


def scoped_rows(auction, days=None):
    """Keep genuine afternoon products only when the target is explicitly IDA3."""
    days = pd.date_range("2025-10-01", periods=12) if days is None else days
    frame = prediction_rows(days, minutes=15)
    if auction == "IDA3":
        frame = frame.loc[frame.product_start.dt.tz_convert("Europe/Rome").dt.hour >= 12].copy()
    frame["auction"] = auction
    return frame


def calibration(auction="DA", days=None):
    """Calibrate synthetic chronological errors on an explicitly chosen auction."""
    rows = scoped_rows(auction, days)
    return calibrate_gaussian_prices(rows, auction=auction, calibration_cutoff=rows.product_end.max(), minimum_complete_days=2)


def small_grid_schedule(auctions=("DA", "IDA1", "IDA2")):
    """Use two afternoon native intervals with residual-grid auction masks."""
    calendar = AuctionCalendar()
    full = calendar.delivery_grid("2025-11-01")
    grid = full.to_domain()
    grid = replace(grid, interval_ids=grid.interval_ids[48:50], starts=grid.starts[48:50], ends=grid.ends[48:50])
    schedule = tuple(replace(event, eligible_intervals=(0, 1)) for event in calendar.schedule("2025-11-01") if event.auction in auctions)
    return grid, schedule


def test_history_mean_sees_only_ancestors_and_changes_descendant_price_means():
    grid, schedule = small_grid_schedule()
    seen = []

    def mean(auction, history):
        seen.append((auction, tuple(history)))
        if auction == "DA":
            assert not history
            return np.full(2, 100.)
        if auction == "IDA1":
            assert set(history) == {"DA"}
            with pytest.raises(TypeError):
                history["IDA3"] = np.zeros(2)
            with pytest.raises(ValueError):
                history["DA"][0] = 999.
            return history["DA"] + 10
        assert set(history) == {"DA", "IDA1"}
        return .5 * history["DA"] + .5 * history["IDA1"] + 7

    def innovations(auction, count, random):
        return np.repeat(np.array([-5., 5.])[:, None], count, axis=1) if auction == "DA" else np.zeros((2, count))

    bundle, tree = conditional_price_tree(grid, np.ones(2), schedule, history_mean=mean,
        innovation_sampler=innovations)
    assert len(bundle.scenario_ids) == 2
    assert np.allclose(bundle.auction_prices_eur_per_mwh["IDA1"], bundle.auction_prices_eur_per_mwh["DA"] + 10)
    assert np.allclose(bundle.auction_prices_eur_per_mwh["IDA2"], bundle.auction_prices_eur_per_mwh["DA"] + 12)
    assert len(set(tree.bid_nodes["IDA1"])) == 2
    assert any(auction == "IDA2" for auction, _ in seen)
    with pytest.raises(ValueError, match="never both"):
        conditional_price_tree(grid, np.ones(2), schedule, history_mean=mean,
            conditional_mean=lambda auction, previous: previous, innovation_sampler=innovations)


def test_callback_cannot_mutate_parent_price_arrays_even_if_it_reenables_its_copy():
    grid, schedule = small_grid_schedule(("DA", "IDA1"))

    def mean(auction, history):
        if auction == "DA":
            return np.full(2, 100.)
        history["DA"].flags.writeable = True
        history["DA"][:] = -999.
        return np.full(2, 105.)

    bundle, _ = conditional_price_tree(grid, np.ones(2), schedule, history_mean=mean,
        innovation_sampler=lambda auction, count, random: np.zeros((1, count)), branch_probabilities=(1.,))
    assert np.array_equal(bundle.auction_prices_eur_per_mwh["DA"], [[100., 100.]])


def test_history_mean_uses_fixed_known_prior_without_future_keys():
    grid, schedule = small_grid_schedule(("DA", "IDA1"))
    known = np.array([120., 130.])
    cutoff = schedule[0].results_at + pd.Timedelta(minutes=1)
    seen = []

    def mean(auction, history):
        assert auction == "IDA1" and set(history) == {"DA"}
        seen.append(history["DA"].copy())
        return history["DA"] + 3

    bundle, tree = conditional_price_tree(grid, np.ones(2), schedule, history_mean=mean,
        innovation_sampler=lambda auction, count, random: np.zeros((1, count)),
        branch_probabilities=(1.,), known_prices={"DA": known}, decision_cutoff=cutoff)
    assert np.array_equal(bundle.auction_prices_eur_per_mwh["DA"], [known])
    assert np.array_equal(bundle.auction_prices_eur_per_mwh["IDA1"], [known + 3])
    assert "DA" not in tree.bid_nodes and len(seen) == 1


def test_ida3_template_pools_exact_48_afternoon_slots_across_dst_without_inference(tmp_path):
    days = ["2025-03-30", "2025-10-26", "2025-11-01"]
    artifact = calibration("IDA3", days)
    assert artifact.auction == "IDA3"
    assert list(artifact.grids) == ["IDA3_15min_48products_ordinary"]
    model = next(iter(artifact.grids.values()))
    assert model.calibration_days == 3
    assert len(model.slot_labels) == 48 and model.slot_labels[0] == "12:00/fold0"
    assert artifact.to_dict()["schema"] == "gaussian-price-calibration-v2"
    path = tmp_path / "ida3.json"
    artifact.save(path)
    loaded = GaussianPriceCalibration.load(path)
    assert loaded.to_dict() == artifact.to_dict()
    for day in ("2026-03-29", "2026-10-25", "2026-11-01"):
        products = scoped_rows("IDA3", [day])
        assert artifact.sample_innovations(products, 3, seed=2).shape == (3, 48)
        assert np.array_equal(artifact.sample_innovations(products, 3, seed=2), loaded.sample_innovations(products, 3, seed=2))
    halfday = scoped_rows("IDA3", days).drop(columns="auction")
    with pytest.raises(ValueError, match="Insufficient complete"):
        calibrate_gaussian_prices(halfday, calibration_cutoff=halfday.product_end.max(), minimum_complete_days=2)
    with pytest.raises(ValueError, match="auction-eligible native"):
        artifact.parameters_for(prediction_rows(["2026-11-01"], minutes=15))


def test_existing_v1_artifacts_and_default_day_ahead_layout_remain_unchanged(tmp_path):
    artifact = calibration()
    assert artifact.to_dict()["schema"] == "gaussian-price-calibration-v1"
    assert "auction" not in artifact.to_dict()
    assert list(artifact.grids) == ["15min_96products_ordinary"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(artifact.to_dict()))
    assert GaussianPriceCalibration.load(path).auction == "DA"
    assert GaussianPriceCalibration.load(path).to_dict() == artifact.to_dict()


def test_ida3_gaps_morning_rows_wrong_scope_and_hourly_labels_are_rejected():
    rows = scoped_rows("IDA3")
    with pytest.raises(ValueError, match="Insufficient complete"):
        calibrate_gaussian_prices(rows.groupby("delivery_day").head(47), auction="IDA3",
            calibration_cutoff=rows.product_end.max(), minimum_complete_days=2)
    with pytest.raises(ValueError, match="noon to midnight"):
        calibrate_gaussian_prices(prediction_rows(minutes=15), auction="IDA3",
            calibration_cutoff=rows.product_end.max(), minimum_complete_days=2)
    with pytest.raises(ValueError, match="explicitly selected auction"):
        calibrate_gaussian_prices(rows, auction="DA", calibration_cutoff=rows.product_end.max(), minimum_complete_days=2)
    with pytest.raises(ValueError, match="quarter-hour"):
        hourly = prediction_rows()
        calibrate_gaussian_prices(hourly, auction="IDA3", calibration_cutoff=hourly.product_end.max(), minimum_complete_days=2)


def test_ida3_untouched_test_scores_48_products_per_day_with_frozen_parameters():
    artifact = calibration("IDA3")
    test = scoped_rows("IDA3", ["2025-11-01", "2025-11-02"])
    before = artifact.to_dict()
    score = evaluate_gaussian_prices(artifact, test)
    assert score["metrics"]["scored_rows"] == 96
    assert score["metrics"]["scored_days"] == 2
    assert artifact.to_dict() == before


def test_gaussian_history_callbacks_rebuild_intraday_means_from_each_sampled_da_branch():
    grid, schedule = small_grid_schedule(("DA", "IDA1"))
    cutoff = schedule[0].submission_at
    calibrations = {auction: calibration(auction) for auction in ("DA", "IDA1")}
    seen = []

    def ida_mean(history):
        assert set(history) == {"DA"}
        seen.append(history["DA"].copy())
        return 1.1 * history["DA"] + 10

    bundle, tree = gaussian_price_tree(grid, np.ones(2), schedule, mean_prices={"DA": np.full(2, 80.)},
        conditional_means={"IDA1": ida_mean}, calibrations=calibrations,
        feature_available_at={auction: cutoff for auction in calibrations},
        model_fit_cutoffs={auction: cutoff for auction in calibrations},
        decision_cutoff=cutoff, branches=2, seed=5, allow_retrospective=True)
    assert len(seen) == 2 and not np.array_equal(seen[0], seen[1])
    assert len(bundle.scenario_ids) == 4 and len(set(tree.bid_nodes["IDA1"])) == 2
    assert "history-aware" in bundle.provenance["conditional_dependence"]
    late = {"DA": cutoff, "IDA1": cutoff + pd.Timedelta(seconds=1)}
    with pytest.raises(ValueError, match="future model/features"):
        gaussian_price_tree(grid, np.ones(2), schedule, mean_prices={"DA": np.full(2, 80.)},
            conditional_means={"IDA1": ida_mean}, calibrations=calibrations,
            feature_available_at=late, model_fit_cutoffs=late,
            decision_cutoff=cutoff, allow_retrospective=True)


def test_fitted_linear_spread_wrapper_is_rebuilt_inside_each_gaussian_branch():
    from bess_pv_trading.forecasting.intraday_price_features import ConditionalSpreadPriceForecast
    from bess_pv_trading.forecasting.prices import fit_ridge

    grid, schedule = small_grid_schedule(("DA", "IDA1"))
    cutoff = schedule[0].submission_at
    prices = np.array([50., 70., 90., 110.])
    publication = pd.date_range("2025-09-01", periods=4, tz="UTC")
    model = fit_ridge(pd.DataFrame({"previous_auction_price": prices}), .1 * prices + 3,
        feature_available_at=publication, target_available_at=publication, fit_cutoff=cutoff, alpha=0)
    wrapper = ConditionalSpreadPriceForecast("IDA1", model, "2025-10-12")
    future = pd.DataFrame({"product_start": grid.starts, "product_end": grid.ends,
                           "previous_auction_price": -999., "scenario_grid_index": [0, 1]})
    seen = []

    def mean(history):
        result = wrapper.predict_absolute(future, history)
        assert np.allclose(result, 1.1 * history["DA"] + 3)
        seen.append(result.copy())
        return result

    artifacts = {auction: calibration(auction) for auction in ("DA", "IDA1")}
    bundle, _ = gaussian_price_tree(grid, np.ones(2), schedule, mean_prices={"DA": np.full(2, 80.)},
        conditional_means={"IDA1": mean}, calibrations=artifacts,
        feature_available_at={auction: cutoff for auction in artifacts},
        model_fit_cutoffs={auction: cutoff for auction in artifacts}, decision_cutoff=cutoff,
        branches=2, seed=11, allow_retrospective=True)
    assert len(seen) == 2 and not np.array_equal(seen[0], seen[1])
    assert len(bundle.scenario_ids) == 4


def test_known_ida3_only_needs_actual_eligible_labels_on_a_partial_residual_grid():
    calendar = AuctionCalendar()
    full = calendar.delivery_grid("2025-11-01").to_domain()
    grid = replace(full, interval_ids=full.interval_ids[44:], starts=full.starts[44:], ends=full.ends[44:])
    event = replace(calendar.schedule("2025-11-01")[3], eligible_intervals=tuple(range(4, 52)))
    actual_afternoon = np.linspace(-30., 80., 48)
    bundle, tree = gaussian_price_tree(grid, np.ones(52), (event,),
        calibrations={}, feature_available_at={}, model_fit_cutoffs={}, decision_cutoff=grid.starts[0],
        known_prices={"IDA3": actual_afternoon})
    assert not tree.bid_nodes
    assert np.array_equal(bundle.auction_prices_eur_per_mwh["IDA3"][0, 4:], actual_afternoon)
    assert np.array_equal(bundle.auction_prices_eur_per_mwh["IDA3"][0, :4], np.zeros(4))
    with pytest.raises(ValueError, match="outside eligible products"):
        gaussian_price_tree(grid, np.ones(52), (event,), calibrations={}, feature_available_at={},
            model_fit_cutoffs={}, decision_cutoff=grid.starts[0], known_prices={"IDA3": np.ones(52)})
    original = calendar.schedule("2025-11-01")
    with pytest.raises(ValueError, match="not published"):
        gaussian_price_tree(full, np.ones(96), (original[3],), calibrations={}, feature_available_at={},
            model_fit_cutoffs={}, decision_cutoff=original[0].submission_at,
            known_prices={"IDA3": actual_afternoon})


@pytest.mark.solver
def test_joint_gaussian_ida3_paths_optimize_only_true_afternoon_products():
    """Exercise 48-dimensional IDA3 covariance inside a 96-interval MILP."""
    gurobi = pytest.importorskip("gurobipy")
    try:
        with gurobi.Model() as model:
            model.Params.OutputFlag = 0
    except gurobi.GurobiError as error:
        pytest.skip(str(error))
    calendar = AuctionCalendar()
    native = calendar.delivery_grid("2025-11-01")
    grid = native.to_domain()
    schedule = (calendar.schedule("2025-11-01")[3],)
    cutoff = calendar.schedule("2025-11-01")[0].submission_at
    artifact = calibration("IDA3")
    bundle, tree = gaussian_price_tree(grid, np.ones(96), schedule,
        mean_prices={"IDA3": np.full(48, 80.)}, calibrations={"IDA3": artifact},
        feature_available_at={"IDA3": cutoff}, model_fit_cutoffs={"IDA3": cutoff},
        decision_cutoff=cutoff, branches=2, seed=9, allow_retrospective=True)
    assert bundle.auction_prices_eur_per_mwh["IDA3"].shape == (2, 96)
    assert np.array_equal(bundle.auction_prices_eur_per_mwh["IDA3"][:, :48], np.zeros((2, 48)))
    assert len(bundle.provenance["auction_eligible_product_indices"]["IDA3"]) == 48
    assert bundle.provenance["untradable_price_placeholders"]["IDA3"]["product_indices"] == list(range(48))
    mask = tuple(index in schedule[0].eligible_intervals for index in range(96))
    inputs = OptimizationInput(grid, PlantParameters(1, 0., 1., BatteryParameters(0., 0., 0.)),
        PhysicalState(grid.starts[0], 0.), MarketState(), bundle, tree, {"IDA3": mask}, .25, .25, 1000., 0.)
    with pytest.raises(ValueError, match="outside the declared"):
        replace(inputs, auction_eligibility={"IDA3": (True,) * 96})
    built = build_model(inputs)
    assert len(built.variables["trades"]) == 48
    result = solve_model(built, SolverSettings())
    assert result.status == "optimal" and validate_solution(inputs, result).valid
    assert set(product for _, _, product in result.trades) == set(grid.interval_ids[48:])
    built.model.dispose()

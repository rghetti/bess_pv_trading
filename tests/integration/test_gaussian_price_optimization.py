"""Gaussian DA-only paths satisfy the actual stochastic MILP's full contract."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.domain import MarketState, OptimizationInput, PhysicalState, SolverSettings
from bess_pv_trading.forecasting.gaussian_prices import calibrate_gaussian_prices, gaussian_price_tree
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.optimization.degradation import degradation_settings_from_config
from bess_pv_trading.simulation.synthetic import synthetic_inputs


def _synthetic_oos_rows(calendar):
    """Supply complete native calibration days with certified earlier-day fits."""
    rows = []
    for index, day in enumerate(pd.date_range("2025-10-01", periods=8)):
        delivery_day = day.date().isoformat()
        products = calendar.delivery_grid(delivery_day)
        issuance = calendar.schedule(delivery_day)[0].submission_at
        point = np.full(len(products), 70.)
        errors = 2 + (index - 3.5) * np.linspace(1, 2, len(products))
        rows.append(pd.DataFrame({"delivery_day": delivery_day,
            "product_start": products.starts, "product_end": products.ends,
            "prediction": point, "actual": point + errors,
            "fit_cutoff": issuance - pd.Timedelta(hours=1), "forecast_cutoff": issuance,
            "training_max_delivery_day": (day - pd.Timedelta(days=1)).date().isoformat(),
            "prediction_provenance": "chronological_out_of_sample", "native_interval_minutes": 15}))
    return pd.concat(rows, ignore_index=True)


@pytest.mark.solver
def test_gaussian_day_ahead_paths_optimize_with_shared_bids_and_native_energy_units():
    """Verify licensed optimization, information sharing and independent accounting."""
    gurobi = pytest.importorskip("gurobipy")
    try:
        with gurobi.Model() as license_check:
            license_check.Params.OutputFlag = 0
    except gurobi.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")

    config = load_config(Path(__file__).resolve().parents[2] / "configs/synthetic_single.yaml")
    config["synthetic"]["delivery_day"] = "2025-11-01"
    fixture = synthetic_inputs(config)
    rows = _synthetic_oos_rows(fixture["calendar"])
    calibration = calibrate_gaussian_prices(rows, calibration_cutoff=rows.product_end.max(),
        covariance_shrinkage=.5, minimum_complete_days=4)
    grid = fixture["grid"]
    schedule = (fixture["schedule"][0],)
    cutoff = schedule[0].submission_at
    scenarios, tree = gaussian_price_tree(grid, fixture["pv"], schedule,
        mean_prices={"DA": fixture["base"]}, calibrations={"DA": calibration},
        feature_available_at={"DA": cutoff}, model_fit_cutoffs={"DA": cutoff - pd.Timedelta(hours=1)},
        decision_cutoff=cutoff, branches=3, seed=18, allow_retrospective=True)
    mask = {"DA": tuple(index in schedule[0].eligible_intervals for index in range(len(grid.interval_ids)))}
    inputs = OptimizationInput(grid, fixture["plant"], PhysicalState(grid.starts[0], .5), MarketState(),
        scenarios, tree, mask, config["trading"]["trade_limit_mwh"], config["trading"]["position_limit_mwh"],
        config["trading"]["imbalance_penalty_eur_per_mwh"], .5,
        degradation=degradation_settings_from_config(config))
    assert set(scenarios.auction_prices_eur_per_mwh) == {"DA"}
    assert set(tree.bid_nodes) == {"DA"}
    assert len(set(tree.bid_nodes["DA"])) == 1
    assert np.allclose(grid.delta_hours, .25)

    built = build_model(inputs)
    assert len(built.variables["trades"]) == 96
    result = solve_model(built, SolverSettings(**config["solver"]))
    assert result.status == "optimal"
    validation = validate_solution(inputs, result)
    assert validation.valid, validation.violations
    assert set(auction for auction, _, _ in result.trades) == {"DA"}
    assert len(result.trades) == 96
    assert result.objective_eur == pytest.approx(float(np.dot(scenarios.probabilities,
        result.components["operating_profit"])))
    # Bid quantities already use MWh. Native 15-minute durations enter physical
    # balances, so the independent financial sum must not multiply them again.
    expected_revenue = 0.0
    for scenario, probability in enumerate(scenarios.probabilities):
        owner = tree.bid_nodes["DA"][scenario]
        quantities = np.array([result.trades["DA", owner, product] for product in grid.interval_ids])
        expected_revenue += probability * float(np.dot(scenarios.auction_prices_eur_per_mwh["DA"][scenario], quantities))
        assert np.allclose(result.arrays["net_position_mwh"][scenario], quantities)
    assert float(np.dot(scenarios.probabilities, result.components["auction_revenue"])) == pytest.approx(expected_revenue)
    assert not scenarios.provenance["operational_eligible"]
    assert built.model.Params.MIPGap == pytest.approx(1e-4)

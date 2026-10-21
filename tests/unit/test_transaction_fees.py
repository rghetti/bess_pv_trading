"""Fee economics, exact tariff bands, monthly state and scenario consistency."""
from dataclasses import replace
from datetime import timedelta
import json

import numpy as np
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.domain import AcceptedTrade, InformationNode, InformationTree, MarketState, SolverSettings
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.transaction_fees import (
    TransactionFeeState, account_trade_fees, transaction_fees_from_config)
from test_optimization_contracts import example_inputs


def fee_settings(mode="gme_real", starting=0.0, as_of="2025-01-01T00:00:00+00:00"):
    """Use the published bands with explicitly assumed historical applicability."""
    config = load_config("configs/full_experiments/synthetic_fees_gme.yaml")
    config["transaction_fees"].update(mode=mode, flat_eur_per_mwh=0.04,
        initial_state={"as_of": as_of, "monthly_volume_mwh": {str(as_of)[:7]: starting},
            "cumulative_fees_eur": 17.0, "provenance": "synthetic known operator state"})
    return transaction_fees_from_config(config)


@pytest.fixture
def solve():
    """Run licensed fee models and independently validate every incumbent."""
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model():
            pass
    except gp.GurobiError as error:
        pytest.skip(str(error))

    def run(inputs, fixed=None):
        built = build_model(inputs)
        try:
            for (auction, node, index), variable in built.variables["trades"].items():
                if fixed is not None and auction in fixed:
                    built.model.addConstr(variable == fixed[auction])
            result = solve_model(built, SolverSettings(threads=1, output_flag=False))
        finally:
            built.model.dispose()
        assert result.has_incumbent
        validation = validate_solution(inputs, result)
        assert validation.valid, validation.violations
        return result
    return run


def arbitrage_inputs(mode, starting=0.0, spread=0.05):
    """Offset one MWh across two auctions with zero physical grid delivery."""
    prices = {"DA": np.array([[100.0]]), "IDA1": np.array([[100.0 + spread]])}
    inputs = example_inputs(auctions=prices, pv=np.zeros((1, 1)), hours=1,
        start="2025-01-02T12:00:00+00:00")
    cutoff = inputs.tree.nodes[0].cutoff
    fees = fee_settings(mode, starting, cutoff.isoformat())
    return replace(inputs, plant=replace(inputs.plant, poi_import_mw=1.0, poi_export_mw=1.0),
        optimize_imbalances=False, transaction_fees=fees,
        auction_submission_times={"DA": cutoff, "IDA1": cutoff + timedelta(hours=3)})


@pytest.mark.parametrize("start,volume,expected", [
    (0, 20000, 0), (19999, 1, 0), (20000, 1, 0.04), (19999, 2, 0.04),
    (999999, 2, 0.07), (1000000, 1, 0.03),
    (9999999, 2, 0.05), (10000000, 1, 0.02),
    (0, 10000001, 309200.02),
])
def test_progressive_marginal_bands(start, volume, expected):
    tariff = fee_settings().tariff("2025-01")
    assert tariff.increment(start, volume) == pytest.approx(expected)
    assert tariff.increment(start, 0) == 0


def test_month_reset_uses_transaction_dates_and_quarter_hour_energy_shares():
    fees = fee_settings(starting=19999, as_of="2024-12-31T10:00:00+00:00")
    rows = [{"auction": auction, "submission_time": at, "quantity_mwh": quantity,
             "interval_id": f"q{index}", "trade_id": f"{auction}:q{index}"}
        for auction, at, quantity in (("DA", "2024-12-31T11:00:00+00:00", 0.5),
            ("IDA3", "2025-01-01T09:55:00+00:00", -0.5)) for index in range(4)]
    final, records, summary = account_trade_fees(fees, fees.initial_state, rows)
    assert summary["gross_traded_mwh"] == 4
    assert summary["purchased_mwh"] == summary["sold_mwh"] == 2
    assert summary["transaction_fee_eur"] == pytest.approx(0.04)
    assert final.monthly_volume_mwh == {"2024-12": 20001, "2025-01": 2}
    assert final.cumulative_fees_eur == pytest.approx(17.04)
    flat = fee_settings("flat", as_of="2024-12-31T10:00:00+00:00")
    assert account_trade_fees(flat, flat.initial_state, rows)[2]["transaction_fee_eur"] == pytest.approx(0.16)
    with pytest.raises(ValueError, match="Duplicate"):
        account_trade_fees(fees, fees.initial_state, rows + rows[:1])


def test_tariff_dates_and_future_operator_state_are_explicit():
    inputs = arbitrage_inputs("gme_real", 19999)
    with pytest.raises(ValueError, match="unavailable"):
        replace(inputs, market=MarketState(transaction_fee_state=replace(
            inputs.transaction_fees.initial_state, as_of=inputs.tree.nodes[0].cutoff + timedelta(seconds=1))))
    with pytest.raises(ValueError, match="No applicable"):
        inputs.transaction_fees.tariff("2024-11")
    with pytest.raises(ValueError, match="Missing known"):
        inputs.transaction_fees.starting_volume(TransactionFeeState(
            as_of=inputs.tree.nodes[0].cutoff), "2025-01")
    with pytest.raises(ValueError, match="Future executed"):
        replace(inputs, market=MarketState(transaction_fee_state=replace(
            inputs.transaction_fees.initial_state, monthly_volume_mwh={"2025-01": 19999, "2025-02": 10})))
    for section in ({"mode": "invalid"}, {"mode": "flat", "flat_eur_per_mwh": -1},
                    {"mode": "gme_real", "operator_scope": "standalone_plant"}):
        with pytest.raises(ValueError):
            transaction_fees_from_config({"transaction_fees": section})


def test_tariff_version_is_selected_by_effective_month():
    from datetime import date
    fees = fee_settings()
    original = replace(fees.tariffs[0], valid_to=date(2025, 12, 31))
    changed = replace(original, version="synthetic_2026_change", valid_from=date(2026, 1, 1),
        valid_to=date(2026, 12, 31), marginal_eur_per_mwh=(0.01,) * 4)
    fees = replace(fees, tariffs=(original, changed))
    assert fees.increment("2025-12", 0, 1) == 0
    assert fees.increment("2026-01", 0, 1) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="overlap"):
        replace(fees, tariffs=(original, original))


def test_fee_modes_change_decisions_and_none_has_no_extra_model_variables(solve):
    none = solve(arbitrage_inputs("none"))
    flat = solve(arbitrage_inputs("flat"))
    tiered = solve(arbitrage_inputs("gme_real", 19999))
    assert none.objective_eur == pytest.approx(0.05)
    assert flat.objective_eur == pytest.approx(0)
    # One free gross MWh supports a half-MWh round trip. Further trading has
    # negative marginal profit, so the optimal order stops at the exemption.
    assert tiered.objective_eur == pytest.approx(0.025)
    assert tiered.components["transaction_fees"][0] == pytest.approx(0)
    assert sum(abs(value) for value in tiered.trades.values()) == pytest.approx(1)
    assert sum(abs(value) for value in flat.trades.values()) == pytest.approx(0)
    assert flat.diagnostics["binary_variables"] == none.diagnostics["binary_variables"]
    zero_flat = replace(arbitrage_inputs("flat"), transaction_fees=replace(
        arbitrage_inputs("flat").transaction_fees, flat_eur_per_mwh=0))
    baseline = solve(zero_flat)
    assert baseline.objective_eur == pytest.approx(none.objective_eur)
    assert baseline.diagnostics["variables"] == none.diagnostics["variables"]
    free = solve(arbitrage_inputs("gme_real", 0))
    assert free.diagnostics["variables"] == none.diagnostics["variables"]


@pytest.mark.parametrize("starting,expected", [(19999, 0.04), (999999, 0.07), (9999999, 0.05)])
def test_exact_milp_cannot_inflate_volume_to_choose_discounted_bands(solve, starting, expected):
    result = solve(arbitrage_inputs("gme_real", starting, spread=0), fixed={"DA": -1, "IDA1": 1})
    assert result.objective_eur == pytest.approx(-expected)
    assert result.components["transaction_fees"][0] == pytest.approx(expected)
    detail = result.diagnostics["transaction_fees_by_scenario"][0]
    assert detail["gross_traded_mwh"] == 2
    assert detail["by_month"]["2025-01"]["transaction_fee_eur"] == pytest.approx(expected)
    assert result.diagnostics["transaction_fee_formulation"]["exact_absolute_constraints"] == 2


def test_stochastic_fees_follow_leaf_trades_without_revealing_future_volume(solve):
    prices = {"DA": np.array([[10.0], [20.0]]), "IDA1": np.full((2, 1), 100.0),
              "IDA2": np.array([[100.01], [110.0]])}
    inputs = example_inputs(prices=prices["DA"], auctions=prices, pv=np.zeros((2, 1)),
        hours=1, start="2025-01-02T12:00:00+00:00")
    root = inputs.tree.nodes[0]
    branches = tuple(InformationNode(f"known{s}", "root", root.cutoff + timedelta(hours=1),
        0.5, observations=("auction:DA",)) for s in range(2))
    leaves = tuple(InformationNode(f"leaf{s}", branches[s].node_id, inputs.grid.ends[-1]) for s in range(2))
    owners = tuple(node.node_id for node in branches)
    tree = InformationTree((root, *branches, *leaves), tuple(node.node_id for node in leaves),
        {"DA": ("root", "root"), "IDA1": owners, "IDA2": owners}, inputs.tree.control_nodes)
    fees = fee_settings(starting=999999, as_of=root.cutoff.isoformat())
    inputs = replace(inputs, tree=tree, plant=replace(inputs.plant, poi_import_mw=1.0, poi_export_mw=1.0),
        transaction_fees=fees, optimize_imbalances=False,
        auction_submission_times={auction: root.cutoff + timedelta(hours=2*index)
            for index, auction in enumerate(prices)})
    result = solve(inputs, fixed={"DA": 0})
    np.testing.assert_allclose(result.components["transaction_fees"], [0, 0.07])
    assert result.objective_eur == pytest.approx((10 - 0.07) / 2)
    assert [item["gross_traded_mwh"] for item in result.diagnostics["transaction_fees_by_scenario"]] == [0, 2]
    tampered = result.components["transaction_fees"].copy()
    result.components["transaction_fees"][:] = 0
    assert not validate_solution(inputs, result).valid
    result.components["transaction_fees"] = tampered


@pytest.mark.parametrize("published", [False, True])
def test_residual_solve_keeps_executed_volume_and_excludes_sunk_fees(solve, published):
    inputs = arbitrage_inputs("gme_real", 999999, spread=0)
    fees, original = inputs.transaction_fees, inputs.tree.nodes[0]
    charged_state, _, booked = account_trade_fees(fees, fees.initial_state, [{
        "auction": "DA", "submission_time": original.cutoff, "interval_id": "t0",
        "quantity_mwh": -1.0, "trade_id": "DA:t0"}])
    accepted = AcceptedTrade("DA", original.cutoff, original.cutoff + timedelta(minutes=30),
        "t0", -1, 100, "DA:t0", transaction_fee_eur=0.04)
    root = replace(original, cutoff=original.cutoff + timedelta(minutes=60 if published else 15),
                   observations=("auction:DA",) if published else ())
    tree = replace(inputs.tree, nodes=(root, *inputs.tree.nodes[1:]), bid_nodes={"IDA1": ("root",)})
    inputs = replace(inputs, tree=tree, auction_eligibility={"IDA1": (True,)},
        market=MarketState((accepted,) if published else (), charged_state),
        pending_position_mwh=None if published else (-1.0,))
    result = solve(inputs)
    assert result.components["transaction_fees"][0] == pytest.approx(0.03)
    assert result.objective_eur == pytest.approx(99.97)
    assert result.diagnostics["sunk_transaction_fees_eur"] == pytest.approx(17.04)
    assert booked["transaction_fee_eur"] == pytest.approx(0.04)
    assert result.objective_eur - 100 - booked["transaction_fee_eur"] == pytest.approx(-0.07)


@pytest.mark.parametrize("mode", ["flat", "gme"])
def test_prepared_forecast_route_carries_fee_settings_into_the_milp(solve, mode):
    from test_forecast_inputs import prepared_case
    from bess_pv_trading.optimization.forecast_inputs import build_forecast_inputs
    config, pv, specs = prepared_case(["DA"])
    config["transaction_fees"] = load_config(f"configs/full_experiments/synthetic_fees_{mode}.yaml")["transaction_fees"]
    specs["DA"]["mean_curve"]["mean_price_eur_mwh"] = 100
    inputs = build_forecast_inputs(config, pv, specs)
    result = solve(inputs)
    gross = sum(abs(value) for value in result.trades.values())
    assert gross > 1
    expected = 0.04 * (gross if mode == "flat" else gross - 1)
    np.testing.assert_allclose(result.components["transaction_fees"], expected)


def test_gme_perfect_information_and_zero_battery_route_include_fees(solve, tmp_path):
    from bess_pv_trading.markets import AuctionCalendar
    from bess_pv_trading.simulation.gme_oracle import _solve_perfect_information_schedule
    calendar = AuctionCalendar(valid_from="2025-01-01", valid_to="2025-12-31")
    grid = calendar.delivery_grid("2025-01-02").to_domain()
    schedule = calendar.schedule("2025-01-02")[:2]
    config = load_config("configs/full_experiments/synthetic_fees_flat.yaml")
    config["transaction_fees"]["flat_eur_per_mwh"] = 0.01
    config["state"].update(terminal_target_mwh=0)
    config["trading"]["optimize_imbalances"] = False
    config["experiment_outputs"] = {"save_model_files": False}
    config["gme_oracle_experiment"] = {"decision_resolution_minutes": 15, "timezone": "Europe/Rome",
        "auction_datasets": {"DA": {"native_interval_minutes": 60}, "IDA1": {"native_interval_minutes": 15}}}
    plant = arbitrage_inputs("none").plant
    timing_records = []
    _, diagnostics = _solve_perfect_information_schedule(config, grid, plant, schedule,
        {"DA": np.full(96, 100), "IDA1": np.full(96, 100.05)}, np.zeros(96), 0, tmp_path,
        timing_records=timing_records)
    timing = json.loads((tmp_path / "json/optimization_timing.json").read_text())
    assert timing["solves"] == timing_records
    assert timing["gurobi_runtime_seconds"]["total"] == diagnostics["runtime_seconds"]
    assert diagnostics["objective_eur"] == pytest.approx(0.72)
    assert diagnostics["components"]["transaction_fees"] == pytest.approx(0.48)
    assert diagnostics["transaction_fee_formulation"]["absolute_volume_variables"] == 24 + 96


def test_milp_groups_by_transaction_month_at_the_delivery_year_boundary(solve):
    prices = {"DA": np.array([[100.0]]), "IDA3": np.array([[100.0]])}
    inputs = example_inputs(auctions=prices, pv=np.zeros((1, 1)), hours=1,
        start="2025-01-01T12:00:00+00:00")
    cutoff = inputs.grid.starts[0] - timedelta(hours=25, minutes=5)
    root = replace(inputs.tree.nodes[0], cutoff=cutoff)
    tree = replace(inputs.tree, nodes=(root, *inputs.tree.nodes[1:]))
    fees = fee_settings(starting=20000, as_of=cutoff.isoformat())
    inputs = replace(inputs, tree=tree, plant=replace(inputs.plant, poi_import_mw=1, poi_export_mw=1),
        optimize_imbalances=False, transaction_fees=fees,
        auction_submission_times={"DA": cutoff, "IDA3": inputs.grid.starts[0] - timedelta(hours=2, minutes=5)})
    result = solve(inputs, fixed={"DA": -1, "IDA3": 1})
    months = result.diagnostics["transaction_fees_by_scenario"][0]["by_month"]
    assert months["2024-12"]["transaction_fee_eur"] == pytest.approx(0.04)
    assert months["2025-01"]["transaction_fee_eur"] == 0
    assert result.objective_eur == pytest.approx(-0.04)


def test_exact_milp_can_cross_multiple_tariff_bands_in_one_solve(solve):
    inputs = arbitrage_inputs("gme_real", 19999, spread=0)
    inputs = replace(inputs, plant=replace(inputs.plant, poi_import_mw=5000000, poi_export_mw=5000000))
    result = solve(inputs, fixed={"DA": -5000000, "IDA1": 5000000})
    expected = 980000 * 0.04 + 9000000 * 0.03 + 19999 * 0.02
    assert result.components["transaction_fees"][0] == pytest.approx(expected, abs=1e-6)
    assert result.objective_eur == pytest.approx(-expected, abs=1e-6)


def test_identical_contingent_trades_share_one_monthly_fee_constraint(solve):
    reference = arbitrage_inputs("gme_real", 19999, spread=0)
    prices = {"DA": np.full((2, 1), 100.0), "IDA1": np.full((2, 1), 100.0)}
    inputs = example_inputs(prices=prices["DA"], auctions=prices, pv=np.zeros((2, 1)),
        hours=1, start="2025-01-02T12:00:00+00:00")
    inputs = replace(inputs, plant=reference.plant, optimize_imbalances=False,
        transaction_fees=reference.transaction_fees, auction_submission_times=reference.auction_submission_times)
    result = solve(inputs, fixed={"DA": -1, "IDA1": 1})
    np.testing.assert_allclose(result.components["transaction_fees"], [0.04, 0.04])
    assert result.objective_eur == pytest.approx(-0.04)
    assert result.diagnostics["transaction_fee_formulation"]["piecewise_constraints"] == 1

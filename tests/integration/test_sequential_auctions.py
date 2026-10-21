"""Sequential price-oracle decisions ignore later auctions but preserve delivery physics."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.simulation import synthetic


@pytest.fixture
def replay_config():
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model():
            pass
    except gp.GurobiError as error:
        pytest.skip(str(error))
    config = load_config("configs/full_experiments/synthetic_small.yaml")
    config["execution"].update(reoptimize_physical_controls=False, dispatch_policy="track_committed_grid",
        infeasible_auction_policy="previous_plan")
    config["trading"]["optimize_imbalances"] = False
    config["outputs"]["plots"] = False
    return config


def test_sequential_uses_actual_current_curve_and_fixed_prior_positions(replay_config, tmp_path, monkeypatch):
    fixture = synthetic.synthetic_inputs(replay_config)
    fixture["auction_planning_mode"] = "current_auction_only"
    delay = load_config("tests/fixtures/configs/participation_cases.yaml")["sequential"]["publication_delay_minutes"]
    fixture["schedule"] = (replace(fixture["schedule"][0], results_at=fixture["schedule"][1].submission_at
        + pd.Timedelta(minutes=delay)), *fixture["schedule"][1:])
    captured = []
    build = synthetic.build_model

    def capture(inputs):
        captured.append(inputs)
        return build(inputs)

    monkeypatch.setattr(synthetic, "build_model", capture)
    summary = synthetic._replay_strategy(replay_config, fixture, "oracle", tmp_path / "sequential")
    solves = json.loads((tmp_path / "sequential/solves.json").read_text())
    ledger = pd.read_csv(tmp_path / "sequential/ledger.csv")
    for event, inputs, diagnostic in zip(fixture["schedule"], captured, solves):
        assert set(inputs.auction_eligibility) == {event.auction}
        assert set(inputs.tree.bid_nodes) == {event.auction}
        assert set(inputs.scenarios.auction_prices_eur_per_mwh) == {event.auction}
        first = fixture["grid"].interval_ids.index(inputs.grid.interval_ids[0])
        np.testing.assert_allclose(inputs.scenarios.auction_prices_eur_per_mwh[event.auction][0],
            fixture["actual_prices"][event.auction][first:])
        earlier = ledger.loc[pd.to_datetime(ledger.submission_time, utc=True) < event.submission_at]
        positions = earlier.groupby("interval_id").quantity_mwh.sum().reindex(inputs.grid.interval_ids, fill_value=0)
        np.testing.assert_allclose(inputs.market.positions(inputs.grid) + np.asarray(inputs.pending_position_mwh), positions)
        assert diagnostic["modeled_auctions"] == [event.auction]
        assert diagnostic["trade_variable_count"] == sum(inputs.auction_eligibility[event.auction])
        assert diagnostic["planning_end_utc"] == pd.Timestamp(fixture["grid"].ends[-1]).isoformat()
    assert summary["auction_decision_solves"] == len(fixture["schedule"])
    assert summary["infeasible_auction_decision_count"] == 0
    assert summary["transaction_fees"]["validation"]["valid"]
    assert solves[1]["pending_position_abs_mwh"] > 0
    # At the final gate, integrated planning has exactly the same free trades
    # when conditioned on this baseline's identical state and commitments.
    from bess_pv_trading.scenarios import synthetic_price_tree
    from bess_pv_trading.domain import SolverSettings
    last = captured[-1]
    first_index = fixture["grid"].interval_ids.index(last.grid.interval_ids[0])
    names = [event.auction for event in fixture["schedule"]]
    scenario, tree = synthetic_price_tree(last.grid, last.scenarios.pv_ac_mw[0], fixture["schedule"],
        base_prices=fixture["actual_prices"][names[0]][first_index:],
        stage_spreads={current: (fixture["actual_prices"][current] - fixture["actual_prices"][prior])[first_index:]
            for prior, current in zip(names, names[1:])}, residual_scale=0,
        branch_values=(0,), branch_probabilities=(1,),
        known_prices={name: fixture["actual_prices"][name][first_index:] for name in names[:-1]},
        decision_cutoff=fixture["schedule"][-1].submission_at)
    integrated = build(replace(last, scenarios=scenario, tree=tree))
    try:
        result = synthetic.solve_model(integrated, SolverSettings(**replay_config["solver"]))
    finally:
        integrated.model.dispose()
    assert result.objective_eur == pytest.approx(solves[-1]["objective_eur"])


def test_future_price_changes_leave_first_sequential_decision_unchanged(replay_config, tmp_path):
    fixture = synthetic.synthetic_inputs(replay_config)
    fixture["auction_planning_mode"] = "current_auction_only"
    changed = deepcopy(fixture)
    multiplier = load_config("tests/fixtures/configs/participation_cases.yaml")["sequential"]["future_price_multiplier"]
    first = fixture["schedule"][0].auction
    for auction in changed["actual_prices"]:
        if auction != first:
            changed["actual_prices"][auction] *= multiplier
    for label, data in (("original", fixture), ("changed", changed)):
        synthetic._replay_strategy(replay_config, data, "oracle", tmp_path / label)
    original = pd.read_csv(tmp_path / "original/expected_dispatch_plans.csv")
    perturbed = pd.read_csv(tmp_path / "changed/expected_dispatch_plans.csv")
    columns = ["auction_trade_mwh", "charge_mw", "discharge_mw", "soc_end_mwh"]
    np.testing.assert_allclose(original.loc[original.auction == first, columns],
        perturbed.loc[perturbed.auction == first, columns])


@pytest.mark.parametrize("selection", load_config("tests/fixtures/configs/participation_cases.yaml")["selected_profiles"])
def test_single_auction_planning_modes_have_same_objective(replay_config, tmp_path, selection):
    if len(selection) != 1:
        return
    fixture = synthetic.synthetic_inputs(replay_config)
    fixture["schedule"] = tuple(event for event in fixture["schedule"] if event.auction in selection)
    fixture["actual_prices"] = {name: curve for name, curve in fixture["actual_prices"].items() if name in selection}
    objectives = []
    for mode in ("integrated", "current_auction_only"):
        fixture["auction_planning_mode"] = mode
        summary = synthetic._replay_strategy(replay_config, fixture, "oracle", tmp_path / mode)
        objectives.append(summary["initial_residual_objective_eur"])
    assert objectives[0] == pytest.approx(objectives[1])


def test_sequential_fallback_keeps_dispatch_and_submits_zero(replay_config, tmp_path, monkeypatch):
    from bess_pv_trading.domain import OptimizationResult
    fixture = synthetic.synthetic_inputs(replay_config)
    fixture["auction_planning_mode"] = "current_auction_only"
    original = synthetic.solve_model
    calls = []

    def fail_final(built, settings):
        calls.append(built.inputs)
        if len(calls) == len(fixture["schedule"]):
            return OptimizationResult("infeasible", False)
        return original(built, settings)

    monkeypatch.setattr(synthetic, "solve_model", fail_final)
    summary = synthetic._replay_strategy(replay_config, fixture, "oracle", tmp_path / "fallback")
    ledger = pd.read_csv(tmp_path / "fallback/ledger.csv")
    final = fixture["schedule"][-1]
    np.testing.assert_allclose(ledger.loc[ledger.auction == final.auction, "quantity_mwh"], 0)
    assert summary["infeasible_auction_decisions"][0]["fallback_order_policy"] == "zero_new_trades"
    timing = json.loads((tmp_path / "fallback/optimization_timing.json").read_text())
    assert timing["solve_count"] == len(fixture["schedule"])
    assert timing["solves"][-1]["status"] == "infeasible"
    assert timing["missing_runtime_count"] == 1  # This injected failure has no native Runtime.
    forecasts = pd.read_csv(tmp_path / "fallback/pv_forecast_inputs.csv")
    assert set(forecasts.auction) == {event.auction for event in fixture["schedule"]}
    plans = pd.read_csv(tmp_path / "fallback/expected_dispatch_plans.csv")
    prior = plans.loc[plans.auction == fixture["schedule"][-2].auction].set_index("interval_id")
    retained = plans.loc[plans.auction == final.auction].set_index("interval_id")
    np.testing.assert_allclose(retained[["charge_mw", "discharge_mw", "curtailment_mw"]],
        prior.loc[retained.index, ["charge_mw", "discharge_mw", "curtailment_mw"]])


def test_sequential_refreshes_pv_and_settles_realized_feedback(replay_config, tmp_path):
    from bess_pv_trading.data import ForecastBundle
    reference = load_config("tests/fixtures/configs/battery_cases.yaml")["full_replay"]
    fixture = synthetic.synthetic_inputs(replay_config)
    fixture["auction_planning_mode"] = "current_auction_only"
    fixture["realized_imbalance_prices_eur_per_mwh"] = np.full(len(fixture["grid"].interval_ids),
        reference["actual_imbalance_price_eur_per_mwh"])
    calls = []

    def refresh(cutoff, starts):
        auction = next(event.auction for event in fixture["schedule"] if event.submission_at == cutoff)
        first = pd.DatetimeIndex(fixture["grid"].starts).get_loc(starts[0])
        pv = fixture["pv"][first:] * reference["rolling_pv"]["factors_by_auction"][auction]
        calls.append(auction)
        return ForecastBundle(fixture["plant"].farm_id,
            cutoff - pd.Timedelta(hours=reference["rolling_pv"]["availability_lag_hours"]), cutoff,
            pd.DataFrame({"pv_ac_mw": pv}, index=starts), {"pv_ac_mw": "MW_AC"})

    fixture["pv_forecast_provider"] = refresh
    summary = synthetic._replay_strategy(replay_config, fixture, "oracle", tmp_path / "rolling")
    forecasts = pd.read_csv(tmp_path / "rolling/pv_forecast_inputs.csv")
    dispatch = pd.read_csv(tmp_path / "rolling/dispatch.csv")
    ledger = pd.read_csv(tmp_path / "rolling/ledger.csv")
    assert calls == [event.auction for event in fixture["schedule"]]
    for auction, frame in forecasts.groupby("auction"):
        first = pd.DatetimeIndex(fixture["grid"].starts).get_loc(pd.Timestamp(frame.interval_start_utc.iloc[0]))
        np.testing.assert_allclose(frame.pv_ac_forecast_mw,
            fixture["pv"][first:] * reference["rolling_pv"]["factors_by_auction"][auction])
    np.testing.assert_allclose(dispatch.energy_start_mwh.iloc[1:], dispatch.energy_end_mwh.iloc[:-1])
    expected = ledger.cashflow_eur.sum() + (dispatch.imbalance_mwh * reference["actual_imbalance_price_eur_per_mwh"]).sum()
    expected -= dispatch.degradation_eur.sum() + dispatch.import_fee_eur.sum() + dispatch.transaction_fee_eur.sum()
    assert summary["realized_profit_after_imbalance_eur"] == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["integrated", "current_auction_only"])
def test_ida3_models_only_its_delivery_suffix_and_preserves_earlier_commands(replay_config, tmp_path, monkeypatch, mode):
    """The cutoff controls information; eligibility controls every free interval."""
    from bess_pv_trading.data import ForecastBundle
    reference = load_config("tests/fixtures/configs/battery_cases.yaml")["full_replay"]["rolling_pv"]
    fixture = synthetic.synthetic_inputs(replay_config)
    fixture["auction_planning_mode"] = mode
    captured = []
    build = synthetic.build_model

    def capture(inputs):
        captured.append(inputs)
        return build(inputs)

    def refresh(cutoff, starts):
        event = next(event for event in fixture["schedule"] if event.submission_at == cutoff)
        first = pd.DatetimeIndex(fixture["grid"].starts).get_loc(starts[0])
        values = fixture["pv"][first:] * reference["factors_by_auction"][event.auction]
        return ForecastBundle(fixture["plant"].farm_id,
            cutoff - pd.Timedelta(hours=reference["availability_lag_hours"]), cutoff,
            pd.DataFrame({"pv_ac_mw": values}, index=starts), {"pv_ac_mw": "MW_AC"})

    fixture["pv_forecast_provider"] = refresh
    monkeypatch.setattr(synthetic, "build_model", capture)
    synthetic._replay_strategy(replay_config, fixture, "oracle", tmp_path / mode)
    solves = json.loads((tmp_path / mode / "solves.json").read_text())
    timing = json.loads((tmp_path / mode / "optimization_timing.json").read_text())
    assert timing["solve_count"] == len(solves)
    np.testing.assert_allclose([row["runtime_seconds"] for row in timing["solves"]],
        [row["runtime_seconds"] for row in solves])
    dispatch = pd.read_csv(tmp_path / mode / "dispatch.csv")
    plans = pd.read_csv(tmp_path / mode / "expected_dispatch_plans.csv")
    ledger = pd.read_csv(tmp_path / mode / "ledger.csv")
    event = fixture["schedule"][-1]
    prior_event = fixture["schedule"][-2]
    final = captured[-1]
    first = min(event.eligible_intervals)
    boundary = pd.Timestamp(fixture["grid"].starts[first])
    assert list(final.grid.interval_ids) == [fixture["grid"].interval_ids[t] for t in event.eligible_intervals]
    assert final.state.timestamp == boundary
    assert set(final.auction_eligibility) == {event.auction}
    assert all(final.auction_eligibility[event.auction])
    assert solves[-1]["trade_variable_count"] == len(event.eligible_intervals)
    assert solves[-1]["physical_planning_intervals"] == len(event.eligible_intervals)
    assert pd.Timestamp(solves[-1]["cutoff"]) == event.submission_at
    assert pd.Timestamp(solves[-1]["pv_forecast_available_at_utc"]) == event.submission_at
    assert pd.Timestamp(solves[-1]["planning_start_utc"]) == boundary
    assert dispatch.loc[:first-1, "active_plan_auction"].eq(prior_event.auction).all()
    assert dispatch.loc[first:, "active_plan_auction"].eq(event.auction).all()
    prior = plans.loc[plans.auction == prior_event.auction].set_index("interval_id")
    commands = ["charge_mw", "discharge_mw", "curtailment_mw"]
    np.testing.assert_allclose(dispatch.iloc[:first][[f"requested_{name}" for name in commands]],
        prior.loc[dispatch.interval_id.iloc[:first], commands])
    np.testing.assert_allclose(dispatch.net_position_mwh.iloc[:first],
        prior.loc[dispatch.interval_id.iloc[:first], "planned_net_position_mwh"])
    assert pd.to_datetime(ledger.loc[ledger.auction == event.auction, "interval_id"], utc=True).min() == boundary
    assert pd.to_datetime(plans.loc[plans.auction == event.auction, "interval_start_utc"], utc=True).min() == boundary
    # Forecast the locked prefix without replacing commands or consuming final PV.
    prefix = solves[-1]["locked_prefix_projection"]
    assert prefix[-1]["energy_end_mwh"] == pytest.approx(final.state.energy_mwh)
    assert pd.Timestamp(prefix[-1]["end"]) == boundary
    assert pd.Timestamp(prefix[0]["start"]) < event.submission_at < boundary
    for step in prefix:
        index = fixture["grid"].interval_ids.index(step["interval_id"])
        assert step["plan_source_auction"] == prior_event.auction
        assert step["pv_ac_mw"] == pytest.approx(fixture["pv"][index] * reference["factors_by_auction"][event.auction])
        np.testing.assert_allclose([step[f"requested_{name}"] for name in commands],
            prior.loc[step["interval_id"], commands].to_numpy(dtype=float))
        assert step["grid_target_mw"] == pytest.approx(dispatch.grid_target_mw.iloc[index])

"""Small deterministic verification of information, physics and monetary-risk contracts."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bess_pv_trading.domain import (BatteryParameters, DegradationSettings, DeliveryGrid,
    PhysicalState, PlantParameters, SolverSettings)
from bess_pv_trading.optimization.mpc import (ForecastPrices, MpcInput, MpcSettings,
    TerminalValueCurve, solve_mpc, validate_mpc_plan)
from bess_pv_trading.optimization.adaptive_mpc import control_history_nodes, weighted_loss_cvar


def case(pv, commitments, *, energy=0.25, policy="ensemble_mpc_adaptive", prices=None, settings=None):
    """Construct an exact quarter-hour fixture, independent of study data and training."""
    pv = np.asarray(pv, dtype=float)
    start = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    starts = tuple(start + timedelta(minutes=15 * t) for t in range(pv.shape[1]))
    grid = DeliveryGrid(tuple(stamp.isoformat() for stamp in starts), starts,
        tuple(stamp + timedelta(minutes=15) for stamp in starts))
    battery = BatteryParameters(capacity_mwh=1, max_power_mw=1, converter_charge_efficiency=1,
        converter_discharge_efficiency=1, energy_min_mwh=0, energy_max_mwh=1)
    forecast = None if prices is None else ForecastPrices(np.asarray(prices, dtype=float), start)
    return MpcInput(start, grid, PlantParameters(1, 1, 1, battery), PhysicalState(start, energy),
        np.asarray(commitments, dtype=float), pv, tuple(f"m{s}" for s in range(len(pv))),
        np.full(len(pv), 1 / len(pv)), DegradationSettings(0), policy=policy,
        settings=settings or MpcSettings(terminal_inventory_mode="none"), price_forecast=forecast)


def checked(inputs):
    """A licensed successful solve is required; solver failures never become skips."""
    result = solve_mpc(inputs, SolverSettings(threads=1, feasibility_tolerance=1e-8,
        integrality_tolerance=1e-8))
    assert result.plan is not None, result.diagnostics
    validate_mpc_plan(inputs, result.plan)
    return result


def test_recourse_observes_only_finished_intervals_and_never_member_labels():
    inputs = case([[0, 0, 1], [0, 1, 0]], [0, 0, 0])
    nodes = control_history_nodes(inputs)
    assert nodes[0, 0] == nodes[1, 0]
    assert nodes[0, 1] == nodes[1, 1]
    assert nodes[0, 2] != nodes[1, 2]
    delayed = replace(inputs, settings=replace(inputs.settings, recourse_observation_lag_intervals=2))
    assert np.unique(control_history_nodes(delayed)[:, 2]).size == 1
    restricted = replace(inputs, settings=replace(inputs.settings, recourse_stages=()))
    assert all(np.unique(control_history_nodes(restricted)[:, t]).size == 1 for t in range(3))


def test_adaptive_future_controls_improve_correlated_paths_with_a_shared_root():
    inputs = case([[0, 0, 1], [1, 1, 0]], [0, 0.25, 0.25])
    adaptive = checked(inputs)
    common = checked(replace(inputs, policy="ensemble_mpc"))
    np.testing.assert_allclose(adaptive.plan.scenario_actions[0, 0], adaptive.plan.scenario_actions[1, 0])
    assert adaptive.plan.components["absolute_imbalance_mwh"] < 1e-7
    assert common.plan.components["absolute_imbalance_mwh"] > 0.1
    assert adaptive.plan.scenario_actions[0, 1, 1] == pytest.approx(1)
    assert adaptive.plan.scenario_actions[1, 2, 1] == pytest.approx(1)
    assert adaptive.diagnostics["representative_path_role"] == "shadow_projection_only"
    assert adaptive.diagnostics["model_variables"] > 0
    assert adaptive.diagnostics["model_binary_variables"] == adaptive.diagnostics["control_node_count"]
    assert adaptive.diagnostics["model_constraints"] > 0


def test_adaptive_validator_rejects_hidden_anticipative_commands():
    inputs = case([[0, 0, 1], [0, 1, 0]], [0, 0, 0])
    plan = checked(inputs).plan
    plan.scenario_actions[1, 0, 2] = 0.5
    with pytest.raises(ValueError, match="nonanticipativity"):
        validate_mpc_plan(inputs, plan)


@pytest.mark.parametrize("policy", ["ensemble_mpc_economic", "ensemble_mpc_adaptive"])
def test_battery_state_is_shared_exactly_by_common_control_history(policy):
    """Shared state preserves the optimized physical plan across scenario paths."""
    from bess_pv_trading.optimization.adaptive_mpc import build_research_model

    inputs = case([[0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 0, 1]], [0, 0, 0, 0],
        prices=np.full((3, 4), 20.0), policy=policy)
    model, variables, _ = build_research_model(inputs)
    try:
        nodes = variables["nodes"]
        state = variables["energy"]
        assert state[0, 0] is state[1, 0] is state[2, 0]
        for t in range(nodes.shape[1]):
            for first in range(len(nodes)):
                for second in range(len(nodes)):
                    if nodes[first, t] == nodes[second, t]:
                        assert state[first, t + 1] is state[second, t + 1]
        node_count = sum(len(np.unique(nodes[:, t])) for t in range(nodes.shape[1]))
        # No soft terminal value, directional prices or import fees add variables.
        expected = 5 * node_count + 1 + 3 * np.prod(inputs.pv_ac_mw.shape)
        assert model.NumVars == expected
    finally:
        model.dispose()
    result = checked(inputs)
    assert np.isfinite(result.plan.components["absolute_imbalance_mwh"])


def test_economic_forecast_uses_signed_cash_without_legacy_tracking_penalty():
    inputs = case([[0, 0]], [0, 0], energy=0, prices=[[10, 100]], policy="deterministic_mpc_economic")
    result = checked(inputs)
    assert result.plan.actions[0].charge_mw == pytest.approx(1)
    assert result.plan.actions[1].discharge_mw == pytest.approx(1)
    assert result.plan.components["settlement_eur"] == pytest.approx(22.5)
    assert result.diagnostics["planning_objective_value"] == pytest.approx(-22.5)
    assert not result.diagnostics["price_oracle_enabled"]


def test_economic_common_controls_and_directional_prices_have_exact_cash():
    inputs = case([[0, 0], [0, 0]], [0, 0], prices=[[-100, 50], [-50, 100]],
        policy="ensemble_mpc_economic")
    forecasts = replace(inputs.price_forecast, negative_eur_per_mwh=np.array([[-200, 25], [-100, 50]]))
    inputs = replace(inputs, price_forecast=forecasts)
    plan = checked(inputs).plan
    np.testing.assert_allclose(plan.scenario_actions[0], plan.scenario_actions[1])
    expected = np.mean(np.sum(np.maximum(plan.imbalance_mwh, 0) * forecasts.positive_eur_per_mwh
        - np.maximum(-plan.imbalance_mwh, 0) * forecasts.negative_eur_per_mwh, axis=1))
    assert plan.components["settlement_eur"] == pytest.approx(expected)


def test_forecast_availability_alignment_and_disclosure_fail_closed():
    inputs = case([[0]], [0])
    with pytest.raises(ValueError, match="requires an eligible"):
        replace(inputs, policy="ensemble_mpc_economic")
    future = ForecastPrices(np.array([[100.0]]), inputs.cutoff + timedelta(seconds=1))
    with pytest.raises(ValueError, match="after the cutoff"):
        replace(inputs, price_forecast=future)
    wrong = ForecastPrices(np.array([[100.0, 100.0]]), inputs.cutoff)
    with pytest.raises(ValueError, match="align"):
        replace(inputs, price_forecast=wrong)
    valid = ForecastPrices(np.array([[100.0]]), inputs.cutoff)
    with pytest.raises(ValueError, match="research"):
        replace(inputs, policy="ensemble_mpc", price_forecast=valid)
    assert not valid.positive_eur_per_mwh.flags.writeable
    with pytest.raises(ValueError, match="timezone aware"):
        ForecastPrices(np.array([[100.0]]), "2026-01-15")


def test_weighted_loss_cvar_integrates_fractional_tail_mass_and_negative_losses():
    assert weighted_loss_cvar([0, 10, 30], [0.5, 0.4, 0.1], 0.8) == pytest.approx(20)
    assert weighted_loss_cvar([-30, -10], [0.5, 0.5], 0.5) == pytest.approx(-10)


def test_monetary_risk_changes_decision_and_tracks_financial_not_volume_loss():
    # Expected settlement rewards discharge, but the rare negative-price member
    # makes its worst-half monetary loss worse than retaining the initial energy.
    inputs = case([[0], [0]], [0], prices=[[100], [-60]], policy="ensemble_mpc_economic")
    expectation = checked(inputs)
    risk = checked(replace(inputs, settings=replace(inputs.settings,
        monetary_cvar_alpha=0.5, monetary_cvar_weight=1)))
    assert expectation.plan.actions[0].discharge_mw == pytest.approx(1)
    assert risk.plan.actions[0].discharge_mw == pytest.approx(0)
    assert expectation.plan.components["monetary_cvar_eur"] == pytest.approx(15)
    assert risk.plan.components["monetary_cvar_eur"] == pytest.approx(0)
    with pytest.raises(ValueError, match="requires eligible"):
        case([[0]], [0], settings=replace(inputs.settings, monetary_cvar_weight=1))


@pytest.mark.parametrize("policy", ["ensemble_mpc_adaptive", "ensemble_mpc"])
def test_exact_nonconcave_continuation_curve_and_coverage(policy):
    settings = MpcSettings(terminal_inventory_mode="piecewise_linear")
    base = case([[0]], [0], energy=0.5, policy=policy)
    curve = TerminalValueCurve(np.array([0, 0.25, 0.5, 0.75, 1]),
        np.array([0, 20, 0, 30, 0]), base.cutoff)
    inputs = replace(base, settings=settings, terminal_value=curve)
    # With no tracking penalty, the higher nonconcave peak is the chosen endpoint.
    inputs = replace(inputs, settings=replace(settings, planning_imbalance_penalty_eur_per_mwh=0))
    plan = checked(inputs).plan
    assert plan.energy_mwh[-1] == pytest.approx(0.75)
    assert plan.components["terminal_cost_eur"] == pytest.approx(-30)
    with pytest.raises(ValueError, match="operating range"):
        replace(inputs, terminal_value=replace(curve, energy_mwh=np.array([0, 0.2, 0.4, 0.6, 0.8])))
    with pytest.raises(ValueError, match="after the cutoff"):
        replace(inputs, terminal_value=replace(curve, available_at=base.cutoff + timedelta(seconds=1)))


def test_adaptive_xu_memory_and_terminal_constraints_hold_on_all_paths():
    inputs = case([[0, 0, 1], [1, 1, 0]], [0, 0.25, 0.25])
    aging = DegradationSettings(mode="xu_segmented", segment_count=3, replacement_cost_eur=1000)
    result = checked(replace(inputs, degradation=aging, hard_terminal_target_mwh=0.1))
    np.testing.assert_allclose(result.plan.scenario_energy_mwh[:, -1], 0.1, atol=1e-7)
    assert result.plan.components["degradation_eur"] >= 0


def test_infeasible_terminal_returns_explicit_tracking_fallback():
    inputs = case([[0]], [0], energy=0)
    result = solve_mpc(replace(inputs, hard_terminal_target_mwh=1), SolverSettings(threads=1))
    assert result.plan is None
    assert result.status == "infeasible"
    assert result.diagnostics["fallback_applied"]


@pytest.mark.parametrize("values", [{"recourse_observation_lag_intervals": 0},
    {"recourse_stages": [2, 1]}, {"monetary_cvar_alpha": 1}, {"monetary_cvar_weight": -1},
    {"recourse_observation_features": "future_member_identity"}])
def test_research_settings_reject_invalid_information_and_risk(values):
    with pytest.raises(ValueError):
        MpcSettings(**values)

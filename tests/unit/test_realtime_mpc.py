"""Independent MPC physics, information contracts and signed economic objectives."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bess_pv_trading.domain import (BatteryParameters, DegradationSettings, DeliveryGrid,
    PhysicalState, PlantParameters, SolverSettings)
from bess_pv_trading.optimization.mpc import (MpcInput, MpcSettings, OraclePrices,
    solve_mpc, validate_mpc_plan)


def inputs_for(pv, commitments, *, energy=0.25, settings=None, policy="deterministic_mpc", prices=None):
    """A lossless quarter-hour battery isolates allocation and settlement signs."""
    pv = np.asarray(pv, dtype=float)
    start = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    length = pv.shape[1]
    starts = tuple(start + timedelta(minutes=15 * t) for t in range(length))
    grid = DeliveryGrid(tuple(stamp.isoformat() for stamp in starts), starts,
                        tuple(stamp + timedelta(minutes=15) for stamp in starts))
    battery = BatteryParameters(capacity_mwh=1, max_power_mw=1, converter_charge_efficiency=1,
                                converter_discharge_efficiency=1, energy_min_mwh=0, energy_max_mwh=1)
    return MpcInput(start, grid, PlantParameters(1, 1, 1, battery), PhysicalState(start, energy),
        np.asarray(commitments), pv, tuple(f"member_{s}" for s in range(len(pv))),
        np.full(len(pv), 1 / len(pv)), DegradationSettings(0), policy=policy,
        settings=settings or MpcSettings(terminal_inventory_mode="none"),
        oracle=None if prices is None else OraclePrices(np.asarray(prices, dtype=float)))


def checked(inputs):
    """Tests require actual successful licensed solves rather than skipped coverage."""
    result = solve_mpc(inputs, SolverSettings(threads=1, feasibility_tolerance=1e-8, integrality_tolerance=1e-8))
    assert result.plan is not None, result.diagnostics
    validate_mpc_plan(inputs, result.plan, tolerance=1e-6)
    return result


def test_oracle_saves_energy_for_later_expensive_shortfall():
    inputs = inputs_for([[0, 0]], [0.25, 0.25], prices=[10, 100],
                        policy="deterministic_mpc_price_oracle")
    result = checked(inputs)
    assert result.plan.actions[0].discharge_mw == pytest.approx(0, abs=1e-7)
    assert result.plan.actions[1].discharge_mw == pytest.approx(1, abs=1e-7)
    assert result.plan.components["settlement_eur"] == pytest.approx(-2.5)
    assert result.diagnostics["stage1_l_min_mwh"] == pytest.approx(0.25)
    assert result.diagnostics["oracle_additional_imbalance_used_mwh"] < 1e-7
    assert result.plan.components["settlement_eur"] > -25  # Greedy tracking spends energy first.


def test_price_blind_plans_whole_horizon_not_current_interval():
    # Early discharge helps two members but harms the third. Later discharge helps all.
    inputs = inputs_for([[0, 0], [0, 0], [0.8, 0]], [0.1, 0.1], energy=0.1, policy="ensemble_mpc")
    result = checked(inputs)
    assert result.plan.actions[0].discharge_mw == pytest.approx(0, abs=1e-7)
    assert result.plan.actions[1].discharge_mw == pytest.approx(0.4, abs=1e-7)
    assert result.plan.components["absolute_imbalance_mwh"] == pytest.approx(0.2 / 3)


@pytest.mark.parametrize("prices", [[-100, 0], [0, 0], [100, -100]])
def test_negative_zero_and_signed_prices_have_exact_accounting(prices):
    result = checked(inputs_for([[0, 0]], [0.25, 0.25], prices=prices,
                                policy="deterministic_mpc_price_oracle"))
    expected = np.dot(prices, result.plan.imbalance_mwh[0])
    assert result.plan.components["settlement_eur"] == pytest.approx(expected)


def test_budget_bounds_speculation_and_supports_directional_prices():
    settings = MpcSettings(terminal_inventory_mode="none", oracle_absolute_imbalance_budget_mwh=0.05)
    inputs = inputs_for([[0]], [0], energy=0.5, prices=[100],
                        policy="deterministic_mpc_price_oracle", settings=settings)
    result = checked(inputs)
    assert result.plan.components["absolute_imbalance_mwh"] == pytest.approx(0.05)
    assert result.plan.components["settlement_eur"] == pytest.approx(5)
    dual = replace(inputs, oracle=OraclePrices(np.array([-100.0]), np.array([-200.0])))
    result = checked(dual)
    assert result.plan.imbalance_mwh[0, 0] == pytest.approx(-0.05)
    assert result.plan.components["settlement_eur"] == pytest.approx(10)


def test_price_blind_interface_rejects_future_realized_prices():
    inputs = inputs_for([[0]], [0.1])
    with pytest.raises(ValueError, match="explicit price-oracle"):
        replace(inputs, oracle=OraclePrices(np.array([100.0])))
    with pytest.raises(ValueError, match="explicit price-oracle"):
        replace(inputs, policy="deterministic_mpc_price_oracle")
    with pytest.raises(ValueError, match="after the cutoff"):
        replace(inputs, forecast_provenance={"pv_forecast_available_at_utc": "2026-01-16T00:00:00Z"})


def test_current_interval_realized_pv_requires_explicit_disclosure():
    inputs = inputs_for([[0.4, 0.6]], [0.15, 0.15])
    with pytest.raises(ValueError, match="explicit current_pv_information"):
        replace(inputs, current_realized_pv_mw=0.4)
    known = replace(inputs, settings=MpcSettings(terminal_inventory_mode="none",
        current_pv_information="realized_interval_average"), current_realized_pv_mw=0.4)
    result = checked(known)
    assert result.diagnostics["current_interval_pv_oracle_enabled"]
    assert not result.diagnostics["price_oracle_enabled"]
    assert abs(result.plan.expected_absolute_imbalance_by_interval_mwh[0]
               - abs(result.plan.imbalance_mwh[0, 0])) < 1e-9
    with pytest.raises(ValueError, match="first interval"):
        replace(known, current_realized_pv_mw=0.3)


def test_disclosed_retrospective_model_never_admits_future_weather():
    inputs = inputs_for([[0.4]], [0.1])
    provenance = {"pv_forecast_model_available_at_utc": "2026-10-27T00:00:00Z",
                  "pv_forecast_model_retrospective": True,
                  "pv_forecast_available_at_utc": "2026-01-15T06:00:00Z"}
    checked(replace(inputs, forecast_provenance=provenance))
    with pytest.raises(ValueError, match="after the cutoff"):
        replace(inputs, forecast_provenance={**provenance,
            "pv_forecast_available_at_utc": "2026-01-16T06:00:00Z"})
    with pytest.raises(ValueError, match="after the cutoff"):
        replace(inputs, forecast_provenance={**provenance, "pv_forecast_model_retrospective": False})


def test_common_controls_fractional_curtailment_and_weight_validation():
    inputs = inputs_for([[0.2, 0.4], [1.0, 0.8]], [0.05, 0.1], energy=0, policy="ensemble_mpc")
    result = checked(inputs)
    assert len(result.plan.actions) == 2
    controls = np.array([[a.charge_mw, a.discharge_mw, a.pv_utilization_fraction] for a in result.plan.actions])
    np.testing.assert_allclose(result.plan.grid_mw,
        inputs.pv_ac_mw * controls[:, 2] + controls[:, 1] - controls[:, 0], atol=1e-7)
    with pytest.raises(ValueError, match="weights"):
        replace(inputs, weights=np.array([0.5, 0.6]))
    with pytest.raises(ValueError, match="Deterministic"):
        replace(inputs, policy="deterministic_mpc")
    assert not inputs.committed_mwh.flags.writeable


@pytest.mark.parametrize("mode", ["none", "soft_target", "linear_value"])
def test_terminal_modes_and_hard_study_target(mode):
    settings = MpcSettings(terminal_inventory_mode=mode, terminal_inventory_value_eur_per_mwh=50)
    inputs = inputs_for([[0]], [0], energy=0.5, settings=settings)
    result = checked(replace(inputs, hard_terminal_target_mwh=0.6))
    assert result.plan.energy_mwh[-1] == pytest.approx(0.6)
    expected = 0 if mode == "none" else 5 if mode == "soft_target" else -30
    assert result.plan.components["terminal_cost_eur"] == pytest.approx(expected)


def test_self_discharge_efficiencies_and_xu_memory():
    inputs = inputs_for([[0, 0]], [0.1, 0.1], energy=0.5)
    battery = replace(inputs.plant.battery, self_discharge_per_hour=0.01,
                      converter_charge_efficiency=0.9, converter_discharge_efficiency=0.9)
    checked(replace(inputs, plant=replace(inputs.plant, battery=battery)))
    aging = DegradationSettings(mode="xu_segmented", segment_count=3, replacement_cost_eur=1000)
    xu = replace(inputs, degradation=aging)
    checked(xu)
    with pytest.raises(ValueError, match="zero self discharge"):
        replace(xu, plant=replace(inputs.plant, battery=battery))


def test_infeasible_study_target_returns_logged_fallback():
    inputs = inputs_for([[0]], [0], energy=0)
    result = solve_mpc(replace(inputs, hard_terminal_target_mwh=1))
    assert result.plan is None
    assert result.diagnostics["fallback_applied"]
    assert result.status == "infeasible"


def test_invalid_oracle_refinement_keeps_validated_stage1_action(monkeypatch):
    import bess_pv_trading.optimization.mpc as module

    original = module._plan_from_solution
    calls = []

    def invalid_second(*args):
        plan = original(*args)
        calls.append(plan)
        if len(calls) == 2:
            plan.actions[0] = replace(plan.actions[0], charge_mw=999)
        return plan

    monkeypatch.setattr(module, "_plan_from_solution", invalid_second)
    result = checked(inputs_for([[0, 0]], [0.25, 0.25], prices=[10, 100],
                                policy="deterministic_mpc_price_oracle"))
    assert len(calls) == 2
    assert result.plan is calls[0]
    assert result.status == "oracle_refinement_rejected"
    assert not result.diagnostics["fallback_applied"]
    assert result.diagnostics["accepted_action_source"] == "price_blind_stage1"


def test_exhausted_oracle_budget_never_authorizes_uncertified_economics():
    inputs = inputs_for([[0, 0]], [0.25, 0.25], prices=[10, 100],
        policy="deterministic_mpc_price_oracle",
        settings=MpcSettings(terminal_inventory_mode="none", solver_time_limit_seconds=1e-9))
    result = solve_mpc(inputs)
    assert not result.diagnostics.get("oracle_stage2_used", False)
    assert all(row["stage"] != "signed_settlement_economics" for row in result.diagnostics["stages"])


@pytest.mark.parametrize("values", [
    {"solver_time_limit_seconds": 0}, {"oracle_relative_imbalance_budget": -0.1},
    {"terminal_inventory_mode": "hard_midnight"}, {"ensemble_include_deterministic_member": 1},
    {"scenario_weights": {"a": 0.8}}, {"reoptimization_minutes": True}])
def test_invalid_settings_fail(values):
    with pytest.raises(ValueError):
        MpcSettings(**values)

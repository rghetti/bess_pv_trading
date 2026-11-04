"""Currency validation must accumulate row error without hiding real cash errors."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bess_pv_trading.domain import (BatteryParameters, DegradationSettings, DeliveryGrid,
    PhysicalState, PlantParameters, SolverSettings)
from bess_pv_trading.optimization import mpc


def oracle_inputs(length=29, scenarios=1):
    """Late-day signed settlement coefficients reproduce the observed error scale."""
    start = datetime(2026, 1, 15, 15, 45, tzinfo=timezone.utc)
    starts = tuple(start + timedelta(minutes=15 * t) for t in range(length))
    grid = DeliveryGrid(tuple(stamp.isoformat() for stamp in starts), starts,
        tuple(stamp + timedelta(minutes=15) for stamp in starts))
    battery = BatteryParameters(1, 1, energy_min_mwh=0, energy_max_mwh=1)
    return mpc.MpcInput(start, grid, PlantParameters(1, 1, 1, battery), PhysicalState(start, 0.25),
        np.zeros(length), np.zeros((scenarios, length)), tuple(str(s) for s in range(scenarios)),
        np.full(scenarios, 1 / scenarios), DegradationSettings(20),
        policy="ensemble_mpc_price_oracle", settings=mpc.MpcSettings(terminal_inventory_mode="none"),
        oracle=mpc.OraclePrices(np.linspace(100, 300, length)))


def test_currency_allowance_accumulates_horizon_but_not_duplicate_scenario_count():
    inputs = oracle_inputs()
    allowance = mpc._component_validation_allowances(inputs, {"grid": 0}, 1e-8, 1e-8)
    assert allowance["settlement_eur"] == pytest.approx(4e-8 * inputs.oracle.positive_eur_per_mwh.sum())
    assert allowance["settlement_eur"] > 0.000217859
    repeated = mpc._component_validation_allowances(oracle_inputs(scenarios=51), {"grid": 0}, 1e-8, 1e-8)
    assert repeated["settlement_eur"] == pytest.approx(allowance["settlement_eur"])
    expensive = replace(inputs, oracle=mpc.OraclePrices(inputs.oracle.positive_eur_per_mwh * 100))
    changed = mpc._component_validation_allowances(expensive, {"grid": 0}, 1e-8, 1e-8)
    assert changed["absolute_imbalance_mwh"] == allowance["absolute_imbalance_mwh"]
    assert changed["settlement_eur"] == pytest.approx(100 * allowance["settlement_eur"])


@pytest.mark.parametrize("factor, accepted", [(0.95, True), (10, False)])
def test_small_accumulated_currency_error_is_accepted_but_large_error_rejected(monkeypatch, factor, accepted):
    inputs = oracle_inputs()
    allowance = mpc._component_validation_allowances(inputs, {"grid": 0}, 1e-8, 1e-8)["settlement_eur"]
    original = mpc._plan_from_solution
    calls = []
    def rounded_accounting(*args):
        plan = original(*args)
        calls.append(plan)
        if len(calls) == 2:
            plan.components["settlement_eur"] += factor * allowance
        return plan
    monkeypatch.setattr(mpc, "_plan_from_solution", rounded_accounting)
    result = mpc.solve_mpc(inputs, SolverSettings(threads=1, feasibility_tolerance=1e-8,
        integrality_tolerance=1e-8))
    assert result.plan is not None, result.diagnostics
    assert len(calls) == 2
    assert result.diagnostics["oracle_stage2_used"] is accepted
    assert not result.diagnostics["fallback_applied"]
    assert result.plan.components["settlement_eur"] == pytest.approx(0, abs=1e-7)
    if not accepted:
        assert result.status == "oracle_refinement_rejected"


def test_directional_allowance_accounts_for_spread_and_sign_integrality():
    inputs = oracle_inputs(length=1)
    directional = replace(inputs, oracle=mpc.OraclePrices(np.array([100.0]), np.array([-200.0])))
    ordinary = mpc._component_validation_allowances(inputs, {"grid": 1e-8}, 1e-8, 1e-8)
    bound = mpc._component_validation_allowances(directional, {"grid": 1e-8}, 1e-8, 1e-8)
    assert bound["settlement_eur"] > ordinary["settlement_eur"]


def test_oracle_cap_checks_reconstructed_delivery_not_only_model_split_variables(monkeypatch):
    inputs = oracle_inputs(length=2)
    original = mpc._plan_from_solution
    calls = []
    def inaccurate_split(*args):
        plan = original(*args)
        calls.append(plan)
        if len(calls) == 2:
            # Model objective splits still claim zero imbalance. A physically
            # coherent tiny discharge reconstructs a larger signed deviation.
            delta = 4e-6
            plan.actions[0] = replace(plan.actions[0], discharge_mw=delta)
            plan.energy_mwh[1:] -= delta * 0.25 / inputs.plant.battery.eta_discharge
            plan.grid_mw[:, 0] += delta
            plan.imbalance_mwh[:, 0] += delta * 0.25
            plan.expected_absolute_imbalance_by_interval_mwh = inputs.weights @ np.abs(plan.imbalance_mwh)
            plan.expected_shortfall_by_interval_mwh = inputs.weights @ np.maximum(-plan.imbalance_mwh, 0)
        return plan
    monkeypatch.setattr(mpc, "_plan_from_solution", inaccurate_split)
    result = mpc.solve_mpc(inputs, SolverSettings(threads=1, feasibility_tolerance=1e-8,
        integrality_tolerance=1e-8))
    assert result.plan is calls[0], result.diagnostics
    assert result.status == "oracle_refinement_rejected"
    assert "reconstructed imbalance cap" in result.diagnostics["failure_reason"]
    assert not result.diagnostics["fallback_applied"]


def test_equal_interval_supports_share_variables_without_losing_full_scenario_outputs():
    inputs = oracle_inputs(length=4, scenarios=51)
    model, variables, _ = mpc.build_mpc_model(inputs)
    try:
        assert variables["interval_support_count"] == 4
        assert variables["scenario_interval_count"] == 204
        assert variables["grid"][0, 2] is variables["grid"][50, 2]
        assert model.NumVars == 4 * 4 + 5 + 3 * 4
    finally:
        model.dispose()
    solved = mpc.solve_mpc(inputs, SolverSettings(threads=1))
    assert solved.plan is not None, solved.diagnostics
    assert solved.plan.grid_mw.shape == (51, 4)
    np.testing.assert_allclose(solved.plan.grid_mw, np.broadcast_to(solved.plan.grid_mw[0], (51, 4)))
    assert solved.plan.components["absolute_imbalance_mwh"] == pytest.approx(0)


def test_support_deduplication_never_rounds_different_pv_or_drops_zero_weight_constraints():
    inputs = oracle_inputs(length=1, scenarios=2)
    distinct = replace(inputs, pv_ac_mw=np.array([[0], [1e-15]]))
    model, variables, _ = mpc.build_mpc_model(distinct)
    try:
        assert variables["interval_support_count"] == 2
        assert variables["grid"][0, 0] is not variables["grid"][1, 0]
    finally:
        model.dispose()
    robust = replace(inputs, pv_ac_mw=np.array([[0], [2.0]]), weights=np.array([1.0, 0.0]))
    result = mpc.solve_mpc(robust, SolverSettings(threads=1))
    assert result.plan is not None, result.diagnostics
    assert result.plan.grid_mw[1, 0] <= robust.plant.poi_export_mw + 1e-7
    assert result.diagnostics["interval_support_count"] == 2


def test_aliased_directional_settlement_and_import_cost_keep_probability_mass():
    inputs = oracle_inputs(length=2, scenarios=3)
    inputs = replace(inputs, weights=np.array([0.2, 0.3, 0.5]),
        committed_mwh=np.array([-0.1, 0.1]), import_fee_eur_per_mwh=7,
        oracle=mpc.OraclePrices(np.array([100.0, 300.0]), np.array([120.0, 350.0])))
    result = mpc.solve_mpc(inputs, SolverSettings(threads=1))
    assert result.plan is not None, result.diagnostics
    components, _ = mpc.validate_mpc_plan(inputs, result.plan)
    assert result.plan.components == pytest.approx(components)
    single = replace(inputs, pv_ac_mw=inputs.pv_ac_mw[:1], weights=np.ones(1), scenario_ids=("0",))
    reference = mpc.solve_mpc(single, SolverSettings(threads=1))
    assert reference.plan is not None, reference.diagnostics
    assert result.plan.components == pytest.approx(reference.plan.components, abs=1e-6)

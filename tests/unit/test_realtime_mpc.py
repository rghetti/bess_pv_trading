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


@pytest.mark.parametrize("policy", ["deterministic_mpc", "ensemble_mpc",
    "ensemble_mpc_adaptive", "deterministic_mpc_price_oracle"])
def test_provisional_target_preserves_energy_without_changing_binding_orders(policy):
    """A forecast afternoon sale gives otherwise curtailed PV a future use."""
    inputs = inputs_for([[0.5, 0]], [0, 0], energy=0, policy=policy,
                        prices=[20, 30] if policy.endswith("price_oracle") else None)
    inputs = replace(inputs, degradation=DegradationSettings(10))
    planned = replace(inputs, provisional_ida3_mwh=np.array([0, 0.125]))
    baseline, revised = checked(inputs), checked(planned)
    if not policy.endswith("price_oracle"):
        assert baseline.plan.actions[0].charge_mw == pytest.approx(0)
    assert revised.plan.actions[0].charge_mw == pytest.approx(0.5)
    assert revised.plan.actions[1].discharge_mw == pytest.approx(0.5)
    np.testing.assert_array_equal(planned.committed_mwh, [0, 0])
    np.testing.assert_array_equal(planned.planning_position_mwh, [0, 0.125])
    assert revised.plan.components["absolute_imbalance_mwh"] == pytest.approx(0, abs=1e-7)


def test_provisional_targets_are_copied_and_cannot_change_current_commitments():
    """Planning forecasts are immutable and separately typed from submitted trades."""
    inputs = inputs_for([[0, 0]], [0.1, 0.2])
    provisional = np.array([0, -0.15])
    planned = replace(inputs, provisional_ida3_mwh=provisional)
    provisional[1] = 99
    np.testing.assert_allclose(planned.planning_position_mwh, [0.1, 0.05])
    assert not planned.planning_position_mwh.flags.writeable
    with pytest.raises(ValueError, match="current or submitted-only"):
        replace(inputs, provisional_ida3_mwh=np.array([0.1, 0]))
    with pytest.raises(ValueError, match="current or submitted-only"):
        replace(planned, settings=replace(planned.settings, provisional_ida3_mode="submitted_only"))
    with pytest.raises(ValueError, match="shape or values"):
        replace(inputs, provisional_ida3_mwh=np.array([0, np.nan]))


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


@pytest.mark.parametrize("invalid_retry", [False, True])
@pytest.mark.parametrize("separate_budget", [False, True])
def test_oracle_numeric_retry_uses_remaining_stage_budget_and_preserves_stage1(monkeypatch, invalid_retry, separate_budget):
    """A retry must validate its result and never discard the safe first stage."""
    import bess_pv_trading.optimization.mpc as module

    original = module._plan_from_solution
    calls = []

    def invalidate_refinements(*args):
        plan = original(*args)
        calls.append(plan)
        if len(calls) == 2 or (len(calls) == 3 and invalid_retry):
            plan.actions[0] = replace(plan.actions[0], charge_mw=999)
        return plan

    monkeypatch.setattr(module, "_plan_from_solution", invalidate_refinements)
    settings = MpcSettings(terminal_inventory_mode="none", solver_time_limit_seconds=10,
        oracle_stage2_time_limit_seconds=10 if separate_budget else None, oracle_retry_numeric_focus=3,
        oracle_retry_feasibility_tolerance=1e-9)
    inputs = inputs_for([[0, 0]], [0.25, 0.25], prices=[10, 100],
        policy="deterministic_mpc_price_oracle", settings=settings)
    result = checked(inputs)
    assert len(calls) == 3
    assert result.diagnostics["oracle_retry_attempted"]
    assert result.diagnostics["oracle_retry_numeric_focus"] == 3
    assert result.diagnostics["oracle_retry_feasibility_tolerance"] == 1e-9
    stages = result.diagnostics["stages"]
    assert stages[-1]["stage"] == "signed_settlement_economics_numeric_retry"
    assert 0 < stages[-1]["time_limit_seconds"] < 10
    if invalid_retry:
        assert result.plan is calls[0]
        assert result.status == "oracle_refinement_rejected"
    else:
        assert result.plan is calls[2]
        assert result.diagnostics["oracle_stage2_used"]


def test_oracle_retry_cannot_loosen_feasibility_tolerance(monkeypatch):
    """A permissive retry request cannot weaken the original solver tolerance."""
    import bess_pv_trading.optimization.mpc as module

    original = module._plan_from_solution
    calls = []

    def reject_first_refinement(*args):
        plan = original(*args)
        calls.append(plan)
        if len(calls) == 2:
            plan.actions[0] = replace(plan.actions[0], charge_mw=999)
        return plan

    monkeypatch.setattr(module, "_plan_from_solution", reject_first_refinement)
    settings = MpcSettings(terminal_inventory_mode="none", solver_time_limit_seconds=10,
        oracle_stage2_time_limit_seconds=10, oracle_retry_numeric_focus=3,
        oracle_retry_feasibility_tolerance=1e-7)
    inputs = inputs_for([[0, 0]], [0.25, 0.25], prices=[10, 100],
        policy="deterministic_mpc_price_oracle", settings=settings)
    result = checked(inputs)
    assert result.diagnostics["oracle_retry_feasibility_tolerance"] == 1e-8


def test_oracle_numeric_retry_does_not_extend_exhausted_stage(monkeypatch):
    """Slow candidate validation cannot grant another economic-stage budget."""
    import time
    import bess_pv_trading.optimization.mpc as module

    original = module._plan_from_solution
    calls = []

    def reject_after_deadline(*args):
        plan = original(*args)
        calls.append(plan)
        if len(calls) == 2:
            time.sleep(0.2)
            plan.actions[0] = replace(plan.actions[0], charge_mw=999)
        return plan

    monkeypatch.setattr(module, "_plan_from_solution", reject_after_deadline)
    settings = MpcSettings(terminal_inventory_mode="none", solver_time_limit_seconds=10,
        oracle_stage2_time_limit_seconds=0.1, oracle_retry_numeric_focus=3)
    inputs = inputs_for([[0, 0]], [0.25, 0.25], prices=[10, 100],
        policy="deterministic_mpc_price_oracle", settings=settings)
    result = checked(inputs)
    assert len(calls) == 2
    assert result.plan is calls[0]
    assert result.status == "oracle_refinement_rejected"
    assert not result.diagnostics.get("oracle_retry_attempted", False)


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
    {"scenario_weights": {"a": 0.8}}, {"reoptimization_minutes": True},
    {"oracle_retry_numeric_focus": True}, {"oracle_retry_numeric_focus": 4},
    {"oracle_retry_feasibility_tolerance": 1e-9},
    {"oracle_retry_numeric_focus": 3, "oracle_retry_feasibility_tolerance": 1e-10}])
def test_invalid_settings_fail(values):
    with pytest.raises(ValueError):
        MpcSettings(**values)


def test_first_interval_weight_and_lexicographic_priority_change_dispatch():
    """Both requested objectives prefer present tracking when wear is expensive."""
    base = inputs_for([[0.0, 0.0]], [0.1, 0.0], energy=0.5,
        settings=MpcSettings(terminal_inventory_mode='none'))
    base = replace(base, degradation=DegradationSettings(600))
    original = checked(base)
    weighted = checked(replace(base, settings=replace(base.settings,
        first_interval_priority='weighted', first_interval_weight=10.0)))
    strict = checked(replace(base, settings=replace(base.settings,
        first_interval_priority='lexicographic')))
    assert original.plan.expected_absolute_imbalance_by_interval_mwh[0] == pytest.approx(0.1)
    assert weighted.plan.expected_absolute_imbalance_by_interval_mwh[0] == pytest.approx(0, abs=1e-6)
    assert strict.plan.expected_absolute_imbalance_by_interval_mwh[0] == pytest.approx(0, abs=1e-6)
    assert weighted.diagnostics['planning_objective_value'] == pytest.approx(60, abs=1e-4)
    assert strict.diagnostics['first_interval_minimum_certified']
    assert strict.diagnostics['first_interval_refinement_used']
    assert strict.diagnostics['first_interval_incumbent_mwh'] == pytest.approx(0, abs=1e-6)
    assert [stage['stage'] for stage in strict.diagnostics['stages']] == [
        'minimum_first_interval_imbalance', 'horizon_given_first_interval_priority']


def test_strict_first_interval_priority_honors_final_study_constraint():
    """The immediate best action must preserve hard terminal feasibility."""
    settings = MpcSettings(terminal_inventory_mode='none', first_interval_priority='lexicographic')
    base = inputs_for([[0.0]], [0.1], energy=0.5, settings=settings)
    result = checked(replace(base, hard_terminal_target_mwh=0.5))
    assert result.plan.expected_absolute_imbalance_by_interval_mwh[0] == pytest.approx(0.1, abs=1e-6)
    assert result.plan.energy_mwh[-1] == pytest.approx(0.5)
    assert result.diagnostics['first_interval_minimum_certified']


def test_first_interval_priority_settings_reject_conflicting_policies():
    """Avoid silently applying the new objective to oracle or ensemble research MPC."""
    with pytest.raises(ValueError, match='at least one'):
        MpcSettings(first_interval_priority='weighted', first_interval_weight=0.5)
    with pytest.raises(ValueError, match='requires weighted'):
        MpcSettings(first_interval_priority='lexicographic', first_interval_weight=10)
    base = inputs_for([[0.0]], [0.1])
    with pytest.raises(ValueError, match='price-blind imbalance-and-costs'):
        replace(base, policy='ensemble_mpc', settings=replace(base.settings, first_interval_priority='lexicographic'))
    with pytest.raises(ValueError, match='price-blind imbalance-and-costs'):
        replace(base, settings=replace(base.settings, first_interval_priority='weighted',
            first_interval_weight=10, planning_objective='imbalance_only'))


def test_strict_priority_retains_valid_first_action_if_refinement_is_invalid(monkeypatch):
    """A rejected second-stage plan cannot discard the certified first action."""
    import bess_pv_trading.optimization.mpc as mpc

    base = inputs_for([[0.0, 0.0]], [0.1, 0.0], energy=0.5,
        settings=MpcSettings(terminal_inventory_mode='none', first_interval_priority='lexicographic'))
    original = mpc._plan_from_solution
    calls = []

    def fail_refinement(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('synthetic refinement rejection')
        return original(*args, **kwargs)

    monkeypatch.setattr(mpc, '_plan_from_solution', fail_refinement)
    result = solve_mpc(base, SolverSettings(threads=1, feasibility_tolerance=1e-8, integrality_tolerance=1e-8))
    assert result.status == 'first_interval_refinement_rejected'
    assert result.plan is not None
    assert result.diagnostics['first_interval_minimum_certified']
    assert not result.diagnostics['first_interval_refinement_used']
    assert result.diagnostics['accepted_action_source'] == 'first_interval_stage1'
    assert result.plan.expected_absolute_imbalance_by_interval_mwh[0] == pytest.approx(0, abs=1e-6)
    validate_mpc_plan(base, result.plan)


def test_shared_first_ensemble_weight_prioritizes_disclosed_current_pv():
    """Weight ten applies to a common observed first interval, with recourse later."""
    settings = MpcSettings(terminal_inventory_mode="none", recourse_mode="first_action_only",
        current_pv_information="realized_interval_average")
    scenarios = [[0.0, 0.0]] * 25 + [[0.0, 0.8]] * 26
    base = inputs_for(scenarios, [0.1, 0.1], energy=0.5,
        settings=replace(settings, current_pv_information="forecast"), policy="ensemble_mpc_adaptive")
    base = replace(base, settings=settings, current_realized_pv_mw=0.0,
        degradation=DegradationSettings(600))
    original = checked(base)
    weighted = checked(replace(base, settings=replace(settings,
        first_interval_priority="weighted", first_interval_weight=10.0)))
    assert original.plan.expected_absolute_imbalance_by_interval_mwh[0] == pytest.approx(0.1, abs=1e-6)
    assert weighted.plan.expected_absolute_imbalance_by_interval_mwh[0] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(weighted.plan.scenario_actions[0][0], weighted.plan.scenario_actions[50][0])
    assert weighted.plan.control_nodes[0][0] == weighted.plan.control_nodes[50][0]
    assert weighted.plan.control_nodes[0][1] != weighted.plan.control_nodes[50][1]
    expected = 500 * (weighted.plan.components["absolute_imbalance_mwh"]
        + 9 * weighted.plan.expected_absolute_imbalance_by_interval_mwh[0])
    expected += sum(weighted.plan.components[key] for key in ("degradation_eur", "terminal_cost_eur", "import_cost_eur"))
    assert weighted.diagnostics["planning_objective_value"] == pytest.approx(expected, abs=1e-5)

"""Verify local-day cycle budgets against independent AC-energy accounting."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from bess_pv_trading.domain import (BatteryParameters, DegradationSettings, LinearSOHSettings, NaumannSettings,
                                    PhysicalState, PlantParameters, SolverSettings)
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.optimization.daily_cycle_limit import cycle_interval_day, daily_cycle_budget, daily_cycle_excess
from bess_pv_trading.optimization.linear_soh import cycle_loss_per_internal_mwh, initialize_linear_health
from bess_pv_trading.optimization.naumann import initialize_health
from bess_pv_trading.simulation.replay import execute_interval
from test_optimization_contracts import example_inputs


def battery(limit=.5):
    """One-MWh battery with asymmetric combined efficiencies."""
    return BatteryParameters(1., 1., energy_max_mwh=1., efficiency_convention="combined",
        combined_charge_efficiency=.8, combined_discharge_efficiency=.9,
        daily_cycle_limit_efc=limit)


def execute(state, end, charge=0., discharge=0., policy="follow_battery_plan", target=None):
    """Execute one real interval with the configured grid safety limits."""
    return execute_interval(state=state, start=state.timestamp, end=end, pv_ac_mw=0.,
        requested_charge_mw=charge, requested_discharge_mw=discharge,
        requested_curtailment_mw=0., plant=PlantParameters(1, 1., 1., battery()),
        dispatch_policy=policy, grid_target_mw=target)


def test_limit_counts_ac_energy_and_tracks_controller_feedback():
    """Grid tracking cannot spend a budget a previous command consumed."""
    start = datetime(2026, 1, 1, 6, tzinfo=timezone.utc)
    state = PhysicalState(start, .5)
    first, first_record = execute(state, start + timedelta(hours=1), charge=.5)
    assert first.daily_ac_throughput_mwh == pytest.approx(.5)
    second, second_record = execute(first, start + timedelta(hours=2),
                                    policy="track_committed_grid", target=.9)
    assert second_record["discharge_mw"] == pytest.approx(.5)
    assert second.daily_ac_throughput_mwh == pytest.approx(1.)
    assert second_record["grid_tracking_error_mw"] == pytest.approx(-.4)
    assert daily_cycle_budget(second, battery()) == pytest.approx(0.)


@pytest.mark.parametrize("date,hours", [((2026, 3, 29), 23), ((2026, 10, 25), 25)])
def test_local_midnight_resets_on_dst_days(date, hours):
    """Reset at real local midnight, including 23- and 25-hour days."""
    zone = ZoneInfo("Europe/Rome")
    local = datetime(*date, tzinfo=zone)
    start = local.astimezone(timezone.utc)
    final = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), zone).astimezone(timezone.utc)
    assert (final - start).total_seconds() == hours * 3600
    state = PhysicalState(start, .5)
    middle, _ = execute(state, final - timedelta(minutes=15), charge=0.)
    finish, record = execute(middle, final, charge=.1)
    assert record["daily_cycle_ac_throughput_mwh"] == pytest.approx(.025)
    assert finish.daily_ac_throughput_mwh == 0.
    assert finish.daily_cycle_day == final.astimezone(zone).date().isoformat()


def test_spanning_midnight_and_missing_history_are_rejected():
    """Never reset an unknown history or hide an interval across midnight."""
    start = datetime(2026, 1, 1, 23, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="split"):
        cycle_interval_day(start, start + timedelta(days=2), "Europe/Rome")
    with pytest.raises(ValueError, match="history"):
        daily_cycle_budget(PhysicalState(start, .5, origin="simulated_carried_state"), battery())


def test_cap_clips_feedback_and_curtailed_pv_stays_within_poi():
    """Exhausted charging budget must not let surplus PV exceed the POI."""
    start = datetime(2026, 1, 1, 6, tzinfo=timezone.utc)
    plant = PlantParameters(1, 1., 1., battery(0.))
    state = PhysicalState(start, .5)
    _, result = execute_interval(state=state, start=start, end=start + timedelta(hours=1),
        pv_ac_mw=2., requested_charge_mw=0., requested_discharge_mw=0.,
        requested_curtailment_mw=0., plant=plant, dispatch_policy="track_committed_grid",
        grid_target_mw=1.)
    assert result["charge_mw"] == 0.
    assert result["curtailment_mw"] == pytest.approx(1.)
    assert result["grid_mw"] == pytest.approx(1.)


def test_ac_limit_and_internal_health_loss_use_distinct_energy_bases():
    """AC throughput caps operation while the aging model counts internal energy."""
    start = datetime(2026, 1, 1, 6, tzinfo=timezone.utc)
    cell = battery()
    settings = DegradationSettings(mode="linear_soh", replacement_cost_eur=1000.,
        linear_soh=LinearSOHSettings(.01, 6000, 1., .17, .8, 31557600., "Europe/Rome", "test"))
    state = PhysicalState(start, .5)
    next_state, record = execute_interval(state=state, start=start, end=start + timedelta(hours=1),
        pv_ac_mw=0., requested_charge_mw=.5, requested_discharge_mw=0.,
        requested_curtailment_mw=0., plant=PlantParameters(1, 1., 1., cell),
        degradation_settings=settings)
    assert record["daily_cycle_ac_throughput_mwh"] == pytest.approx(.5)
    assert next_state.battery_health.cycle_loss == pytest.approx(
        cycle_loss_per_internal_mwh(cell, settings) * .4)


def test_aged_battery_budget_uses_today_committed_capacity():
    """A 90%-SOH battery has a smaller AC budget than a fresh battery."""
    start = datetime(2026, 1, 1, 6, tzinfo=timezone.utc)
    cell = battery(.5)
    settings = DegradationSettings(mode="linear_soh", replacement_cost_eur=1000.,
        linear_soh=LinearSOHSettings(.01, 6000, 1., .17, .8, 31557600., "Europe/Rome", "test"))
    state = PhysicalState(start, .5)
    health = replace(initialize_linear_health(state, cell, settings), calendar_loss=.1, committed_soh=.9)
    state = replace(state, battery_health=health, daily_cycle_day="2026-01-01", daily_ac_throughput_mwh=.6)
    assert daily_cycle_budget(state, cell) == pytest.approx(.3)
    final, record = execute_interval(state=state, start=start, end=start + timedelta(hours=1),
        pv_ac_mw=0., requested_charge_mw=.5, requested_discharge_mw=0.,
        requested_curtailment_mw=0., plant=PlantParameters(1, 1., 1., cell),
        degradation_settings=settings)
    assert record["charge_mw"] == pytest.approx(.3)
    assert record["daily_cycle_capacity_mwh"] == pytest.approx(.9)
    assert final.daily_ac_throughput_mwh == pytest.approx(.9)


def test_aged_naumann_health_uses_committed_daily_capacity():
    """The detailed health mode supplies the same 90%-SOH cap basis."""
    start = datetime(2026, 1, 1, 6, tzinfo=timezone.utc)
    cell = battery(.5)
    settings = DegradationSettings(mode="naumann_lfp", replacement_cost_eur=1000.,
                                   naumann=NaumannSettings())
    state = PhysicalState(start, .5)
    health = replace(initialize_health(state, cell, settings), calendar_exposure=.01, committed_soh=.9)
    state = replace(state, battery_health=health, daily_cycle_day="2026-01-01", daily_ac_throughput_mwh=.6)
    assert daily_cycle_budget(state, cell) == pytest.approx(.3)


def test_future_day_validation_uses_scenario_forecast_capacity():
    """A forecast capacity drop tightens the following day's AC energy cap."""
    inputs = example_inputs(prices=((0., 0.),), pv=np.zeros((1, 2)), battery=battery(),
                            hours=1., start="2026-01-01T22:00:00+00:00")
    settings = DegradationSettings(mode="linear_soh", replacement_cost_eur=1000.,
        linear_soh=LinearSOHSettings(.01, 6000, 1., .17, .8, 31557600., "Europe/Rome", "test"))
    inputs = replace(inputs, degradation=settings)
    charge = np.array([[0., .9]])
    discharge = np.zeros((1, 2))
    arrays = {"linear_capacity_mwh": np.array([[1., .8, .8]])}
    assert daily_cycle_excess(inputs, charge, discharge, arrays) == pytest.approx(.1)


def test_future_day_validation_uses_each_scenario_capacity():
    """A lower-health scenario has the tighter limit even with common inputs."""
    inputs = example_inputs(prices=((0., 0.), (0., 0.)), pv=np.zeros((2, 2)), battery=battery(),
                            hours=1., start="2026-01-01T22:00:00+00:00")
    settings = DegradationSettings(mode="linear_soh", replacement_cost_eur=1000.,
        linear_soh=LinearSOHSettings(.01, 6000, 1., .17, .8, 31557600., "Europe/Rome", "test"))
    inputs = replace(inputs, degradation=settings)
    arrays = {"linear_capacity_mwh": np.array([[1., .8, .8], [1., .6, .6]])}
    charge = np.array([[0., .7], [0., .9]])
    assert daily_cycle_excess(inputs, charge, np.zeros((2, 2)), arrays) == pytest.approx(.3)


def test_exhausted_budget_does_not_silently_violate_midnight_health():
    """Fail explicitly when no remaining cycle budget can protect faded capacity."""
    start = datetime(2026, 1, 1, 22, 45, tzinfo=timezone.utc)
    plant = PlantParameters(1, 1., 1., battery(0.))
    state = PhysicalState(start, 1.)
    settings = DegradationSettings(mode="linear_soh", replacement_cost_eur=1000.,
        linear_soh=LinearSOHSettings(.01, 6000, 1., .17, .8, 31557600., "Europe/Rome", "test"))
    with pytest.raises(ValueError, match="conflicts with the remaining daily cycle budget"):
        execute_interval(state=state, start=start, end=start + timedelta(minutes=15),
            pv_ac_mw=0., requested_charge_mw=0., requested_discharge_mw=0.,
            requested_curtailment_mw=0., plant=plant, degradation_settings=settings)


@pytest.mark.solver
def test_optimization_accounts_for_carried_daily_budget():
    """The solver and independent validator agree on remaining daily energy."""
    pytest.importorskip("gurobipy")
    inputs = example_inputs(prices=((0., 0.),), pv=np.zeros((1, 2)), battery=battery(),
                            hours=1., start="2026-01-01T10:00:00+00:00")
    state = replace(inputs.state, energy_mwh=.5, daily_cycle_day="2026-01-01",
                    daily_ac_throughput_mwh=.95)
    inputs = replace(inputs, state=state, terminal_target_mwh=.5)
    built = build_model(inputs)
    result = solve_model(built, SolverSettings())
    assert result.has_incumbent
    assert validate_solution(inputs, result).valid
    assert built.model.getConstrByName("daily_cycle_limit[s0,day2026-01-01]") is not None
    owner = inputs.tree.control_nodes[0][0]
    built.model.addConstr(built.variables["charge"][0, owner] >= .1)
    built.model.optimize()
    assert built.model.Status == pytest.importorskip("gurobipy").GRB.INFEASIBLE


@pytest.mark.solver
def test_future_day_health_capacity_keeps_daily_limit_linear():
    """The next-day cap depends on forecast health without quadratic rows."""
    pytest.importorskip("gurobipy")
    inputs = example_inputs(prices=((0., 0.),), pv=np.zeros((1, 2)), battery=battery(),
                            hours=1., start="2026-01-01T22:00:00+00:00")
    settings = DegradationSettings(mode="linear_soh", replacement_cost_eur=1000.,
        linear_soh=LinearSOHSettings(.01, 6000, 1., .17, .8, 31557600., "Europe/Rome", "test"))
    inputs = replace(inputs, degradation=settings, terminal_target_mwh=.5)
    built = build_model(inputs)
    result = solve_model(built, SolverSettings())
    assert result.has_incumbent and validate_solution(inputs, result).valid
    assert result.arrays["linear_capacity_mwh"][0, 1] < result.arrays["linear_capacity_mwh"][0, 0]
    assert built.model.getConstrByName("daily_cycle_limit[s0,day2026-01-02]") is not None
    assert built.model.NumQConstrs == 0

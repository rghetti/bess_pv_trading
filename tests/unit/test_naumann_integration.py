"""Small contract checks for health-aware auction and MPC model integration."""

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from bess_pv_trading.domain import (BatteryParameters, DegradationSettings, DeliveryGrid,
    NaumannSettings, PhysicalState, PlantParameters, SolverSettings)
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.optimization.mpc import MpcInput, MpcSettings, solve_mpc, validate_mpc_plan
from bess_pv_trading.optimization.naumann import initialize_health
from bess_pv_trading.optimization.prepared import load_prepared_inputs, terminal_minimum_energy_mwh
from bess_pv_trading.config import ConfigurationError, load_config
from bess_pv_trading.simulation.gme_oracle import _soc_energy_targets
from test_optimization_contracts import example_inputs, without_new_trades
from test_prepared_optimization import prepared_config


def health_settings():
    """Use a small declared numerical grid for deterministic software fixtures."""
    return DegradationSettings(mode="naumann_lfp", replacement_cost_eur=1000,
        naumann=NaumannSettings(depth_knots=(0, .25, .5, .75, 1),
            c_rate_bin_width=.25, sqrt_loss_error_budget=1e-4,
            calendar_annual_soh_error_budget=1e-3))


def unit_battery():
    """Keep AC/internal conversion exact while retaining a nonzero SOC minimum."""
    return BatteryParameters(1, .5, energy_min_mwh=.2, energy_max_mwh=.9,
        converter_charge_efficiency=1, converter_discharge_efficiency=1)


def fresh_state(start, energy=.5):
    """Fresh initialization is explicit; carried operation cannot infer history."""
    return PhysicalState(start, energy, origin="known_delivery_start_assumption")


def require_solver():
    """Skip only an unavailable pre-existing solver or license."""
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model() as model:
            model.Params.OutputFlag = 0
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")


def mpc_inputs(*, policy="deterministic_mpc", terminal_mode="none", aged=False):
    """Two quarter-hours cross one Italy delivery-day boundary without market data."""
    start = datetime(2026, 1, 15, 22, 45, tzinfo=timezone.utc)
    starts = (start, start + timedelta(minutes=15))
    grid = DeliveryGrid(("a", "b"), starts,
        tuple(stamp + timedelta(minutes=15) for stamp in starts))
    battery = unit_battery()
    settings = health_settings()
    state = fresh_state(start, .175 if aged else .5)
    if aged:
        # A valid aged SOC lies below the original nominal lower-energy bound.
        health = replace(initialize_health(state, battery, settings),
            calendar_exposure=.15 ** 2, committed_soh=.85)
        state = replace(state, battery_health=health, origin="simulated_carried_state")
    count = 2 if policy == "ensemble_mpc_adaptive" else 1
    return MpcInput(start, grid, PlantParameters(1, 1, 1, battery), state,
        np.zeros(2), np.zeros((count, 2)), tuple(f"s{s}" for s in range(count)),
        np.full(count, 1 / count), settings, policy=policy,
        settings=MpcSettings(terminal_inventory_mode=terminal_mode,
            solver_time_limit_seconds=20))


def test_soc_floor_conversion_accepts_explicit_available_capacity():
    """Nominal energy is preserved while an SOC floor follows available energy."""
    battery = replace(unit_battery(), energy_min_mwh=.17, energy_max_mwh=.765)
    floor = terminal_minimum_energy_mwh({"terminal_soc_min_percent": 30}, battery,
        available_capacity_mwh=.85)
    assert floor == pytest.approx(.255)
    assert battery.capacity_mwh == 1


def test_used_battery_soc_conversion_honors_available_and_explicit_nominal_bases():
    """An SOC percentage has a declared capacity basis; absolute MWh takes priority."""
    state = {"initial_soc_percent": 50, "terminal_soc_percent": 30,
        "battery_health": {"committed_soh": .85}}
    assert _soc_energy_targets(state, unit_battery()) == pytest.approx((.425, .255))
    nominal = {**state, "initial_soc_basis": "nominal_capacity"}
    assert _soc_energy_targets(nominal, unit_battery()) == pytest.approx((.5, .3))
    explicit = {**state, "initial_energy_mwh": .31, "terminal_target_mwh": None}
    assert _soc_energy_targets(explicit, unit_battery()) == (.31, None)
    with pytest.raises(ValueError, match="outside battery"):
        _soc_energy_targets({**state, "initial_energy_mwh": .8}, unit_battery())


def test_opt_in_naumann_yaml_loads_without_starting_an_experiment():
    """The real study configuration selects typed health settings explicitly."""
    config = load_config(Path(__file__).resolve().parents[2]
        / "configs/full_experiments/naumann_lfp.yaml")
    assert config["degradation"]["model"] == "naumann_lfp"
    assert config["degradation"]["throughput_lifetime"] is None
    assert config["state"]["initial_soc_basis"] == "available_capacity"
    assert config["state"]["terminal_soc_basis"] == "available_capacity"
    # The selected workflow ends at one midnight and does not restore the removed IDA3 policy.
    assert config["continuous_optimization"]["ida3_next_day_horizon"] is False
    assert config["continuous_optimization"]["ida3_enforce_intermediate_soc"] is False


@pytest.mark.parametrize("overrides,match", [
    ({"degradation": {"naumann": {"c_rate_bin_width": 0}}}, "c_rate_bin_width"),
    ({"degradation": {"naumann": {"reference_temperature_c": 30}}}, "25 C"),
    ({"degradation": {"naumann": {"unknown_parameter": 1}}}, "unknown_parameter"),
    ({"battery": {"self_discharge_per_hour": .001}}, "zero self-discharge"),
])
def test_naumann_config_validation_checks_physical_settings_before_execution(overrides, match):
    """Invalid health settings fail on YAML load, including inherited null costs."""
    path = Path(__file__).resolve().parents[2] / "configs/full_experiments/naumann_lfp.yaml"
    with pytest.raises(ConfigurationError, match=match):
        load_config(path, overrides=overrides)


def test_prepared_relative_targets_keep_percentages_until_forecast_health_is_known(tmp_path):
    """Percent-only targets stay symbolic; explicit null MWh disables each one."""
    config = prepared_config(tmp_path)
    config["degradation"] = {"model": "naumann_lfp", "replacement_cost_eur": 1000,
        "assumptions": "software verification fixture", "naumann": asdict(health_settings().naumann)}
    config["state"].pop("terminal_target_mwh")
    config["state"].update(terminal_soc_percent=55, terminal_soc_min_percent=30,
        terminal_soc_basis="available_capacity")
    inputs = load_prepared_inputs(config)
    assert inputs.terminal_target_mwh is None
    assert inputs.terminal_minimum_mwh is None
    assert inputs.terminal_target_soc_percent == 55
    assert inputs.terminal_minimum_soc_percent == 30
    config["state"].update(terminal_target_mwh=None, terminal_minimum_mwh=None)
    disabled = load_prepared_inputs(config)
    assert disabled.terminal_target_soc_percent is None
    assert disabled.terminal_minimum_soc_percent is None


def test_relative_mpc_target_contract_rejects_absolute_conflicts_and_other_modes():
    """Available-capacity percentages are unambiguous opt-in health contracts."""
    inputs = mpc_inputs()
    with pytest.raises(ValueError, match="not both"):
        replace(inputs, hard_terminal_target_soc_percent=50, hard_terminal_target_mwh=.5)
    with pytest.raises(ValueError, match="require naumann_lfp"):
        replace(inputs, degradation=DegradationSettings(0),
            state=replace(inputs.state, battery_health=None), hard_terminal_target_soc_percent=50)


def test_mpc_accepts_aged_state_without_resetting_exposure_or_depth_memory():
    """Input validation uses the faded box and keeps both supplied bank histories."""
    inputs = mpc_inputs(aged=True)
    assert inputs.state.energy_mwh < inputs.plant.battery.energy_min_mwh
    assert inputs.state.battery_health.committed_soh == .85
    assert inputs.state.battery_health.calendar_exposure == pytest.approx(.0225)
    assert sum(inputs.state.battery_health.discharge_segment_energy_mwh) == pytest.approx(.175)
    with pytest.raises(ValueError, match="explicit health"):
        replace(inputs, state=replace(inputs.state, battery_health=None))


@pytest.mark.solver
def test_auction_idle_health_forecasts_cross_midnight_and_share_scenario_cost_once():
    """Node-owned idle operation fades capacity and books only incremental wear."""
    require_solver()
    inputs = example_inputs(prices=((0, 0), (0, 0)), pv=np.zeros((2, 2)),
        probabilities=(.3, .7), battery=replace(unit_battery(), energy_min_mwh=0), hours=.25,
        start="2026-01-15T22:45:00+00:00")
    settings = health_settings()
    state = fresh_state(inputs.state.timestamp)
    state = replace(state, battery_health=initialize_health(state, unit_battery(), settings))
    inputs = replace(without_new_trades(inputs), state=state,
        plant=replace(inputs.plant, battery=unit_battery()),
        degradation=settings, terminal_target_mwh=.5,
        imbalance_penalty_eur_per_mwh=0)
    built = build_model(inputs)
    # Fix physical idle operation so this checks aging bookkeeping, not a policy.
    for t in range(2):
        built.model.addConstr(built.variables["charge"][t, "root"] == 0)
        built.model.addConstr(built.variables["discharge"][t, "root"] == 0)
    result = solve_model(built, SolverSettings(threads=1, feasibility_tolerance=1e-8,
        integrality_tolerance=1e-8))
    assert result.has_incumbent, result.message
    report = validate_solution(inputs, result, tolerance=1e-6)
    assert report.valid, report.violations
    capacity = result.arrays["naumann_capacity_mwh"]
    assert capacity[0, 0] == pytest.approx(1)
    assert capacity[0, 1] < 1
    assert capacity[0, 2] == pytest.approx(capacity[0, 1])
    np.testing.assert_allclose(capacity[0], capacity[1])
    wear = result.components["battery_degradation_cost"]
    assert wear[0] > 0
    np.testing.assert_allclose(wear, wear[0])
    assert result.objective_eur == pytest.approx(-wear[0])
    assert inputs.state.battery_health.calendar_exposure == 0
    result.arrays["naumann_cycle_exposure"][0, -1] += .001
    assert not validate_solution(inputs, result).valid


@pytest.mark.solver
@pytest.mark.parametrize("policy", ["deterministic_mpc", "ensemble_mpc_adaptive"])
def test_mpc_health_forecasts_and_available_capacity_terminal_soc(policy):
    """Common and conditional controllers retain health without mutating execution."""
    require_solver()
    inputs = mpc_inputs(policy=policy, terminal_mode="hard_target")
    result = solve_mpc(inputs, SolverSettings(threads=1, feasibility_tolerance=1e-8,
        integrality_tolerance=1e-8))
    assert result.plan is not None, result.diagnostics
    plan = result.plan
    components, residuals = validate_mpc_plan(inputs, plan)
    capacity = plan.aging_arrays["naumann_capacity_mwh"]
    energies = (plan.energy_mwh[None, :] if plan.scenario_energy_mwh is None
        else plan.scenario_energy_mwh)
    np.testing.assert_allclose(energies[:, -1], capacity[:, -1] * .5, atol=1e-6)
    assert components["degradation_eur"] > 0
    assert max(residuals.values()) <= 1e-6
    assert inputs.state.battery_health.calendar_exposure == 0
    assert "forecast_replay_error_bound_eur" in plan.aging_diagnostics


@pytest.mark.solver
def test_adaptive_health_follows_conditional_controls_after_the_shared_first_action():
    """Forecast wear follows each allowed control history without updating actual SOH."""
    require_solver()
    inputs = mpc_inputs(policy="ensemble_mpc_adaptive")
    inputs = replace(inputs, committed_mwh=np.array([0, .1]),
        pv_ac_mw=np.array([[0, 0], [0, .4]]),
        settings=replace(inputs.settings, recourse_mode="first_action_only",
            planning_imbalance_penalty_eur_per_mwh=1000))
    result = solve_mpc(inputs, SolverSettings(threads=1, feasibility_tolerance=1e-8,
        integrality_tolerance=1e-8))
    assert result.plan is not None, result.diagnostics
    plan = result.plan
    np.testing.assert_allclose(plan.scenario_actions[0, 0], plan.scenario_actions[1, 0])
    assert plan.scenario_actions[0, 1, 1] > plan.scenario_actions[1, 1, 1]
    exposure = plan.aging_arrays["naumann_cycle_exposure"]
    assert exposure[0, 1] == pytest.approx(exposure[1, 1])
    assert exposure[0, -1] > exposure[1, -1]
    validate_mpc_plan(inputs, plan)
    assert inputs.state.battery_health.cycle_exposure == 0


@pytest.mark.solver
@pytest.mark.parametrize("policy", ["deterministic_mpc", "ensemble_mpc_adaptive"])
def test_explicit_relative_mpc_target_uses_terminal_health_even_without_inventory_penalty(policy):
    """A hard study target survives independently of the MPC inventory objective."""
    require_solver()
    inputs = replace(mpc_inputs(policy=policy), hard_terminal_target_soc_percent=55,
        hard_terminal_minimum_soc_percent=52)
    result = solve_mpc(inputs, SolverSettings(threads=1, feasibility_tolerance=1e-8,
        integrality_tolerance=1e-8))
    assert result.plan is not None, result.diagnostics
    plan = result.plan
    capacity = plan.aging_arrays["naumann_capacity_mwh"]
    energies = (plan.energy_mwh[None, :] if plan.scenario_energy_mwh is None
        else plan.scenario_energy_mwh)
    np.testing.assert_allclose(energies[:, -1], .55 * capacity[:, -1], atol=1e-6)
    assert np.all(np.abs(energies[:, -1] - .55 * capacity[:, 0]) > 1e-6)
    validate_mpc_plan(inputs, plan)


@pytest.mark.solver
def test_auction_relative_terminal_and_boundary_targets_follow_capacity_after_midnight():
    """No target is converted to initial-Q MWh before future health is optimized."""
    require_solver()
    inputs = example_inputs(prices=(0, 0), pv=np.zeros((1, 2)),
        battery=replace(unit_battery(), energy_min_mwh=0), hours=.25,
        start="2026-01-15T22:45:00+00:00")
    inputs = replace(without_new_trades(inputs), state=fresh_state(inputs.state.timestamp),
        plant=replace(inputs.plant, battery=unit_battery()), degradation=health_settings(),
        terminal_target_mwh=None, terminal_target_soc_percent=55,
        terminal_minimum_soc_percent=50, target_soc_percent_by_boundary={1: 50},
        minimum_soc_percent_by_boundary={1: 40}, imbalance_penalty_eur_per_mwh=0)
    result = solve_model(build_model(inputs), SolverSettings(threads=1,
        feasibility_tolerance=1e-8, integrality_tolerance=1e-8))
    assert result.has_incumbent, result.message
    report = validate_solution(inputs, result)
    assert report.valid, report.violations
    capacity = result.arrays["naumann_capacity_mwh"]
    energy = result.arrays["energy_mwh"]
    assert energy[0, 1] == pytest.approx(.5 * capacity[0, 1], abs=1e-6)
    assert energy[0, -1] == pytest.approx(.55 * capacity[0, -1], abs=1e-6)
    assert abs(energy[0, -1] - .55 * capacity[0, 0]) > 1e-6

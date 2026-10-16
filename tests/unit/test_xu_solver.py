"""Licensed Xu MILP checks against cycle equations and independent replay."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bess_pv_trading.domain import PhysicalState, SolverSettings
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.optimization.degradation import evaluate_xu_dispatch, rainflow_discharge_depths
from test_optimization_contracts import example_inputs
from test_xu_degradation import battery, settings

pytestmark = pytest.mark.solver


@pytest.fixture(autouse=True)
def licensed_gurobi():
    """Solver checks explicitly require the pre-existing licensed environment."""
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model() as model:
            model.Params.OutputFlag = 0
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")


def solve_forced(energy, *, eta_charge=1, eta_discharge=1, count=4):
    """Fix aggregate commands from an energy path, leaving segment allocation free."""
    energy = np.asarray(energy)
    inputs = example_inputs(prices=np.zeros(len(energy)-1), pv=np.zeros((1, len(energy)-1)),
                            battery=battery(eta_charge, eta_discharge), hours=2)
    inputs = replace(inputs, state=replace(inputs.state, energy_mwh=float(energy[0])),
        terminal_target_mwh=float(energy[-1]), degradation=replace(settings(), segment_count=count),
        imbalance_penalty_eur_per_mwh=0, trade_limit_mwh=0)
    built = build_model(inputs)
    for t, change in enumerate(np.diff(energy)):
        built.model.addConstr(built.variables["charge"][t, "root"] == max(change, 0) / (2*eta_charge))
        built.model.addConstr(built.variables["discharge"][t, "root"] == max(-change, 0) * eta_discharge / 2)
    result = solve_model(built, SolverSettings())
    assert result.status == "optimal", result
    report = validate_solution(inputs, result)
    assert report.valid, report.violations
    return inputs, result


def test_fixed_nested_cycles_match_rainflow_reference_and_independent_evaluator():
    energy = np.array([1, .6, .8, .4, .7, 0])
    inputs, result = solve_forced(energy, eta_charge=.9, eta_discharge=.8, count=10)
    # The .2, .3 nested cycles and full discharge depth 1 are independent
    # aggregate rainflow counts, unrelated to the optimizer's segment variables.
    depths = rainflow_discharge_depths(energy, 1)
    assert result.components["battery_degradation_cost"][0] == pytest.approx(1000*.01*sum(depths**2))
    independent = evaluate_xu_dispatch(battery=inputs.plant.battery, settings=inputs.degradation,
        state=inputs.state, charge_mw=result.arrays["charge_mw"][0],
        discharge_mw=result.arrays["discharge_mw"][0], delta_hours=inputs.grid.delta_hours)
    np.testing.assert_allclose(result.arrays["xu_segment_energy_mwh"][0], independent["segment_energy_mwh"], atol=1e-8)
    assert result.components["battery_degradation_cost"][0] == pytest.approx(independent["degradation_eur"])


def test_depth_marginal_cost_selects_only_profitable_shallow_segments():
    inputs = example_inputs(prices=(0, 10), pv=np.zeros((1, 2)), battery=battery())
    inputs = replace(inputs, degradation=settings())
    result = solve_model(build_model(inputs), SolverSettings())
    assert result.status == "optimal", result
    assert validate_solution(inputs, result).valid
    assert result.arrays["charge_mw"][0, 0] == pytest.approx(.5)
    assert result.arrays["discharge_mw"][0, 1] == pytest.approx(.5)
    assert result.components["battery_degradation_cost"][0] == pytest.approx(2.5)
    assert result.objective_eur == pytest.approx(2.5)
    np.testing.assert_allclose(result.arrays["xu_segment_energy_mwh"][0, 1], [.25, .25, 0, 0], atol=1e-8)
    assert result.diagnostics["relative_mip_gap_tolerance"] == 1e-4
    assert result.diagnostics["absolute_mip_gap_configured"] is False


def test_identical_two_leaf_histories_share_segment_commands_and_cost():
    inputs = example_inputs(prices=((0, 10), (0, 10)), probabilities=(.3, .7),
                            pv=np.zeros((2, 2)), battery=battery())
    inputs = replace(inputs, degradation=settings())
    built = build_model(inputs)
    assert len(built.variables["xu_segment_charge"]) == 2 * 4
    assert len(built.variables["xu_segment_discharge"]) == 2 * 4
    result = solve_model(built, SolverSettings())
    assert result.status == "optimal"
    assert validate_solution(inputs, result).valid
    for name in ("xu_segment_charge_mw", "xu_segment_discharge_mw", "xu_segment_energy_mwh"):
        np.testing.assert_allclose(result.arrays[name][0], result.arrays[name][1])
    assert result.objective_eur == pytest.approx(2.5)


def test_residual_solve_starts_from_segment_memory_without_rebooking_aging():
    initial_inputs, initial_result = solve_forced([1, .5])
    base = example_inputs(prices=(0,), pv=np.zeros((1, 1)), battery=battery())
    state = PhysicalState(base.grid.starts[0], .5, .5, 2.5, "simulated_carried_state",
                          tuple(initial_result.arrays["xu_segment_energy_mwh"][0, -1]))
    inputs = replace(base, state=state, terminal_target_mwh=.25, degradation=settings(),
                     imbalance_penalty_eur_per_mwh=0, trade_limit_mwh=0)
    result = solve_model(build_model(inputs), SolverSettings())
    assert result.status == "optimal"
    assert validate_solution(inputs, result).valid
    # It must continue through the third segment (12.5 EUR/MWh), rather than
    # initializing cheap shallow segments again from the aggregate .5 MWh.
    assert result.components["battery_degradation_cost"][0] == pytest.approx(3.125)
    assert result.objective_eur == pytest.approx(-3.125)
    np.testing.assert_allclose(result.arrays["xu_segment_energy_mwh"][0, -1], [0, 0, 0, .25], atol=1e-8)


def test_validator_detects_priority_state_and_aging_cost_tampering():
    inputs, result = solve_forced([0, .5, 0])
    result.arrays["xu_segment_energy_mwh"][0, 1] = [.25, 0, .25, 0]
    report = validate_solution(inputs, result)
    assert not report.valid
    assert report.max_errors["xu_priority_energy_mwh_s0"] == pytest.approx(.25)
    _, result = solve_forced([0, .5, 0])
    result.components["battery_degradation_cost"][0] += 1
    report = validate_solution(inputs, result)
    assert not report.valid
    assert report.max_errors["component_battery_degradation_cost_eur"] == pytest.approx(1)


def test_auction_replay_preserves_xu_state_through_committed_boundary_projection(tmp_path):
    from bess_pv_trading.config import load_config
    from bess_pv_trading.domain import DeliveryGrid
    from bess_pv_trading.simulation.synthetic import synthetic_inputs, _replay_strategy
    import pandas as pd

    config = load_config(Path(__file__).resolve().parents[2] / "configs/synthetic_xu.yaml")
    config["degradation"].update(replacement_cost_eur=1000, stress_coefficient=.01,
                                   stress_exponent=2, segment_count=4)
    config["synthetic"]["daily_price_amplitude"] = 100
    config["outputs"]["plots"] = False
    fixture = synthetic_inputs(config)
    selected = slice(40, 50)
    full = fixture["grid"]
    fixture["grid"] = DeliveryGrid(full.interval_ids[selected], full.starts[selected], full.ends[selected])
    for key in ("pv", "base"):
        fixture[key] = fixture[key][selected]
    fixture["actual_prices"] = {key: value[selected] for key, value in fixture["actual_prices"].items()}
    fixture["schedule"] = tuple(replace(event, eligible_intervals=tuple(t-selected.start
        for t in event.eligible_intervals if selected.start <= t < selected.stop))
        for event in fixture["schedule"])
    summary = _replay_strategy(config, fixture, "deterministic_integrated", tmp_path / "replay")
    dispatch = pd.read_csv(tmp_path / "replay" / "dispatch.csv")
    initial = PhysicalState(fixture["grid"].starts[0], config["state"]["initial_energy_mwh"])
    reference = evaluate_xu_dispatch(battery=fixture["plant"].battery,
        settings=settings(), state=initial, charge_mw=dispatch.charge_mw.to_numpy(),
        discharge_mw=dispatch.discharge_mw.to_numpy(), delta_hours=fixture["grid"].delta_hours)
    assert summary["components"]["degradation_eur"] > 0
    assert summary["components"]["degradation_eur"] == pytest.approx(reference["degradation_eur"])
    assert summary["final_state"]["cumulative_degradation_eur"] == pytest.approx(dispatch.degradation_eur.sum())
    np.testing.assert_allclose(summary["final_state"]["cycle_segment_energy_mwh"], reference["segment_energy_mwh"][-1], atol=1e-7)
    assert summary["terminal_error_mwh"] < 1e-7

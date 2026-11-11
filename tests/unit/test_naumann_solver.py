"""Licensed, fixed-trajectory checks of the Naumann health MILP surrogate."""
from dataclasses import replace

import numpy as np
import pytest

from bess_pv_trading.domain import DegradationSettings, NaumannSettings
from bess_pv_trading.optimization.naumann import (
    calendar_graph, calendar_increment, cycle_rate_factor, depth_exposure,
    evaluate_naumann_dispatch, naumann_segment_parameters, rate_bin, sqrt_graph,
)
from bess_pv_trading.optimization.degradation import rainflow_discharge_depths
from bess_pv_trading.optimization.naumann_milp import (
    _bank_reference,
    build_naumann_degradation, extract_naumann_arrays,
    naumann_degradation_from_arrays, naumann_planner_error_bounds,
    validate_naumann_arrays,
)
from bess_pv_trading.optimization.physical import add_battery_controls
from test_optimization_contracts import example_inputs
from test_xu_degradation import battery

pytestmark = pytest.mark.solver


@pytest.fixture(autouse=True)
def licensed_gurobi():
    """Require the existing installed solver and license without modifying them."""
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model() as model:
            model.Params.OutputFlag = 0
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")


def _solve_path(path, *, hours=.9, start="2026-01-02T10:00:00Z",
                probabilities=(1.,), prior=None, settings=None, parameters=None,
                expect_optimal=True):
    """Fix an independently specified internal-energy path in a small aging MILP."""
    import gurobipy as gp

    path = np.asarray(path, dtype=float)
    count = len(probabilities)
    intervals = len(path) - 1
    settings = settings or DegradationSettings(mode="naumann_lfp",
        replacement_cost_eur=1000., naumann=NaumannSettings())
    parameters = battery() if parameters is None else parameters
    # The shared legacy fixture starts empty; install the requested SOC box and
    # the independently specified starting energy together in the final input.
    fixture_battery = replace(parameters, energy_min_mwh=0.)
    inputs = example_inputs(prices=np.zeros((count, intervals)),
        pv=np.zeros((count, intervals)), probabilities=probabilities,
        battery=fixture_battery, hours=hours, start=start)
    state = replace(inputs.state, energy_mwh=float(path[0])) if prior is None else prior
    inputs = replace(inputs, state=state, degradation=settings,
                     plant=replace(inputs.plant, battery=parameters),
                     terminal_target_mwh=float(path[-1]))
    model = gp.Model("naumann_fixed_path_verification")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-4
    charge = {}
    discharge = {}
    mode = {}
    energy = {}
    # These controls are fixed from energy differences, so this is numerical
    # software verification rather than an economic dispatch experiment.
    for t, change in enumerate(np.diff(path)):
        charge[t, "root"], discharge[t, "root"], mode[t, "root"] = add_battery_controls(
            model, inputs.plant.battery, gp, f"t{t}")
        model.addConstr(charge[t, "root"] == max(0., change) / (hours * parameters.eta_charge))
        model.addConstr(discharge[t, "root"] == max(0., -change) * parameters.eta_discharge / hours)
    for s in range(count):
        for boundary, stored in enumerate(path):
            energy[s, boundary] = model.addVar(lb=float(stored), ub=float(stored),
                name=f"fixed_energy[s{s},boundary{boundary}]")
    variables = {"charge": charge, "discharge": discharge, "mode": mode,
                 "energy": energy}
    wear = build_naumann_degradation(model, inputs, variables, gp)
    model.setObjective(gp.quicksum(float(probabilities[s]) * wear[s]
                                  for s in range(count)), gp.GRB.MINIMIZE)
    model.optimize()
    if expect_optimal:
        assert model.Status == gp.GRB.OPTIMAL
        assert model.NumQConstrs == 0
        assert model.NumQNZs == 0
        arrays = extract_naumann_arrays(inputs, variables)
        arrays.update(energy_mwh=np.tile(path, (count, 1)),
            charge_mw=np.tile(np.maximum(np.diff(path), 0) / (hours * parameters.eta_charge), (count, 1)),
            discharge_mw=np.tile(np.maximum(-np.diff(path), 0) * parameters.eta_discharge / hours, (count, 1)))
        errors = validate_naumann_arrays(inputs, arrays)
        assert max(errors.values()) < 2e-6, errors
        np.testing.assert_allclose(naumann_degradation_from_arrays(inputs, arrays),
            [wear[s].getValue() for s in range(count)], atol=1e-7)
        return inputs, arrays, variables, model
    return inputs, None, variables, model


def test_idle_calendar_aging_is_integrated_and_not_lost_to_solver_tolerance():
    inputs, arrays, _, _ = _solve_path([.8, .8], hours=.25)
    actual = calendar_increment(.8, .8, 900., inputs.degradation)
    _, _, maximum_excess = calendar_graph(inputs.degradation)
    planned = arrays["naumann_calendar_exposure"][0, -1]
    # Exposure is much smaller than the default solver feasibility tolerance;
    # internal scaling must still preserve this strictly positive calendar dose.
    assert 0 < actual < 1e-6
    assert planned >= actual - 1e-13
    assert planned - actual <= 900 * maximum_excess + 1e-12
    assert arrays["naumann_cycle_exposure"][0, -1] == pytest.approx(0., abs=1e-13)
    assert naumann_degradation_from_arrays(inputs, arrays)[0] > 0


def test_complete_cycle_matches_transformed_naumann_depth_and_rate_law():
    inputs, arrays, _, _ = _solve_path([.5, .7, .5])
    _, multiplier = rate_bin(.2 / .9, inputs.degradation)
    expected = depth_exposure(.2, inputs.degradation) * multiplier
    assert arrays["naumann_cycle_exposure"][0, -1] == pytest.approx(expected,
        rel=1e-7, abs=1e-12)
    # Symmetric counting charges half of this full-cycle dose to each direction.
    np.testing.assert_allclose(arrays["naumann_cycle_increment"][0],
                               [expected / 2, expected / 2], atol=1e-12)


def test_nonconvex_depth_marginals_keep_forced_memory_and_match_independent_replay():
    inputs, arrays, _, _ = _solve_path([.5, .3, .55, .4, .5])
    reference = evaluate_naumann_dispatch(battery=inputs.plant.battery,
        settings=inputs.degradation, state=inputs.state,
        charge_mw=arrays["charge_mw"][0], discharge_mw=arrays["discharge_mw"][0],
        delta_hours=inputs.grid.delta_hours,
        calendar_reference_capacity_mwh=inputs.plant.battery.capacity_mwh)
    np.testing.assert_allclose(arrays["naumann_bank_energy_mwh"][0, :, 0],
        reference["discharge_segment_energy_mwh"], atol=1e-9)
    np.testing.assert_allclose(arrays["naumann_bank_energy_mwh"][0, :, 1],
        reference["charge_segment_energy_mwh"], atol=1e-9)
    np.testing.assert_allclose(arrays["naumann_cycle_exposure"][0],
        reference["cycle_exposure"], atol=1e-11)
    bounds = naumann_planner_error_bounds(inputs, arrays)
    assert abs(naumann_degradation_from_arrays(inputs, arrays)[0]
               - reference["degradation_eur"]) <= bounds["wear_eur"][0]


def test_zero_probability_path_shares_banks_rate_bins_and_exposure():
    inputs, arrays, variables, _ = _solve_path([.5, .7, .5], probabilities=(1., 0.))
    count = len(inputs.degradation.naumann.depth_knots) - 1
    # The state has two leaves, while depth transfers remain owned by one root.
    assert len(variables["naumann_bank_in"]) == 2 * 2 * count
    for name in ("naumann_bank_energy_mwh", "naumann_bank_in_mwh",
                 "naumann_bank_out_mwh", "naumann_rate_bin_weight",
                 "naumann_calendar_exposure", "naumann_cycle_exposure",
                 "naumann_calendar_guard_exposure", "naumann_calendar_guard_loss_pwl"):
        np.testing.assert_allclose(arrays[name][0], arrays[name][1], atol=1e-9)


def test_capacity_changes_at_midnight_and_stays_committed_during_the_next_day():
    inputs, arrays, _, _ = _solve_path([.7, .7, .7], hours=1,
                                      start="2026-01-01T22:00:00Z")
    capacity = arrays["naumann_capacity_mwh"][0]
    assert capacity[0] == 1
    assert capacity[1] < capacity[0]
    assert capacity[2] == capacity[1]
    assert arrays["naumann_calendar_exposure"][0, 2] > arrays["naumann_calendar_exposure"][0, 1]
    _, _, error = sqrt_graph(inputs.degradation)
    expected = (1 - arrays["naumann_calendar_loss_pwl"][0, 1]
                  - arrays["naumann_cycle_loss_pwl"][0, 1] - 2 * error)
    assert capacity[1] == pytest.approx(expected)


def test_midnight_capacity_guard_rejects_energy_that_would_need_silent_clipping():
    import gurobipy as gp

    _, _, _, model = _solve_path([.9999, .9999], hours=1,
        start="2026-01-01T22:00:00Z", expect_optimal=False)
    assert model.Status == gp.GRB.INFEASIBLE


def test_validator_detects_carried_memory_and_exposure_tampering():
    inputs, arrays, _, _ = _solve_path([.5, .7, .5])
    changed = {name: values.copy() for name, values in arrays.items()}
    changed["naumann_bank_energy_mwh"][0, 1, 0, 0] += .001
    assert max(validate_naumann_arrays(inputs, changed).values()) >= .001 - 1e-10
    changed = {name: values.copy() for name, values in arrays.items()}
    changed["naumann_calendar_exposure"][0, -1] += .0001
    assert validate_naumann_arrays(inputs, changed)["naumann_calendar_balance"] >= .0001 - 1e-10
    changed = {name: values.copy() for name, values in arrays.items()}
    changed["naumann_calendar_guard_exposure"][0, -1] -= 1e-9
    # A squared-loss error below an ordinary 1e-6 physical tolerance still
    # invalidates the guard.  Its validator residual uses solver micro-units.
    assert validate_naumann_arrays(inputs, changed)["naumann_calendar_guard_balance"] >= .001 - 1e-8


def test_rate_bin_uses_nominal_total_power_and_one_scalar_dose():
    inputs, arrays, variables, _ = _solve_path([.5, .7, .5])
    selected = arrays["naumann_rate_bin"][0]
    upper_rate = (selected + 1) * inputs.degradation.naumann.c_rate_bin_width
    expected = cycle_rate_factor(upper_rate, inputs.degradation) ** 2
    np.testing.assert_allclose(arrays["naumann_cycle_increment"][0],
        arrays["naumann_depth_dose"][0] * expected, atol=1e-12)
    assert len(variables["naumann_depth_dose"]) == 2


def test_internal_energy_and_rate_apply_asymmetric_efficiencies_once():
    _, ideal, _, _ = _solve_path([.5, .7, .5])
    _, lossy, _, _ = _solve_path([.5, .7, .5], parameters=battery(.9, .8))
    np.testing.assert_allclose(lossy["naumann_cycle_exposure"],
                               ideal["naumann_cycle_exposure"], atol=1e-12)
    np.testing.assert_allclose(lossy["naumann_rate_bin"], ideal["naumann_rate_bin"])
    assert lossy["charge_mw"][0, 0] == pytest.approx(.2 / (.9 * .9))
    assert lossy["discharge_mw"][0, 1] == pytest.approx(.2 * .8 / .9)


def test_residual_model_carries_both_banks_and_books_only_new_wear():
    inputs, first, _, _ = _solve_path([.5, .3])
    reference = evaluate_naumann_dispatch(battery=inputs.plant.battery,
        settings=inputs.degradation, state=inputs.state,
        charge_mw=first["charge_mw"][0], discharge_mw=first["discharge_mw"][0],
        delta_hours=inputs.grid.delta_hours)
    carried = reference["final_state"]
    residual, arrays, _, _ = _solve_path([.3, .55, .4], prior=carried,
        start=carried.timestamp.isoformat(), settings=inputs.degradation)
    np.testing.assert_allclose(arrays["naumann_bank_energy_mwh"][0, 0, 0],
        carried.battery_health.discharge_segment_energy_mwh, atol=1e-12)
    np.testing.assert_allclose(arrays["naumann_bank_energy_mwh"][0, 0, 1],
        carried.battery_health.charge_segment_energy_mwh, atol=1e-12)
    assert arrays["naumann_cycle_exposure"][0, 0] == pytest.approx(
        carried.battery_health.cycle_exposure, abs=1e-12)
    future = evaluate_naumann_dispatch(battery=residual.plant.battery,
        settings=residual.degradation, state=carried,
        charge_mw=arrays["charge_mw"][0], discharge_mw=arrays["discharge_mw"][0],
        delta_hours=residual.grid.delta_hours)
    np.testing.assert_allclose(arrays["naumann_cycle_exposure"][0],
        future["cycle_exposure"], atol=1e-12)
    assert abs(naumann_degradation_from_arrays(residual, arrays)[0]
               - future["degradation_eur"]) <= naumann_planner_error_bounds(
                   residual, arrays)["wear_eur"][0]


def test_delivery_interval_cannot_hide_a_midnight_capacity_update():
    with pytest.raises(ValueError, match="split at local midnight"):
        _solve_path([.5, .5], hours=1, start="2026-01-01T22:30:00Z")


def _rainflow_depths_both_directions(path):
    """Count full/downward residues on the path and its complement independently."""
    path = np.asarray(path, dtype=float)
    return np.r_[rainflow_discharge_depths(path, 1.),
                 rainflow_discharge_depths(1 - path, 1.)]


@pytest.mark.parametrize("path", [
    [.5, .3, .6, .4, .5],
    [.413, .207, .581, .304, .449],
])
def test_nonuniform_bank_basis_matches_independent_pwl_rainflow(path):
    inputs, arrays, _, _ = _solve_path(path)
    widths, _ = naumann_segment_parameters(inputs.plant.battery, inputs.degradation)
    knots = np.asarray(inputs.degradation.naumann.depth_knots)
    depths = _rainflow_depths_both_directions(path)
    # Every independent PWL knot increment has a ramp basis.  Equality of the
    # entire vector checks all stress functions, including nonconvex ones, rather
    # than checking only one convenient quadratic stress curve.
    rainflow_basis = .5 * np.sum(np.clip(depths[:, None] - knots[:-1], 0, widths)
                                  / widths, axis=0)
    bank_basis = .5 * arrays["naumann_bank_out_mwh"][0].sum(axis=(0, 1)) / widths
    np.testing.assert_allclose(bank_basis, rainflow_basis, atol=1e-9)
    knot_stress = depth_exposure(knots, inputs.degradation)
    bank_dose = np.dot(bank_basis, np.diff(knot_stress))
    interpolated_rainflow = .5 * np.sum(np.interp(depths, knots, knot_stress))
    assert bank_dose == pytest.approx(interpolated_rainflow, abs=1e-12)


def test_between_knot_analytic_stress_error_converges_under_depth_refinement():
    path = np.array([.413, .207, .581, .304, .449])
    base = NaumannSettings()
    settings = DegradationSettings(mode="naumann_lfp", replacement_cost_eur=1000.,
                                   naumann=base)
    depths = _rainflow_depths_both_directions(path)
    exact = .5 * np.sum(depth_exposure(depths, settings))
    errors = []
    # These are deterministic checks of one fixed analytical path.  Refinement
    # changes only the PWL approximation, without fitting or dispatch experiments.
    for subdivisions in (1, 2, 4, 8):
        knots = np.r_[np.concatenate([np.linspace(left, right, subdivisions,
            endpoint=False) for left, right in zip(base.depth_knots, base.depth_knots[1:])]), 1.]
        refined = replace(settings, naumann=replace(base, depth_knots=tuple(knots)))
        widths, marginal = naumann_segment_parameters(battery(), refined)
        counted = np.zeros(len(widths))
        for trace in (path, 1 - path):
            initial = np.clip(trace[0] - np.r_[0., np.cumsum(widths[:-1])], 0, widths)
            _, _, removed = _bank_reference(initial, np.maximum(np.diff(trace), 0),
                np.maximum(-np.diff(trace), 0), widths)
            counted += .5 * removed.sum(axis=0)
        bank_dose = np.dot(counted, marginal)
        pwl_rainflow = .5 * np.sum(np.interp(depths, knots, depth_exposure(knots, refined)))
        assert bank_dose == pytest.approx(pwl_rainflow, abs=1e-12)
        errors.append(abs(bank_dose - exact))
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < errors[0] / 10


def test_second_midnight_guard_rejects_the_frozen_normalization_counterexample():
    import gurobipy as gp

    parameters = replace(battery(), energy_min_mwh=.07, energy_max_mwh=.93)
    # A frozen-reference-only guard admitted these energies even though exact
    # daily renormalization makes the second midnight ceiling about 7.5e-6 MWh
    # lower.  The independent future-day exposure guard must reject that path.
    _, _, _, model = _solve_path([.93, .9271398305131042, .9259742495242764],
        hours=24, start="2026-01-01T23:00:00Z", parameters=parameters,
        expect_optimal=False)
    assert model.Status == gp.GRB.INFEASIBLE


def test_multi_day_guard_protects_actual_capacity_without_changing_wear_forecast():
    parameters = replace(battery(), energy_min_mwh=.07, energy_max_mwh=.93)
    inputs, arrays, variables, _ = _solve_path([.93, .9271, .9258], hours=24,
        start="2026-01-01T23:00:00Z", parameters=parameters)
    reference = evaluate_naumann_dispatch(battery=parameters,
        settings=inputs.degradation, state=inputs.state,
        charge_mw=arrays["charge_mw"][0], discharge_mw=arrays["discharge_mw"][0],
        delta_hours=inputs.grid.delta_hours)
    guard = arrays["naumann_calendar_guard_exposure"][0]
    forecast = arrays["naumann_calendar_exposure"][0]
    np.testing.assert_allclose(guard[:2], forecast[:2], atol=1e-12)
    assert guard[2] > forecast[2]
    assert guard[2] >= reference["calendar_exposure"][2]
    np.testing.assert_array_less(arrays["naumann_capacity_mwh"][0],
                                reference["available_capacity_mwh"] + 1e-10)
    assert np.all(arrays["energy_mwh"][0]
                  <= .93 * reference["available_capacity_mwh"] + 1e-10)
    # Identical current-day expressions are shared; only genuinely future-day
    # boundaries need an extra guard variable and graph.
    assert variables["naumann_calendar_guard_exposure"][0, 1] is variables["naumann_calendar_exposure"][0, 1]
    assert variables["naumann_calendar_guard_exposure"][0, 2] is not variables["naumann_calendar_exposure"][0, 2]
    value = inputs.degradation.replacement_cost_eur / (1 - inputs.degradation.naumann.end_of_life_soh)
    guard_wear = value * (arrays["naumann_calendar_guard_loss_pwl"][0, -1]
                          + arrays["naumann_cycle_loss_pwl"][0, -1])
    assert naumann_degradation_from_arrays(inputs, arrays)[0] < guard_wear

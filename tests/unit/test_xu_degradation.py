"""Cycle-depth equations and carried memory checked without Gurobi."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bess_pv_trading.domain import BatteryParameters, DegradationSettings, PhysicalState, PlantParameters
from bess_pv_trading.optimization.degradation import (
    degradation_settings_from_config, evaluate_xu_dispatch, initial_segment_energy,
    rainflow_discharge_depths, xu_segment_parameters,
)
from bess_pv_trading.simulation.replay import execute_interval
from test_optimization_contracts import example_inputs
from battery_case_config import case_config, configured_battery


def settings(**overrides):
    """Quadratic stress permits simple hand calculations in EUR."""
    return replace(degradation_settings_from_config(case_config("xu_reference")), **overrides)


def battery(eta_charge=None, eta_discharge=None):
    """Declare internal nameplate energy separately from AC conversion losses."""
    parameters = configured_battery("lossless")
    return replace(parameters,
        combined_charge_efficiency=parameters.eta_charge if eta_charge is None else eta_charge,
        combined_discharge_efficiency=parameters.eta_discharge if eta_discharge is None else eta_discharge)


def trace(energy, *, count=4, eta_charge=1.0, eta_discharge=1.0, exponent=2):
    """Convert an independently specified SoC trajectory into physical AC powers."""
    parameters = battery(eta_charge, eta_discharge)
    aging = replace(settings(), segment_count=count, stress_exponent=exponent)
    change = np.diff(energy)
    state = PhysicalState(datetime(2026, 1, 1, tzinfo=timezone.utc), float(energy[0]))
    return evaluate_xu_dispatch(battery=parameters, settings=aging, state=state,
        charge_mw=np.maximum(change, 0) / eta_charge,
        discharge_mw=np.maximum(-change, 0) * eta_discharge,
        delta_hours=np.ones(len(change)))


def test_original_equation_5_and_efficiency_basis():
    # R=1000, Phi=0.01*depth^2, E=1, J=4, eta_dis=0.8.
    capacity, marginal = xu_segment_parameters(battery(0.9, 0.8), settings())
    np.testing.assert_allclose(capacity, [.25] * 4)
    np.testing.assert_allclose(marginal, [3.125, 9.375, 15.625, 21.875])
    # A full internal discharge sells eta_dis MWh; efficiency cancels the
    # coefficient's inverse eta_dis and charges precisely R*Phi(1)=10 EUR.
    values = trace([0, 1, 0], eta_charge=.9, eta_discharge=.8)
    assert values["degradation_eur"] == pytest.approx(10)
    assert values["segment_charge_mw"].sum() == pytest.approx(1 / .9)
    assert values["segment_discharge_mw"].sum() == pytest.approx(.8)


def test_paper_stress_parameters_and_nameplate_depth():
    paper = DegradationSettings(mode="xu_segmented", replacement_cost_eur=300000,
                                segment_count=10)
    parameters = replace(battery(), capacity_mwh=2, energy_min_mwh=.2, energy_max_mwh=1.8)
    capacity, costs = xu_segment_parameters(parameters, paper)
    assert paper.stress_coefficient == 5.24e-4
    assert paper.stress_exponent == 2.03
    assert sum(capacity) == pytest.approx(2)
    # A .2 MWh segment is 10% of nameplate, regardless of usable energy limits.
    assert costs[0] * .2 == pytest.approx(300000 * 5.24e-4 * .1**2.03)
    assert np.all(np.diff(costs) > 0)


def test_shallow_repeated_cycles_cost_less_than_equal_deep_throughput():
    deep = trace([0, 1, 0])
    shallow = trace([0, .25, 0, .25, 0, .25, 0, .25, 0])
    assert deep["degradation_eur"] == pytest.approx(10)
    assert shallow["degradation_eur"] == pytest.approx(4 * 1000 * .01 * .25**2)
    assert shallow["degradation_eur"] == pytest.approx(deep["degradation_eur"] / 4)
    assert shallow["segment_discharge_mw"].sum() == deep["segment_discharge_mw"].sum()


def test_segments_preserve_cycle_memory_rather_than_absolute_soc_buckets():
    values = trace([1, .5, .75, .5, .25])
    np.testing.assert_allclose(values["segment_energy_mwh"][1], [0, 0, .25, .25])
    np.testing.assert_allclose(values["segment_energy_mwh"][2], [.25, 0, .25, .25])
    np.testing.assert_allclose(values["interval_degradation_eur"], [2.5, 0, .625, 3.125])


def test_rainflow_independent_nested_cycle_reference_at_segment_knots():
    energy = np.array([1, .6, .8, .4, .7, 0])
    depths = rainflow_discharge_depths(energy, 1)
    np.testing.assert_allclose(sorted(depths), [.2, .3, 1])
    values = trace(energy, count=10, exponent=2.03)
    assert values["degradation_eur"] == pytest.approx(1000 * .01 * sum(depths**2.03))


def test_non_knot_cycle_cost_converges_to_independent_rainflow_benchmark():
    energy = np.array([0, .733, .198, .615, 0])
    depths = rainflow_discharge_depths(energy, 1)
    np.testing.assert_allclose(sorted(depths), [.417, .733])
    nonlinear = 1000 * .01 * sum(depths**2)
    coarse, fine, very_fine = [trace(energy, count=count)["degradation_eur"] for count in (4, 64, 512)]
    assert nonlinear <= very_fine < fine < coarse
    assert (very_fine - nonlinear) / nonlinear < 1e-5


def test_charging_half_cycle_has_zero_aging_and_discharge_half_cycle_full_cost():
    assert trace([0, .75])["degradation_eur"] == 0
    assert trace([.75, 0])["degradation_eur"] == pytest.approx(1000 * .01 * .75**2)


def test_executor_carries_history_and_cost_after_safety_adjustment():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    parameters = battery()
    plant = PlantParameters(1, 1, .25, parameters)
    initial = PhysicalState(start, 1)
    first, record = execute_interval(state=initial, start=start, end=start + timedelta(hours=1),
        pv_ac_mw=0, requested_charge_mw=0, requested_discharge_mw=.5,
        requested_curtailment_mw=0, plant=plant, degradation_settings=settings())
    # POI safety discharges .25, so only the first shallow segment ages.
    assert record["discharge_mw"] == pytest.approx(.25)
    assert first.cycle_segment_energy_mwh == pytest.approx((0, .25, .25, .25))
    second, record = execute_interval(state=first, start=first.timestamp,
        end=first.timestamp + timedelta(hours=1), pv_ac_mw=0,
        requested_charge_mw=0, requested_discharge_mw=.25,
        requested_curtailment_mw=0, plant=plant, degradation_settings=settings())
    assert record["degradation_eur"] == pytest.approx(1000 * .01 * (.5**2 - .25**2))
    assert second.cumulative_degradation_eur == pytest.approx(1000 * .01 * .5**2)
    assert second.cumulative_throughput_mwh == pytest.approx(.5)
    assert second.cycle_segment_energy_mwh == pytest.approx((0, 0, .25, .25))


def test_partial_cycle_state_survives_an_arbitrary_split():
    energy = np.array([1, .6, .8, .4, .7, 0])
    whole = trace(energy, count=10)
    initial = PhysicalState(datetime(2026, 1, 1, tzinfo=timezone.utc), float(energy[0]))
    parameters = battery()
    aging = replace(settings(), segment_count=10)
    plant = PlantParameters(1, 1, 1, parameters)
    state = initial
    for change in np.diff(energy):
        state, _ = execute_interval(state=state, start=state.timestamp,
            end=state.timestamp + timedelta(hours=1), pv_ac_mw=0,
            requested_charge_mw=max(change, 0), requested_discharge_mw=max(-change, 0),
            requested_curtailment_mw=0, plant=plant, degradation_settings=aging)
    assert state.cumulative_degradation_eur == pytest.approx(whole["degradation_eur"])
    np.testing.assert_allclose(state.cycle_segment_energy_mwh, whole["segment_energy_mwh"][-1], atol=1e-12)


@pytest.mark.parametrize("overrides", [{"replacement_cost_eur": 0}, {"stress_exponent": .9},
    {"stress_coefficient": 0}, {"segment_count": 0}, {"segment_count": 2.5},
    {"cost_eur_per_mwh_throughput": 2}])
def test_invalid_xu_parameters_rejected(overrides):
    with pytest.raises(ValueError):
        replace(settings(), **overrides)


def test_initialization_and_carried_state_validation_prevent_resets():
    inputs = example_inputs(battery=battery())
    valid = replace(inputs, degradation=settings(), state=replace(inputs.state, energy_mwh=.6))
    np.testing.assert_allclose(initial_segment_energy(valid.state, battery(), settings()), [.25, .25, .1, 0])
    with pytest.raises(ValueError, match="history cannot reset"):
        replace(valid, state=replace(valid.state, cumulative_throughput_mwh=1))
    with pytest.raises(ValueError, match="known_delivery_start_assumption"):
        replace(valid, state=replace(valid.state, origin="carried_from_yesterday"))
    with pytest.raises(ValueError, match="shape or capacity"):
        replace(valid, state=replace(valid.state, cycle_segment_energy_mwh=(.3, .1, .1, .1)))
    with pytest.raises(ValueError, match="sum to"):
        replace(valid.state, cycle_segment_energy_mwh=(0, 0, .1, .1))
    with pytest.raises(ValueError, match="zero self discharge"):
        replace(valid, plant=replace(valid.plant, battery=replace(battery(), self_discharge_per_hour=.001)))
    with pytest.raises(ValueError, match="silently discarded"):
        replace(valid, degradation=DegradationSettings(), state=replace(valid.state, cycle_segment_energy_mwh=(.25, .25, .1, 0)))


def test_config_parser_supports_existing_baseline_and_explicit_xu():
    assert degradation_settings_from_config({"model": "throughput", "cost_eur_per_mwh_throughput": 2}).mode == "constant_throughput"
    parsed = degradation_settings_from_config({"degradation": {"model": "xu_segmented", "replacement_cost_eur": 300000}})
    assert parsed.mode == "xu_segmented"
    assert parsed.segment_count == 16


def throughput_cost_config():
    """Published-cost inputs with an intentionally asymmetric AC loss path."""
    return case_config("replacement_benchmark")


def test_replacement_cost_amortizes_once_over_reference_ac_lifetime():
    config = throughput_cost_config()
    expected = case_config("replacement_expected")
    parsed = degradation_settings_from_config(config)
    assert parsed.replacement_cost_eur == pytest.approx(expected["replacement_cost_eur"])
    ac_energy_per_cycle = expected["ac_throughput_per_cycle_mwh"]
    cycles = config["degradation"]["throughput_lifetime"]["cycle_life"]
    assert parsed.cost_eur_per_mwh_throughput * ac_energy_per_cycle == pytest.approx(expected["cycle_cost_eur"])
    assert parsed.cost_eur_per_mwh_throughput * ac_energy_per_cycle * cycles == pytest.approx(expected["replacement_cost_eur"])
    # Changing battery size scales replacement expense and lifetime throughput
    # equally, leaving the marginal EUR/MWh wear coefficient unchanged.
    config["battery"]["capacity_mwh"] *= expected["resizing_factor"]
    larger = degradation_settings_from_config(config)
    assert larger.replacement_cost_eur == pytest.approx(expected["replacement_cost_eur"] * expected["resizing_factor"])
    assert larger.cost_eur_per_mwh_throughput == pytest.approx(parsed.cost_eur_per_mwh_throughput)


@pytest.mark.parametrize("case", case_config("replacement_invalid"))
def test_invalid_replacement_lifetime_inputs_rejected(case):
    config = throughput_cost_config()
    config["degradation"][case["group"]][case["key"]] = case["value"]
    with pytest.raises(ValueError):
        degradation_settings_from_config(config)


def test_derived_and_manual_wear_costs_cannot_be_double_counted():
    config = throughput_cost_config()
    config["degradation"]["cost_eur_per_mwh_throughput"] = case_config("replacement_expected")["conflicting_direct_cost_eur_per_mwh"]
    with pytest.raises(ValueError, match="not both"):
        degradation_settings_from_config(config)

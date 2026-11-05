"""Physical feedback changes SOC and cannot discharge energy that failed to arrive."""
from datetime import datetime, timedelta

import pytest

from battery_case_config import case_config, configured_battery
from bess_pv_trading.domain import PhysicalState, PlantParameters
from bess_pv_trading.simulation.replay import execute_interval


def feedback_case(name):
    defaults = case_config("feedback_defaults")
    values = {**defaults, **case_config("feedback_cases")[name]}
    plant = PlantParameters(**values["plant"], battery=configured_battery(values["battery"]))
    state = PhysicalState(datetime.fromisoformat(values["start"]), values["initial_energy_mwh"])
    return values, plant, state


@pytest.mark.parametrize("name", list(case_config("feedback_cases")))
def test_feedback_flow_priority_and_actual_energy_conservation(name):
    case, plant, state = feedback_case(name)
    throughput = 0.0
    for step in case["steps"]:
        previous = state
        state, record = execute_interval(state=state, start=state.timestamp,
            end=state.timestamp+timedelta(hours=case["duration_hours"]), plant=plant,
            dispatch_policy="track_committed_grid", degradation_eur_per_mwh=case["degradation_eur_per_mwh"], **step)
        assert record["charge_mw"] * record["discharge_mw"] == 0
        assert plant.battery.energy_min_mwh <= state.energy_mwh <= plant.battery.maximum_energy
        assert record["grid_mw"] == pytest.approx(step["pv_ac_mw"]-record["curtailment_mw"]+record["discharge_mw"]-record["charge_mw"])
        energy = previous.energy_mwh + case["duration_hours"]*(
            plant.battery.eta_charge*record["charge_mw"]-record["discharge_mw"]/plant.battery.eta_discharge)
        assert state.energy_mwh == pytest.approx(energy)
        throughput += case["duration_hours"]*(record["charge_mw"]+record["discharge_mw"])
    for field, expected in case["expected"].items():
        assert record[field] == pytest.approx(expected)
    assert state.cumulative_degradation_eur == pytest.approx(throughput*case["degradation_eur_per_mwh"])


def test_fixed_battery_policy_and_feedback_are_distinct_configurable_choices():
    case, plant, state = feedback_case("reduced_charge")
    step = case["steps"][0]
    end = state.timestamp+timedelta(hours=case["duration_hours"])
    fixed, fixed_record = execute_interval(state=state, start=state.timestamp, end=end,
        plant=plant, dispatch_policy="follow_battery_plan", **step)
    actual, _ = execute_interval(state=state, start=state.timestamp, end=end,
        plant=plant, dispatch_policy="track_committed_grid", **step)
    assert fixed_record["charge_mw"] == step["requested_charge_mw"]
    assert actual.energy_mwh < fixed.energy_mwh


def test_feedback_requires_an_explicit_finite_committed_target():
    case, plant, state = feedback_case("reduced_charge")
    step = dict(case["steps"][0]); del step["grid_target_mw"]
    with pytest.raises(ValueError, match="target"):
        execute_interval(state=state, start=state.timestamp,
            end=state.timestamp+timedelta(hours=case["duration_hours"]), plant=plant,
            dispatch_policy="track_committed_grid", **step)


def test_recovered_pv_is_carried_to_later_commitment():
    """Recovered PV changes real SOC and can support a later grid target."""
    case, plant, state = feedback_case("scheduled_curtailment_replaces_discharge_then_charges")
    state = PhysicalState(state.timestamp, plant.battery.energy_min_mwh)
    end = state.timestamp + timedelta(hours=case["duration_hours"])
    state, first = execute_interval(state=state, start=state.timestamp, end=end, plant=plant,
        pv_ac_mw=1.0, requested_charge_mw=0.0, requested_discharge_mw=0.0,
        requested_curtailment_mw=0.5, dispatch_policy="track_committed_grid", grid_target_mw=0.5)
    assert first["grid_mw"] == pytest.approx(0.5)
    assert first["charge_mw"] == pytest.approx(0.5)
    state, second = execute_interval(state=state, start=state.timestamp,
        end=state.timestamp + timedelta(hours=case["duration_hours"]), plant=plant,
        pv_ac_mw=0.0, requested_charge_mw=0.0, requested_discharge_mw=0.3,
        requested_curtailment_mw=0.0, dispatch_policy="track_committed_grid", grid_target_mw=0.3)
    assert second["grid_mw"] == pytest.approx(0.3)
    assert second["discharge_mw"] == pytest.approx(0.3)


@pytest.mark.parametrize('requested_charge,requested_discharge,initial_fraction', [
    (0.0, 0.0, 0.3), (0.0, 0.2, 0.3), (0.2, 0.0, 0.3), (0.0, 0.0, 0.99),
])
def test_mpc_curtailment_recovery_preserves_grid_and_respects_battery(requested_charge, requested_discharge, initial_fraction):
    """Recovery keeps protected grid exchange fixed and advances real SOC."""
    case, plant, state = feedback_case('scheduled_curtailment_replaces_discharge_then_charges')
    battery = plant.battery
    energy = battery.energy_min_mwh + initial_fraction * (battery.maximum_energy - battery.energy_min_mwh)
    state = PhysicalState(state.timestamp, energy)
    kwargs = dict(state=state, start=state.timestamp, end=state.timestamp + timedelta(hours=case['duration_hours']),
        plant=plant, pv_ac_mw=1.0, requested_charge_mw=requested_charge,
        requested_discharge_mw=requested_discharge, requested_curtailment_mw=0.4)
    old_state, old = execute_interval(dispatch_policy='follow_battery_plan', **kwargs)
    new_state, new = execute_interval(dispatch_policy='follow_battery_plan_recover_curtailment', **kwargs)
    assert new['grid_mw'] == pytest.approx(old['grid_mw'])
    # Recompute the maximum grid-neutral AC recovery from physical headroom.
    from bess_pv_trading.optimization.physical import retained_energy
    retained = retained_energy(state.energy_mwh, battery, case['duration_hours'])
    charge_limit = min(battery.max_power_mw, max(0.0, battery.maximum_energy - retained)
        / (case['duration_hours'] * battery.eta_charge))
    maximum = min(old['curtailment_mw'], max(0.0, old['discharge_mw'] - old['charge_mw'] + charge_limit))
    assert new['curtailment_recovered_mw'] == pytest.approx(maximum)
    assert new['curtailment_mw'] <= old['curtailment_mw']
    assert new['curtailment_recovered_mw'] == pytest.approx(old['curtailment_mw'] - new['curtailment_mw'])
    assert new['charge_mw'] * new['discharge_mw'] == 0
    assert new_state.energy_mwh >= old_state.energy_mwh
    assert battery.energy_min_mwh <= new_state.energy_mwh <= battery.maximum_energy
    if initial_fraction == 0.99:
        assert new['curtailment_mw'] > 0
    else:
        assert new['curtailment_recovered_mw'] > 0


def test_mpc_recovery_can_absorb_safety_curtailment():
    """POI clipping occurs first, then recovery preserves the protected grid."""
    case, plant, state = feedback_case('scheduled_curtailment_replaces_discharge_then_charges')
    kwargs = dict(state=state, start=state.timestamp, end=state.timestamp + timedelta(hours=case['duration_hours']),
        plant=plant, pv_ac_mw=plant.poi_export_mw + 0.2, requested_charge_mw=0.0,
        requested_discharge_mw=0.0, requested_curtailment_mw=0.0)
    _, fixed = execute_interval(dispatch_policy='follow_battery_plan', **kwargs)
    _, recovered = execute_interval(dispatch_policy='follow_battery_plan_recover_curtailment', **kwargs)
    assert recovered['grid_mw'] == pytest.approx(fixed['grid_mw'])
    assert recovered['curtailment_mw'] < fixed['curtailment_mw']
    assert recovered['curtailment_recovered_mw'] > 0

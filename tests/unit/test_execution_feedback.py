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

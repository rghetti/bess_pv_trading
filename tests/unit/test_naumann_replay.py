"""Health-aware clipping, replay persistence and midnight stopping contracts."""

from dataclasses import asdict, replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from bess_pv_trading.domain import (
    BatteryHealthState, BatteryParameters, DegradationSettings, PhysicalState, PlantParameters,
)
from bess_pv_trading.simulation.battery_health import (
    BatteryEndOfLifeReached, initialize_physical_health, relative_terminal_soc, restore_battery_health,
)
from bess_pv_trading.simulation.continuous import ContinuousReplay, _physical_state
from bess_pv_trading.simulation.events import Event, EventKind
from bess_pv_trading.simulation.frozen_plans import _actual_boundary
from bess_pv_trading.simulation.replay import condition_and_slice, execute_interval


def _equipment():
    """Use different efficiencies to expose directional AC power mistakes."""
    return BatteryParameters(capacity_mwh=1, max_power_mw=4,
        energy_min_mwh=.07, energy_max_mwh=.93, efficiency_convention="combined",
        combined_charge_efficiency=.9, combined_discharge_efficiency=.8)


def _aging():
    """Keep economic replacement value separate from the physical calibration."""
    return DegradationSettings(mode="naumann_lfp", replacement_cost_eur=1000)


def _carried_state(*, energy=.4, soh=.9):
    """Provide explicit exposure and both banks for a used battery."""
    aging = _aging()
    instant = pd.Timestamp("2026-01-01T12:00:00Z").to_pydatetime()
    health = BatteryHealthState(calendar_exposure=(1 - soh) ** 2,
        cycle_exposure=0, committed_soh=soh, active_day="2026-01-01",
        discharge_segment_energy_mwh=(energy,), charge_segment_energy_mwh=(1 - energy,),
        calibration_id=aging.naumann.calibration_id)
    return PhysicalState(instant, energy, origin="simulated_carried_state", battery_health=health)


def test_checkpoint_restores_typed_health_and_both_memories():
    # JSON converts tuples into lists; restoration must recover immutable banks
    # without inferring either history from aggregate energy.
    state = _carried_state()
    payload = asdict(state)
    payload["timestamp"] = state.timestamp.isoformat()
    restored = _physical_state(json.loads(json.dumps(payload)))
    assert restored == state
    assert isinstance(restored.battery_health.discharge_segment_energy_mwh, tuple)
    assert isinstance(restored.battery_health.charge_segment_energy_mwh, tuple)
    assert restore_battery_health(str(asdict(state.battery_health))) == state.battery_health


def test_real_midnight_checkpoint_resumes_health_and_issued_mpc_action(tmp_path):
    from test_mpc_history import _checkpoint_replay, horizon_rows
    from bess_pv_trading.markets import AuctionCalendar
    from bess_pv_trading.optimization.mpc import MpcControllerState

    equipment, aging = _equipment(), _aging()
    replay = _checkpoint_replay(tmp_path)
    replay.checkpoint_enabled = True
    replay.aging = aging
    replay.plant = PlantParameters(1, 2, 2, equipment)
    calendar = AuctionCalendar(timezone="Europe/Rome")
    replay.daily = {day: {"grid": calendar.delivery_grid(day).to_domain()}
                    for day in ("2026-01-01", "2026-01-02")}
    instant = pd.Timestamp("2026-01-01T23:45:00", tz="Europe/Rome").to_pydatetime()
    midnight = pd.Timestamp("2026-01-02T00:00:00", tz="Europe/Rome")
    replay.state = initialize_physical_health(PhysicalState(instant, .5), equipment, aging)
    replay.state, _ = execute_interval(state=replay.state, start=instant, end=midnight,
        pv_ac_mw=0, requested_charge_mw=.2, requested_discharge_mw=0,
        requested_curtailment_mw=0, plant=replay.plant, degradation_settings=aging)
    action = {"interval_id": "next-quarter", "issued_at": midnight.isoformat(),
        "charge_mw": 0., "discharge_mw": .1, "pv_utilization_fraction": 1.}
    replay.pending[action["interval_id"]] = {"commands": [0., .1, 0.]}
    replay.realtime.state = MpcControllerState(
        plans=horizon_rows("2026-01-01") + horizon_rows("2026-01-02", 1),
        scenarios=horizon_rows("2026-01-01") + horizon_rows("2026-01-02", 1),
        latest_plan={action["interval_id"]: action}, outstanding_action=action,
        issued_actions=[{**action, "battery_health_observed_at_cutoff": asdict(replay.state.battery_health)}])
    replay.save_checkpoint(Event(midnight, EventKind.SETTLEMENT, 10, 0))
    checkpoint = json.loads((tmp_path / "json/checkpoint.json").read_text())
    resumed = _checkpoint_replay(tmp_path)
    resumed.checkpoint_enabled = True
    resumed.aging = aging
    resumed.plant = replay.plant
    resumed._restore(checkpoint)
    assert resumed.state == replay.state
    assert resumed.state.battery_health.active_day == "2026-01-02"
    assert resumed.realtime.state.outstanding_action == action
    assert resumed.realtime.state.latest_plan == replay.realtime.state.latest_plan
    assert json.loads(json.dumps(asdict(resumed.realtime.state))) == json.loads(
        json.dumps(asdict(replay.realtime.state)))
    assert resumed.pending == replay.pending
    # Replaying the already issued command must continue both cycle banks and
    # exposures identically, without another midnight commit or another solve.
    for current in (replay, resumed):
        issued = current.realtime.state.outstanding_action
        current.state, _ = execute_interval(state=current.state, start=midnight,
            end=midnight + pd.Timedelta(minutes=15), pv_ac_mw=0,
            requested_charge_mw=issued["charge_mw"], requested_discharge_mw=issued["discharge_mw"],
            requested_curtailment_mw=0, plant=current.plant, degradation_settings=aging)
    assert resumed.state == replay.state
    assert resumed.state.battery_health.calendar_exposure > checkpoint["state"]["battery_health"]["calendar_exposure"]


@pytest.mark.parametrize("overrides,mode,expected", [
    ({}, "naumann_lfp", (50., 40.)),
    ({"terminal_target_mwh": None}, "naumann_lfp", (None, 40.)),
    ({"terminal_minimum_mwh": None}, "naumann_lfp", (50., None)),
    ({"terminal_target_mwh": .4}, "naumann_lfp", (None, 40.)),
    ({"terminal_soc_basis": "nominal_capacity"}, "naumann_lfp", (None, None)),
    ({}, "xu_segmented", (None, None)),
])
def test_relative_soc_preserves_explicit_mwh_and_null_overrides(overrides, mode, expected):
    configured = {"terminal_soc_percent": 50, "terminal_soc_min_percent": 40,
                  "terminal_soc_basis": "available_capacity", **overrides}
    assert relative_terminal_soc(configured, mode) == expected


def test_continuation_freezes_only_observed_capacity_and_discloses_omitted_health(monkeypatch):
    from bess_pv_trading.simulation import energy_price_inputs

    equipment, aging = _equipment(), _aging()
    instant = pd.Timestamp("2026-01-01T12:00:00Z").to_pydatetime()
    state = initialize_physical_health(PhysicalState(instant, .5), equipment, aging)
    state = replace(state, battery_health=replace(state.battery_health,
        calendar_exposure=.01, committed_soh=.9))
    replay = SimpleNamespace(config={"realtime_mpc": {"terminal_inventory_mode": "piecewise_linear"},
        "continuation_value": {"horizon_hours": .25}},
        aging=aging, state=state, plant=PlantParameters(1, 2, 2, equipment),
        grid=SimpleNamespace(starts=(), interval_ids=()), trading={"grid_import_fee_eur_per_mwh": 0},
        _forecast=lambda *args, **kwargs: (np.zeros(1), {}))
    monkeypatch.setattr(energy_price_inputs, "settlement_forecasts", lambda *args:
        SimpleNamespace(positive_eur_per_mwh=np.zeros((1, 1)), negative_eur_per_mwh=None, provenance={}))
    captured = {}

    def capture(**values):
        captured.update(values)
        return "forecast curve"

    monkeypatch.setattr(energy_price_inputs, "build_continuation_value", capture)
    curve = energy_price_inputs.conditional_terminal_value(replay, {}, instant,
        instant + timedelta(hours=1))
    assert curve == "forecast curve"
    assert captured["plant"].battery.energy_max_mwh == pytest.approx(.837)
    assert captured["plant"].battery.max_power_mw == pytest.approx(.72)
    assert equipment.max_power_mw == 4
    approximation = captured["provenance"]["battery_health_approximation"]
    assert approximation["capacity_reference"] == "committed_observed_at_cutoff"
    assert approximation["future_capacity_fade_modelled"] is False
    assert approximation["cycle_memory_modelled"] is False
    assert approximation["calendar_and_cycle_wear_modelled"] is False


@pytest.mark.parametrize("day,partial", [("2026-01-01", True), ("2026-01-01", False),
                                        ("2026-03-29", False)])
def test_health_daily_export_uses_real_midnight_and_step_plots(tmp_path, monkeypatch, day, partial):
    from bess_pv_trading.reporting import continuous_outputs

    timezone = "Europe/Rome"
    first = pd.Timestamp(day, tz=timezone)
    final = (first + pd.Timedelta(hours=12) if partial else
             (pd.Timestamp(day) + pd.DateOffset(days=1)).tz_localize(timezone))
    boundaries = pd.date_range(first, final, freq="15min").tz_convert("UTC")
    starts, ends = boundaries[:-1], boundaries[1:]
    active_days = ends.tz_convert(timezone).strftime("%Y-%m-%d")
    end_soh = np.where(active_days == day, .9, .89)
    start_soh = np.r_[.9, end_soh[:-1]]
    frame = pd.DataFrame({"start": starts, "end": ends, "delivery_day": day,
        "battery_soh_start": start_soh, "battery_soh_end": end_soh,
        "available_capacity_mwh_start": start_soh, "available_capacity_mwh_end": end_soh,
        "naumann_active_day_end": active_days, "degradation_eur": .1,
        "realized_market_cashflow_eur": 1.})
    (tmp_path / "csv").mkdir()
    (tmp_path / "json").mkdir()
    styles = []
    original_plot = continuous_outputs.plot_boundary_series

    def capture_style(*args, **kwargs):
        styles.append(kwargs.get("drawstyle"))
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(continuous_outputs, "plot_boundary_series", capture_style)
    continuous_outputs._save_battery_health({"market": {"timezone": timezone},
        "experiment_outputs": {"plots": True}}, tmp_path, frame)
    row = pd.read_csv(tmp_path / "csv/battery_health_daily.csv").iloc[0]
    expected_commit = pd.Timestamp(active_days[-1], tz=timezone).tz_convert("UTC")
    assert pd.Timestamp(row.committed_at_utc) == expected_commit
    assert pd.Timestamp(row.observed_through_utc) == final.tz_convert("UTC")
    assert bool(row.completed_calendar_day) is not partial
    assert styles == ["steps-post", "steps-post"]
    if day == "2026-03-29":
        assert len(frame) == 92


def test_residual_slicing_remaps_future_health_soc_boundaries():
    from test_optimization_contracts import example_inputs
    from battery_case_config import configured_battery

    inputs = example_inputs(prices=[[0., 0., 0.]], battery=configured_battery("lossless"))
    aging = _aging()
    state = initialize_physical_health(replace(inputs.state, energy_mwh=.5), inputs.plant.battery, aging)
    inputs = replace(inputs, state=state, degradation=aging, terminal_target_mwh=None,
        target_soc_percent_by_boundary={3: 50}, minimum_soc_percent_by_boundary={0: 0, 1: 10, 3: 20})
    carried = replace(state, timestamp=inputs.grid.starts[1])
    residual = condition_and_slice(inputs, first_interval=1, state=carried, market=inputs.market,
        observed_prices={}, open_auctions=tuple(inputs.tree.bid_nodes))
    assert residual.target_soc_percent_by_boundary == {2: 50}
    assert residual.minimum_soc_percent_by_boundary == {0: 10, 2: 20}


@pytest.mark.parametrize("policy", [
    "follow_battery_plan", "track_committed_grid", "follow_battery_plan_recover_curtailment",
])
@pytest.mark.parametrize("charging", [True, False])
def test_every_execution_policy_uses_directional_health_limits(monkeypatch, policy, charging):
    from bess_pv_trading.optimization import naumann
    from bess_pv_trading.simulation import replay

    equipment, aging, state = _equipment(), _aging(), _carried_state(energy=.5)
    observed = {}
    monkeypatch.setattr(naumann, "initialize_health", lambda state, battery, settings: state.battery_health)
    monkeypatch.setattr(naumann, "effective_battery", lambda state, battery, settings:
        replace(battery, energy_min_mwh=.063, energy_max_mwh=.837))
    monkeypatch.setattr(naumann, "power_limits", lambda state, battery, settings: (.7, 1.1))
    monkeypatch.setattr(replay, "health_execution_record", lambda *args: {})

    def advance(**values):
        # Aging receives the original nameplate, even after capacity clipping.
        observed.update(values)
        energy = values["energy_mwh"]
        health = replace(values["state"].battery_health,
            discharge_segment_energy_mwh=(energy,), charge_segment_energy_mwh=(1 - energy,))
        return replace(values["state"], timestamp=values["end"], energy_mwh=energy,
                       battery_health=health), 0.0

    monkeypatch.setattr(replay, "advance_degradation_state", advance)
    plant = PlantParameters(1, 10, 10, equipment)
    end = state.timestamp + timedelta(minutes=15)
    updated, record = execute_interval(state=state, start=state.timestamp, end=end,
        pv_ac_mw=0, requested_charge_mw=4 if charging else 0,
        requested_discharge_mw=0 if charging else 4, requested_curtailment_mw=0,
        plant=plant, degradation_settings=aging, dispatch_policy=policy,
        grid_target_mw=-4 if charging else 4)
    assert record["charge_mw"] == pytest.approx(.7 if charging else 0)
    assert record["discharge_mw"] == pytest.approx(0 if charging else 1.1)
    assert observed["battery"] is equipment
    assert observed["charge_mw"] == record["charge_mw"]
    assert observed["discharge_mw"] == record["discharge_mw"]
    assert .063 <= updated.energy_mwh <= .837
    assert state.energy_mwh == .5


def test_idle_calendar_loss_commits_at_local_midnight_and_survives_restore():
    # Actual operation and a forecast projection return new immutable states.
    # An idle battery loses health without losing charge or creating cycle wear.
    equipment, aging = _equipment(), _aging()
    instant = pd.Timestamp("2026-01-01T23:45:00", tz="Europe/Rome").to_pydatetime()
    state = initialize_physical_health(PhysicalState(instant, .5), equipment, aging)
    plant = PlantParameters(1, 10, 10, equipment)
    updated, record = execute_interval(state=state, start=instant,
        end=instant + timedelta(minutes=15), pv_ac_mw=0,
        requested_charge_mw=0, requested_discharge_mw=0, requested_curtailment_mw=0,
        plant=plant, degradation_settings=aging)
    assert updated.energy_mwh == state.energy_mwh
    assert state.battery_health.calendar_exposure == 0
    assert updated.battery_health.calendar_exposure > 0
    assert updated.battery_health.cycle_exposure == 0
    assert updated.battery_health.active_day == "2026-01-02"
    assert updated.battery_health.committed_soh < state.battery_health.committed_soh
    assert record["degradation_eur"] > 0
    assert record["degradation_eur"] == pytest.approx(
        record["calendar_degradation_eur"] + record["cycle_degradation_eur"])
    payload = asdict(updated)
    payload["timestamp"] = updated.timestamp.isoformat()
    restored = _physical_state(json.loads(json.dumps(payload)))
    continued, _ = execute_interval(state=restored, start=restored.timestamp,
        end=restored.timestamp + timedelta(minutes=15), pv_ac_mw=0,
        requested_charge_mw=0, requested_discharge_mw=0, requested_curtailment_mw=0,
        plant=plant, degradation_settings=aging)
    assert continued.battery_health.calendar_exposure > restored.battery_health.calendar_exposure
    assert continued.battery_health.committed_soh == restored.battery_health.committed_soh


def test_frozen_plan_boundary_carries_actual_health_without_inventing_history():
    equipment, aging = _equipment(), _aging()
    instant = pd.Timestamp("2026-01-01T12:00:00Z").to_pydatetime()
    state = initialize_physical_health(PhysicalState(instant, .5), equipment, aging)
    updated, record = execute_interval(state=state, start=instant,
        end=instant + timedelta(minutes=15), pv_ac_mw=0, requested_charge_mw=0,
        requested_discharge_mw=0, requested_curtailment_mw=0,
        plant=PlantParameters(1, 10, 10, equipment), degradation_settings=aging)
    dispatch = pd.DataFrame([record, {"energy_start_mwh": .5}])
    config = {"degradation": {"model": "naumann_lfp"}, "state": {}}
    grid = SimpleNamespace(starts=(instant, updated.timestamp))
    carried = _actual_boundary(config, grid, dispatch, 1)
    assert carried.battery_health == updated.battery_health
    assert carried.cumulative_degradation_eur == record["degradation_eur"]
    assert initialize_physical_health(carried, equipment, aging) == carried
    with pytest.raises(ValueError, match="complete actual battery health"):
        _actual_boundary(config, grid, dispatch.drop(columns="battery_health_end"), 1)


@pytest.mark.parametrize("initial_energy,grid_target,export_limit", [(.92, 0., 2.), (.93, 1., 1.)])
def test_midnight_protection_changes_commands_and_curtailment_without_clipping_energy(
        initial_energy, grid_target, export_limit):
    # Unexpected PV feedback can fill the old ceiling. At an already full POI,
    # curtailing a little PV permits the physical discharge needed after fading.
    equipment, aging = _equipment(), _aging()
    instant = pd.Timestamp("2026-01-01T23:45:00", tz="Europe/Rome").to_pydatetime()
    state = initialize_physical_health(PhysicalState(instant, initial_energy), equipment, aging)
    updated, record = execute_interval(state=state, start=instant,
        end=instant + timedelta(minutes=15), pv_ac_mw=1,
        requested_charge_mw=0, requested_discharge_mw=0, requested_curtailment_mw=0,
        plant=PlantParameters(1, 2, export_limit, equipment), degradation_settings=aging,
        dispatch_policy="track_committed_grid", grid_target_mw=grid_target)
    expected_energy = initial_energy + .25 * (equipment.eta_charge * record["charge_mw"]
        - record["discharge_mw"] / equipment.eta_discharge)
    assert record["health_protection_adjustment_mw"] > 0
    assert updated.energy_mwh == pytest.approx(expected_energy, abs=1e-12)
    assert updated.energy_mwh <= equipment.maximum_energy * updated.battery_health.committed_soh + 1e-10
    assert record["grid_mw"] <= export_limit + 1e-12
    assert record["grid_mw"] == pytest.approx(1 - record["curtailment_mw"]
        + record["discharge_mw"] - record["charge_mw"])
    assert min(record["charge_mw"], record["discharge_mw"]) == 0
    assert state.battery_health.calendar_exposure == 0
    assert state.battery_health.cycle_exposure == 0
    assert record["degradation_eur"] == pytest.approx(
        record["calendar_degradation_eur"] + record["cycle_degradation_eur"])
    assert record["charge_c_rate_effective"] == pytest.approx(
        record["charge_c_rate_nominal"] / state.battery_health.committed_soh)
    assert record["discharge_c_rate_effective"] == pytest.approx(
        record["discharge_c_rate_nominal"] / state.battery_health.committed_soh)
    if initial_energy == .93:
        assert record["discharge_mw"] > 0
        assert record["curtailment_mw"] == pytest.approx(record["discharge_mw"])


def test_midnight_protection_fails_explicitly_when_no_export_can_remove_energy():
    equipment, aging = _equipment(), _aging()
    instant = pd.Timestamp("2026-01-01T23:45:00", tz="Europe/Rome").to_pydatetime()
    state = initialize_physical_health(PhysicalState(instant, .93), equipment, aging)
    with pytest.raises(ValueError, match="No bounded battery command"):
        execute_interval(state=state, start=instant, end=instant + timedelta(minutes=15),
            pv_ac_mw=0, requested_charge_mw=0, requested_discharge_mw=0,
            requested_curtailment_mw=0, plant=PlantParameters(1, 0, 0, equipment),
            degradation_settings=aging)


def test_execution_projects_only_tolerance_sized_command_residuals():
    """Record tiny overlap and preserve the requested internal energy rate."""
    equipment, aging = _equipment(), _aging()
    instant = pd.Timestamp("2026-01-01T12:00:00Z").to_pydatetime()
    state = initialize_physical_health(PhysicalState(instant, .5), equipment, aging)
    values = dict(state=state, start=instant, end=instant + timedelta(minutes=15),
                  pv_ac_mw=0, requested_charge_mw=4e-6, requested_discharge_mw=2e-6,
                  requested_curtailment_mw=0, plant=PlantParameters(1, 1, 1, equipment),
                  degradation_settings=aging, numerical_tolerance=1e-4)
    updated, record = execute_interval(**values)
    expected_rate = .9 * 4e-6 - 2e-6 / .8
    assert updated.energy_mwh == pytest.approx(.5 + .25 * expected_rate)
    assert record["discharge_mw"] == 0
    assert record["command_projection_mw"] > 0
    assert record["command_projection_grid_delta_mw"] == pytest.approx(
        record["discharge_mw"] - record["charge_mw"] - (2e-6 - 4e-6))
    assert record["safety_adjustment_mw"] > 0
    with pytest.raises(ValueError, match="command projection"):
        execute_interval(**{**values, "requested_charge_mw": .01, "requested_discharge_mw": .01})


def test_eol_finishes_settlement_and_checkpoint_before_stopping_new_actions(monkeypatch):
    # The old day's accepted commitments remain present when EOL prevents
    # the next physical action and the replay explicitly stops after settlement.
    replay = object.__new__(ContinuousReplay)
    state = _carried_state(soh=.8)
    midnight = pd.Timestamp("2026-01-02T00:00:00", tz="Europe/Rome")
    replay.state = replace(state, timestamp=midnight.to_pydatetime())
    replay.aging = _aging()
    replay.grid = SimpleNamespace(interval_ids=("product",))
    replay.day_for_product = {"product": "2026-01-01"}
    replay.prepared = {"calendar": SimpleNamespace(timezone="Europe/Rome")}
    replay.event_trace = []
    replay.progress = None
    replay.market = SimpleNamespace(accepted_trades=("unchanged commitment",))
    calls = []
    monkeypatch.setattr(replay, "_control", lambda event: calls.append("new_control"))
    monkeypatch.setattr(replay, "_settle", lambda event: calls.append("settle"))
    monkeypatch.setattr(replay, "save_checkpoint", lambda event: calls.append("checkpoint"))
    replay._handle(Event(midnight, EventKind.PHYSICAL_CONTROL, 1, 0))
    with pytest.raises(BatteryEndOfLifeReached):
        replay._handle(Event(midnight, EventKind.SETTLEMENT, 2, 0))
    assert calls == ["settle", "checkpoint"]
    assert replay.market.accepted_trades == ("unchanged commitment",)
    assert replay.event_trace[0]["battery_health_action"] == "skipped_at_end_of_life"

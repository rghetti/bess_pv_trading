"""Daily horizon partitions preserve control state and survive checkpoint restarts."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.domain import BatteryParameters, DeliveryGrid, MarketState, PhysicalState, PlantParameters
from bess_pv_trading.optimization.mpc import MpcControllerState
from bess_pv_trading.reporting.mpc_history import (
    archive_completed_history, iter_mpc_history, validate_mpc_history_settings, verify_history_archives,
)
from bess_pv_trading.simulation.continuous import ContinuousReplay
from bess_pv_trading.simulation.events import Event, EventKind
from bess_pv_trading.simulation.realtime_control import RealtimeControl
from bess_pv_trading.simulation.replay import execute_interval
from bess_pv_trading.transaction_fees import TransactionFeeState


def horizon_rows(day, count=2):
    """Repeated nested metadata exercises Arrow-safe JSON encoding and deduplication."""
    metadata = {"weights": [0.5, 0.5], "nested": {}, "forecast": "available"}
    return [{"delivery_day": day, "solve_id": f"solve-{day}", "interval_id": f"{day}-{index}",
             "pv_ac_mw": float(index), "provenance": metadata, "optional": None}
            for index in range(count)]


def test_completed_day_spill_preserves_next_day_first_action_and_streams_small_batches(tmp_path):
    first, second = horizon_rows("2026-01-01"), horizon_rows("2026-01-02", 1)
    state = MpcControllerState(plans=first + second, scenarios=deepcopy(first + second),
        latest_plan={"next": second[0]}, outstanding_action={"charge_mw": 0.2},
        solves=[{"cutoff": "retained"}], issued_actions=[{"interval_id": "next"}])
    before = deepcopy(asdict(state))
    index = archive_completed_history(tmp_path, state, [], ["2026-01-01"])
    assert state.plans == second
    assert state.scenarios == second
    for name in ("latest_plan", "outstanding_action", "solves", "issued_actions"):
        assert getattr(state, name) == before[name]
    verify_history_archives(tmp_path, index)
    batches = list(iter_mpc_history(tmp_path, batch_size=1))
    assert [len(frame) for frame in batches] == [1, 1]
    assert batches[0].provenance.iloc[0] == first[0]["provenance"]
    repeated = archive_completed_history(tmp_path, state, index, ["2026-01-01"])
    assert repeated == index
    assert len(list(tmp_path.rglob("*.parquet"))) == 2


def test_checkpoint_partition_checksums_and_paths_are_enforced(tmp_path):
    state = MpcControllerState(plans=horizon_rows("2026-01-01"), scenarios=horizon_rows("2026-01-01"))
    index = archive_completed_history(tmp_path, state, [], ["2026-01-01"])
    escaped = deepcopy(index)
    escaped[0]["plans"]["path"] = "../outside.parquet"
    with pytest.raises(ValueError, match="escapes"):
        verify_history_archives(tmp_path, escaped)
    path = tmp_path / index[0]["scenarios"]["path"]
    with path.open("ab") as handle:
        handle.write(b"unexpected modification")
    with pytest.raises(ValueError, match="checksum"):
        verify_history_archives(tmp_path, index)


def test_failed_spill_leaves_buffers_and_content_addressed_orphans_are_safe(tmp_path, monkeypatch):
    import bess_pv_trading.reporting.mpc_history as module

    state = MpcControllerState(plans=horizon_rows("2026-01-01"), scenarios=horizon_rows("2026-01-01"))
    original = deepcopy(asdict(state))
    writer = module._write_partition

    def fail_scenarios(destination, day, kind, rows):
        if kind == "scenarios":
            raise OSError("interrupted before checkpoint")
        return writer(destination, day, kind, rows)

    monkeypatch.setattr(module, "_write_partition", fail_scenarios)
    with pytest.raises(OSError):
        archive_completed_history(tmp_path, state, [], ["2026-01-01"])
    assert asdict(state) == original
    assert not (tmp_path / "json/mpc_horizon_history.json").exists()
    monkeypatch.setattr(module, "_write_partition", writer)
    # A replay from an older checkpoint may produce a different feasible shadow plan.
    state.plans[0]["pv_ac_mw"] = 99.0
    index = archive_completed_history(tmp_path, state, [], ["2026-01-01"])
    verify_history_archives(tmp_path, index)
    assert len(list((tmp_path / "diagnostics/mpc_horizon/2026-01-01").glob("plans-*.parquet"))) == 2
    assert list(iter_mpc_history(tmp_path))[0].pv_ac_mw.iloc[0] == 99


def _checkpoint_replay(destination):
    """Construct the real checkpoint/execution boundary without an optimizer dependency."""
    replay = object.__new__(ContinuousReplay)
    replay.config = {"realtime_mpc_outputs": {"horizon_history": "daily_parquet"}}
    replay.destination = destination
    replay.identity = {"test": "midnight-storage-boundary"}
    replay.state = PhysicalState(pd.Timestamp("2026-01-01T23:45:00Z"), 0.5)
    replay.market = MarketState(transaction_fee_state=TransactionFeeState())
    battery = BatteryParameters(1, 1, converter_charge_efficiency=1, converter_discharge_efficiency=1)
    replay.plant = PlantParameters(1, 1, 1, battery)
    replay.daily = {}
    for day in ("2026-01-01", "2026-01-02"):
        start = pd.Timestamp(day, tz="UTC")
        replay.daily[day] = {"grid": DeliveryGrid((day,), (start.to_pydatetime(),),
            ((start + pd.Timedelta(days=1)).to_pydatetime(),))}
    for name in ("submitted", "known", "fee_records", "control_plan", "pending", "records", "plan_snapshots",
                 "initial_expectations", "initial_revenues"):
        setattr(replay, name, {})
    for name in ("solves", "projections", "event_trace", "publications", "forecast_inputs"):
        setattr(replay, name, [])
    replay.realtime = RealtimeControl(replay)
    return replay


def test_midnight_checkpoint_resume_preserves_actual_state_and_issued_first_action(tmp_path):
    replay = _checkpoint_replay(tmp_path)
    midnight = pd.Timestamp("2026-01-02T00:00:00Z")
    replay.state, _ = execute_interval(state=replay.state, start=replay.state.timestamp, end=midnight,
        pv_ac_mw=0.0, requested_charge_mw=0.2, requested_discharge_mw=0.0,
        requested_curtailment_mw=0.0, plant=replay.plant)
    first = {"interval_id": "2026-01-02-first", "issued_at": midnight.isoformat(),
             "charge_mw": 0.0, "discharge_mw": 0.1, "pv_utilization_fraction": 1.0}
    next_rows = horizon_rows("2026-01-02", 1)
    replay.realtime.state = MpcControllerState(plans=horizon_rows("2026-01-01") + next_rows,
        scenarios=horizon_rows("2026-01-01") + deepcopy(next_rows), latest_plan={first["interval_id"]: first},
        outstanding_action=first, solves=[{"day": "2026-01-01"}], issued_actions=[first])
    replay.pending[first["interval_id"]] = {"commands": [0.0, 0.1, 0.0]}
    event = Event(midnight, EventKind.SETTLEMENT, 10, 0)
    replay.save_checkpoint(event)
    checkpoint = json.loads((tmp_path / "json/checkpoint.json").read_text())
    assert len(checkpoint["mpc_history_archives"]) == 1
    assert all(row["delivery_day"] == "2026-01-02" for row in checkpoint["mpc_state"]["plans"])
    resumed = _checkpoint_replay(tmp_path)
    resumed._restore(checkpoint)
    assert resumed.realtime.state.outstanding_action == first
    assert resumed.realtime.state.latest_plan == replay.realtime.state.latest_plan
    assert resumed.realtime.state.issued_actions == replay.realtime.state.issued_actions
    assert resumed.state == replay.state
    # Execute the same already-issued next-day action in both continuations.
    for current in (replay, resumed):
        action = current.realtime.state.outstanding_action
        current.state, _ = execute_interval(state=current.state, start=midnight,
            end=midnight + pd.Timedelta(minutes=15), pv_ac_mw=0,
            requested_charge_mw=action["charge_mw"], requested_discharge_mw=action["discharge_mw"],
            requested_curtailment_mw=0, plant=current.plant)
    assert resumed.state == replay.state
    assert resumed.state.energy_mwh == pytest.approx(0.525)


def test_final_partition_reporting_does_not_reload_archived_horizons(tmp_path, monkeypatch):
    import bess_pv_trading.reporting.realtime_mpc as reporting

    state = MpcControllerState(plans=horizon_rows("2026-01-01"), scenarios=horizon_rows("2026-01-01"))
    index = archive_completed_history(tmp_path, state, [], ["2026-01-01"])
    (tmp_path / "csv").mkdir()
    frame = pd.DataFrame({"delivery_day": ["2026-01-01"], "imbalance_mwh": [0.0],
        "imbalance_cashflow_eur": [0.0], "discharge_mw": [0.0], "charge_mw": [0.0],
        "requested_discharge_mw": [0.0], "requested_charge_mw": [0.0]})
    monkeypatch.setattr(reporting, "_financial_dispatch", lambda *args: frame.copy())
    monkeypatch.setattr(reporting, "controller_metrics", lambda *args: {"maximum_cumulative_shortage_mwh": 0.0})
    monkeypatch.setattr(pd, "read_parquet", lambda *args: pytest.fail("Final reporting must not reload old horizons"))
    config = {"execution": {"dispatch_policy": "ensemble_mpc"},
              "realtime_mpc_outputs": {"horizon_history": "daily_parquet"}}
    result = reporting.save_mpc_outputs(config, tmp_path, frame, state, [], history_archives=index)
    assert result["mpc_metrics"]["horizon_history"]["plan_records"] == 2
    assert not (tmp_path / "csv/mpc_plans.csv").exists()
    assert not (tmp_path / "json/mpc_scenarios.json").exists()
    assert (tmp_path / "csv/mpc_actions.csv").exists()


@pytest.mark.parametrize("settings", [{"horizon_history": "delete"}, {"unknown": True}, "daily_parquet"])
def test_invalid_history_configuration_fails(settings):
    with pytest.raises(ValueError):
        validate_mpc_history_settings({"realtime_mpc_outputs": settings})


def test_legacy_history_configuration_is_unchanged():
    assert validate_mpc_history_settings({}) == "memory"

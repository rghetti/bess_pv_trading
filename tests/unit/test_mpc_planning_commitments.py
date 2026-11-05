"""IDA2 forecast orders affect MPC planning until actual IDA3 submission only."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.optimization.mpc import MpcSettings
from bess_pv_trading.simulation.realtime_control import RealtimeControl


def controller_case():
    """Use signed forecast and actual IDA3 orders on two afternoon products."""
    day = "2026-01-15"
    products = ("morning0", "morning1", "afternoon0", "afternoon1")
    grid = SimpleNamespace(interval_ids=products)
    ida3 = SimpleNamespace(auction="IDA3", available=True,
        submission_at=pd.Timestamp("2026-01-15T09:00Z"), eligible_intervals=(2, 3))
    snapshot = [{"interval_id": key, "planned_trade_IDA3_mwh": quantity,
                 "optimizer_cutoff_utc": "2026-01-14T20:55Z"}
                for key, quantity in zip(products, [99, 99, -0.15, 0.25])]
    base = dict(zip(products, [0.2, 0.3, 0.4, 0.5]))
    replay = SimpleNamespace(config={}, submitted={day: {"IDA2": base}},
        daily={day: {"grid": grid, "schedule": [ida3]}},
        plan_snapshots={day: {"IDA2": snapshot}})
    replay._position = lambda key: sum(orders.get(key, 0) for orders in replay.submitted[day].values())
    controller = RealtimeControl.__new__(RealtimeControl)
    controller.replay = replay
    controller.settings = MpcSettings()
    return controller, day, grid


def test_planned_ida3_adds_only_eligible_orders_and_retains_the_ledger():
    controller, day, grid = controller_case()
    before = deepcopy(controller.replay.submitted)
    actual, provisional, evidence = controller.planning_commitments(day, grid, pd.Timestamp("2026-01-15T08:00Z"))
    np.testing.assert_allclose(actual + provisional, [0.2, 0.3, 0.25, 0.75])
    np.testing.assert_allclose(provisional, [0, 0, -0.15, 0.25])
    assert controller.replay.submitted == before
    assert evidence["provisional_ida3_active"]
    assert evidence["source_auction"] == "IDA2"
    assert evidence["provisional_orders_are_binding"] is False


@pytest.mark.parametrize("missing", ["snapshot", "participation", "availability", "ida2", "disabled"])
def test_missing_or_disabled_future_market_never_creates_forecast_commitments(missing):
    controller, day, grid = controller_case()
    if missing == "snapshot":
        controller.replay.plan_snapshots[day].clear()
    elif missing == "participation":
        controller.replay.config = {"gme_oracle_experiment": {"participating_auctions": ["DA", "IDA2"]}}
    elif missing == "availability":
        controller.replay.daily[day]["schedule"][0].available = False
    elif missing == "ida2":
        controller.replay.submitted[day].clear()
    else:
        controller.settings = replace(controller.settings, provisional_ida3_mode="submitted_only")
    _, provisional, evidence = controller.planning_commitments(day, grid, pd.Timestamp("2026-01-15T08:00Z"))
    np.testing.assert_array_equal(provisional, 0)
    assert not evidence["provisional_ida3_active"]


def test_submission_replaces_forecast_without_double_counting_before_price_publication():
    controller, day, grid = controller_case()
    controller.replay.submitted[day]["IDA3"] = {"afternoon0": 0.1, "afternoon1": -0.2}
    actual, provisional, _ = controller.planning_commitments(day, grid, pd.Timestamp("2026-01-15T09:00Z"))
    np.testing.assert_allclose(actual, [0.2, 0.3, 0.5, 0.3])
    np.testing.assert_array_equal(provisional, 0)


def test_future_snapshot_is_rejected_and_gate_clock_stops_provisional_orders():
    controller, day, grid = controller_case()
    for row in controller.replay.plan_snapshots[day]["IDA2"]:
        row["optimizer_cutoff_utc"] = "2026-01-15T08:30Z"
    with pytest.raises(ValueError, match="unavailable"):
        controller.planning_commitments(day, grid, pd.Timestamp("2026-01-15T08:00Z"))
    _, provisional, _ = controller.planning_commitments(day, grid, pd.Timestamp("2026-01-15T09:00Z"))
    np.testing.assert_array_equal(provisional, 0)

"""Continuous economic control, scenario transformations and information ablations."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from test_continuous import continuous_config
from test_realtime_continuous import mpc_prepared
from bess_pv_trading.forecasting.energy_prices import PricePrediction
from bess_pv_trading.simulation.continuous import ContinuousReplay
from bess_pv_trading.simulation.events import EventKind
from bess_pv_trading.simulation.research_inputs import auction_price_predictions


def test_reduced_adaptive_controller_preserves_conditional_plans(continuous_config, tmp_path):
    config = continuous_config
    config["execution"]["dispatch_policy"] = "ensemble_mpc_adaptive"
    config["realtime_scenarios"] = {"dependence": "independent_shuffle", "seed": 7, "reduce_to": 2}
    prepared = mpc_prepared(config, one_day=True)
    replay = ContinuousReplay(config, prepared, "integrated", tmp_path / "adaptive")
    replay.realtime.issue(0, pd.Timestamp(replay.grid.starts[0]))
    diagnostic = replay.realtime.state.solves[-1]
    assert not diagnostic["fallback_applied"]
    assert diagnostic["scenario_count"] == 2
    assert diagnostic["adaptive_recourse_enabled"]
    rows = pd.DataFrame(replay.realtime.state.scenarios)
    assert {"control_node", "charge_mw", "energy_end_mwh"} <= set(rows)
    first = rows[rows.interval_id == replay.grid.interval_ids[0]]
    assert first.control_node.nunique() == 1
    assert first.charge_mw.nunique() == 1
    assert all(row["future_plan_role"] == "representative_shadow_path" for row in replay.realtime.state.plans)


def test_forecast_updates_can_freeze_without_reading_later_actuals(continuous_config, tmp_path):
    config = continuous_config
    config["research_information"] = {"pv_updates": "frozen_initial"}
    prepared = mpc_prepared(config, one_day=True)
    replay = ContinuousReplay(config, prepared, "integrated", tmp_path / "freeze")
    day = replay.days[0]
    daily = replay.daily[day]
    starts = pd.DatetimeIndex(daily["grid"].starts)
    cutoff = starts[10]
    before, provenance = replay._forecast(cutoff, starts[10:], "IDA3", causal=True)
    prepared["full_realized_pv"][:] = 1000
    prepared["forecast_provider"] = lambda *args: pytest.fail("Frozen ablation must use pinned initial forecasts")
    after, _ = replay._forecast(cutoff, starts[10:], "IDA3", causal=True)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(after, daily["initial_forecast"][10:])
    assert provenance["pv_forecast_policy"] == "frozen_initial"


def test_unpublished_auction_prices_are_forecast_and_published_prices_are_fixed(continuous_config, tmp_path):
    config = continuous_config
    config["energy_price_forecast"] = {"auction_enabled": True}
    prepared = mpc_prepared(config, one_day=True)
    calls = []

    class Prices:
        def predict(self, name, cutoff, starts, scenario_count=1):
            calls.append((name, cutoff, starts))
            assert all(starts >= cutoff)
            point = np.full(len(starts), 123.0)
            return PricePrediction(point, point[None, :], cutoff, {"source": "synthetic_forecast"})

    prepared["energy_price_provider"] = Prices()
    replay = ContinuousReplay(config, prepared, "integrated", tmp_path / "auctions")
    day = replay.days[0]
    daily = replay.daily[day]
    event = next(event for event in replay.events if event.kind == EventKind.AUCTION_SUBMISSION
        and event.payload.event.auction == "IDA3")
    replay.known[day] = ["DA", "IDA1", "IDA2"]
    predicted, provenance = auction_price_predictions(replay, day, event.at)
    assert len(calls) == 1 and calls[0][0] == "IDA3"
    np.testing.assert_array_equal(predicted["DA"], daily["prices"]["DA"])
    before = predicted["IDA3"].copy()
    daily["prices"]["IDA3"] = np.full_like(before, -999)
    after, _ = auction_price_predictions(replay, day, event.at)
    np.testing.assert_array_equal(before, after["IDA3"])
    assert provenance["auction_price_information"] == "causal_point_forecast"


def test_conditional_continuation_uses_only_forecasts(continuous_config, tmp_path):
    config = continuous_config
    config["execution"]["dispatch_policy"] = "deterministic_mpc_economic"
    config["realtime_mpc"] = {"terminal_inventory_mode": "piecewise_linear"}
    config["continuation_value"] = {"horizon_hours": 1, "settings": {"energy_grid_points": 11}}
    prepared = mpc_prepared(config, one_day=True)
    original = prepared["forecast_provider"]

    def provider(cutoff, starts):
        bundle = original(cutoff, starts)
        bundle.values.loc[:, "pv_ac_mw"] = 0.0
        return bundle

    prepared["forecast_provider"] = provider
    replay = ContinuousReplay(config, prepared, "integrated", tmp_path / "continuation")
    replay.realtime.issue(0, pd.Timestamp(replay.grid.starts[0]))
    diagnostic = replay.realtime.state.solves[-1]
    assert not diagnostic["fallback_applied"]
    evidence = diagnostic["terminal_value_provenance"]
    assert evidence["forecast_is_oracle"] is False
    assert pd.Timestamp(evidence["continuation_start_utc"]) == pd.Timestamp(replay.grid.ends[-1])
    assert evidence["degradation_approximation"] == "linear_throughput_without_Xu_cycle_memory"

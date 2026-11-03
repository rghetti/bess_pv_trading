"""MPC event chronology, immutable ledgers, DST and open-loop projection checks."""

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from test_continuous import continuous_config, _prepared
from bess_pv_trading.data import ForecastBundle
from bess_pv_trading.optimization.mpc import MPC_POLICIES
from bess_pv_trading.simulation.continuous import ContinuousReplay, ContinuousAuction
from bess_pv_trading.simulation.events import EventKind


def mpc_prepared(config, one_day=False):
    """Extend the shared synthetic fixture to provide every causal control cutoff."""
    prepared, fixtures = _prepared(config, one_day=one_day)
    starts = pd.DatetimeIndex(prepared["full_grid"].starts)
    pv = pd.Series(np.concatenate([item["initial_forecast"] for item in prepared["days_data"].values()]), index=starts)

    def provider(cutoff, requested):
        return ForecastBundle(config["farm_id"], pd.Timestamp(cutoff) - pd.Timedelta(hours=6),
            pd.Timestamp(cutoff), pd.DataFrame({"pv_ac_mw": pv.reindex(requested).to_numpy()}, index=requested),
            {"pv_ac_mw": "MW_AC"})

    prepared["forecast_provider"] = provider
    prepared["provenance"]["pv_ensemble"] = {}
    for day, daily in prepared["days_data"].items():
        daily["ensemble_members"] = ("1", "2")
        daily["ensemble_pv"] = np.stack([daily["initial_forecast"] * 0.8, daily["initial_forecast"] * 1.2])
        cutoff = min(event.submission_at for event in daily["schedule"])
        prepared["provenance"]["pv_ensemble"][day] = {
            "forecast_vintage_utc": (cutoff - pd.Timedelta(hours=6)).isoformat(),
            "available_at_utc": cutoff.isoformat()}
    return prepared


@pytest.mark.parametrize("policy", sorted(MPC_POLICIES))
def test_all_controllers_reconcile_actual_cash_and_store_full_plans(continuous_config, tmp_path, policy):
    config = continuous_config
    config["execution"]["dispatch_policy"] = policy
    prepared = mpc_prepared(config, one_day=True)
    replay = ContinuousReplay(config, prepared, "integrated", tmp_path / policy)
    before = prepared["full_realized_pv"].copy()
    summary = replay.run()
    dispatch = pd.DataFrame(replay.records.values()).sort_values("start")
    assert len(replay.realtime.state.solves) == len(prepared["full_grid"].interval_ids)
    assert not any(row["fallback_applied"] for row in replay.realtime.state.solves)
    assert len(replay.realtime.state.plans) > len(dispatch)
    assert dispatch.dispatch_policy.eq("follow_battery_plan").all()
    assert dispatch.imbalance_penalty_eur.eq(0).all()
    np.testing.assert_allclose(dispatch.energy_start_mwh.iloc[1:], dispatch.energy_end_mwh.iloc[:-1])
    assert summary["net_profit_eur"] == pytest.approx(replay.market.cashflow_eur
        + dispatch.imbalance_cashflow_eur.sum() - dispatch.degradation_eur.sum()
        - dispatch.import_fee_eur.sum() - replay.market.transaction_fee_state.cumulative_fees_eur)
    np.testing.assert_array_equal(before, prepared["full_realized_pv"])
    assert all(row["price_oracle_enabled"] == policy.endswith("price_oracle") for row in replay.realtime.state.solves)


@pytest.mark.parametrize("auction", ["DA", "IDA3"])
@pytest.mark.parametrize("known_current", [False, True])
def test_noon_and_midnight_projection_preserve_locked_action_and_never_resolve(continuous_config, tmp_path, monkeypatch, auction, known_current):
    config = continuous_config
    config["continuous_optimization"]["end_date"] = "2025-01-02"
    config["execution"]["dispatch_policy"] = "ensemble_mpc"
    config["realtime_mpc"] = {"current_pv_information": "realized_interval_average" if known_current else "forecast"}
    prepared = mpc_prepared(config)
    replay = ContinuousReplay(config, prepared, "integrated", tmp_path / "projection")
    target_day = prepared["days"][1 if auction == "DA" else 0]
    target = next(event for event in replay.events if event.kind == EventKind.AUCTION_SUBMISSION
                  and event.payload.delivery_day == target_day and event.payload.event.auction == auction)
    # The projection contract needs the latest issued MPC plan, not dozens of
    # earlier optimizations. Advance its actual prefix with the existing tracker.
    controller = replay.realtime
    replay.realtime = None
    replay.dispatch_policy = "track_committed_grid"
    # Stop just before DA, which occurs within an unfinished quarter-hour.
    for event in replay.events:
        if event >= target:
            break
        if event.kind == EventKind.PHYSICAL_CONTROL and event.at == target.at.floor("15min"):
            replay.realtime = controller
            replay.dispatch_policy = "ensemble_mpc"
        replay._handle(event)
    assert replay.pending
    state_before, pending_before = asdict(replay.state), deepcopy(replay.pending)
    solve_count = len(replay.realtime.state.solves)
    import bess_pv_trading.simulation.realtime_control as control

    def forbidden(*args, **kwargs):
        raise AssertionError("An auction projection must not solve MPC")

    monkeypatch.setattr(control, "solve_mpc", forbidden)
    inputs, diagnostic = replay._planning_inputs(target.payload, target.at)
    boundary = prepared["days_data"][target_day]["grid"].starts[0 if auction == "DA" else 48]
    assert inputs.state.timestamp == boundary
    assert asdict(replay.state) == state_before
    assert replay.pending == pending_before
    assert len(replay.realtime.state.solves) == solve_count
    assert inputs.state.origin == "projected_soc_mpc_deterministic"
    assert all(not row["projection_fallback"] for row in diagnostic["locked_prefix_projection"])
    expected_source = "disclosed_current_interval_realized_pv" if known_current else "eligible_deterministic_forecast"
    assert diagnostic["locked_prefix_projection"][0]["projection_pv_information_source"] == expected_source
    # Change only unseen realized PV: the shadow state remains identical.
    unseen = replay.starts > target.at.floor("15min") if known_current else replay.starts >= target.at.floor("15min")
    prepared["full_realized_pv"][unseen] = 999
    repeated, _ = replay._planning_inputs(target.payload, target.at)
    assert repeated.state == inputs.state
    # Clearing plan coverage is an explicit tracking fallback, still without MPC solves.
    replay.realtime.state.latest_plan = {}
    _, fallback = replay._planning_inputs(target.payload, target.at)
    assert any(row.get("projection_fallback") for row in fallback["locked_prefix_projection"])
    expected_kind = "delivery_midnight" if auction == "DA" else "ida3_noon"
    assert replay.projections[-1]["boundary_kind"] == expected_kind
    assert replay.projections[-1]["actual_initial_energy_mwh"] is None


@pytest.mark.parametrize("day,count", [("2025-03-30", 92), ("2025-01-01", 96), ("2025-10-26", 100)])
def test_mpc_dst_grid_and_first_horizon(continuous_config, tmp_path, day, count):
    config = continuous_config
    config["continuous_optimization"].update(start_date=day, end_date=day)
    config["market"].update(valid_from=day, valid_to=day)
    config["execution"]["dispatch_policy"] = "deterministic_mpc"
    prepared = mpc_prepared(config, one_day=True)
    replay = ContinuousReplay(config, prepared, "integrated", tmp_path / day)
    replay.realtime.issue(0, pd.Timestamp(replay.grid.starts[0]))
    assert len(replay.realtime.state.latest_plan) == count
    assert replay.realtime.state.solves[0]["horizon_intervals"] == count


def test_price_blind_current_action_ignores_unseen_pv_and_settlement(continuous_config, tmp_path):
    config = continuous_config
    config["execution"]["dispatch_policy"] = "deterministic_mpc"
    prepared = mpc_prepared(config, one_day=True)
    replay = ContinuousReplay(config, prepared, "integrated", tmp_path / "original")
    replay.realtime.issue(0, pd.Timestamp(replay.grid.starts[0]))
    issued = replay.pending[replay.grid.interval_ids[0]]
    prepared["full_realized_pv"][:] = 999
    prepared["imbalance_prices_eur_per_mwh"][:] = -999999
    changed = ContinuousReplay(config, prepared, "integrated", tmp_path / "changed")
    changed.realtime.issue(0, pd.Timestamp(changed.grid.starts[0]))
    assert changed.pending[changed.grid.interval_ids[0]] == issued


def test_current_pv_disclosure_replaces_every_members_first_step_only(continuous_config, tmp_path, monkeypatch):
    import bess_pv_trading.simulation.realtime_control as control

    config = continuous_config
    config["execution"]["dispatch_policy"] = "ensemble_mpc"
    config["realtime_mpc"] = {"current_pv_information": "realized_interval_average"}
    prepared = mpc_prepared(config, one_day=True)
    prepared["full_realized_pv"][0] = 0.123
    prepared["full_realized_pv"][1:] = 999
    original = control.solve_mpc
    captured = []

    def capture(inputs, settings):
        captured.append(inputs)
        return original(inputs, settings)

    monkeypatch.setattr(control, "solve_mpc", capture)
    replay = ContinuousReplay(config, prepared, "integrated", tmp_path / "known_current")
    replay.realtime.issue(0, pd.Timestamp(replay.grid.starts[0]))
    inputs = captured[0]
    np.testing.assert_array_equal(inputs.pv_ac_mw[:, 0], 0.123)
    assert not np.any(inputs.pv_ac_mw[:, 1:] == 999)
    assert inputs.current_realized_pv_mw == 0.123
    assert inputs.oracle is None
    current = [row for row in replay.realtime.state.scenarios if row["interval_id"] == replay.grid.interval_ids[0]]
    assert all(row["pv_input_role"] == "realized_current_interval" for row in current)
    assert len(replay.realtime.state.issued_actions) == 1


def test_checkpoint_roundtrips_mpc_state(continuous_config, tmp_path):
    config = continuous_config
    config["execution"]["dispatch_policy"] = "deterministic_mpc"
    prepared = mpc_prepared(config, one_day=True)
    destination = tmp_path / "resume"
    replay = ContinuousReplay(config, prepared, "integrated", destination)
    replay.run()
    restored = ContinuousReplay(config, prepared, "integrated", destination, resume=True)
    assert restored.state == replay.state
    assert restored.realtime.state.latest_plan == replay.realtime.state.latest_plan
    assert len(restored.realtime.state.solves) == len(replay.realtime.state.solves)
    assert restored.run()["net_profit_eur"] == pytest.approx(json.loads((destination / "json/kpis.json").read_text())["net_profit_eur"])


def test_frozen_workflow_replays_source_submissions_and_rejects_changed_data(continuous_config, tmp_path, monkeypatch):
    from bess_pv_trading.simulation import realtime_mpc_study as study

    config = continuous_config
    config["continuous_optimization"].update(output_root=str(tmp_path), run_id="frozen",
        perfect_pv_reference=False, planning_modes=["integrated"])
    config["realtime_mpc_comparison"] = {"mode": "frozen_commitments",
        "controllers": ["track_committed_grid", "deterministic_mpc"]}
    config["experiment_outputs"]["plots"] = True
    prepared = mpc_prepared(config, one_day=True)
    prepared["provenance"]["days"] = {day: {"source": "synthetic"} for day in prepared["days"]}
    monkeypatch.setattr(study, "prepare_continuous_inputs", lambda *args, **kwargs: prepared)
    result = study.run_realtime_mpc_experiment(config)
    output = tmp_path / "farm_1/frozen/frozen_commitments"
    assert result["status"] == "complete"
    assert len(list((output / "plots").glob("*.png"))) >= 9
    frozen = json.loads((output / "frozen_commitments.json").read_text())
    study._validate_frozen(config, prepared, frozen)
    for row in result["comparison"]:
        assert row["auction_revenue_difference_eur"] == pytest.approx(0)
        assert row["trading_fee_difference_eur"] == pytest.approx(0)
    source = pd.read_csv(output / "source/csv/realized_dispatch.csv")
    baseline = pd.read_csv(output / "controllers/track_committed_grid/csv/realized_dispatch.csv")
    for field in ("grid_mw", "charge_mw", "discharge_mw", "curtailment_mw", "energy_end_mwh"):
        np.testing.assert_allclose(source[field], baseline[field])
    changed = deepcopy(prepared)
    changed["full_realized_pv"][0] += 1
    with pytest.raises(ValueError, match="incompatible"):
        study._validate_frozen(config, changed, frozen)
    def do_not_regenerate(*args):
        raise AssertionError("A verified existing frozen ledger must be reused")

    monkeypatch.setattr(study, "_freeze_source", do_not_regenerate)
    def do_not_prepare_again(*args, **kwargs):
        raise AssertionError("The comparison launcher must be able to share pinned inputs")

    monkeypatch.setattr(study, "prepare_continuous_inputs", do_not_prepare_again)
    repeated = study.run_realtime_mpc_experiment(config, prepared_inputs=prepared)
    assert repeated["status"] == "complete"
    assert json.loads((output / "manifest.json").read_text())["frozen_ledger_reused"]
    assert json.loads((output / "frozen_commitments.json").read_text())["sha256"] == frozen["sha256"]
    assert (output / "source/json/optimization_attempts.json").exists()


def test_mpc_runtime_failure_keeps_interval_and_uses_tracking(continuous_config, tmp_path, monkeypatch):
    from bess_pv_trading.optimization.mpc import MpcResult
    import bess_pv_trading.simulation.realtime_control as control

    config = continuous_config
    config["execution"]["dispatch_policy"] = "deterministic_mpc"
    replay = ContinuousReplay(config, mpc_prepared(config, one_day=True), "integrated", tmp_path / "fallback")
    monkeypatch.setattr(control, "solve_mpc", lambda *args: MpcResult("time_limit_no_incumbent", diagnostics={
        "status": "time_limit_no_incumbent", "fallback_applied": True, "stages": [], "runtime_seconds": 0.0}))
    replay.realtime.issue(0, pd.Timestamp(replay.grid.starts[0]))
    product = replay.grid.interval_ids[0]
    assert replay.pending[product]["policy"] == "track_committed_grid"
    assert replay.pending[product]["detail"]["mpc_fallback"]
    assert replay.submitted == {day: {} for day in replay.days}

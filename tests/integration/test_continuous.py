"""Synthetic acceptance coverage for one causal, consecutive operating trajectory."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data import ForecastBundle
from bess_pv_trading.domain import DeliveryGrid, OptimizationResult
from bess_pv_trading.markets import AuctionCalendar
from bess_pv_trading.simulation import synthetic
from bess_pv_trading.simulation.continuous_inputs import _fingerprint


CASES = Path(__file__).resolve().parents[1] / "fixtures/configs/continuous_cases.yaml"


@pytest.fixture
def continuous_config():
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model():
            pass
    except gp.GurobiError as error:
        pytest.skip(str(error))
    config = load_config(CASES)
    config["state"].update(
        initial_energy_mwh=config["battery"]["capacity_mwh"] * config["state"]["initial_soc_percent"] / 100,
        terminal_target_mwh=config["battery"]["capacity_mwh"] * config["state"]["terminal_soc_percent"] / 100)
    config["decision_resolution_minutes"] = config["gme_oracle_experiment"]["decision_resolution_minutes"]
    config["native_interval_minutes_by_auction"] = {
        name: item["native_interval_minutes"]
        for name, item in config["gme_oracle_experiment"]["auction_datasets"].items()}
    return config


def _prepared(config, *, one_day=False, cancel=False):
    """Create absolute-time prices/PV and a causal provider without source readers."""
    section, reference = config["continuous_optimization"], config["test_case"]
    days = tuple(stamp.date().isoformat() for stamp in pd.date_range(
        section["start_date"], section["end_date"], freq="D"))
    if one_day:
        days = days[:1]
    calendar = AuctionCalendar.from_config(config)
    data, fixtures = {}, {}
    for day in days:
        daily_config = deepcopy(config)
        daily_config["synthetic"]["delivery_day"] = day
        fixture = synthetic.synthetic_inputs(daily_config)
        grid = fixture["grid"]
        # A native hourly DA curve is constant over its four physical quarters.
        # UTC keeps the two repeated autumn local hours distinct and unambiguous.
        hours = pd.DatetimeIndex(grid.starts).floor("h")
        da = fixture["actual_prices"]["DA"]
        if config["native_interval_minutes_by_auction"]["DA"] == 60:
            da = pd.Series(da, index=hours).groupby(level=0).transform("mean").to_numpy()
        fixture["actual_prices"]["DA"] = da
        fixture["base"] = da.copy()
        fixture["kwargs"]["base_prices"] = da.copy()
        fixture["realized_pv"] = fixture["pv"] * reference["realized_pv_factor"]
        if cancel and day == days[reference["cancelled_day_index"]]:
            name = reference["cancelled_auction"]
            fixture["schedule"] = tuple(event for event in fixture["schedule"] if event.auction != name)
            fixture["actual_prices"].pop(name)
        data[day] = {"grid": grid, "schedule": fixture["schedule"],
            "prices": fixture["actual_prices"], "realized_pv": fixture["realized_pv"],
            "initial_forecast": fixture["pv"].copy(), "skipped": [],
            "price_provenance": {"source": "synthetic_test"},
            "pv_provenance": {"realized_power_missing_fallback_intervals": 0},
            "daily_config": daily_config,
            "native_interval_minutes_by_auction": {
                name: config["native_interval_minutes_by_auction"][name] for name in fixture["actual_prices"]}}
        fixtures[day] = fixture
    grids = [data[day]["grid"] for day in days]
    full = DeliveryGrid(tuple(value for grid in grids for value in grid.interval_ids),
        tuple(value for grid in grids for value in grid.starts),
        tuple(value for grid in grids for value in grid.ends))
    base_pv = pd.Series(np.concatenate([fixtures[day]["pv"] for day in days]),
                        index=pd.DatetimeIndex(full.starts))
    event_names = {event.submission_at: event.auction for fixture in fixtures.values()
                   for event in fixture["schedule"]}

    def provider(cutoff, starts):
        starts = pd.DatetimeIndex(starts)
        name = event_names[pd.Timestamp(cutoff)]
        values = base_pv.reindex(starts).to_numpy() * reference["factors_by_auction"][name]
        assert np.isfinite(values).all(), "Synthetic provider request exceeds the study grid"
        return ForecastBundle(config["plant"]["farm_id"],
            pd.Timestamp(cutoff) - pd.Timedelta(hours=config["source"]["availability_lag_hours"]),
            pd.Timestamp(cutoff), pd.DataFrame({"pv_ac_mw": values}, index=starts),
            {"pv_ac_mw": "MW_AC"})

    for day, item in data.items():
        first = item["schedule"][0].submission_at
        bundle = provider(first, item["grid"].starts)
        item["initial_pv_bundle"] = bundle
        item["pv_metadata"] = {"pv_forecast_vintage_utc": bundle.forecast_vintage.isoformat(),
                               "pv_forecast_available_at_utc": bundle.available_at.isoformat()}
        fixture = fixtures[day]
        fixture.update(pv_forecast_provider=provider,
            forecast_available_at=bundle.available_at, pv_forecast_vintage=bundle.forecast_vintage,
            pv_forecast_policy=config["forecast"]["reoptimization_pv_policy"],
            realized_imbalance_prices_eur_per_mwh=np.full(len(item["grid"].interval_ids),
                reference["settlement_price_eur_per_mwh"]))
    prepared = {"calendar": calendar, "days": days, "days_data": data,
        "plant": fixtures[days[0]]["plant"], "full_grid": full,
        "initial_energy_mwh": config["state"]["initial_energy_mwh"],
        "terminal_energy_mwh": config["state"]["terminal_target_mwh"],
        "full_realized_pv": np.concatenate([data[day]["realized_pv"] for day in days]),
        "forecast_provider": provider,
        "imbalance_prices_eur_per_mwh": np.full(len(full.interval_ids), reference["settlement_price_eur_per_mwh"]),
        "settlement_frame": None, "provenance": {"source": "synthetic_test", "offline": True}}
    prepared["input_fingerprint"] = _fingerprint({"days": days,
        "actual_pv": prepared["full_realized_pv"],
        "prices": {day: item["prices"] for day, item in data.items()},
        "config": config})
    return prepared, fixtures


def _module():
    return importlib.import_module("bess_pv_trading.simulation.continuous")


def _run(config, prepared, mode, destination, *, resume=False):
    replay = _module().ContinuousReplay(config, prepared, mode, destination, resume=resume)
    summary = replay.run()
    return replay, summary


def _read(destination, filename):
    return json.loads((destination / "json" / filename).read_text())


@pytest.mark.parametrize("mode", ["integrated", "current_auction_only"])
def test_one_day_matches_existing_price_oracle(continuous_config, tmp_path, mode):
    prepared, fixtures = _prepared(continuous_config, one_day=True)
    fixture = fixtures[prepared["days"][0]]
    fixture["auction_planning_mode"] = mode
    old = tmp_path / "independent"
    independent = synthetic._replay_strategy(deepcopy(continuous_config), fixture, "oracle", old)
    destination = tmp_path / "continuous"
    replay, summary = _run(continuous_config, prepared, mode, destination)
    old_solves = json.loads((old / "solves.json").read_text())
    solves = _read(destination, "solves.json")
    assert [row["current_auction"] for row in solves] == [row["current_auction"] for row in old_solves]
    np.testing.assert_allclose([row["objective_eur"] for row in solves],
        [row["objective_eur"] for row in old_solves], atol=continuous_config["test_case"]["comparison_atol"])
    assert replay.state.energy_mwh == pytest.approx(independent["final_state"]["energy_mwh"],
        abs=continuous_config["test_case"]["comparison_atol"])
    assert summary["net_profit_eur"] == pytest.approx(independent["realized_profit_after_imbalance_eur"],
        abs=continuous_config["test_case"]["comparison_atol"])


@pytest.mark.parametrize("mode", ["integrated", "current_auction_only"])
def test_consecutive_states_commitments_scope_and_final_only_target(continuous_config, tmp_path, monkeypatch, mode):
    tmp_path = tmp_path / "replay"
    prepared, _ = _prepared(continuous_config)
    module, captured = _module(), []
    build = module.build_model

    def capture(inputs):
        captured.append(inputs)
        return build(inputs)

    monkeypatch.setattr(module, "build_model", capture)
    replay, summary = _run(continuous_config, prepared, mode, tmp_path)
    dispatch = pd.read_csv(tmp_path / "csv/realized_dispatch.csv")
    ledger = pd.read_csv(tmp_path / "csv/accepted_trades.csv")
    solves = _read(tmp_path, "solves.json")
    np.testing.assert_allclose(dispatch.energy_start_mwh.iloc[1:], dispatch.energy_end_mwh.iloc[:-1])
    assert dispatch.energy_start_mwh.iloc[0] == pytest.approx(prepared["initial_energy_mwh"])
    assert pd.Timestamp(dispatch.start.iloc[0]) == prepared["full_grid"].starts[0]
    assert pd.Timestamp(dispatch.end.iloc[-1]) == prepared["full_grid"].ends[-1]
    assert len(solves) == sum(len(item["schedule"]) for item in prepared["days_data"].values())
    assert len(captured) == len(solves)
    assert solves[0]["initial_state_is_supplied_boundary"]
    assert pd.Timestamp(solves[0]["cutoff"]) < prepared["full_grid"].starts[0]
    for diagnostic, inputs in zip(solves, captured):
        day, auction = diagnostic["delivery_day"], diagnostic["current_auction"]
        item = prepared["days_data"][day]
        event = next(event for event in item["schedule"] if event.auction == auction)
        eligible_start = item["grid"].starts[min(event.eligible_intervals)]
        assert inputs.grid.starts[0] == eligible_start
        assert inputs.grid.ends[-1] == item["grid"].ends[-1]
        expected_target = prepared["terminal_energy_mwh"] if day == prepared["days"][-1] else None
        assert inputs.terminal_target_mwh == expected_target
        assert diagnostic["terminal_target_mwh"] == expected_target
        expected_pv = prepared["forecast_provider"](event.submission_at, inputs.grid.starts)
        np.testing.assert_allclose(inputs.scenarios.pv_ac_mw[0], expected_pv.values.pv_ac_mw)
        assert pd.Timestamp(diagnostic["pv_forecast_available_at_utc"]) <= event.submission_at
        committed = ledger.loc[pd.to_datetime(ledger.submission_time, utc=True) < event.submission_at]
        expected_positions = committed.groupby("interval_id").quantity_mwh.sum().reindex(inputs.grid.interval_ids, fill_value=0)
        np.testing.assert_allclose(inputs.market.positions(inputs.grid) + np.asarray(inputs.pending_position_mwh),
                                   expected_positions, atol=continuous_config["test_case"]["comparison_atol"])
        if mode == "current_auction_only":
            assert set(inputs.auction_eligibility) == {auction}
            assert set(inputs.tree.bid_nodes) == {auction}
        if auction == "IDA3":
            assert len(inputs.grid.interval_ids) == continuous_config["test_case"]["expected_ida3_intervals"]
    # All actual intervals keep the current delivery day's plan; tomorrow's DA
    # cannot replace today's IDA3 physical commands even though it closes today.
    for day in prepared["days"][:-1]:
        grid = prepared["days_data"][day]["grid"]
        event = next(event for event in prepared["days_data"][day]["schedule"] if event.auction == "IDA3")
        after_noon = dispatch.loc[(pd.to_datetime(dispatch.start, utc=True) >= grid.starts[min(event.eligible_intervals)])
            & (pd.to_datetime(dispatch.start, utc=True) < grid.ends[-1])]
        assert after_noon.active_plan_auction.eq("IDA3").all()
    assert replay.state.cumulative_throughput_mwh == pytest.approx(dispatch.throughput_mwh.sum())
    assert replay.state.cumulative_degradation_eur == pytest.approx(dispatch.degradation_eur.sum())
    fee_state = replay.market.transaction_fee_state
    assert fee_state.cumulative_fees_eur == pytest.approx(ledger.quantity_mwh.abs().sum()
        * continuous_config["transaction_fees"]["flat_eur_per_mwh"])
    months = pd.to_datetime(ledger.submission_time, utc=True).dt.tz_convert(
        continuous_config["market"]["timezone"]).dt.strftime("%Y-%m")
    assert set(fee_state.monthly_volume_mwh) == set(months)
    for month, rows in ledger.groupby(months):
        assert fee_state.monthly_volume_mwh[month] == pytest.approx(rows.quantity_mwh.abs().sum())
    expected_settlement = dispatch.imbalance_mwh * continuous_config["test_case"]["settlement_price_eur_per_mwh"]
    np.testing.assert_allclose(dispatch.imbalance_cashflow_eur, expected_settlement)
    assert summary["net_profit_eur"] == pytest.approx(ledger.cashflow_eur.sum() + expected_settlement.sum()
        - dispatch.degradation_eur.sum() - dispatch.import_fee_eur.sum()
        - fee_state.cumulative_fees_eur)
    da = ledger.loc[ledger.auction == "DA"].copy()
    da["native_hour"] = pd.to_datetime(da.interval_id, utc=True).dt.floor("h")
    assert da.groupby("native_hour").quantity_mwh.agg(np.ptp).max() <= continuous_config["test_case"]["comparison_atol"]


def test_next_day_midnight_projection_never_reads_future_realized_pv(continuous_config, tmp_path, monkeypatch):
    prepared, _ = _prepared(continuous_config)
    target_day = prepared["days"][1]
    event = next(event for event in prepared["days_data"][target_day]["schedule"] if event.auction == "DA")
    changed = deepcopy(prepared)
    starts = pd.DatetimeIndex(changed["full_grid"].starts)
    unseen = starts >= event.submission_at.floor("15min")
    changed["full_realized_pv"][unseen] *= continuous_config["test_case"]["future_realized_pv_factor"]
    offset = 0
    for day in changed["days"]:
        count = len(changed["days_data"][day]["grid"].interval_ids)
        changed["days_data"][day]["realized_pv"] = changed["full_realized_pv"][offset:offset + count].copy()
        offset += count
    selected = []
    module, build = _module(), _module().build_model

    def capture(inputs):
        if inputs.tree.nodes[0].cutoff == event.submission_at:
            selected.append(inputs)
        return build(inputs)

    monkeypatch.setattr(module, "build_model", capture)
    for label, inputs in (("original", prepared), ("changed", changed)):
        _run(continuous_config, inputs, "current_auction_only", tmp_path / label)
    assert len(selected) == 2
    assert selected[0].state.energy_mwh == pytest.approx(selected[1].state.energy_mwh)
    np.testing.assert_allclose(selected[0].scenarios.pv_ac_mw, selected[1].scenarios.pv_ac_mw)
    assert selected[0].state.timestamp == prepared["days_data"][target_day]["grid"].starts[0]


def test_cancelled_auction_has_no_solve_or_submission(continuous_config, tmp_path):
    tmp_path = tmp_path / "replay"
    prepared, _ = _prepared(continuous_config, cancel=True)
    _run(continuous_config, prepared, "integrated", tmp_path)
    day = prepared["days"][continuous_config["test_case"]["cancelled_day_index"]]
    auction = continuous_config["test_case"]["cancelled_auction"]
    assert not any(row["delivery_day"] == day and row["current_auction"] == auction
                   for row in _read(tmp_path, "solves.json"))


def test_sequential_infeasible_later_auction_keeps_plan_and_submits_zero(continuous_config, tmp_path, monkeypatch):
    tmp_path = tmp_path / "replay"
    prepared, _ = _prepared(continuous_config)
    day = prepared["days"][continuous_config["test_case"]["fallback_day_index"]]
    event = next(event for event in prepared["days_data"][day]["schedule"]
        if event.auction == continuous_config["test_case"]["fallback_auction"])
    module, solve = _module(), _module().solve_model

    def fail_selected(built, settings):
        if (set(built.inputs.auction_eligibility) == {event.auction}
                and built.inputs.grid.starts[0] == prepared["days_data"][day]["grid"].starts[min(event.eligible_intervals)]):
            return OptimizationResult("infeasible", False)
        return solve(built, settings)

    monkeypatch.setattr(module, "solve_model", fail_selected)
    replay, summary = _run(continuous_config, prepared, "current_auction_only", tmp_path)
    submitted = replay.submitted[day][event.auction]
    np.testing.assert_allclose(list(submitted.values()), 0)
    failed = next(row for row in replay.solves if row["delivery_day"] == day
                  and row["current_auction"] == event.auction)
    assert failed["fallback_applied"]
    assert failed["fallback_order_policy"] == "zero_new_trades"
    retained = replay.plan_snapshots[day][event.auction]
    prior = {row["interval_id"]: row for row in replay.plan_snapshots[day][failed["plan_source_auction"]]}
    for row in retained:
        for field in ("charge_mw", "discharge_mw", "curtailment_mw"):
            assert row[field] == pytest.approx(prior[row["interval_id"]][field])
    assert replay.state.timestamp == prepared["full_grid"].ends[-1]


def test_first_target_day_failure_cannot_borrow_another_days_plan(continuous_config, tmp_path, monkeypatch):
    prepared, _ = _prepared(continuous_config)
    day = prepared["days"][continuous_config["test_case"]["fallback_day_index"]]
    module, solve = _module(), _module().solve_model

    def fail_tomorrows_da(built, settings):
        if (set(built.inputs.auction_eligibility) == {"DA"}
                and built.inputs.grid.starts[0] == prepared["days_data"][day]["grid"].starts[0]):
            return OptimizationResult("infeasible", False)
        return solve(built, settings)

    monkeypatch.setattr(module, "solve_model", fail_tomorrows_da)
    with pytest.raises(RuntimeError, match=f"{day}/DA: infeasible"):
        _run(continuous_config, prepared, "current_auction_only", tmp_path / "failure")


def test_perfect_pv_reference_preserves_daily_horizon_and_chronology(continuous_config, tmp_path, monkeypatch):
    prepared, _ = _prepared(continuous_config)
    module, build, captured = _module(), _module().build_model, []

    def no_operational_forecast(*args):
        pytest.fail("Perfect-PV reference must use its explicitly declared perfect production")

    def capture(inputs):
        captured.append(inputs)
        return build(inputs)

    prepared["forecast_provider"] = no_operational_forecast
    monkeypatch.setattr(module, "build_model", capture)
    replay, summary = _run(continuous_config, prepared, "perfect_pv", tmp_path / "perfect")
    for inputs in captured:
        indices = pd.DatetimeIndex(prepared["full_grid"].starts).get_indexer(inputs.grid.starts)
        np.testing.assert_allclose(inputs.scenarios.pv_ac_mw[0], prepared["full_realized_pv"][indices])
        local_day = pd.Timestamp(inputs.grid.starts[0]).tz_convert(prepared["calendar"].timezone).date().isoformat()
        assert inputs.grid.ends[-1] == prepared["days_data"][local_day]["grid"].ends[-1]
    assert replay.state.energy_mwh == pytest.approx(prepared["terminal_energy_mwh"],
        abs=continuous_config["test_case"]["comparison_atol"])
    assert summary["absolute_imbalance_mwh"] <= continuous_config["test_case"]["comparison_atol"]


def test_checkpoint_resume_matches_uninterrupted_execution(continuous_config, tmp_path, monkeypatch):
    prepared, _ = _prepared(continuous_config)
    module = _module()
    uninterrupted = tmp_path / "uninterrupted"
    _run(continuous_config, prepared, "current_auction_only", uninterrupted)
    resumed = tmp_path / "resumed"
    save = module.ContinuousReplay.save_checkpoint

    def interrupt_after_midnight(self, event):
        save(self, event)
        raise RuntimeError("synthetic interruption after durable checkpoint")

    monkeypatch.setattr(module.ContinuousReplay, "save_checkpoint", interrupt_after_midnight)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _run(continuous_config, prepared, "current_auction_only", resumed)
    assert (resumed / "json/checkpoint.json").is_file()
    checkpoint = _read(resumed, "checkpoint.json")
    following_day = prepared["days"][1]
    pre_delivery = {event.auction for event in prepared["days_data"][following_day]["schedule"]
                    if event.submission_at < prepared["days_data"][following_day]["grid"].starts[0]}
    assert set(checkpoint["submitted"][following_day]) == pre_delivery
    monkeypatch.setattr(module.ContinuousReplay, "save_checkpoint", save)
    _run(continuous_config, prepared, "current_auction_only", resumed, resume=True)
    expected = pd.read_csv(uninterrupted / "csv/realized_dispatch.csv")
    actual = pd.read_csv(resumed / "csv/realized_dispatch.csv")
    pd.testing.assert_frame_equal(expected, actual, check_exact=False,
        atol=continuous_config["test_case"]["comparison_atol"], rtol=0)
    for name in ("accepted_trades.csv",):
        pd.testing.assert_frame_equal(pd.read_csv(uninterrupted / "csv" / name),
                                      pd.read_csv(resumed / "csv" / name))


@pytest.mark.parametrize("case", load_config(CASES)["test_case"]["dst_cases"])
def test_dst_day_lengths_utc_identity_and_native_products(continuous_config, tmp_path, monkeypatch, case):
    day = case["delivery_day"]
    continuous_config["continuous_optimization"].update(start_date=day, end_date=day)
    continuous_config["market"].update(valid_from=day, valid_to=day)
    continuous_config["native_interval_minutes_by_auction"]["DA"] = case["native_da_minutes"]
    continuous_config["gme_oracle_experiment"]["auction_datasets"]["DA"]["native_interval_minutes"] = case["native_da_minutes"]
    prepared, _ = _prepared(continuous_config)
    grid = prepared["full_grid"]
    assert len(grid.interval_ids) == case["expected_physical_intervals"]
    assert len(set(grid.interval_ids)) == len(grid.interval_ids)
    assert tuple(grid.starts[1:]) == tuple(grid.ends[:-1])
    assert all(pd.Timestamp(product) == start for product, start in zip(grid.interval_ids, grid.starts))
    assert np.all(np.diff(pd.DatetimeIndex(grid.starts).asi8) > 0)
    from bess_pv_trading.optimization.market_regime import configured_auction_product_groups
    groups = configured_auction_product_groups(grid, "DA", case["native_da_minutes"], prepared["calendar"].timezone)
    assert len(groups) == case["expected_da_products"]
    module, build, captured = _module(), _module().build_model, []

    def capture(inputs):
        captured.append(inputs)
        return build(inputs)

    monkeypatch.setattr(module, "build_model", capture)
    _run(continuous_config, prepared, "current_auction_only", tmp_path / "dst")
    ida3 = next(inputs for inputs in captured if set(inputs.auction_eligibility) == {"IDA3"})
    assert len(ida3.grid.interval_ids) == continuous_config["test_case"]["expected_ida3_intervals"]
    assert pd.Timestamp(ida3.grid.starts[0]).tz_convert(prepared["calendar"].timezone).strftime("%H:%M") == continuous_config["market"]["delivery_start_times"]["IDA3"]
    da = next(inputs for inputs in captured if set(inputs.auction_eligibility) == {"DA"})
    assert da.native_interval_minutes_by_auction["DA"] == case["native_da_minutes"]


def test_checkpoint_rejects_changed_config_and_input_fingerprints(continuous_config, tmp_path, monkeypatch):
    prepared, _ = _prepared(continuous_config)
    module = _module()
    destination = tmp_path / "checkpointed"
    save = module.ContinuousReplay.save_checkpoint

    def interrupt(self, event):
        save(self, event)
        raise RuntimeError("durable checkpoint")

    monkeypatch.setattr(module.ContinuousReplay, "save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="durable checkpoint"):
        _run(continuous_config, prepared, "integrated", destination)
    changed_config = deepcopy(continuous_config)
    changed_config["state"]["terminal_soc_percent"] = changed_config["state"]["initial_soc_percent"]
    with pytest.raises(ValueError, match="different configuration, inputs or policy"):
        module.ContinuousReplay(changed_config, prepared, "integrated", destination, resume=True)
    changed_input = dict(prepared, input_fingerprint=_fingerprint({"changed": prepared["input_fingerprint"]}))
    with pytest.raises(ValueError, match="different configuration, inputs or policy"):
        module.ContinuousReplay(continuous_config, changed_input, "integrated", destination, resume=True)


def test_future_prices_cannot_change_current_sequential_order(continuous_config, tmp_path):
    prepared, _ = _prepared(continuous_config, one_day=True)
    changed = deepcopy(prepared)
    day = prepared["days"][0]
    first = prepared["days_data"][day]["schedule"][0].auction
    for auction, prices in changed["days_data"][day]["prices"].items():
        if auction != first:
            prices *= continuous_config["test_case"]["future_price_multiplier"]
    original, _ = _run(continuous_config, prepared, "current_auction_only", tmp_path / "original")
    perturbed, _ = _run(continuous_config, changed, "current_auction_only", tmp_path / "perturbed")
    assert original.submitted[day][first] == pytest.approx(perturbed.submitted[day][first],
        abs=continuous_config["test_case"]["comparison_atol"])
    columns = ["charge_mw", "discharge_mw", "curtailment_mw", "soc_end_mwh"]
    np.testing.assert_allclose(pd.DataFrame(original.plan_snapshots[day][first])[columns],
        pd.DataFrame(perturbed.plan_snapshots[day][first])[columns],
        atol=continuous_config["test_case"]["comparison_atol"])

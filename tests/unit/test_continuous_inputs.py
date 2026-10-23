"""Offline continuous preparation pins causal shared inputs before any replay."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data import ForecastBundle
from bess_pv_trading.simulation import continuous_inputs

FARM_IDS = load_config(Path(__file__).resolve().parents[2] / "configs/standard.yaml")["farms"]


@pytest.fixture
def prepared_case(monkeypatch):
    """Use configured scientific values and substitute local in-memory sources."""
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "tests/fixtures/configs/continuous_inputs_cases.yaml")
    case = config["test_case"]
    calls = {"pv": [], "weather": [], "prices": []}

    def bundle(cutoff, starts):
        starts = pd.DatetimeIndex(starts).as_unit("ns")
        frame = pd.DataFrame({"pv_ac_mw": np.full(len(starts), case["forecast_mw"]),
            case["marker_column"]: case["marker_value"]}, index=starts)
        return ForecastBundle(config["forecast"]["farm_id"],
            pd.Timestamp(cutoff) - pd.Timedelta(hours=case["cycle_age_hours"]),
            pd.Timestamp(cutoff), frame, {"pv_ac_mw": "MW"})

    def load_pv(daily, calendar, grid, *, decision_cutoff):
        calls["pv"].append((daily["gme_oracle_experiment"]["delivery_day"],
                             pd.Timestamp(decision_cutoff), grid))
        pv = bundle(decision_cutoff, grid.starts)
        return (pv.values.pv_ac_mw.to_numpy(copy=True),
            np.full(len(grid.starts), case["realized_mw"]), pv,
            {"farm_metadata": {"farm_id": config["forecast"]["farm_id"]},
             "realized_power_missing_fallback_intervals": 0})

    def load_weather(config, calendar, cutoff, starts):
        calls["weather"].append((pd.Timestamp(cutoff), pd.DatetimeIndex(starts)))
        return bundle(cutoff, starts)

    def load_prices(daily, calendar, grid, events):
        calls["prices"].append(deepcopy(daily["gme_oracle_experiment"]))
        prices = {event.auction: np.full(len(grid.starts), case["auction_price_eur_mwh"])
                  for event in events}
        return prices, {name: {"snapshots": [{"storage": {"snapshot_id": name}}]}
                        for name in prices}, []

    monkeypatch.setattr(continuous_inputs, "_load_pv", load_pv)
    monkeypatch.setattr(continuous_inputs, "_load_pv_forecast", load_weather)
    monkeypatch.setattr(continuous_inputs, "_load_auction_prices", load_prices)
    monkeypatch.setattr(continuous_inputs, "load_realized_imbalance_prices", lambda config, grid:
        (pd.DataFrame({"settlement_price_eur_mwh": np.full(len(grid.starts), case["settlement_eur_mwh"])}),
         {"snapshots": [{"storage": {"snapshot_id": "settlement"}}]}))
    return config, calls, bundle


@pytest.mark.parametrize("farm_id", FARM_IDS)
def test_continuous_preflight_instantiates_selected_farm_battery(prepared_case, farm_id):
    """The real preflight constructs per-farm assets using in-memory data providers."""
    config, _, _ = prepared_case
    path = Path(__file__).resolve().parents[2] / "tests/fixtures/configs/continuous_inputs_cases.yaml"
    config.update(load_config(path, overrides={"farm_id": int(farm_id)}))
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    battery = prepared["plant"].battery
    expected = config["farms"][farm_id]["battery"]
    assert prepared["plant"].farm_id == int(farm_id)
    assert battery.capacity_mwh == expected["capacity_mwh"]
    assert battery.max_power_mw == expected["max_power_mw"]
    assert prepared["initial_energy_mwh"] == pytest.approx(
        battery.capacity_mwh * config["state"]["initial_soc_percent"] / 100)
    assert prepared["terminal_energy_mwh"] == pytest.approx(
        battery.capacity_mwh * config["state"]["terminal_soc_percent"] / 100)
    assert prepared["provenance"]["asset_input_settings"]["battery"] == config["battery"]


def test_continuous_preflight_shares_inputs_and_reads_no_previous_day_observations(prepared_case):
    config, calls, _ = prepared_case
    before = deepcopy(config)
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    assert config == before
    assert [day for day, _, _ in calls["pv"]] == list(prepared["days"])
    first_day = prepared["days"][0]
    assert calls["pv"][0][1].tz_convert(prepared["calendar"].timezone).date() < pd.Timestamp(first_day).date()
    assert calls["pv"][0][2].starts[0] == prepared["full_grid"].starts[0]
    np.testing.assert_array_equal(prepared["full_realized_pv"], config["test_case"]["realized_mw"])
    np.testing.assert_array_equal(prepared["imbalance_prices_eur_per_mwh"], config["test_case"]["settlement_eur_mwh"])
    assert prepared["initial_energy_mwh"] == pytest.approx(config["battery"]["capacity_mwh"] * config["state"]["initial_soc_percent"] / 100)
    for item in prepared["days_data"].values():
        assert item["native_interval_minutes_by_auction"]["DA"] == config["test_case"]["expected_da_hourly_minutes"]
        assert item["native_interval_minutes_by_auction"]["IDA3"] == config["test_case"]["expected_da_quarter_minutes"]


def test_forecast_provider_covers_projection_and_target_and_is_defensive(prepared_case):
    config, calls, _ = prepared_case
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    second = prepared["days_data"][prepared["days"][-1]]
    event = next(event for event in second["schedule"] if event.auction == "DA")
    global_starts = pd.DatetimeIndex(prepared["full_grid"].starts)
    requested = global_starts[(global_starts >= event.submission_at.floor("15min"))
                             & (global_starts < second["grid"].ends[-1])]
    provider = prepared["forecast_provider"]
    forecast = provider(event.submission_at, requested)
    assert forecast.values.index.equals(requested)
    assert requested[0] < second["grid"].starts[0]
    n_reads = len(calls["weather"])
    forecast.values.loc[:, "pv_ac_mw"] = config["test_case"]["changed_forecast_mw"]
    forecast.provenance["mutated"] = True
    repeated = provider(event.submission_at, requested)
    np.testing.assert_array_equal(repeated.values.pv_ac_mw, config["test_case"]["forecast_mw"])
    assert "mutated" not in repeated.provenance
    assert len(calls["weather"]) == n_reads
    assert provider(event.submission_at, second["grid"].starts).values.index.equals(pd.DatetimeIndex(second["grid"].starts))


def test_inputs_fingerprint_includes_weather_not_only_final_pv(prepared_case):
    config, _, _ = prepared_case
    original = continuous_inputs.prepare_continuous_inputs(config)
    again = continuous_inputs.prepare_continuous_inputs(config)
    assert original["input_fingerprint"] == again["input_fingerprint"]
    config["test_case"]["marker_value"] = config["test_case"]["changed_marker_value"]
    changed = continuous_inputs.prepare_continuous_inputs(config)
    assert original["input_fingerprint"] != changed["input_fingerprint"]
    np.testing.assert_array_equal(original["full_realized_pv"], changed["full_realized_pv"])


def test_cancelled_auctions_are_not_preflighted_for_submission(prepared_case, monkeypatch):
    config, _, _ = prepared_case
    canceled = config["test_case"]["cancelled_auction"]
    original = continuous_inputs._load_auction_prices

    def prices(*args):
        result, sources, _ = original(*args)
        del result[canceled]
        return result, sources, [{"auction": canceled, "reason": "documented_cancelled_auction"}]

    monkeypatch.setattr(continuous_inputs, "_load_auction_prices", prices)
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    for day in prepared["days_data"].values():
        assert canceled not in [event.auction for event in day["schedule"]]
        assert day["skipped"][0]["reason"] == "documented_cancelled_auction"


@pytest.mark.parametrize("day_key", ["alternate_day", "dst_day"])
def test_native_resolution_and_dst_are_derived_per_delivery_day(prepared_case, day_key):
    config, _, _ = prepared_case
    day = config["test_case"][day_key]
    config["continuous_optimization"].update(start_date=day, end_date=day)
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    item = prepared["days_data"][day]
    assert sum(item["grid"].delta_hours) == (
        pd.Timestamp(item["grid"].ends[-1]) - pd.Timestamp(item["grid"].starts[0])).total_seconds() / 3600
    expected = config["test_case"]["expected_da_quarter_minutes" if day_key == "alternate_day" else "expected_da_hourly_minutes"]
    assert item["native_interval_minutes_by_auction"]["DA"] == expected


def test_forecast_availability_and_unprepared_requests_fail(prepared_case, monkeypatch):
    config, _, bundle = prepared_case
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    provider = prepared["forecast_provider"]
    first = prepared["days_data"][prepared["days"][0]]
    event = first["schedule"][0]
    with pytest.raises(ValueError, match="not preflighted"):
        provider(event.submission_at - pd.Timedelta(minutes=config["market"]["decision_buffer_minutes"]), first["grid"].starts)
    unavailable = lambda config, calendar, cutoff, starts: replace(bundle(cutoff, starts),
        available_at=pd.Timestamp(cutoff) + pd.Timedelta(hours=config["test_case"]["cycle_age_hours"]))
    monkeypatch.setattr(continuous_inputs, "_load_pv_forecast", unavailable)
    with pytest.raises(ValueError, match="after submission"):
        continuous_inputs.prepare_continuous_inputs(config)


def test_missing_prices_remain_an_error_and_settlement_can_be_disabled(prepared_case, monkeypatch):
    config, _, _ = prepared_case
    monkeypatch.setattr(continuous_inputs, "load_realized_imbalance_prices", lambda config, grid:
        (None, {"status": "disabled"}))
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    assert prepared["imbalance_prices_eur_per_mwh"] is None
    monkeypatch.setattr(continuous_inputs, "_load_auction_prices", lambda *args:
        (_ for _ in ()).throw(ValueError("missing prices do not establish cancellation")))
    with pytest.raises(ValueError, match="Continuous preflight.*missing prices"):
        continuous_inputs.prepare_continuous_inputs(config)


def test_delayed_telemetry_is_rejected_before_loading_inputs(prepared_case):
    config, calls, _ = prepared_case
    config["source"]["telemetry_latency_minutes"] = config["test_case"]["unsupported_telemetry_latency_minutes"]
    with pytest.raises(ValueError, match="immediate completed-interval observations"):
        continuous_inputs.prepare_continuous_inputs(config)
    assert not any(calls.values())

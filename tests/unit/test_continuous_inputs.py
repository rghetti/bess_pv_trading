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


def test_ensemble_selection_uses_latest_available_covering_cycle():
    """New member paths must not enter before release or outside their coverage."""
    from types import SimpleNamespace

    starts = pd.date_range("2026-01-16T05:30Z", periods=6, freq="15min")
    base_info = {"forecast_vintage_utc": "2026-01-15T00:00:00+00:00",
                 "available_at_utc": "2026-01-15T06:00:00+00:00"}
    update_info = {"forecast_vintage_utc": "2026-01-16T00:00:00+00:00",
                   "available_at_utc": "2026-01-16T06:00:00+00:00"}
    daily = {"grid": SimpleNamespace(starts=starts), "ensemble_pv": np.ones((2, 6)),
             "ensemble_members": ("a", "b"), "ensemble_updates": [{"starts": starts[2:],
             "values": np.full((2, 4), 2.0), "members": ("a", "b"), "provenance": update_info}]}
    early, _, early_info = continuous_inputs.ensemble_at_cutoff(daily, base_info, starts[1], starts[2:])
    latest, _, latest_info = continuous_inputs.ensemble_at_cutoff(daily, base_info, starts[2], starts[2:])
    uncovered, _, _ = continuous_inputs.ensemble_at_cutoff(daily, base_info, starts[2], starts[1:])
    assert np.all(early == 1.0)
    assert early_info is base_info
    assert np.all(latest == 2.0)
    assert latest_info is update_info
    assert np.all(uncovered == 1.0)
    with pytest.raises(ValueError, match="No available ensemble"):
        continuous_inputs.ensemble_at_cutoff(daily, base_info, pd.Timestamp("2026-01-15T05:59Z"), starts)


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

    def load_prices(daily, calendar, grid, events, **kwargs):
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


def test_curated_continuous_provenance_discloses_synthetic_execution(prepared_case):
    """Top-level labels must agree with the selected reviewed production source."""
    config, _, _ = prepared_case
    config["forecast"]["curated_realized"] = {"enabled": True, "dataset_root": "unused"}
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    provenance = prepared["provenance"]
    assert "retrospective" in provenance["measured_pv_basis"]
    assert "no forecast fallback" in provenance["realized_pv_fallback"]


def test_continuous_preflight_shares_inputs_and_reads_no_previous_day_observations(prepared_case):
    config, calls, _ = prepared_case
    before = deepcopy(config)
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    assert config == before
    assert "forecast_model_artifacts" not in prepared["provenance"]
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


def test_extended_ida3_preflight_pins_next_day_without_scoring_it(prepared_case, monkeypatch):
    """An IDA3 cutoff can request the next full delivery day as context."""
    config, _, _ = prepared_case
    config["continuous_optimization"].update(ida3_next_day_horizon=True, context_end_date="2025-01-03")
    seen = []
    monkeypatch.setattr(continuous_inputs, "_replacement_evidence", lambda config: seen.append(True))
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    assert seen == [True]
    assert prepared["days"] == ("2025-01-01", "2025-01-02")
    assert prepared["context_days"] == ("2025-01-03",)
    assert len(prepared["full_grid"].interval_ids) == 192
    current = prepared["days_data"]["2025-01-02"]
    following = prepared["days_data"]["2025-01-03"]
    cutoff = next(event.submission_at for event in current["schedule"] if event.auction == "IDA3")
    bundle = prepared["forecast_provider"](cutoff, following["grid"].starts)
    assert len(bundle.values) == 96
    assert prepared["provenance"]["scored_end_date"] == "2025-01-02"


def test_extended_ida3_requires_context_day(prepared_case):
    """A replay cannot silently shorten the last IDA3 planning horizon."""
    config, _, _ = prepared_case
    config["continuous_optimization"]["ida3_next_day_horizon"] = True
    with pytest.raises(ValueError, match="requires at least one context"):
        continuous_inputs.prepare_continuous_inputs(config)


def test_mpc_preflight_pins_every_physical_cutoff_without_enabling_auction_ensemble(prepared_case):
    config, _, _ = prepared_case
    config["execution"]["dispatch_policy"] = "deterministic_mpc"
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    for day in prepared["days"]:
        grid = prepared["days_data"][day]["grid"]
        for index, cutoff in enumerate(grid.starts):
            bundle = prepared["forecast_provider"](cutoff, grid.starts[index:])
            assert bundle.available_at <= cutoff
    assert not config["continuous_optimization"]["pv_ensemble"]["enabled"]


def test_selected_bundle_initialization_and_artifacts_are_pinned(prepared_case, monkeypatch, tmp_path):
    import hashlib

    config, _, _ = prepared_case
    config["solver"]["threads"] = 1
    config["execution"]["dispatch_policy"] = "deterministic_mpc"
    config["pv"]["forecast_replacement"]["enabled"] = True
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"checksums": {}, "training_period": "2025"}')
    model = config["pv"]["forecast_replacement"]["models_by_farm"][str(config["farm_id"])]
    model.update(artifact_path=str(manifest), available_at_utc="2020-01-01T00:00:00Z")
    monkeypatch.setattr("bess_pv_trading.forecasting.inference.pv_forecast_bundle.predict_bundle",
        lambda frame, *args, **kwargs: np.full(len(frame), 0.2))
    prepared = continuous_inputs.prepare_continuous_inputs(config)
    for item in prepared["days_data"].values():
        np.testing.assert_array_equal(item["initial_forecast"], 0.2)
        assert item["pv_provenance"]["initial_forecast_model"] == model["name"]
    evidence = prepared["provenance"]["forecast_model_artifacts"]
    assert evidence["checksums"]["manifest.json"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert evidence["training_period"] == "2025"


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

    def prices(*args, **kwargs):
        result, sources, _ = original(*args, **kwargs)
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
    monkeypatch.setattr(continuous_inputs, "_load_auction_prices", lambda *args, **kwargs:
        (_ for _ in ()).throw(ValueError("missing prices do not establish cancellation")))
    with pytest.raises(ValueError, match="Continuous preflight.*missing prices"):
        continuous_inputs.prepare_continuous_inputs(config)


def test_delayed_telemetry_is_rejected_before_loading_inputs(prepared_case):
    config, calls, _ = prepared_case
    config["source"]["telemetry_latency_minutes"] = config["test_case"]["unsupported_telemetry_latency_minutes"]
    with pytest.raises(ValueError, match="immediate completed-interval observations"):
        continuous_inputs.prepare_continuous_inputs(config)
    assert not any(calls.values())


def test_reviewed_issue_context_is_fixed_before_slicing(monkeypatch, tmp_path):
    """Overlapping requests use one complete vintage and identical predictions."""
    import json
    from bess_pv_trading.forecasting.inference import pv_reviewed_hpo

    artifact = tmp_path / "result.json"
    artifact.write_text(json.dumps({"input_contract": {"issue_horizon_hours": 48}}))
    config = {"forecast": {"farm_id": 1}, "market": {"timezone": "Europe/Rome"},
        "pv": {"forecast_replacement": {"models_by_farm": {"1": {"kind": "reviewed_hpo"}}}},
        "policy": {"forecast_pv_night": {"enabled": True, "monthly_windows": {}}}}
    evidence = {"status": "pinned", "artifact": str(artifact), "model": "selected"}
    cycle = pd.Timestamp("2026-01-15T00:00:00Z")
    available = cycle + pd.Timedelta(hours=6)
    issue = pd.date_range(cycle, periods=192, freq="15min")
    calls = []

    def load(config, calendar, cutoff, starts):
        calls.append(pd.DatetimeIndex(starts))
        values = pd.DataFrame({"pv_ac_mw": np.zeros(len(starts)), "weather": np.arange(len(starts))}, index=starts)
        return ForecastBundle(1, cycle, available, values, {"pv_ac_mw": "MW"})

    def predict(frame, directory, *, farm_id, night_policy):
        assert len(frame) == 192
        assert frame.forecast_cycle_utc.nunique() == 1
        return np.arange(len(frame), dtype=float)

    monkeypatch.setattr(continuous_inputs, "_load_pv_forecast", load)
    monkeypatch.setattr(pv_reviewed_hpo, "predict_reviewed_hpo", predict)
    cache = {}
    first = load(config, None, available, issue[24:60])
    second = load(config, None, available, issue[30:40])
    a = continuous_inputs._issue_context_prediction(config, None, available, first, evidence, cache)
    b = continuous_inputs._issue_context_prediction(config, None, available, second, evidence, cache)
    np.testing.assert_array_equal(a.values.loc[issue[30:40], "pv_ac_mw"], b.values.pv_ac_mw)
    np.testing.assert_array_equal(b.values.pv_ac_mw, np.arange(30, 40))
    assert len(calls) == 3
    assert len(calls[-1]) == 192
    assert a.provenance["model_issue_context_hours"] == 48
    outside = load(config, None, available, pd.DatetimeIndex([issue[-1] + pd.Timedelta(minutes=15)]))
    with pytest.raises(ValueError, match="outside.*context"):
        continuous_inputs._issue_context_prediction(config, None, available, outside, evidence, cache)


def test_ensemble_context_fill_uses_only_same_vintage_before_delivery(monkeypatch):
    """Context repair never substitutes delivery weather or a later forecast issue."""
    cycle = pd.Timestamp("2025-12-31T00:00Z")
    issue = pd.date_range(cycle, periods=4, freq="15min")
    delivery = issue[2:]
    frame = pd.DataFrame({"interval_start_utc": list(issue) * 2,
        "forecast_ghi": [np.nan, 1.0, np.nan, 3.0] * 2})
    donor = pd.DataFrame({"forecast_ghi": [7.0, 8.0, 9.0, 10.0]}, index=issue)
    config = {"forecast_mapping": {"forecast_ghi": "shortwave_radiation"},
        "continuous_optimization": {"pv_ensemble": {"issue_context_missing_policy": "same_vintage_deterministic"}}}
    def load(config, calendar, cutoff, starts):
        return ForecastBundle(1, cycle, cycle + pd.Timedelta(hours=6), donor, {"forecast_ghi": "W/m2"})
    monkeypatch.setattr(continuous_inputs, "_load_pv_forecast", load)
    cutoff = cycle + pd.Timedelta(hours=6)
    filled, evidence = continuous_inputs._fill_prior_issue_context(frame, delivery, issue,
        config=config, calendar=None, cutoff=cutoff, cycle=cycle)
    assert filled.forecast_ghi.iloc[0] == 7 and filled.forecast_ghi.iloc[4] == 7
    assert pd.isna(filled.forecast_ghi.iloc[2]) and pd.isna(filled.forecast_ghi.iloc[6])
    assert evidence["filled_member_values_by_feature"] == {"forecast_ghi": 2}
    assert evidence["forecast_vintage_utc"] == cycle.isoformat()
    with pytest.raises(ValueError, match="same eligible"):
        continuous_inputs._fill_prior_issue_context(frame, delivery, issue,
            config=config, calendar=None, cutoff=cutoff, cycle=cycle + pd.Timedelta(hours=6))

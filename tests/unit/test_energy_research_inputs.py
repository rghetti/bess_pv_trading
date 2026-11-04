"""Integration contracts for price artifacts, information refreshes and MPC inputs."""

from copy import deepcopy
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.domain import BatteryParameters, DegradationSettings, DeliveryGrid, PhysicalState, PlantParameters
from bess_pv_trading.forecasting.energy_prices import CausalPriceForecaster, PriceForecastSettings, PricePrediction
from bess_pv_trading.optimization.mpc import ForecastPrices, MpcAction, MpcPlan, MpcResult
from bess_pv_trading.simulation import energy_price_inputs, realtime_control
from bess_pv_trading.simulation.continuous import ContinuousReplay
from bess_pv_trading.simulation.research_inputs import (
    auction_price_predictions, frozen_initial_pv, initial_information_cutoff,
    price_information_cutoff, validate_information_settings,
)


def short_grid(start="2026-01-02T12:00:00Z", count=4):
    """Small physical grids exercise integration without solving or reading catalogs."""
    boundaries = pd.date_range(start, periods=count + 1, freq="15min")
    return DeliveryGrid(tuple(stamp.isoformat() for stamp in boundaries[:-1]),
        tuple(boundaries[:-1].to_pydatetime()), tuple(boundaries[1:].to_pydatetime()))


def plant_for():
    return PlantParameters(1, 1, 1, BatteryParameters(capacity_mwh=1, max_power_mw=1,
        converter_charge_efficiency=1, converter_discharge_efficiency=1, energy_min_mwh=0, energy_max_mwh=1))


def daily_for(grid):
    initial = pd.Timestamp("2026-01-01T11:00:00Z")
    schedule = [SimpleNamespace(auction=name, submission_at=initial + pd.Timedelta(hours=offset),
        results_at=initial + pd.Timedelta(hours=offset, minutes=30))
        for name, offset in (("DA", 0), ("IDA1", 3), ("IDA3", 23))]
    return {"grid": grid, "schedule": schedule, "research_initial_cutoff": initial.isoformat(),
        "prices": {event.auction: np.full(len(grid.starts), 100 + index) for index, event in enumerate(schedule)},
        "initial_forecast": np.arange(len(grid.starts), dtype=float) / 10,
        "pv_metadata": {"pv_forecast_vintage_utc": "2026-01-01T00:00:00Z",
                        "pv_forecast_available_at_utc": "2026-01-01T06:00:00Z"}}


class RecordingPriceProvider:
    """A clock-sensitive provider identifies exact preflight requests and disclosures."""

    def __init__(self, negative=False, future_negative=False, fail_continuation=False, point_only=False):
        self.models = {"settlement": object()}
        if negative:
            self.models["settlement_negative"] = object()
        self.calls = []
        self.provenance = {"source": "synthetic_software_fixture"}
        self.future_negative = future_negative
        self.fail_continuation = fail_continuation
        self.point_only = point_only

    def predict(self, name, cutoff, starts, scenario_count=1):
        starts = pd.DatetimeIndex(starts)
        cutoff = pd.Timestamp(cutoff)
        self.calls.append((name, cutoff, starts, scenario_count))
        if self.point_only and scenario_count > 1:
            raise ValueError("No residual calibration for a deterministic predictor")
        if self.fail_continuation and name == "settlement_negative" and starts[0].hour >= 13:
            raise ValueError("Negative continuation curve lacks eligible history")
        available = cutoff + pd.Timedelta(hours=1) if name == "settlement_negative" and self.future_negative else cutoff
        point = np.full(len(starts), -10 if name == "settlement_negative" else 20)
        return PricePrediction(point, np.tile(point, (scenario_count, 1)), available,
            {"forecast_cutoff_utc": cutoff.isoformat(), "historical_availability": "declared_test_clock"})


@pytest.fixture(scope="module")
def synthetic_price_model():
    """A short deterministic fit is software verification, not historical training."""
    starts = pd.date_range("2025-12-01", "2026-01-05", freq="15min", inclusive="left", tz="UTC")
    history = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(minutes=15),
        "available_at": starts + pd.Timedelta(hours=4, minutes=15),
        "price_eur_per_mwh": 50 + 10 * np.sin(np.arange(len(starts)) / 96 * 2 * np.pi)})
    model = CausalPriceForecaster.fit(history, cutoff="2026-01-01T00:00:00Z",
        settings=PriceForecastSettings(min_training_rows=96, calibration_days=3),
        provenance={"availability": {"kind": "assumed_lag_after_product_end", "lag_hours": 4}})
    return model, history


def test_provider_pins_model_and_history_hashes_and_filters_future_sidecar_labels(tmp_path, synthetic_price_model):
    model, history = synthetic_price_model
    artifact = tmp_path / "model.json"
    model.save(artifact)
    original = tmp_path / "history.parquet"
    history.to_parquet(original)
    config = {"_project_root": str(tmp_path), "energy_price_forecast": {
        "settlement_enabled": True, "artifacts": {"settlement": "model.json", "DA": "unused_missing.json"},
        "history_paths": {"settlement": "history.parquet"}, "seed": 10}}
    provider = energy_price_inputs.EnergyPriceProvider(config)
    assert set(provider.models) == {"settlement"}
    assert provider.provenance["settlement"]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert provider.provenance["settlement"]["observation_history"]["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    cutoff = pd.Timestamp("2026-01-02T00:00:00Z")
    starts = pd.DatetimeIndex(short_grid(start=cutoff).starts)
    expected = provider.predict("settlement", cutoff, starts, scenario_count=3)
    changed = history.copy()
    changed.loc[changed.available_at > cutoff, "price_eur_per_mwh"] += 1e6
    changed.to_parquet(original)
    revised = energy_price_inputs.EnergyPriceProvider(config).predict("settlement", cutoff, starts, scenario_count=3)
    np.testing.assert_array_equal(expected.scenarios_eur_per_mwh, revised.scenarios_eur_per_mwh)
    assert expected.provenance["availability"]["kind"] == "assumed_lag_after_product_end"
    assert expected.provenance["pv_price_coupling"] == "independent_residual_sampling_assumption"
    # The loaded provider owns the already prepared sidecar, not a live mutable file.
    np.testing.assert_array_equal(expected.scenarios_eur_per_mwh,
        provider.predict("settlement", cutoff, starts, scenario_count=3).scenarios_eur_per_mwh)


def test_provider_preserves_residual_members_as_horizon_shrinks(tmp_path, synthetic_price_model):
    model, _ = synthetic_price_model
    model.save(tmp_path / "model.json")
    config = {"_project_root": str(tmp_path), "energy_price_forecast": {
        "settlement_enabled": True, "artifacts": {"settlement": "model.json"}, "seed": 3}}
    provider = energy_price_inputs.EnergyPriceProvider(config)
    starts = pd.DatetimeIndex(short_grid().starts)
    cutoff = pd.Timestamp("2026-01-01T12:00:00Z")
    full = provider.predict("settlement", cutoff, starts, scenario_count=4)
    suffix = provider.predict("settlement", cutoff, starts[2:], scenario_count=4)
    np.testing.assert_array_equal(full.scenarios_eur_per_mwh[:, 2:], suffix.scenarios_eur_per_mwh)
    with pytest.raises(ValueError, match="not available"):
        provider.predict("settlement", pd.Timestamp("2025-12-31T00:00:00Z"), starts)


def prepared_for(grid):
    return {"days_data": {"2026-01-02": daily_for(grid)}, "provenance": {},
            "calendar": SimpleNamespace(timezone="Europe/Rome")}


def test_preflight_checks_each_information_cutoff_before_replay(monkeypatch):
    grid = short_grid()
    provider = RecordingPriceProvider(negative=True)
    monkeypatch.setattr(energy_price_inputs, "EnergyPriceProvider", lambda config: provider)
    config = {"energy_price_forecast": {"settlement_enabled": True, "auction_enabled": True},
        "execution": {"dispatch_policy": "ensemble_mpc_adaptive"}}
    prepared = prepared_for(grid)
    energy_price_inputs.prepare_energy_price_provider(config, prepared)
    settlement = [call for call in provider.calls if call[0] == "settlement"]
    negative = [call for call in provider.calls if call[0] == "settlement_negative"]
    assert [call[1] for call in settlement] == list(pd.DatetimeIndex(grid.starts))
    assert len(negative) == len(grid.starts)
    assert all(call[3] > 1 for call in settlement)
    assert prepared["provenance"]["energy_price_prediction_checks"] == len(provider.calls)
    assert prepared["energy_price_provider"] is provider
    for name, cutoff, starts, _ in provider.calls:
        assert (starts >= cutoff).all()


def test_preflight_frozen_information_reuses_initial_cutoff_and_keeps_horizon_products(monkeypatch):
    grid = short_grid()
    provider = RecordingPriceProvider()
    monkeypatch.setattr(energy_price_inputs, "EnergyPriceProvider", lambda config: provider)
    config = {"energy_price_forecast": {"settlement_enabled": True},
        "research_information": {"price_updates": "frozen_initial"},
        "execution": {"dispatch_policy": "ensemble_mpc_adaptive"}}
    prepared = prepared_for(grid)
    energy_price_inputs.prepare_energy_price_provider(config, prepared)
    initial = pd.Timestamp(prepared["days_data"]["2026-01-02"]["research_initial_cutoff"])
    assert all(call[1] == initial for call in provider.calls)
    assert [len(call[2]) for call in provider.calls] == [4, 3, 2, 1]


def test_preflight_deterministic_economics_does_not_require_unused_residuals(monkeypatch):
    provider = RecordingPriceProvider(point_only=True)
    monkeypatch.setattr(energy_price_inputs, "EnergyPriceProvider", lambda config: provider)
    config = {"energy_price_forecast": {"settlement_enabled": True},
              "execution": {"dispatch_policy": "deterministic_mpc_economic"}}
    energy_price_inputs.prepare_energy_price_provider(config, prepared_for(short_grid()))
    assert all(call[3] == 1 for call in provider.calls)


def test_preflight_rejects_missing_directional_continuation_history(monkeypatch):
    provider = RecordingPriceProvider(negative=True, fail_continuation=True)
    monkeypatch.setattr(energy_price_inputs, "EnergyPriceProvider", lambda config: provider)
    config = {"energy_price_forecast": {"settlement_enabled": True},
        "execution": {"dispatch_policy": "ensemble_mpc_adaptive"},
        "realtime_mpc": {"terminal_inventory_mode": "piecewise_linear"},
        "continuation_value": {"horizon_hours": 1}}
    with pytest.raises(ValueError, match="Negative continuation"):
        energy_price_inputs.prepare_energy_price_provider(config, prepared_for(short_grid()))


def test_directional_settlement_uses_latest_availability_and_rejects_future_negative_prices():
    grid = short_grid()
    daily = daily_for(grid)
    cutoff = pd.Timestamp(grid.starts[0])
    replay = SimpleNamespace(config={"energy_price_forecast": {"settlement_enabled": True}},
        prepared={"energy_price_provider": RecordingPriceProvider(negative=True)})
    prices = energy_price_inputs.settlement_forecasts(replay, daily, cutoff, pd.DatetimeIndex(grid.starts), 3)
    assert prices.positive_eur_per_mwh.shape == (3, 4)
    assert prices.available_at == cutoff
    np.testing.assert_array_equal(prices.negative_eur_per_mwh, -10)
    replay.prepared["energy_price_provider"] = RecordingPriceProvider(negative=True, future_negative=True)
    with pytest.raises(ValueError, match="unavailable"):
        energy_price_inputs.settlement_forecasts(replay, daily, cutoff, pd.DatetimeIndex(grid.starts), 3)


@pytest.mark.parametrize("config", [
    {"energy_price_forecast": {"unknown": True}},
    {"energy_price_forecast": {"settlement_enabled": 1}},
    {"energy_price_forecast": {"settlement_enabled": True}},
    {"energy_price_forecast": {"seed": -1}},
    {"realtime_scenarios": {"dependence": "ecc"}},
    {"realtime_scenarios": {"reduce_to": True}},
    {"realtime_scenarios": {"seed": False}},
    {"continuation_value": {"unknown": 2}},
    {"research_information": {"pv_updates": "hourly"}},
])
def test_invalid_study_schema_fails_before_source_access(config):
    with pytest.raises((ValueError, TypeError)):
        validate_information_settings(config)
        energy_price_inputs.validate_energy_research_settings(config)


def test_scenario_transform_preserves_joint_price_samples_through_reduction():
    grid = short_grid()
    values = np.arange(16).reshape(4, 4) / 20
    prices = ForecastPrices(100 * values + 1, grid.starts[0], -50 * values - 3)
    config = {"realtime_scenarios": {"dependence": "independent_shuffle", "seed": 7, "reduce_to": 2}}
    bundle, transformed, selected, evidence = energy_price_inputs.transform_mpc_scenarios(
        config, grid, pd.Timestamp(grid.starts[0]), plant_for(), values, tuple(f"s{i}" for i in range(4)),
        np.full(4, 0.25), np.zeros(4), prices)
    np.testing.assert_allclose(transformed.positive_eur_per_mwh, 100 * bundle.pv_ac_mw + 1)
    np.testing.assert_allclose(transformed.negative_eur_per_mwh, -50 * bundle.pv_ac_mw - 3)
    assert transformed.available_at == prices.available_at
    assert len(selected) == 2
    assert bundle.weights.sum() == pytest.approx(1)
    assert evidence["dependence_transform"]["joint_interval_pairs_preserved"]
    assert isinstance(evidence["scenario_reduction"]["settings"], dict)


@pytest.mark.parametrize("availability,cover_horizon", [
    ("2026-01-03T00:00:00Z", True), ("2026-01-01T00:00:00", True), ("2026-01-01T00:00:00Z", False),
])
def test_ecc_artifact_must_be_available_and_cover_exact_products(tmp_path, availability, cover_horizon):
    grid = short_grid()
    starts = [stamp.isoformat() for stamp in grid.starts]
    path = tmp_path / "calibrated.npz"
    np.savez(path, available_at_utc=np.array(availability), interval_starts_utc=np.array(starts if cover_horizon else starts[:-1]),
        pv_ac_mw=np.ones((2, 4 if cover_horizon else 3)))
    config = {"_project_root": str(tmp_path), "realtime_scenarios": {
        "dependence": "ecc", "calibrated_marginals_path": "calibrated.npz"}}
    with pytest.raises(ValueError, match="unavailable|cover"):
        energy_price_inputs.transform_mpc_scenarios(config, grid, pd.Timestamp(grid.starts[0]), plant_for(),
            np.ones((2, 4)), ("a", "b"), np.array([0.5, 0.5]), np.zeros(4))


def test_ecc_subselects_products_by_identity_and_records_checksum(tmp_path):
    grid = short_grid()
    starts = pd.date_range(pd.Timestamp(grid.starts[0]) - pd.Timedelta(minutes=15), periods=5, freq="15min")
    path = tmp_path / "calibrated.npz"
    samples = np.array([[9, 0.3, 0.4, 0.5, 0.6], [8, 0.1, 0.2, 0.3, 0.4]])
    np.savez(path, available_at_utc=np.array("2026-01-01T00:00:00Z"),
        interval_starts_utc=np.array(starts.astype(str), dtype=str), pv_ac_mw=samples)
    config = {"_project_root": str(tmp_path), "realtime_scenarios": {
        "dependence": "ecc", "calibrated_marginals_path": "calibrated.npz"}}
    bundle, _, _, evidence = energy_price_inputs.transform_mpc_scenarios(config, grid, pd.Timestamp(grid.starts[0]),
        plant_for(), np.array([[0, 1, 0, 1], [1, 0, 1, 0]]), ("a", "b"), np.array([0.5, 0.5]), np.zeros(4))
    np.testing.assert_array_equal(np.sort(bundle.pv_ac_mw, axis=0), np.sort(samples[:, 1:], axis=0))
    assert evidence["calibrated_marginals_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def affine_calibration_artifact(path, *, kind_key="artifactkind", **overrides):
    """Stored evaluation samples deliberately differ from the later runtime grid."""
    payload = {
        kind_key: np.array("affine_marginal_calibration"),
        "available_at_utc": np.array("2026-01-14T23:00:00Z"),
        "calibration_fit_cutoff_utc": np.array("2026-01-14T23:00:00Z"),
        "calibration_bias_normalized": np.array(0.1),
        "calibration_spread_scale": np.array(1.5),
        "ac_nameplate_mw": np.array(1.0),
        "farm_id": np.array(1),
        "interval_starts_utc": np.array(["2026-01-15T12:00:00Z", "2026-01-15T12:15:00Z"]),
        "forecast_available_at_utc": np.array(["2026-01-14T06:00:00Z"] * 2),
        "pv_ac_mw": np.full((2, 2), 0.99),
        "scenario_ids": np.array(["old_a", "old_b"]),
    }
    payload.update({key: np.array(value) for key, value in overrides.items()})
    np.savez(path, **payload)


@pytest.mark.parametrize("kind_key", ["artifactkind", "artifact_kind"])
def test_affine_calibration_uses_current_forecasts_on_future_grids_and_new_members(tmp_path, kind_key):
    grid = short_grid("2026-08-02T12:00:00Z")
    path = tmp_path / "calibrated.npz"
    affine_calibration_artifact(path, kind_key=kind_key)
    config = {"_project_root": str(tmp_path), "policy": {"forecast_pv_night": {"enabled": False}},
        "realtime_scenarios": {"dependence": "ecc", "calibrated_marginals_path": "calibrated.npz"}}
    values = np.array([[0.05, 0.1, 0.3, 0.5], [0.2, 0.3, 0.6, 0.7], [0.4, 0.5, 0.8, 0.9]])
    weights = np.full(3, 1 / 3)
    ids = ("new_a", "new_b", "new_c")
    prices = ForecastPrices(100 * values + 1, grid.starts[0], -50 * values - 3)
    bundle, transformed_prices, _, evidence = energy_price_inputs.transform_mpc_scenarios(
        config, grid, pd.Timestamp(grid.starts[0]), plant_for(), values, ids, weights, np.zeros(4), prices)
    expected = np.clip(values.mean(axis=0) + 0.1 + 1.5 * (values - values.mean(axis=0)), 0, 1)
    np.testing.assert_allclose(bundle.pv_ac_mw, expected)
    np.testing.assert_array_equal(transformed_prices.positive_eur_per_mwh, prices.positive_eur_per_mwh)
    assert bundle.scenario_ids == ids
    assert evidence["calibration_kind"] == "affine_marginal_calibration"
    assert evidence["calibrated_marginals_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # A newly eligible deterministic/ensemble forecast must change the result.
    updated = values * 0.8
    changed, _, _, _ = energy_price_inputs.transform_mpc_scenarios(config, grid, pd.Timestamp(grid.starts[0]),
        plant_for(), updated, ids, weights, np.zeros(4))
    assert not np.array_equal(changed.pv_ac_mw, bundle.pv_ac_mw)
    # Dependence contrasts retain exactly the same calibrated marginal samples.
    config["realtime_scenarios"]["dependence"] = "independent_shuffle"
    shuffled, _, _, _ = energy_price_inputs.transform_mpc_scenarios(config, grid, pd.Timestamp(grid.starts[0]),
        plant_for(), values, ids, weights, np.zeros(4))
    np.testing.assert_array_equal(np.sort(shuffled.pv_ac_mw, axis=0), np.sort(bundle.pv_ac_mw, axis=0))


@pytest.mark.parametrize("overrides, message", [
    ({"available_at_utc": "2026-08-03T00:00:00Z"}, "unavailable"),
    ({"calibration_fit_cutoff_utc": "2026-01-15T00:00:00Z"}, "fit cutoff"),
    ({"calibration_fit_cutoff_utc": "2026-01-14T23:00:00"}, "fit cutoff"),
    ({"calibration_fit_cutoff_utc": "NaT"}, "fit cutoff"),
    ({"farm_id": 345}, "different farm"),
    ({"calibration_spread_scale": -0.1}, "Invalid affine"),
])
def test_affine_calibration_checks_model_clock_farm_and_parameters(tmp_path, overrides, message):
    grid = short_grid("2026-08-02T12:00:00Z")
    affine_calibration_artifact(tmp_path / "calibrated.npz", **overrides)
    config = {"_project_root": str(tmp_path), "realtime_scenarios": {
        "dependence": "ecc", "calibrated_marginals_path": "calibrated.npz"}}
    with pytest.raises(ValueError, match=message):
        energy_price_inputs.transform_mpc_scenarios(config, grid, pd.Timestamp(grid.starts[0]), plant_for(),
            np.full((3, 4), 0.2), ("a", "b", "c"), np.full(3, 1 / 3), np.zeros(4))


def test_affine_artifact_cache_freezes_parameters_and_hash_for_one_replay(tmp_path):
    grid = short_grid("2026-08-02T12:00:00Z")
    path = tmp_path / "calibrated.npz"
    affine_calibration_artifact(path)
    config = {"_project_root": str(tmp_path), "realtime_scenarios": {
        "dependence": "ecc", "calibrated_marginals_path": "calibrated.npz"}}
    cache = {}
    arguments = (config, grid, pd.Timestamp(grid.starts[0]), plant_for(),
        np.full((3, 4), 0.2), ("a", "b", "c"), np.full(3, 1 / 3), np.zeros(4))
    original, _, _, evidence = energy_price_inputs.transform_mpc_scenarios(*arguments, artifact_cache=cache)
    affine_calibration_artifact(path, calibration_bias_normalized=0.6)
    frozen, _, _, same_evidence = energy_price_inputs.transform_mpc_scenarios(*arguments, artifact_cache=cache)
    np.testing.assert_array_equal(frozen.pv_ac_mw, original.pv_ac_mw)
    assert same_evidence["calibrated_marginals_sha256"] == evidence["calibrated_marginals_sha256"]
    refreshed, _, _, _ = energy_price_inputs.transform_mpc_scenarios(*arguments, artifact_cache={})
    assert not np.array_equal(refreshed.pv_ac_mw, original.pv_ac_mw)


def frozen_replay():
    grid = short_grid("2026-01-02T23:30:00Z", 4)
    days = ("2026-01-02", "2026-01-03")
    daily = {day: daily_for(DeliveryGrid(grid.interval_ids[i:i + 2], grid.starts[i:i + 2], grid.ends[i:i + 2]))
             for day, i in zip(days, (0, 2))}
    daily[days[0]]["initial_forecast"] = np.array([0.1, 0.2])
    daily[days[1]]["initial_forecast"] = np.array([0.3, 0.4])
    daily[days[0]]["pv_provenance"] = {"initial_forecast_model_retrospective": True,
        "forecast_model_evidence": {"model": "retrospective_tcn", "available_at_utc": "2026-10-01T00:00:00Z"}}
    return SimpleNamespace(grid=grid, starts=pd.DatetimeIndex(grid.starts), daily=daily,
        day_for_product={product: days[index // 2] for index, product in enumerate(grid.interval_ids)},
        day_offsets={days[0]: 0, days[1]: 2})


def test_frozen_pv_uses_each_pinned_book_and_preserves_retrospective_model_labels():
    replay = frozen_replay()
    values, evidence = frozen_initial_pv(replay, pd.Timestamp(replay.grid.starts[0]), pd.DatetimeIndex(replay.grid.starts)[1:])
    np.testing.assert_array_equal(values, [0.2, 0.3, 0.4])
    assert evidence["frozen_initial_pv_provenance"]["2026-01-02"]["initial_forecast_model_retrospective"]
    assert evidence["frozen_forecast_delivery_days"] == ["2026-01-02", "2026-01-03"]
    replay.daily["2026-01-03"]["pv_metadata"]["pv_forecast_available_at_utc"] = "2026-01-03T06:00:00Z"
    with pytest.raises(ValueError, match="unavailable"):
        frozen_initial_pv(replay, pd.Timestamp(replay.grid.starts[0]), replay.starts)


def test_forecast_refresh_freeze_is_independent_of_auction_access():
    replay = frozen_replay()
    replay.perfect = False
    replay.config = {"research_information": {"pv_updates": "frozen_initial"}}
    # The latest provider is deliberately unavailable; frozen data must suffice.
    replay.prepared = {"forecast_provider": lambda *args: pytest.fail("Latest forecast read in frozen ablation")}
    cutoff = pd.Timestamp(replay.grid.starts[0])
    baseline, _ = ContinuousReplay._forecast(replay, cutoff, replay.starts, "DA", causal=True)
    for daily in replay.daily.values():
        daily["schedule"] = [event for event in daily["schedule"] if event.auction == "IDA3"]
    after_removing_auctions, _ = ContinuousReplay._forecast(replay, cutoff, replay.starts, "IDA3", causal=True)
    np.testing.assert_array_equal(baseline, after_removing_auctions)
    assert initial_information_cutoff(replay.daily["2026-01-02"]) == pd.Timestamp("2026-01-01T11:00:00Z")


def test_frozen_price_predictions_do_not_freeze_publication_of_actual_cash_curves():
    grid = short_grid()
    daily = daily_for(grid)
    provider = RecordingPriceProvider()
    replay = SimpleNamespace(config={"research_information": {"price_updates": "frozen_initial"},
        "energy_price_forecast": {"auction_enabled": True}}, daily={"2026-01-02": daily},
        prepared={"energy_price_provider": provider, "calendar": SimpleNamespace(timezone="Europe/Rome")},
        known={"2026-01-02": ["DA"]})
    curves, provenance = auction_price_predictions(replay, "2026-01-02", pd.Timestamp(grid.starts[0]))
    np.testing.assert_array_equal(curves["DA"], daily["prices"]["DA"])
    np.testing.assert_array_equal(curves["IDA1"], 20)
    assert all(call[1] == initial_information_cutoff(daily) for call in provider.calls)
    assert provenance["auction_price_forecasts"]["DA"]["source"] == "published_actual_clearing_price"
    with pytest.raises(ValueError, match="after"):
        price_information_cutoff(replay.config, daily, pd.Timestamp("2026-01-01T10:00:00Z"))


def control_replay():
    grid = short_grid(count=2)
    day = "2026-01-02"
    daily = daily_for(grid)
    daily.update(ensemble_members=(0, 1), ensemble_pv=np.array([[0.1, 0.2], [0.6, 0.7]]))
    config = {"execution": {"dispatch_policy": "ensemble_mpc_adaptive"}, "solver": {"threads": 1},
        "realtime_mpc": {"terminal_inventory_mode": "none", "current_pv_information": "realized_interval_average"},
        "realtime_scenarios": {"dependence": "independent_shuffle", "reduce_to": 2, "seed": 2},
        "energy_price_forecast": {"settlement_enabled": True}}
    replay = SimpleNamespace(config=config, grid=grid, starts=pd.DatetimeIndex(grid.starts), days=(day,),
        day_for_product=dict.fromkeys(grid.interval_ids, day), day_offsets={day: 0}, daily={day: daily},
        dispatch_policy="ensemble_mpc_adaptive", plant=plant_for(), state=PhysicalState(grid.starts[0], 0.5),
        aging=DegradationSettings(0), trading={"grid_import_fee_eur_per_mwh": 0}, pending={}, progress=None,
        index={product: index for index, product in enumerate(grid.interval_ids)})
    replay.prepared = {"full_realized_pv": np.array([0.35, 0.99]), "terminal_energy_mwh": None,
        "terminal_minimum_energy_mwh": None, "energy_price_provider": RecordingPriceProvider(),
        "provenance": {"pv_ensemble": {day: {"available_at_utc": "2026-01-01T06:00:00Z",
            "forecast_vintage_utc": "2026-01-01T00:00:00Z"}}}}
    replay._forecast = lambda *args, **kwargs: (np.array([0.3, 0.4]), {})
    replay._commands = lambda product: ([0.0, 0.0, 0.0], {"auction": "DA"}, "track_committed_grid", 0.4)
    replay._position = lambda product: 0.1
    return replay


@pytest.mark.parametrize("success", [True, False])
def test_realtime_links_use_only_current_actual_pv_and_label_adaptive_shadows_or_fallback(monkeypatch, success):
    replay = control_replay()
    captured = []

    def solve(inputs, solver):
        captured.append(inputs)
        diagnostic = {"current_realized_pv_mw": inputs.current_realized_pv_mw, "runtime_seconds": 0,
                      "failure_reason": None if success else "synthetic_verification_failure"}
        if not success:
            return MpcResult("solver_error", diagnostics=diagnostic)
        actions = [MpcAction(product, inputs.cutoff.isoformat(), 0, 0, 1) for product in inputs.grid.interval_ids]
        scenarios, length = inputs.pv_ac_mw.shape
        commands = np.zeros((scenarios, length, 3))
        commands[:, :, 2] = 1
        imbalance = inputs.pv_ac_mw * inputs.grid.delta_hours - inputs.committed_mwh
        plan = MpcPlan("fixture", actions, np.full(length + 1, 0.5), inputs.pv_ac_mw, imbalance,
            inputs.weights @ np.abs(imbalance), inputs.weights @ np.maximum(-imbalance, 0), {},
            scenario_actions=commands, scenario_energy_mwh=np.full((scenarios, length + 1), 0.5),
            control_nodes=tuple(tuple(range(length)) for _ in range(scenarios)))
        return MpcResult("optimal", plan=plan, diagnostics=diagnostic)

    monkeypatch.setattr(realtime_control, "solve_mpc", solve)
    controller = realtime_control.RealtimeControl(replay)
    controller.issue(0, pd.Timestamp(replay.grid.starts[0]))
    inputs = captured[0]
    assert inputs.pv_ac_mw.shape == (2, 2)
    np.testing.assert_array_equal(inputs.pv_ac_mw[:, 0], 0.35)
    assert not np.any(inputs.pv_ac_mw[:, 1] == 0.99)
    assert inputs.oracle is None
    assert inputs.price_forecast is not None
    pending = replay.pending[replay.grid.interval_ids[0]]
    if success:
        assert pending["policy"] == "follow_battery_plan"
        assert all(row["future_plan_role"] == "representative_shadow_path" for row in controller.state.plans)
        assert len(controller.state.scenarios) == 4
        assert all("forecast_positive_price_eur_per_mwh" in row for row in controller.state.scenarios)
        _, detail, _, _ = controller.projection_commands(replay.grid.interval_ids[1], 0.5)
        assert detail["future_plan_role"] == "representative_shadow_path"
    else:
        assert pending["policy"] == "track_committed_grid"
        assert pending["detail"]["mpc_fallback"]
        assert pending["detail"]["mpc_failure_reason"] == "synthetic_verification_failure"
    assert controller.projection_pv(replay.grid.interval_ids[0], 0.1) == (0.35, "disclosed_current_interval_realized_pv")
    assert controller.projection_pv(replay.grid.interval_ids[1], 0.2) == (0.2, "eligible_deterministic_forecast")


def test_continuous_ensemble_preflight_selects_archive_per_day_and_discovers_member_counts(tmp_path, monkeypatch):
    """The existing converter accepts new months and variable-size ensemble axes."""
    import xarray as xr
    import bess_pv_trading.data as data_module
    from bess_pv_trading.forecasting.data import pv_probabilistic_data
    from bess_pv_trading.forecasting.models import pv
    from bess_pv_trading.simulation.continuous_inputs import _prepare_pv_ensembles

    paths, daily, calls = [], {}, []
    for day, count in (("2026-01-02", 3), ("2026-02-02", 5)):
        cycle = pd.Timestamp(day) - pd.Timedelta(days=1)
        leads = pd.timedelta_range("0h", periods=49, freq="h")
        path = tmp_path / f"members_{count}.zarr"
        dataset = xr.Dataset({"test_weather": (("farm_id", "cycle_time", "lead_time", "member"),
            np.ones((1, 1, 49, count)))}, coords={"farm_id": [1], "cycle_time": [cycle],
                "lead_time": leads, "member": np.arange(count)})
        dataset.to_zarr(path, consolidated=True, zarr_format=2)
        paths.append(str(path))
        grid = short_grid(f"{day}T12:00:00Z")
        daily[day] = daily_for(grid)
        daily[day]["schedule"] = [SimpleNamespace(auction="DA", submission_at=cycle.tz_localize("UTC") + pd.Timedelta(hours=11))]

    def align(dataset, farm_id, cycle, starts, config, ensemble):
        members = dataset.member.values.astype(str)
        calls.append((cycle, tuple(members)))
        frame = pd.DataFrame({"member": np.repeat(members, len(starts)),
            "pv_ac_mw": np.repeat(np.arange(len(members)) / 10, len(starts))})
        return SimpleNamespace(values=frame, available_at=cycle + pd.Timedelta(hours=6), provenance={"fixture": True})

    monkeypatch.setattr(data_module, "load_metadata", lambda path, farms: {1: object()})
    monkeypatch.setattr(pv_probabilistic_data, "align_cycle", align)
    monkeypatch.setattr(pv, "forecast_pv", lambda weather, *args: weather)
    config = {"continuous_optimization": {"pv_ensemble": {"sources": paths, "member_count": "auto"}},
        "policy": {"forecast_pv_night": {"enabled": False}}, "forecast": {"farm_id": 1,
            "missing_forecast_policy": "zero_only_when_policy_marks_night"},
        "source": {"availability_lag_hours": 6}, "market": {"timezone": "Europe/Rome"},
        "pv": {"forecast_replacement": {"enabled": False}}, "inputs": {"metadata_parquet": "unused_fixture.parquet"}}
    evidence = _prepare_pv_ensembles(config, daily)
    assert daily["2026-01-02"]["ensemble_pv"].shape == (3, 4)
    assert daily["2026-02-02"]["ensemble_pv"].shape == (5, 4)
    assert len(calls) == 2
    assert evidence["2026-01-02"]["source"] == paths[0]
    assert evidence["2026-02-02"]["source"] == paths[1]
    assert evidence["2026-02-02"]["source_inventory"]["member_count"] == 5
    config["continuous_optimization"]["pv_ensemble"]["member_count"] = 3
    with pytest.raises(ValueError, match="unexpected member axis"):
        _prepare_pv_ensembles(config, daily)

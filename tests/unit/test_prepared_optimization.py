"""Prepared real-input planning boundaries are independently solver-free."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.markets import AuctionCalendar
from bess_pv_trading.optimization.prepared import PreparedInputError, load_prepared_inputs
from battery_case_config import battery_config


def prepared_config(tmp_path):
    calendar = AuctionCalendar()
    day = "2025-10-02"
    grid = calendar.delivery_grid(day)
    cutoff = calendar.schedule(day)[0].submission_at
    source = cutoff - pd.Timedelta(hours=8)
    available = source + pd.Timedelta(hours=4)
    pv = pd.DataFrame({"farm_id": 1, "member_id": "deterministic", "member_origin": "deterministic_weather_forecast",
        "member_probability": 1., "forecast_cycle_time": source.isoformat(), "forecast_available_at": available.isoformat(),
        "interval_start_utc": grid.starts, "interval_end_utc": grid.ends, "pv_ac_mw": 0.5,
        "input_eligibility": True, "review_status": "no_configured_observation_review_exclusion",
        "review_excluded": False, "derived_review_excluded": False})
    prices = []
    for event in calendar.schedule(day):
        for t in event.eligible_intervals:
            prices.append({"auction": event.auction, "product_start": grid.starts[t].isoformat(),
                "product_end": grid.ends[t].isoformat(), "price_eur_mwh": -5. if event.auction == "DA" else 25.,
                "available_at": available.isoformat(), "source_time": source.isoformat(), "fit_cutoff": source.isoformat(),
                "origin": "supplied_point_forecast", "zone": "NORD"})
    pv.to_csv(tmp_path / "pv.csv", index=False)
    pd.DataFrame(prices).to_csv(tmp_path / "prices.csv", index=False)
    return {"optimization": {"pv_forecast_path": str(tmp_path / "pv.csv"), "price_forecast_path": str(tmp_path / "prices.csv"),
        "decision_cutoff": cutoff.isoformat(), "delivery_day": day, "pv_model_fit_status": "unfitted_explicit_assumption",
        "settlement_point_price_eur_per_mwh": 20., "settlement_point_price_available_at": source.isoformat(),
        "settlement_point_price_status": "study_assumption"},
        "plant": {"farm_id": 1, "poi_import_mw": 1., "poi_export_mw": 1.},
        "battery": battery_config("half_power"),
        "state": {"initial_energy_mwh": .5, "terminal_target_mwh": .5, "origin": "known_delivery_start_assumption"},
        "trading": {"trade_limit_mwh": .25, "position_limit_mwh": .25, "imbalance_penalty_eur_per_mwh": 100., "grid_import_fee_eur_per_mwh": 0.},
        "degradation": {"model": "throughput", "cost_eur_per_mwh_throughput": 5., "assumptions": "synthetic fixture throughput coefficient"}}


def test_complete_prepared_inputs_preserve_ac_negative_prices_and_masks(tmp_path):
    inputs = load_prepared_inputs(prepared_config(tmp_path))
    assert inputs.scenarios.pv_ac_mw.shape == (1, 96)
    assert np.all(inputs.scenarios.pv_ac_mw == .5)
    assert np.all(inputs.scenarios.auction_prices_eur_per_mwh["DA"] == -5)
    assert inputs.scenarios.probabilities.tolist() == [1.]
    assert sum(inputs.auction_eligibility["IDA3"]) == 48
    assert "synthetic" not in inputs.scenarios.origin
    assert inputs.scenarios.provenance["planning_scope"].startswith("initial pre-DA")
    assert len(inputs.tree.leaf_nodes) == 1
    assert not inputs.market.accepted_trades


def test_prepared_inputs_allow_explicit_null_terminal_soc(tmp_path):
    config = prepared_config(tmp_path)
    config["state"]["terminal_target_mwh"] = None
    assert load_prepared_inputs(config).terminal_target_mwh is None
    del config["state"]["terminal_target_mwh"]
    with pytest.raises(PreparedInputError, match="terminal_target_mwh"):
        load_prepared_inputs(config)


@pytest.mark.parametrize("field", ["available_at", "source_time", "fit_cutoff"])
def test_future_price_information_is_rejected(tmp_path, field):
    config = prepared_config(tmp_path)
    path = config["optimization"]["price_forecast_path"]
    frame = pd.read_csv(path)
    frame.loc[0, field] = "2025-10-02T10:00:00Z"
    frame.to_csv(path, index=False)
    with pytest.raises(PreparedInputError, match="Future/unavailable"):
        load_prepared_inputs(config)


@pytest.mark.parametrize("change,match", [({"pv_ac_mw": np.nan}, "finite"),
    ({"input_eligibility": False}, "Invalid PV"), ({"review_excluded": True}, "review exclusions"),
    ({"member_probability": .5}, "probability one"), ({"forecast_available_at": "2025-10-02T12:00:00Z"}, "Future/unavailable")])
def test_invalid_pv_information_is_rejected(tmp_path, change, match):
    config = prepared_config(tmp_path)
    path = config["optimization"]["pv_forecast_path"]
    frame = pd.read_csv(path)
    for key, value in change.items():
        frame.loc[0, key] = value
    frame.to_csv(path, index=False)
    with pytest.raises(PreparedInputError, match=match):
        load_prepared_inputs(config)


def test_incomplete_or_oracle_prices_rejected(tmp_path):
    config = prepared_config(tmp_path)
    path = config["optimization"]["price_forecast_path"]
    frame = pd.read_csv(path)
    frame.iloc[1:].to_csv(path, index=False)
    with pytest.raises(PreparedInputError, match="cover its eligible"):
        load_prepared_inputs(config)
    frame["origin"] = "future_realized_price"
    frame.to_csv(path, index=False)
    with pytest.raises(PreparedInputError, match="realized/oracle"):
        load_prepared_inputs(config)


def test_missing_template_has_actionable_error_and_null_equipment_rejected(tmp_path):
    config = load_config("configs/full_experiments/prepared_optimization.example.yaml")
    with pytest.raises(PreparedInputError, match="Missing explicitly configured optimization"):
        load_prepared_inputs(config)
    config = prepared_config(tmp_path)
    config["battery"]["capacity_mwh"] = None
    with pytest.raises(PreparedInputError, match="battery equipment"):
        load_prepared_inputs(config)

"""Small analytic storage examples verify causal continuation-value construction."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.domain import BatteryParameters, DeliveryGrid, PlantParameters
from bess_pv_trading.forecasting.continuation import ContinuationSettings, build_continuation_value


def inputs(price=100, pv=0, commitment=0, **settings):
    """A one-hour lossless battery has an analytic one-period inventory value."""
    start = pd.Timestamp("2026-01-02T00:00:00Z")
    grid = DeliveryGrid((start.isoformat(),), (start.to_pydatetime(),),
        ((start + pd.Timedelta(hours=1)).to_pydatetime(),))
    battery = BatteryParameters(1, 1, converter_charge_efficiency=1, converter_discharge_efficiency=1)
    return {"cutoff": "2026-01-01T00:00:00Z", "available_at": "2026-01-01T00:00:00Z",
        "grid": grid, "plant": PlantParameters(1, 1, 1, battery), "positive_eur_per_mwh": [price],
        "pv_ac_mw": [pv], "committed_mwh": [commitment],
        "settings": ContinuationSettings(energy_grid_points=11, **settings)}


def test_positive_price_values_stored_energy_after_wear():
    curve = build_continuation_value(**inputs(throughput_cost_eur_per_mwh=10))
    np.testing.assert_allclose(curve.value_eur, curve.energy_mwh * 90)
    assert curve.provenance["forecast_is_oracle"] is False
    assert curve.provenance["future_auction_recourse"].startswith("none")


def test_negative_prices_value_headroom_and_curtail_pv():
    curve = build_continuation_value(**inputs(price=-50, pv=2))
    np.testing.assert_allclose(curve.value_eur, (1 - curve.energy_mwh) * 50)


def test_efficiencies_and_poi_limit_battery_discharge():
    arguments = inputs()
    plant = arguments["plant"]
    arguments["plant"] = replace(plant, poi_export_mw=0.3,
        battery=replace(plant.battery, converter_discharge_efficiency=0.8))
    curve = build_continuation_value(**arguments)
    # Discrete SOC transitions constrain the approximation from below.
    assert curve.value_eur[-1] <= 30 + 1e-10
    assert curve.value_eur[-1] >= 24 - 1e-10
    assert curve.value_eur[0] == pytest.approx(0)


def test_directional_settlement_and_fixed_commitments():
    arguments = inputs(price=100, commitment=0.5)
    arguments["negative_eur_per_mwh"] = [200]
    curve = build_continuation_value(**arguments)
    assert curve.value_eur[0] == pytest.approx(-100)
    assert curve.value_eur[-1] == pytest.approx(50)


def test_forecast_availability_and_missing_values_fail():
    arguments = inputs()
    arguments["available_at"] = "2026-01-01T01:00:00Z"
    with pytest.raises(ValueError, match="available by cutoff"):
        build_continuation_value(**arguments)
    arguments = inputs()
    arguments["positive_eur_per_mwh"] = [np.nan]
    with pytest.raises(ValueError, match="finite aligned"):
        build_continuation_value(**arguments)


def test_self_discharge_and_terminal_salvage_use_actual_physical_retention():
    arguments = inputs(price=0, final_inventory_value_eur_per_mwh=100)
    plant = arguments["plant"]
    arguments["plant"] = replace(plant, poi_import_mw=0,
        battery=replace(plant.battery, self_discharge_per_hour=0.1))
    curve = build_continuation_value(**arguments)
    np.testing.assert_allclose(curve.value_eur, curve.energy_mwh * 90, atol=1e-10)

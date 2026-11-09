"""Prevent mismatched observations from becoming an unqualified comparison."""

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


@pytest.fixture
def compare_module():
    """Load the report script and its sibling batch reader."""
    directory = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("compare_k25_formulations_observed", directory / "compare_k25_formulations_observed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observed_input_comparison_discloses_forecast_differences(compare_module):
    """A changed issued forecast is recorded even when actual data match."""
    old = pd.DataFrame([{"interval_id": "a", "pv_ac_mw": 1, "imbalance_price_eur_mwh": 20,
                         "pv_ac_forecast_mw": 1, "pv_forecast_vintage_utc": "first"}])
    new = old.copy()
    new["pv_ac_forecast_mw"] = 1.01
    trades = pd.DataFrame([{"auction": "DA", "interval_id": "a", "price_eur_per_mwh": 30}])
    result = compare_module.compare_observed_inputs(old, new, trades, trades)
    assert result["forecast_values_changed"] == 1
    assert result["maximum_forecast_difference_mw"] == pytest.approx(0.01)
    assert result["forecast_vintages_changed"] == 0


def test_observed_input_comparison_rejects_different_prices(compare_module):
    """A change to actual settlement or auction prices blocks a matched claim."""
    dispatch = pd.DataFrame([{"interval_id": "a", "pv_ac_mw": 1, "imbalance_price_eur_mwh": 20,
                              "pv_ac_forecast_mw": 1, "pv_forecast_vintage_utc": "first"}])
    trades = pd.DataFrame([{"auction": "DA", "interval_id": "a", "price_eur_per_mwh": 30}])
    changed = trades.copy()
    changed["price_eur_per_mwh"] = 31
    with pytest.raises(ValueError, match="clearing prices"):
        compare_module.compare_observed_inputs(dispatch, dispatch, trades, changed)

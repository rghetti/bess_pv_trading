"""Ex-post imbalance counterfactuals compare reoptimization plans fairly."""
import numpy as np
import pandas as pd
import pytest

from battery_case_config import case_config, configured_battery
from bess_pv_trading.domain import PlantParameters
from bess_pv_trading.simulation.gme_oracle import _counterfactual_reoptimization_settlement
from test_optimization_contracts import example_inputs


def test_reoptimization_settlements_use_realized_pv_prices_and_common_horizon():
    case = case_config("poi_trading")
    inputs = example_inputs(prices=(0,), pv=np.zeros((1, 1)),
        hours=case["import_interval_hours"])
    battery = configured_battery("no_bess")
    plant = PlantParameters(inputs.plant.farm_id, case["poi_import_mw"],
        case["poi_export_mw"], battery)
    start, end = inputs.grid.starts[0], inputs.grid.ends[0]
    target = case["poi_export_mw"]
    plan_rows = []
    for auction in ("DA", "IDA3"):
        plan_rows.append({"auction": auction, "interval_id": inputs.grid.interval_ids[0],
            "interval_start_utc": start.isoformat(), "interval_end_utc": end.isoformat(),
            "optimizer_cutoff_utc": start.isoformat(), "expected_grid_mw": target,
            "charge_mw": 0.0, "discharge_mw": 0.0,
            "curtailment_mw": case["pv_ac_mw"] - target,
            "pv_ac_forecast_mw": case["pv_ac_mw"], "soc_start_mwh": 0.0})
    plans = pd.DataFrame(plan_rows)
    dispatch = pd.DataFrame({"interval_id": inputs.grid.interval_ids,
        "net_position_mwh": [target * inputs.grid.delta_hours[0]],
        "imbalance_price_eur_mwh": [case["sale_price_eur_per_mwh"]],
        "realized_imbalance_cashflow_eur": [0.0]})
    realized = np.asarray([case["pv_ac_mw"] * case["realized_pv_fraction"]])
    summary, details = _counterfactual_reoptimization_settlement(
        plans, dispatch, realized, inputs.grid, plant, ("DA", "IDA3"))

    assert summary.auction.tolist() == ["DA", "IDA3"]
    assert summary.common_horizon_intervals.tolist() == [1, 1]
    expected_imbalance = (realized[0] - (case["pv_ac_mw"] - target) - target) * inputs.grid.delta_hours[0]
    expected_cashflow = expected_imbalance * case["sale_price_eur_per_mwh"]
    np.testing.assert_allclose(summary.common_horizon_absolute_imbalance_mwh,
        abs(expected_imbalance))
    np.testing.assert_allclose(summary.common_horizon_imbalance_cashflow_eur,
        expected_cashflow)
    np.testing.assert_allclose(summary.common_horizon_final_contract_absolute_imbalance_mwh,
        abs(expected_imbalance))
    np.testing.assert_allclose(summary.common_horizon_final_contract_cashflow_eur,
        expected_cashflow)
    assert details.common_comparison_horizon.all()

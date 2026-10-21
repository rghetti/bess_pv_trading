"""Independent frozen-plan accounting, actual boundary state and future-trade tests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from battery_case_config import configured_battery
from bess_pv_trading.config import ConfigurationError, load_config, validate_config
from bess_pv_trading.domain import DeliveryGrid, PlantParameters
from bess_pv_trading.simulation.frozen_plans import simulate_frozen_auction_plans


def frozen_case(degradation="throughput"):
    """Load separately tabulated financial and physical trajectories from YAML."""
    case = load_config(Path(__file__).resolve().parents[1] / "fixtures/configs/frozen_plan_cases.yaml")
    settings = case["degradation_cases"][degradation]
    config = deepcopy(case["config"])
    config["degradation"] = settings["settings"]
    starts, ends = pd.to_datetime(case["starts"], utc=True), pd.to_datetime(case["ends"], utc=True)
    products = tuple(stamp.isoformat() for stamp in starts)
    grid = DeliveryGrid(products, tuple(starts.to_pydatetime()), tuple(ends.to_pydatetime()))
    plant = PlantParameters(case["farm_id"], case["poi_import_mw"], case["poi_export_mw"],
                            configured_battery(case["battery_reference"]))
    events = tuple(SimpleNamespace(**{**item,
        "submission_at": pd.Timestamp(item["submission_at"]),
        "results_at": pd.Timestamp(item["results_at"])}) for item in case["auctions"])
    dispatch = pd.DataFrame({"interval_id": products, "start": starts, "end": ends,
        **case["reference"], "degradation_eur": settings["reference_wear_eur"]})
    if "reference_segment_energy_end_mwh" in settings:
        dispatch["cycle_segment_energy_end_mwh"] = list(map(str, settings["reference_segment_energy_end_mwh"]))
    ledger = pd.DataFrame([{"auction": event.auction, "interval_id": products[index],
        "quantity_mwh": case["reference_trades"][event.auction][index]}
        for event in events for index in event.eligible_intervals])
    plans = []
    for auction, plan in case["plans"].items():
        for offset, index in enumerate(range(plan["first_interval"], len(products))):
            plans.append({"auction": auction, "interval_id": products[index],
                "interval_start_utc": starts[index].isoformat(), "interval_end_utc": ends[index].isoformat(),
                "optimizer_cutoff_utc": next(item.submission_at.isoformat() for item in events if item.auction == auction),
                "snapshot_scenario_count": plan["snapshot_scenario_count"],
                "planned_net_position_mwh": plan["planned_net_position_mwh"][offset],
                "soc_start_mwh": plan["snapshot_soc_start_mwh"],
                **{name: plan[name][offset] for name in ("charge_mw", "discharge_mw", "curtailment_mw")},
                **{f"planned_trade_{name}_mwh": quantities[offset] for name, quantities in plan["trades"].items()}})
    summary = {"imbalance_settlement_status": "priced",
        "realized_profit_after_imbalance_eur": settings["reference_profit_eur"]}
    return case, (config, grid, plant, events, case["prices"], pd.DataFrame(plans), dispatch, ledger, summary)


@pytest.mark.parametrize("aging", ["throughput", "xu"])
def test_frozen_plans_reconcile_total_profit_with_feedback_actual_soc_and_future_trades(aging):
    case, arguments = frozen_case(aging)
    expected = case["expected"]
    summary, dispatch, ledger = simulate_frozen_auction_plans(*arguments)
    assert summary.frozen_after_auction.tolist() == expected["frozen_auctions"]
    assert summary.common_prefix_intervals.tolist() == expected["common_prefix_intervals"]
    for column, values in (("auction_revenue_eur", expected["frozen_revenues_eur"]),
            ("transaction_fee_eur", expected["transaction_fees_eur"]),
            ("realized_imbalance_cashflow_eur", expected["imbalance_cashflows_eur"]),
            ("reoptimization_gain_eur", expected["reoptimization_gains_eur"]),
            ("final_soc_percent", expected["final_soc_percent"]),
            ("frozen_plan_profit_eur", case["degradation_cases"][aging]["frozen_profits_eur"])):
        np.testing.assert_allclose(summary[column], values)
    np.testing.assert_allclose(summary.degradation_cost_eur, case["degradation_cases"][aging]["frozen_wear_eur"])
    np.testing.assert_allclose(summary.additional_optimizer_solves, 0)
    for index, row in summary.iterrows():
        trajectory = dispatch.loc[dispatch.frozen_after_auction == row.frozen_after_auction]
        trades = ledger.loc[ledger.frozen_after_auction == row.frozen_after_auction]
        np.testing.assert_allclose(trades.loc[trades.auction == arguments[3][-1].auction, "quantity_mwh"],
            expected["future_ida3_mwh"][index])
        assert trajectory.iloc[-1].energy_end_mwh == pytest.approx(row.final_energy_mwh)
        assert trajectory.degradation_eur.sum() == pytest.approx(row.degradation_cost_eur)
        assert trajectory.transaction_fee_eur.sum() == pytest.approx(row.transaction_fee_eur)
        np.testing.assert_allclose(trajectory.loc[~trajectory.replayed_from_frozen_plan, "grid_mw"],
            arguments[6].iloc[:row.common_prefix_intervals].grid_mw)
        assert ((trajectory.energy_end_mwh >= arguments[2].battery.energy_min_mwh)
                & (trajectory.energy_end_mwh <= arguments[2].battery.maximum_energy)).all()


def test_unavailable_actual_settlement_never_becomes_zero_priced_profit():
    _, arguments = frozen_case()
    arguments[-1].update(imbalance_settlement_status="unavailable", realized_profit_after_imbalance_eur=None)
    summary, _, _ = simulate_frozen_auction_plans(*arguments)
    assert summary.frozen_plan_profit_eur.isna().all()
    assert summary.reoptimization_gain_eur.isna().all()
    assert summary.profit_before_imbalance_settlement_eur.notna().all()


def test_frozen_plans_reject_contingent_selected_leaf_and_incomplete_trade_capture():
    case, arguments = frozen_case()
    arguments[5]["snapshot_scenario_count"] = case["invalid_scenario_count"]
    with pytest.raises(ValueError, match="deterministic"):
        simulate_frozen_auction_plans(*arguments)
    _, arguments = frozen_case()
    missing = arguments[5].drop(columns=f"planned_trade_{arguments[3][-1].auction}_mwh")
    with pytest.raises(ValueError, match="complete auction trade snapshots"):
        simulate_frozen_auction_plans(*arguments[:5], missing, *arguments[6:])


def test_only_one_active_auction_has_no_later_reoptimization_to_compare():
    _, arguments = frozen_case()
    summary, dispatch, ledger = simulate_frozen_auction_plans(
        *arguments[:3], arguments[3][-1:], *arguments[4:])
    assert summary.empty and dispatch.empty and ledger.empty


def test_configured_freeze_selection_and_disabled_settings_are_explicit():
    case, arguments = frozen_case()
    arguments[0]["reoptimization_baseline"] = {
        "enabled": True, "freeze_after_auctions": case["selected_freezes"], "plots": False}
    validate_config(arguments[0])
    summary, _, _ = simulate_frozen_auction_plans(*arguments)
    assert summary.frozen_after_auction.tolist() == case["selected_freezes"]
    arguments[0]["reoptimization_baseline"]["enabled"] = False
    validate_config(arguments[0])
    for invalid in case["invalid_baseline_configs"]:
        with pytest.raises(ConfigurationError, match="reoptimization_baseline"):
            validate_config({"reoptimization_baseline": invalid})

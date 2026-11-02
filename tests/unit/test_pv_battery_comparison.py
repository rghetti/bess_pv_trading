"""Verify paired battery comparisons and reject mismatched PV evidence."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts/compare_pv_battery_cases.py"
    spec = importlib.util.spec_from_file_location("compare_pv_battery_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experiment(root, *, battery, scenarios):
    (root / "json").mkdir(parents=True)
    (root / "json/manifest.json").write_text(json.dumps({"status": "complete"}))
    day = "2026-01-01"
    provenance = {"days": {day: {"realized_pv_sha256": "pv", "auction_price_sha256": "prices"}},
        "weather_forecasts": [], "settlement_values_sha256": "settlement",
        "plant": {"farm_id": 1, "battery": {"capacity_mwh": 1 if battery else 0,
                                               "max_power_mw": 1 if battery else 0}},
        "asset_input_settings": {"state": {"initial_soc_percent": 50}, "battery": {},
                                 "degradation": {}, "pv": {}},
        "pv_ensemble": {day: {"values_sha256": "ensemble"}} if scenarios else None}
    (root / "json/input_provenance.json").write_text(json.dumps(provenance))
    for strategy in (("integrated", "perfect_pv") if scenarios else ("integrated",)):
        directory = root / "strategies" / strategy / "csv"
        directory.mkdir(parents=True)
        pd.DataFrame([{"delivery_day": day, "net_profit_eur": 10 if battery else 8,
            "auction_revenue_eur": 12, "imbalance_cashflow_eur": -1, "degradation_eur": 1,
            "transaction_fee_eur": 0, "signed_imbalance_mwh": -0.02,
            "absolute_imbalance_mwh": 0.02, "curtailment_mwh": 0.1,
            "pv_realized_energy_mwh": 1, "fallback_count": 0}]).to_csv(directory / "daily_kpis.csv", index=False)


def test_six_policy_asset_comparison_checks_input_identity(tmp_path):
    roots = [tmp_path / name for name in ("battery_det", "battery_ens", "no_det", "no_ens")]
    for root, battery, scenarios in zip(roots, (True, True, False, False), (False, True, False, True)):
        _experiment(root, battery=battery, scenarios=scenarios)
    output = tmp_path / "comparison"
    _module().compare(*roots, output)
    monthly = pd.read_csv(output / "monthly_comparison.csv")
    assert len(monthly) == 6
    deltas = pd.read_csv(output / "battery_effect_by_policy.csv")
    assert (deltas.net_profit_eur_battery_minus_no_battery == 2).all()
    assert (output / "profit_imbalances_curtailment.png").is_file()
    path = roots[3] / "json/input_provenance.json"
    data = json.loads(path.read_text())
    data["pv_ensemble"]["2026-01-01"]["values_sha256"] = "different"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="PV ensemble differs"):
        _module().compare(*roots, output)

"""Reject mismatched input histories in the three-way PV comparison."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def load_comparison():
    path = Path(__file__).resolve().parents[2] / "scripts/compare_pv_deterministic_scenarios.py"
    spec = importlib.util.spec_from_file_location("compare_pv_deterministic_scenarios", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_experiment(root, *, price_hash="same", strategies=("integrated",)):
    (root / "json").mkdir(parents=True)
    (root / "json/manifest.json").write_text(json.dumps({"status": "complete"}))
    provenance = {"days": {"2026-01-01": {"realized_pv_sha256": "pv", "auction_price_sha256": price_hash}},
        "settlement_values_sha256": "settlement", "plant": {}, "asset_input_settings": {}, "weather_forecasts": []}
    (root / "json/input_provenance.json").write_text(json.dumps(provenance))
    for strategy in strategies:
        directory = root / "strategies" / strategy / "csv"
        directory.mkdir(parents=True)
        pd.DataFrame([{"delivery_day": "2026-01-01", "net_profit_eur": 10,
            "auction_revenue_eur": 12, "imbalance_cashflow_eur": -1, "degradation_eur": 1,
            "import_fee_eur": 0, "transaction_fee_eur": 0, "signed_imbalance_mwh": -0.02,
            "absolute_imbalance_mwh": 0.02, "fallback_count": 0, "final_soc_percent": 50}]).to_csv(
                directory / "daily_kpis.csv", index=False)


def test_three_way_comparison_checks_common_prices(tmp_path):
    deterministic, scenarios = tmp_path / "deterministic", tmp_path / "scenarios"
    write_experiment(deterministic)
    write_experiment(scenarios, strategies=("integrated", "perfect_pv"))
    output = tmp_path / "comparison"
    load_comparison().compare(deterministic, scenarios, output)
    monthly = pd.read_csv(output / "monthly_comparison.csv")
    assert set(monthly.comparison_label) == {"deterministic", "scenarios_51", "perfect_pv"}
    assert (output / "profit_and_imbalances.png").is_file()
    path = scenarios / "json/input_provenance.json"
    changed = json.loads(path.read_text())
    changed["days"]["2026-01-01"]["auction_price_sha256"] = "different"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="auction_price_sha256"):
        load_comparison().compare(deterministic, scenarios, output)

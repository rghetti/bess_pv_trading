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
        "settlement_values_sha256": "settlement", "plant": {"farm_id": 1}, "asset_input_settings": {}, "weather_forecasts": []}
    (root / "json/input_provenance.json").write_text(json.dumps(provenance))
    for strategy in strategies:
        directory = root / "strategies" / strategy / "csv"
        directory.mkdir(parents=True)
        (directory.parent / "json").mkdir()
        (directory.parent / "json/solves.json").write_text(json.dumps([{"status": "optimal"}]))
        pd.DataFrame([{"delivery_day": "2026-01-01", "net_profit_eur": 10,
            "auction_revenue_eur": 12, "imbalance_cashflow_eur": -1, "degradation_eur": 1,
            "import_fee_eur": 0, "transaction_fee_eur": 0, "signed_imbalance_mwh": -0.02,
            "absolute_imbalance_mwh": 0.02, "fallback_count": 0, "final_soc_percent": 50,
            "pv_realized_energy_mwh": 2.0, "charge_mwh": 0.4, "discharge_mwh": 0.3,
            "curtailment_mwh": 0.1}]).to_csv(
                directory / "daily_kpis.csv", index=False)


def test_three_way_comparison_checks_common_prices(tmp_path):
    deterministic, scenarios = tmp_path / "deterministic", tmp_path / "scenarios"
    write_experiment(deterministic)
    write_experiment(scenarios, strategies=("integrated", "perfect_pv"))
    output = tmp_path / "comparison"
    load_comparison().compare(deterministic, scenarios, output)
    monthly = pd.read_csv(output / "monthly_comparison.csv")
    assert set(monthly.comparison_label) == {"deterministic", "scenarios_51", "perfect_pv"}
    assert monthly.set_index("comparison_label").loc["scenarios_51", "charge_mwh"] == pytest.approx(0.4)
    summary = (output / "summary.md").read_text()
    assert "Farm 1" in summary
    assert "51 PV scenarios" in summary and "Perfect PV reference (DA only)" in summary
    assert "10.00" in summary and "-0.020" in summary and "0.400" in summary
    assert "Non-optimal solves" in summary
    (scenarios / "strategies/integrated/json/solves.json").unlink()
    load_comparison().compare(deterministic, scenarios, output)
    assert "unknown" in (output / "summary.md").read_text()
    assert (output / "profit_and_imbalances.png").is_file()
    path = scenarios / "json/input_provenance.json"
    changed = json.loads(path.read_text())
    changed["days"]["2026-01-01"]["auction_price_sha256"] = "different"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="auction_price_sha256"):
        load_comparison().compare(deterministic, scenarios, output)

"""Compare only complete experiments with matching delivery days."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def load_comparison():
    path = Path(__file__).resolve().parents[2] / "scripts/compare_pv_scenario_replays.py"
    spec = importlib.util.spec_from_file_location("compare_pv_scenario_replays", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_run(root, status="complete"):
    (root / "json").mkdir(parents=True)
    (root / "json/manifest.json").write_text(json.dumps({"status": status}))
    for strategy in ("integrated", "perfect_pv"):
        directory = root / "strategies" / strategy / "csv"
        directory.mkdir(parents=True)
        pd.DataFrame([{"delivery_day": "2026-01-01", "net_profit_eur": 10,
            "auction_revenue_eur": 12, "imbalance_cashflow_eur": -1,
            "degradation_eur": 1, "import_fee_eur": 0, "transaction_fee_eur": 0,
            "illustrative_imbalance_penalty_eur": 2, "signed_imbalance_mwh": -0.02,
            "absolute_imbalance_mwh": 0.02, "fallback_count": 0}]).to_csv(
                directory / "daily_kpis.csv", index=False)


def test_comparison_produces_four_matched_results(tmp_path):
    exact, penalty = tmp_path / "exact", tmp_path / "penalty"
    prepare_run(exact)
    prepare_run(penalty)
    output = tmp_path / "comparison"
    load_comparison().compare(exact, penalty, output)
    monthly = pd.read_csv(output / "monthly_comparison.csv")
    assert set(monthly.comparison_label) == {"exact_integrated", "exact_perfect_pv",
                                             "penalty_integrated", "penalty_perfect_pv"}
    assert (output / "profit_and_imbalances.png").is_file()


def test_comparison_rejects_incomplete_runs(tmp_path):
    exact, penalty = tmp_path / "exact", tmp_path / "penalty"
    prepare_run(exact, status="failed")
    prepare_run(penalty)
    with pytest.raises(ValueError, match="incomplete"):
        load_comparison().compare(exact, penalty, tmp_path / "comparison")

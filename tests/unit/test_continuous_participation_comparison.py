"""Matched forecast comparisons preserve profile pairing and reject changed inputs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.reporting.continuous_summary import SUMMED_METRICS


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("participation_comparison", ROOT / "scripts/run_continuous_participation_comparison.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root, variant):
    """Create complete small saved studies with identical pinned source data."""
    config = load_config(ROOT / f"configs/full_experiments/continuous_participation_2026_01_{variant}.yaml")
    config["continuous_participation_study"].update(candidate_auctions=["DA"], profiles=None)
    days = ["2026-01-01", "2026-01-02"]
    names = ["DA", "sequential_all_auctions", "perfect_pv_all_auctions"]
    _write(root / "json/study_config.json", config)
    _write(root / "json/manifest.json", {"status": "complete", "delivery_days": days, "completed": names})
    inputs = {"plant": {"farm_id": 1}, "asset_input_settings": {}, "weather_forecasts": [],
        "settlement_values_sha256": "settlement", "days": {day: {"realized_pv_sha256": "pv",
            "auction_price_sha256": {"DA": "price"}, "skipped_auctions": []} for day in days}}
    _write(root / "json/input_provenance.json", inputs)
    records = []
    for name in names:
        for day in days:
            row = {metric: 0 for metric in SUMMED_METRICS}
            row.update(profile=name, delivery_day=day, net_profit_eur=10 if variant == "ensemble" else 4,
                absolute_imbalance_mwh=2, signed_imbalance_mwh=-1, curtailment_mwh=3, final_soc_percent=50)
            records.append(row)
    (root / "csv").mkdir()
    pd.DataFrame(records).to_csv(root / "csv/daily_participation.csv", index=False)
    return inputs


def test_compare_pairs_complete_profiles_and_exposes_inventory(tmp_path):
    det, ens, output = (tmp_path / name for name in ("det", "ens", "comparison"))
    _fixture(det, "deterministic")
    _fixture(ens, "ensemble")
    paired = MODULE.compare(det, ens, output)
    assert list(paired.profile) == ["DA", "sequential_all_auctions", "perfect_pv_all_auctions"]
    row = paired.iloc[0]
    assert row.net_profit_eur_deterministic == 8
    assert row.net_profit_eur_ensemble == 20
    assert row.profit_difference_ensemble_minus_deterministic_eur == 12
    assert row.absolute_imbalance_mwh_ensemble == 4
    assert row.curtailment_mwh_deterministic == 6
    text = (output / "summary.md").read_text()
    assert "Sequential (all auctions)" in text and "Perfect PV (all auctions)" in text
    assert "Final SOC" in text


def test_compare_rejects_changed_source_and_incomplete_profiles(tmp_path):
    det, ens, output = (tmp_path / name for name in ("det", "ens", "comparison"))
    _fixture(det, "deterministic")
    original = _fixture(ens, "ensemble")
    changed = deepcopy(original)
    changed["days"]["2026-01-02"]["realized_pv_sha256"] = "changed"
    _write(ens / "json/input_provenance.json", changed)
    with pytest.raises(ValueError, match="realized_pv_sha256"):
        MODULE.compare(det, ens, output)
    _write(ens / "json/input_provenance.json", original)
    frame = pd.read_csv(ens / "csv/daily_participation.csv")
    frame.iloc[1:].to_csv(ens / "csv/daily_participation.csv", index=False)
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        MODULE.compare(det, ens, output)


@pytest.mark.parametrize("farm_id", [1, 345, 901, 910])
def test_paired_shipped_configs_have_matched_nonforecast_settings(farm_id):
    prefix = "" if farm_id == 1 else f"farm{farm_id}_"
    configs = [load_config(ROOT / f"configs/full_experiments/continuous_participation_2026_01_{prefix}{variant}.yaml")
               for variant in ("deterministic", "ensemble")]
    assert MODULE._comparable_config(configs[0]) == MODULE._comparable_config(configs[1])

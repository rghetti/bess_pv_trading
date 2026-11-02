"""Continuous participation reports compare complete fixed-strategy trajectories."""
from pathlib import Path

import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.simulation.continuous_participation import _save_comparison


def test_comparison_uses_monthly_totals_and_requires_matched_days(tmp_path):
    days = ["2026-01-01", "2026-01-02"]
    def summary(values):
        return {"daily_kpis": [{"delivery_day": day, "net_profit_eur": profit}
                               for day, profit in zip(days, values)],
                "final_energy_mwh": 1.0, "final_soc_percent": 50.0, "fallback_count": 0}
    summaries = {"DA": summary([10, 2]), "IDA1": summary([5, 8]),
                 "sequential_all_auctions": summary([4, 4])}
    ranking = _save_comparison(tmp_path, summaries, [("DA",), ("IDA1",)], days, False)
    assert ranking[0]["profile"] == "IDA1"
    assert ranking[0]["total_net_profit_eur"] == 13
    assert len(pd.read_csv(tmp_path / "csv/daily_participation.csv")) == 6
    summaries["DA"]["daily_kpis"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        _save_comparison(tmp_path, summaries, [("DA",), ("IDA1",)], days, False)


@pytest.mark.parametrize("kind", ["deterministic", "ensemble"])
def test_monthly_configs_are_matched_and_envelope_compatible(kind):
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / f"configs/full_experiments/continuous_participation_2026_01_{kind}.yaml")
    assert config["continuous_optimization"]["pv_ensemble"]["enabled"] == (kind == "ensemble")
    assert config["continuous_optimization"]["start_date"] == "2026-01-01"
    assert config["continuous_optimization"]["end_date"] == "2026-01-31"
    assert not config["trading"]["optimize_imbalances"]
    assert config["continuous_participation_study"]["profiles"] is None


@pytest.mark.parametrize("farm_id", [1, 345, 901, 910])
@pytest.mark.parametrize("kind", ["deterministic", "ensemble"])
def test_participation_configs_select_each_farm(farm_id, kind):
    """Each farm resolves its own assets while both variants share assumptions."""
    root = Path(__file__).resolve().parents[2]
    prefix = "continuous_participation_2026_01_"
    if farm_id != 1:
        prefix += f"farm{farm_id}_"
    config = load_config(root / f"configs/full_experiments/{prefix}{kind}.yaml")
    assert config["farm_id"] == config["plant"]["farm_id"] == config["forecast"]["farm_id"] == farm_id
    farm = config["farms"][str(farm_id)]
    assert config["battery"]["capacity_mwh"] == farm["battery"]["capacity_mwh"]
    assert config["plant"]["poi_export_mw"] == farm["ac_capacity_kw"] / 1000
    assert config["continuous_optimization"]["pv_ensemble"]["enabled"] == (kind == "ensemble")
    assert config["continuous_optimization"]["resume"]

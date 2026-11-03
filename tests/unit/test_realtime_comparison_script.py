"""The full-January launcher plans safely and shares inputs between comparisons."""

import importlib.util
from pathlib import Path

from bess_pv_trading.config import load_config


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("realtime_comparison_script", ROOT / "scripts/run_realtime_mpc_comparison.py")
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)
CONFIG = ROOT / "configs/full_experiments/realtime_farm1_january_2026_all_auctions.yaml"


def test_default_full_january_plan_starts_no_data_reads_or_solves(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not start historical acquisition, inference or optimization")

    monkeypatch.setattr(script, "prepare_continuous_inputs", forbidden)
    monkeypatch.setattr(script, "run_realtime_mpc_experiment", forbidden)
    assert script.main(["--config", str(CONFIG)]) == 0
    config = load_config(CONFIG)
    plan = script.comparison_plan(config, ("frozen_commitments", "continuous"))
    assert plan["delivery_days"] == 31
    assert plan["physical_intervals_per_trajectory"] == 2976
    assert plan["mpc_decisions"] == 5952
    assert plan["auction_solves_if_sources_are_generated"] == 372
    assert plan["detailed_daily_plot_day_limit"] == 3
    assert plan["solves_started"] == 0


def test_preflight_shares_one_prepared_input_set_between_modes(monkeypatch, capsys):
    prepared, calls = {}, []
    monkeypatch.setattr(script, "prepare_continuous_inputs", lambda *args, **kwargs: prepared)

    def checked(config, *, prepared_inputs, progress):
        assert prepared_inputs is prepared
        assert config["realtime_mpc_comparison"]["check_only"]
        calls.append(config["realtime_mpc_comparison"]["mode"])
        return {"status": "preflight_passed"}

    monkeypatch.setattr(script, "run_realtime_mpc_experiment", checked)
    assert script.main(["--config", str(CONFIG), "--check-only"]) == 0
    assert calls == ["frozen_commitments", "continuous"]

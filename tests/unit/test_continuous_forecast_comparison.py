"""Check saved-run accounting and efficiency-aware battery usage comparisons."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SPEC = spec_from_file_location("continuous_forecast_comparison",
    ROOT / "scripts/compare_continuous_forecasts.py")
comparison = module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)
CASES = yaml.safe_load((ROOT / "tests/fixtures/configs/battery_cases.yaml").read_text())


@pytest.mark.parametrize("case_name", ["grid_cycle", "initial_energy_sale"])
def test_internal_cycles_follow_stored_energy_changes(case_name):
    """A full round trip and an initial-stock sale need distinct EFC conventions."""
    case = CASES["solver_cases"][case_name]
    config = {"battery": CASES["batteries"][case["battery"]]}
    duration = CASES["optimizer_defaults"]["interval_hours"]
    expected = case["expected"]
    daily = pd.DataFrame({
        "charge_mwh": np.asarray(expected["charge_mw"]) * duration,
        "discharge_mwh": np.asarray(expected["discharge_mw"]) * duration})

    result = comparison.battery_usage_metrics(config, daily)
    changes = np.diff(expected["energy_mwh"])
    capacity = config["battery"]["capacity_mwh"]
    assert result["equivalent_full_cycles_internal_discharge"] == pytest.approx(
        -changes[changes < 0].sum() / capacity)
    assert result["equivalent_full_cycles_internal_two_sided"] == pytest.approx(
        np.abs(changes).sum() / (2 * capacity))
    assert result["battery_charge_ac_mwh"] == pytest.approx(daily.charge_mwh.sum())
    assert result["battery_discharge_ac_mwh"] == pytest.approx(daily.discharge_mwh.sum())


def test_paired_january_configs_use_the_same_tighter_solver_precision():
    """Keep both forecast comparisons at the established continuous-study precision."""
    from bess_pv_trading.config import load_config

    directory = ROOT / "configs/full_experiments"
    reference = load_config(directory / "continuous_optimization_2025-01.yaml")
    for name in ("continuous_optimization_2026-01_forecast_pilot.yaml",
            "continuous_optimization_2026-01_pvlib_soc40.yaml"):
        config = load_config(directory / name)
        for key in ("feasibility_tolerance", "integrality_tolerance", "mip_gap"):
            assert config["solver"][key] == reference["solver"][key]

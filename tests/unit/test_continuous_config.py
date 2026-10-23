"""Continuous study configuration stays external, portable and explicitly selected."""
from copy import deepcopy
from datetime import date
from pathlib import Path
import sys
from types import ModuleType

import pytest
import yaml

from bess_pv_trading import cli
from bess_pv_trading.config import (ConfigurationError, load_config, resolve_config,
                                    validate_continuous_optimization)


ROOT = Path(__file__).resolve().parents[2]
STANDARD = ROOT / "configs/standard.yaml"
JANUARY = ROOT / "configs/full_experiments/continuous_optimization_2025-01.yaml"


def test_continuous_template_requires_explicit_delivery_dates():
    config = load_config(STANDARD)
    assert config["continuous_optimization"]["start_date"] is None
    assert config["continuous_optimization"]["end_date"] is None
    # Loading shared settings is valid; selecting the command requires dates.
    with pytest.raises(ConfigurationError, match="both start_date and end_date"):
        cli.dispatch("experiment-continuous", config)


def test_january_inherits_shared_settings_and_declares_only_overrides():
    config, defaults = load_config(JANUARY), load_config(STANDARD)
    raw = yaml.safe_load(JANUARY.read_text())
    assert (JANUARY.parent / raw["include"]).resolve() == STANDARD
    section = config["continuous_optimization"]
    assert section["planning_modes"] == defaults["continuous_optimization"]["planning_modes"]
    assert section["perfect_pv_reference"] == defaults["continuous_optimization"]["perfect_pv_reference"]
    assert section["perfect_pv_reoptimize_at_each_auction"] is False
    assert config["state"]["initial_soc_percent"] == raw["state"]["initial_soc_percent"]
    assert config["state"]["terminal_soc_percent"] == raw["state"]["terminal_soc_percent"]
    assert config["solver"]["mip_gap"] == defaults["solver"]["mip_gap"]
    assert config["experiment_outputs"] == defaults["experiment_outputs"]
    assert config["source"]["availability_lag_hours"] == defaults["source"]["availability_lag_hours"]
    assert date.fromisoformat(section["start_date"]) <= date.fromisoformat(section["end_date"])
    assert config["market"]["valid_from"] == section["start_date"]
    assert config["market"]["valid_to"] == section["end_date"]


@pytest.mark.parametrize("field", ["start_date", "end_date"])
def test_one_missing_delivery_date_fails(field):
    config = load_config(JANUARY)
    config["continuous_optimization"][field] = None
    with pytest.raises(ConfigurationError, match="both start_date and end_date"):
        resolve_config(config)


def test_date_range_is_inclusive_and_one_day_is_valid():
    config = load_config(JANUARY)
    section = config["continuous_optimization"]
    same_day = {**section, "end_date": section["start_date"]}
    validated = resolve_config({**config, "continuous_optimization": same_day})
    assert validated["continuous_optimization"]["start_date"] == validated["continuous_optimization"]["end_date"]
    section["start_date"], section["end_date"] = section["end_date"], section["start_date"]
    with pytest.raises(ConfigurationError, match="must not precede"):
        resolve_config(config)


def test_timestamp_is_not_a_delivery_date():
    config = load_config(JANUARY)
    config["continuous_optimization"]["start_date"] += "T00:00:00Z"
    with pytest.raises(ConfigurationError, match="ISO delivery dates"):
        resolve_config(config)


@pytest.mark.parametrize("invalid", [[], ["unknown_mode"], "integrated", [["integrated"]]])
def test_invalid_planning_modes_fail(invalid):
    config = load_config(JANUARY)
    config["continuous_optimization"]["planning_modes"] = invalid
    with pytest.raises(ConfigurationError, match="nonempty unique list"):
        resolve_config(config)


def test_duplicate_planning_mode_fails_and_one_mode_is_allowed():
    config = load_config(JANUARY)
    mode = config["continuous_optimization"]["planning_modes"][0]
    config["continuous_optimization"]["planning_modes"] = [mode]
    validate_continuous_optimization(config, require_dates=True)
    config["continuous_optimization"]["planning_modes"].append(mode)
    with pytest.raises(ConfigurationError, match="nonempty unique list"):
        resolve_config(config)


@pytest.mark.parametrize("field", ["resume", "perfect_pv_reference", "perfect_pv_reoptimize_at_each_auction"])
def test_continuous_switches_require_yaml_booleans(field):
    config = load_config(JANUARY)
    config["continuous_optimization"][field] = str(config["continuous_optimization"][field])
    with pytest.raises(ConfigurationError, match="YAML boolean"):
        resolve_config(config)


@pytest.mark.parametrize("field,bound", [
    ("initial_soc_percent", "soc_min_percent"),
    ("terminal_soc_percent", "soc_max_percent"),
])
def test_continuous_state_must_start_and_target_inside_battery_box(field, bound):
    config = load_config(JANUARY)
    box = config["battery"][bound]
    config["state"][field] = box / 2 if bound == "soc_min_percent" else (box + 100) / 2
    with pytest.raises(ConfigurationError, match="within battery SOC"):
        resolve_config(config)


def test_terminal_soc_can_be_disabled_without_changing_initial_state():
    initial = load_config(JANUARY)["state"]["initial_soc_percent"]
    config = load_config(JANUARY, overrides={"state": {"terminal_soc_percent": None}})
    assert config["state"]["terminal_soc_percent"] is None
    assert config["state"]["initial_soc_percent"] == initial


def test_continuous_study_supports_other_farms_and_date_overrides():
    original = load_config(JANUARY)
    farm_id = next(key for key in original["farms"] if int(key) != original["farm_id"])
    last = original["continuous_optimization"]["end_date"]
    config = load_config(JANUARY, overrides={"farm_id": int(farm_id),
        "continuous_optimization": {"start_date": last, "end_date": last}})
    assert config["plant"]["farm_id"] == config["forecast"]["farm_id"] == int(farm_id)
    assert config["plant"]["poi_export_mw"] == config["farms"][farm_id]["ac_capacity_kw"] / 1000
    assert f"farm{farm_id}" in config["continuous_optimization"]["run_id"]
    for field, value in config["farms"][farm_id]["battery"].items():
        assert config["battery"][field] == value


def test_continuous_rerun_overwrites_same_id_and_honors_separate_output_root(monkeypatch, tmp_path):
    """Exercise actual output orchestration with data preparation and solves stubbed."""
    import json
    from types import SimpleNamespace
    from bess_pv_trading.simulation import continuous
    from bess_pv_trading.reporting import continuous_outputs

    config = load_config(JANUARY, overrides={"continuous_optimization": {
        "output_root": str(tmp_path / "first"), "resume": False,
        "planning_modes": ["integrated"], "perfect_pv_reference": False}})
    settings = config["continuous_optimization"]
    prepared = {"imbalance_prices_eur_per_mwh": [], "input_fingerprint": "fixture",
        "provenance": {"source": "in_memory_fixture"},
        "calendar": SimpleNamespace(timezone=config["market"]["timezone"]),
        "days": [settings["start_date"]]}
    monkeypatch.setattr(continuous, "prepare_continuous_inputs", lambda *args, **kwargs: prepared)
    calls = []

    def replay(config, prepared, strategy, destination, **kwargs):
        calls.append(kwargs["resume"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "result.csv").write_text(str(config["battery"]["max_power_mw"]))
        return {"output_dir": str(destination), "total_gain_eur": config["battery"]["capacity_mwh"]}

    monkeypatch.setattr(continuous, "replay_continuous_strategy", replay)
    monkeypatch.setattr(continuous_outputs, "save_continuous_comparison", lambda *args: {})
    first = continuous.run_continuous_experiment(config)
    destination = Path(first["output_dir"])
    (destination / "stale.csv").write_text("old")
    config["battery"]["max_power_mw"] /= 2
    second = continuous.run_continuous_experiment(config)
    assert second["output_dir"] == first["output_dir"]
    assert not (destination / "stale.csv").exists()
    saved = json.loads((destination / "json/study_config.json").read_text())
    assert saved["battery"] == config["battery"]

    settings["output_root"] = str(tmp_path / "second")
    third = continuous.run_continuous_experiment(config)
    assert third["output_dir"] != first["output_dir"]
    assert (destination / "json/study_config.json").is_file()
    retained = Path(third["output_dir"]) / "checkpoint-marker.json"
    retained.write_text("retained on resume")
    settings["resume"] = True
    continuous.run_continuous_experiment(config)
    assert retained.read_text() == "retained on resume"
    assert calls == [False, False, False, True]


def test_continuous_overwrite_cannot_clear_an_active_run(monkeypatch, tmp_path):
    """The existing lock is acquired before any stale artifact is removed."""
    import fcntl
    from bess_pv_trading.simulation import continuous

    config = load_config(JANUARY, overrides={"continuous_optimization": {
        "output_root": str(tmp_path), "resume": False}})
    destination = tmp_path / config["continuous_optimization"]["run_id"]
    destination.mkdir()
    (destination / "keep.csv").write_text("active")
    monkeypatch.setattr(continuous, "prepare_continuous_inputs", lambda *args, **kwargs:
        {"imbalance_prices_eur_per_mwh": []})
    with (destination / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            continuous.run_continuous_experiment(config)
    assert (destination / "keep.csv").read_text() == "active"


def test_continuous_command_dispatches_lazily_and_reports_progress(monkeypatch, capsys):
    config = load_config(JANUARY)
    calls = []
    module = ModuleType("bess_pv_trading.simulation.continuous")

    def run(received, progress):
        calls.append(received)
        progress("continuous progress")
        return {"status": "complete"}

    module.run_continuous_experiment = run
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert cli.dispatch("experiment-continuous", config) == {"status": "complete"}
    assert calls == [config]
    assert "continuous progress" in capsys.readouterr().err


def test_cli_accepts_continuous_command_without_selecting_other_workflows(monkeypatch, capsys):
    selected = []

    def dispatch(command, config):
        selected.append((command, deepcopy(config)))
        return {"status": "complete"}

    monkeypatch.setattr(cli, "dispatch", dispatch)
    assert cli.main(["experiment-continuous", "--config", str(JANUARY)]) == 0
    assert selected[0][0] == "experiment-continuous"
    assert selected[0][1]["continuous_optimization"] == load_config(JANUARY)["continuous_optimization"]
    assert '"status": "complete"' in capsys.readouterr().out

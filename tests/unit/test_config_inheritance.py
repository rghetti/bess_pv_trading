"""Shared defaults, safe references, farm switching and legacy workflow routing."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from bess_pv_trading import cli
from bess_pv_trading.config import ConfigurationError, load_config, resolve_config, with_standard_defaults

ROOT = Path(__file__).resolve().parents[2]
STANDARD = ROOT / "configs/standard.yaml"
CURRENT = ROOT / "configs/full_experiments/gme_participation_2025-01.yaml"
FARMS = load_config(STANDARD)["farms"]


def test_all_shipped_configs_import_standard_directly():
    for path in (ROOT / "configs").rglob("*.yaml"):
        raw = yaml.safe_load(path.read_text())
        if path != STANDARD:
            assert (path.parent / raw["include"]).resolve() == STANDARD
        assert "include" not in load_config(path)


@pytest.mark.parametrize("farm_id", FARMS)
def test_switch_farm_updates_data_selection_limits_and_labels(farm_id):
    original = load_config(CURRENT)
    config = load_config(CURRENT, overrides={"farm_id": int(farm_id)})
    assert config["plant"]["farm_id"] == config["forecast"]["farm_id"] == config["farm_id"]
    assert config["plant"]["poi_import_mw"] == FARMS[farm_id]["ac_capacity_kw"] / 1000
    assert config["plant"]["poi_export_mw"] == FARMS[farm_id]["ac_capacity_kw"] / 1000
    assert config["pv"]["output"]["ac_limit_mw"] == FARMS[farm_id]["ac_capacity_kw"] / 1000
    assert f"farm{farm_id}" in config["participation_study"]["run_id"]
    assert f"farm{farm_id}" in config["experiment_outputs"]["run_id"]
    assert config["battery"] == original["battery"]


def test_per_farm_limits_merge_integer_and_json_string_keys():
    standard = load_config(STANDARD)
    farm_id = standard["farm_id"]
    rating = standard["plant"]["poi_import_mw"] / 2
    config = load_config(STANDARD, overrides={"farms": {farm_id: {"poi_import_mw": rating}}})
    assert config["plant"]["poi_import_mw"] == rating
    assert config["plant"]["poi_export_mw"] == standard["plant"]["poi_export_mw"]
    assert config["farms"][str(farm_id)]["tracking_type_id"] == standard["farms"][str(farm_id)]["tracking_type_id"]
    assert len(config["farms"]) == len(standard["farms"])
    explicit = load_config(STANDARD, overrides={"plant": {"poi_import_mw": rating}})
    assert explicit["plant"]["poi_import_mw"] == rating


def test_null_terminal_target_and_list_overrides_are_retained():
    config = load_config(CURRENT, overrides={"state": {"terminal_soc_percent": None},
        "gme_oracle_experiment": {"participating_auctions": ["IDA1"]}})
    assert config["state"]["terminal_soc_percent"] is None
    assert config["gme_oracle_experiment"]["participating_auctions"] == ["IDA1"]


def test_shared_gme_fee_default_covers_first_delivery_submission_and_keeps_free_example():
    """Shared tariffs apply to Dec-31 submissions without changing explicit fee modes."""
    from bess_pv_trading.markets import AuctionCalendar
    from bess_pv_trading.transaction_fees import transaction_fees_from_config

    config = load_config(ROOT / "configs/full_experiments/continuous_optimization_2025-01.yaml")
    fees = transaction_fees_from_config(config)
    assert fees.mode == "gme_real"
    assert fees.operator_scope == "standalone_plant"
    first_submission = min(event.submission_at for event in AuctionCalendar.from_config(config).schedule(
        config["continuous_optimization"]["start_date"]))
    assert fees.initial_state.as_of <= first_submission
    month = fees.month(first_submission)
    assert fees.starting_volume(fees.initial_state, month) == 0
    tariff = fees.tariff(month)
    free_volume = tariff.thresholds_mwh[0]
    probe_volume = config["battery"]["capacity_mwh"]
    assert fees.increment(month, 0, free_volume) == 0
    assert fees.increment(month, free_volume, probe_volume) == pytest.approx(
        probe_volume * tariff.marginal_eur_per_mwh[1])
    for mode in ("none", "flat", "gme"):
        example = load_config(ROOT / f"configs/full_experiments/synthetic_fees_{mode}.yaml")
        expected = "gme_real" if mode == "gme" else mode
        assert transaction_fees_from_config(example).mode == expected


def test_include_merges_before_reference_resolution(tmp_path):
    (tmp_path / "base.yaml").write_text("selected: 1\nvalue: ${selected}\nitems: [1, 2]\nnested: {keep: true}\n")
    path = tmp_path / "child.yaml"
    path.write_text("include: base.yaml\nselected: 2\nitems: []\nnested: {extra: false}\n")
    config = load_config(path, overrides={"selected": 3})
    assert config["value"] == config["selected"]
    assert config["items"] == []
    assert config["nested"] == {"keep": True, "extra": False}
    (tmp_path / "base.yaml").write_text("include: child.yaml\n")
    with pytest.raises(ConfigurationError, match="Circular configuration include"):
        load_config(path)


@pytest.mark.parametrize("config, message", [
    ({"value": "${missing}"}, "Unknown configuration reference"),
    ({"a": "${b}", "b": "${a}"}, "Circular configuration reference"),
    ({"a": None, "value": "prefix_${a}"}, "must have a scalar value"),
    ({"plant": {"farm_id": "unknown", "poi_import_mw": "auto"}}, "requires selected-farm"),
])
def test_invalid_references_fail_clearly(config, message):
    with pytest.raises(ConfigurationError, match=message):
        resolve_config(config)


def test_resolve_command_prints_yaml_without_dispatch(capsys, monkeypatch):
    def unexpected(*args):
        pytest.fail("Resolving config must not dispatch experiments or API requests")
    monkeypatch.setattr(cli, "dispatch", unexpected)
    farm_id = next(iter(FARMS))
    assert cli.main(["resolve-config", "--config", str(CURRENT), "--farm-id", farm_id]) == 0
    config = yaml.safe_load(capsys.readouterr().out)
    assert config["plant"]["farm_id"] == int(farm_id)
    assert "include" not in config


def test_api_validation_does_not_select_prepared_optimizer(monkeypatch):
    def unexpected(*args):
        pytest.fail("Shared safety flags must not select prepared inputs")
    monkeypatch.setattr(cli, "_optimization_inputs", unexpected)
    config = load_config(ROOT / "configs/api_experiments/terna_api.yaml")
    assert cli.dispatch("validate-config", config)["status"] == "valid"


def test_point_prepared_route_survives_shared_forecast_safety_flag(monkeypatch):
    from bess_pv_trading.optimization import prepared
    marker = object()
    monkeypatch.setattr(prepared, "load_prepared_inputs", lambda config: marker)
    config = load_config(ROOT / "configs/full_experiments/prepared_optimization.example.yaml")
    assert cli._optimization_inputs(config) is marker


def test_legacy_saved_study_gains_defaults_but_still_detects_changes():
    config = load_config(CURRENT)
    legacy = deepcopy(config)
    for key in ("api", "price_scenarios", "optimization", "report", "outputs"):
        legacy.pop(key)
    assert with_standard_defaults(legacy) == with_standard_defaults(config)
    changed = deepcopy(legacy)
    changed["battery"]["capacity_mwh"] *= 2
    assert with_standard_defaults(changed) != with_standard_defaults(config)

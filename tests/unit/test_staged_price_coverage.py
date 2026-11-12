"""Audit staged feature coverage, target guards and native export horizons."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies import price_feature_data, price_staged_search


ROOT = Path(__file__).resolve().parents[2]


def _save(root, name, value):
    """Write a small immutable array with its expected descriptor."""
    path = root / f"{name}.npy"
    np.save(path, value, allow_pickle=False)
    return {"path": path.name, "sha256": price_feature_data.digest(path)}


def test_feature_coverage_counts_missing_rows_and_unavailable_blocks(tmp_path):
    """Every subset reports its exclusions against the same core rows per split."""
    matrix = np.array([[10, 1, 2], [11, np.nan, 2], [12, 1, np.nan], [13, 3, 4]])
    group = {
        "files": {"x": _save(tmp_path, "x", matrix)},
        "feature_names": ["core", "feature_a", "feature_b"],
        "block_columns": {"a": [1], "b": [2], "retrospective": []},
        "cohorts": {"stage_core": {"indices": {
            "train": _save(tmp_path, "train", np.array([0, 1, 2])),
            "validation": _save(tmp_path, "validation", np.array([3])),
        }}},
    }
    settings = {"feature_blocks": {
        "a": {"features": [{"name": "feature_a"}, {"name": "omitted_a"}]},
        "b": {"features": [{"name": "feature_b"}]},
        "retrospective": {"features": [{"name": "missing_realized"}]},
    }}
    report = price_staged_search.feature_coverage(tmp_path, group, settings)
    train = report["train"]
    assert train["core_eligible_rows"] == 3
    assert train["features"]["feature_a"]["excluded_rows"] == 1
    assert train["features"]["omitted_a"]["status"] == "omitted"
    assert train["blocks"]["a"]["status"] == "partial"
    assert train["blocks"]["a"]["available_rows"] == 2
    assert train["blocks"]["a"]["configured_available_rows"] == 0
    assert train["blocks"]["retrospective"]["available_rows"] is None
    combinations = {tuple(item["blocks"]): item for item in train["combinations"]}
    assert len(combinations) == 8
    assert combinations[()]["available_rows"] == 3
    assert combinations[("a", "b")]["available_rows"] == 1
    assert combinations[("a", "b")]["excluded_rows"] == 2
    assert combinations[("a", "retrospective")]["status"] == "unavailable"
    assert report["validation"]["core_eligible_rows"] == 1
    assert report["test"]["status"] == "unavailable"


@pytest.mark.parametrize("change", [
    {"source": "nord_auctions", "column": "value"},
    {"selectors": {"variable": "spread_it_nord_ch"}},
    {"column": "afrr_positive_price_eur_mwh"},
])
def test_retrospective_whitelist_rejects_disguised_price_source(tmp_path, change):
    """An allowed quantity name cannot conceal a target price, price field or selector."""
    base = load_config(ROOT / "configs/research/price_feature_selection_full_year.yaml")["price_feature_search"]
    options = load_config(ROOT / "configs/research/price_staged_search.yaml")["price_staged_search"]
    feature = next(item for item in base["feature_blocks"]["realized_flows"]["features"]
                   if item["name"] == "ch_generation_same_day")
    feature.update(change)
    with pytest.raises(ValueError, match="(?i)(forbidden|target|source|selector)"):
        price_staged_search.staged_settings(base, options, tmp_path)


def test_staged_snapshot_end_is_configurable_and_fingerprinted(tmp_path):
    """December targets are included without changing the November test boundary."""
    base = load_config(ROOT / "configs/research/price_feature_selection_full_year.yaml")["price_feature_search"]
    options = load_config(ROOT / "configs/research/price_staged_search.yaml")["price_staged_search"]
    settings = price_staged_search.staged_settings(base, options, tmp_path)
    assert settings["staged_contract"]["snapshot_end_exclusive"] == "2027-01-01"
    assert price_staged_search.staged_inputs(settings)["end_exclusive"] == "2027-01-01"
    for group in settings["product_groups"].values():
        assert group["native_end_exclusive"] == "2027-01-01"
        assert group["test_end"] == "2026-12-01"
    options["snapshot_end_exclusive"] = "2027-02-01"
    changed = price_staged_search.staged_settings(base, options, tmp_path)
    assert price_staged_search.engine.signature(settings) != price_staged_search.engine.signature(changed)
    options["snapshot_end_exclusive"] = "2026-11-01"
    with pytest.raises(ValueError, match="cover"):
        price_staged_search.staged_settings(base, options, tmp_path)


@pytest.mark.parametrize("extended", [False, True])
def test_native_horizon_retains_december_without_expanding_test(tmp_path, monkeypatch, extended):
    """Older callers stop at test_end; staged callers may retain later targets for export."""
    days = pd.to_datetime(["2026-09-15T10:00Z", "2026-10-15T10:00Z",
                           "2026-11-15T10:00Z", "2026-12-15T10:00Z"], utc=True)
    prices = pd.DataFrame({"product_start": days, "product_end": days + pd.Timedelta(minutes=15),
                           "value": [1., 2., 3., 4.], "available_at": days - pd.Timedelta(days=2)})
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"selection": {"start": "2026-09-01",
        "end_exclusive": "2027-01-01", "timezone": "Europe/Rome"},
        "tables": {"nord_auctions": {"path": "unused.parquet"}}}))
    monkeypatch.setattr(price_feature_data.pd, "read_parquet", lambda _path: prices)
    monkeypatch.setattr(price_feature_data, "_external_grid", lambda _frames, grid, _settings:
                        (np.empty((len(grid), 0)), [], {}))
    monkeypatch.setattr(price_feature_data, "_price_sources", lambda _frame, _settings: {"DA": prices})
    monkeypatch.setattr(price_feature_data, "_feature_histories", lambda *_args: {"DA": {}})
    monkeypatch.setattr(price_feature_data, "_base_matrix", lambda _history, _histories, starts, _settings, _group:
                        (np.ones((len(starts), 1)), ["core"], np.zeros(len(starts)),
                         np.ones(len(starts)), starts, 0))
    group = {"market": "DA", "native_minutes": 15, "train_start": "2026-09-01",
             "train_end": "2026-10-01", "validation_start": "2026-10-01", "validation_end": "2026-11-01",
             "test_start": "2026-11-01", "test_end": "2026-12-01"}
    if extended:
        group["native_end_exclusive"] = "2027-01-01"
    settings = {"product_groups": {"da": group}, "feature_blocks": {}, "cohorts": {"core": []},
                "timezone": "Europe/Rome", "embargo_days": 0, "inner_min_days": 1,
                "inner_validation_fraction": 0.2, "minimum_training_rows": 0,
                "minimum_inner_rows": 0, "minimum_evaluation_rows": 0}
    result = price_feature_data.prepare_matrices(settings, source, tmp_path / "prepared")["da"]
    starts = np.load(tmp_path / "prepared" / result["files"]["starts"]["path"])
    test = np.load(tmp_path / "prepared" / result["cohorts"]["core"]["indices"]["test"]["path"])
    assert len(starts) == (4 if extended else 3)
    assert pd.to_datetime(starts[test], utc=True).strftime("%Y-%m-%d").tolist() == ["2026-11-15"]

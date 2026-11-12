"""Independently verify staged price-search scheduling and information contracts."""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies import price_staged_search as staged


ROOT = Path(__file__).resolve().parents[2]


def _settings():
    """Load the actual nine-family contract and staged study options."""
    base = load_config(ROOT / "configs/research/price_feature_selection_full_year.yaml")["price_feature_search"]
    options = load_config(ROOT / "configs/research/price_staged_search.yaml")["price_staged_search"]
    return base, options


def test_candidate_settings_are_fixed_across_feature_sets_and_bounded():
    """Each family/product receives exactly four deterministic candidate configurations."""
    base, options = _settings()
    base = staged.staged_settings(base, options, ROOT / "outputs/unused-staged-test")
    for family in base["parameter_spaces"]:
        first = staged.candidates(base, "da_quarter_hourly", family)
        second = staged.candidates(base, "da_quarter_hourly", family)
        assert first == second
        assert len(first) == options["configurations_per_family"] == 4
        assert len({item["seed"] for item in first}) == 4


def test_optional_retrospective_quantity_requires_a_finite_source_value():
    """A missing selected quantity is detected without requiring actual market data."""
    feature = {"source": "swiss", "column": "value", "selectors": {"variable": "schedule"}}
    frames = {"swiss": pd.DataFrame({"variable": ["other", "schedule"], "value": [1, np.nan]})}
    assert not staged._available_feature(frames, feature)
    frames["swiss"].loc[1, "value"] = 4.0
    assert staged._available_feature(frames, feature)


@pytest.mark.parametrize("retrospective_available", [False, True])
def test_stages_keep_matched_cohorts_and_never_exceed_four_configurations(
    tmp_path, monkeypatch, retrospective_available,
):
    """Simulated job results exercise orchestration without fitting or reading a catalog."""
    base, options = _settings()
    block_names = options["initial_blocks"]
    columns = {name: [index + 1] for index, name in enumerate(block_names)}
    columns["retrospective"] = [5] if retrospective_available else []
    group = {
        "base_width": 1,
        "block_columns": columns,
        "cohorts": {
            "stage_main": {"training_eligible": True},
            "stage_retrospective": {"training_eligible": True},
        },
    }
    prepared = {"groups": {"da_quarter_hourly": group}, "scientific_config_sha256": "test"}
    (tmp_path / "prepared.json").write_text(json.dumps(prepared))
    monkeypatch.setattr(staged.engine, "prepare_search", lambda *_args, **_kwargs: prepared)
    submitted = defaultdict(list)

    def fake_jobs(stage, jobs, *_args, **_kwargs):
        """Score synthetic job descriptors and preserve their selected cohorts."""
        submitted[stage].extend(deepcopy(jobs))
        rows = []
        for job in jobs:
            selected = job["arm"]
            # Prefer the first two optional features so ablation scheduling is exercised.
            error = 20.0 - sum(column in selected["columns"] for column in (1, 2))
            rows.append({**job, "status": "complete", "train_rows": 20, "validation_rows": 5,
                         "metrics": {"wape_percent": error, "mae_eur_per_mwh": error,
                                     "rmse_eur_per_mwh": error, "observations": 5}})
        return rows

    monkeypatch.setattr(staged, "_run_jobs", fake_jobs)
    result = staged.run_staged(base, options, tmp_path, workers=1)
    assert result["status"] == "complete"
    assert "outer periods not evaluated" in result["selection"]
    assert not (tmp_path / "holdout_scores.csv").exists()
    for stage in ("screen", "combinations", "ablation", "refinement"):
        assert all(job["arm"]["cohort"] == "stage_main" for job in submitted[stage])
    assert submitted["ablation"]
    configurations = defaultdict(set)
    for jobs in submitted.values():
        for job in jobs:
            key = (job["arm"]["group"], job["candidate"]["family"], tuple(job["arm"]["columns"]))
            configurations[key].add(json.dumps(job["candidate"]["parameters"], sort_keys=True))
    assert all(len(values) <= 4 for values in configurations.values())

    paired = submitted["retrospective"]
    winner = result["groups"]["da_quarter_hourly"]
    if retrospective_available:
        assert len(paired) == 2
        assert {job["arm"]["cohort"] for job in paired} == {"stage_retrospective"}
        assert paired[0]["candidate"] == paired[1]["candidate"]
        assert winner["retrospective"] is not None
    else:
        assert paired == []
        assert winner["retrospective"] is None


def test_matched_fit_passes_identical_training_and_validation_rows(tmp_path, monkeypatch):
    """Adding a block must preserve both the fit rows and chronological validation rows."""
    arrays = {
        "x": np.array([[0, 10], [1, 11], [2, 12], [3, 13], [4, 14]], dtype=float),
        "y": np.array([100, 101, 102, 103, 104], dtype=float),
        "anchor": np.zeros(5),
        "reference": np.ones(5),
        "inner_train": np.array([0, 2]),
        "inner_validation": np.array([3, 4]),
    }
    inputs = []

    def fake_fit(_family, _candidate, x, y, test_x):
        inputs.append((x.copy(), y.copy(), test_x.copy()))
        return None, np.full(len(test_x), 100.0), []

    monkeypatch.setattr(staged.engine, "_WORKER_OUTPUT", str(tmp_path))
    monkeypatch.setattr(staged.engine, "_WORKER_PREPARED", {"groups": {"da": {"feature_names": ["core", "external"]}}})
    monkeypatch.setattr(staged.engine, "_load_arrays", lambda *_args: arrays)
    monkeypatch.setattr(staged.engine, "_fit", fake_fit)
    candidate = {"family": "ridge", "trial": 0, "parameters": {"alpha": 1.0}, "seed": 11}
    for columns in ([0], [0, 1]):
        arm = {"group": "da", "cohort": "stage_main", "columns": columns, "blocks": [],
               "availability_class": "assumed_previous_day"}
        result = staged._staged_fit({"arm": arm, "candidate": candidate})
        assert result["status"] == "complete"
    assert np.array_equal(inputs[0][0][:, 0], inputs[1][0][:, 0])
    assert np.array_equal(inputs[0][1], inputs[1][1])
    assert np.array_equal(inputs[0][2][:, 0], inputs[1][2][:, 0])
    assert inputs[0][0][:, 0].tolist() == [0.0, 2.0]
    assert inputs[0][2][:, 0].tolist() == [3.0, 4.0]


def test_main_feature_block_rejects_target_derived_price():
    """Target-containing spreads cannot enter by being relabelled as a main feature."""
    base, options = _settings()
    base["feature_blocks"]["france"]["features"].append({
        "name": "it_ch_da_spread_same_day", "source": "swiss_commercial_border", "column": "value",
    })
    with pytest.raises(ValueError, match="(?i)(spread|target|forbidden)"):
        staged.staged_settings(base, options, ROOT / "outputs/unused-staged-test")


def test_renamed_target_price_cannot_bypass_feature_guard():
    """The source selector is checked even if a leaking feature has an innocuous name."""
    base, options = _settings()
    base["feature_blocks"]["france"]["features"].append({
        "name": "neighbor_signal", "source": "other", "column": "value",
        "selectors": {"variable": "spread_it_nord_ch"},
    })
    with pytest.raises(ValueError, match="(?i)(spread|target|forbidden)"):
        staged.staged_settings(base, options, ROOT / "outputs/unused-staged-test")


def test_failed_screen_stops_before_selection(tmp_path, monkeypatch):
    """A failed fit must not let the search report a complete winner selection."""
    base, options = _settings()
    group = {
        "base_width": 1,
        "block_columns": {name: [index + 1] for index, name in enumerate(options["initial_blocks"])},
        "cohorts": {"stage_main": {"training_eligible": True}},
    }
    prepared = {"groups": {"da_quarter_hourly": group}, "scientific_config_sha256": "test"}
    (tmp_path / "prepared.json").write_text(json.dumps(prepared))
    monkeypatch.setattr(staged.engine, "prepare_search", lambda *_args, **_kwargs: prepared)
    phases = []

    def fail_screen(stage, _jobs, *_args, **_kwargs):
        phases.append(stage)
        return [{"id": "failed-fit", "status": "failed", "error": "fixture failure"}]

    monkeypatch.setattr(staged, "_run_jobs", fail_screen)
    result = staged.run_staged(base, options, tmp_path, workers=1)
    assert phases == ["screen"]
    assert result["status"] == "partial"
    assert result["phase"] == "screen"
    assert result["failed_candidates"][0]["id"] == "failed-fit"

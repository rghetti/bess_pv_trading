"""Check holiday-feature insertion preserves external block indexing."""

import numpy as np
import pandas as pd
from contextlib import nullcontext
from types import SimpleNamespace

from bess_pv_trading.forecasting.studies.price_feature_data import digest
from bess_pv_trading.forecasting.studies.price_full_year_features import annotate_cohort_coverage, append_calendar_features
from scripts.search_price_full_year import run_inner_only


def test_holiday_features_are_inserted_before_optional_blocks(tmp_path):
    """Base and external values stay associated with their original feature names."""
    matrix = np.array([[1, 10, 20], [2, 30, 40]], dtype=np.float32)
    x_path = tmp_path / "x.npy"
    starts_path = tmp_path / "starts.npy"
    np.save(x_path, matrix)
    starts = pd.DatetimeIndex(["2025-04-25T10:00Z", "2025-04-26T10:00Z"]).as_unit("ns")
    np.save(starts_path, starts.asi8)
    groups = {"da": {"files": {"x": {"path": "x.npy", "sha256": digest(x_path)},
        "starts": {"path": "starts.npy", "sha256": digest(starts_path)}},
        "base_width": 1, "feature_names": ["base", "gas", "france"],
        "block_columns": {"gas": [1], "france": [2]}}}
    calendar = {"timezone": "Europe/Rome", "fixed_holidays": ["04-25"], "easter_holidays": False}
    holiday = {"hour_interactions": True, "include_adjacent_days": True, "include_bridge_day": True}
    result = append_calendar_features(groups, tmp_path, calendar, holiday)["da"]
    expanded = np.load(x_path)
    assert expanded.shape == (2, 30)
    assert expanded[:, 0].tolist() == [1, 2]
    assert expanded[:, result["block_columns"]["gas"][0]].tolist() == [10, 30]
    assert expanded[:, result["block_columns"]["france"][0]].tolist() == [20, 40]
    assert result["base_width"] == 28
    assert expanded[:, result["feature_names"].index("holiday_hour_12")].tolist() == [1, 0]
    assert digest(x_path) == result["files"]["x"]["sha256"]


def test_missing_future_rows_do_not_reclassify_training_eligibility():
    """Swiss candidates can train while unavailable validation and test are flagged."""
    groups = {"ida": {"cohorts": {"swiss": {"eligible": True, "counts": {"validation": 0, "test": 0}},
                                    "complete": {"eligible": True, "counts": {"validation": 100, "test": 120}}}}}
    result = annotate_cohort_coverage(groups, 96)["ida"]["cohorts"]
    assert result["swiss"]["training_eligible"] is True
    assert result["swiss"]["evaluation_minimum_met"] is False
    assert result["complete"]["evaluation_minimum_met"] is True


def test_inner_only_does_not_schedule_outer_evaluation(tmp_path):
    """Training-only execution schedules inner trials and stops after saving status."""
    prepared = {"inner_fits": 2, "scientific_config_sha256": "study", "groups": {},
                "tasks": [{"id": "candidate-family"}]}
    calls = []
    writes = []
    trial_task = object()

    def pool(function, jobs, workers, progress, output, groups):
        calls.append((function, jobs, workers))
        rows = [{"successful": 2, "failed": 0}]
        progress(rows)
        return rows

    engine = SimpleNamespace(run_lock=lambda output: nullcontext(),
        prepare_search=lambda settings, inputs, output: prepared,
        _pool_jobs=pool, _trial_task=trial_task,
        write_json=lambda path, state: writes.append(dict(state)))
    result = run_inner_only({"workers": 1}, {}, tmp_path, engine)
    assert calls == [(trial_task, [(prepared["tasks"][0], False)], 1)]
    assert result["status"] == "inner_complete"
    assert result["phase"] == "inner_finished"
    assert result["inner_fits_complete"] == 2
    assert writes[-1]["status"] == "inner_complete"

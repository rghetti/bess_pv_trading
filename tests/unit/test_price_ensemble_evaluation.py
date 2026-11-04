"""Chronological forecast-combination selection and alignment checks."""

import numpy as np
import pytest

from scripts.evaluate_price_ensembles import evaluate_group


def test_ensemble_ranking_and_size_selection_never_consult_later_test_values():
    """Altering April labels must not affect rankings or selected ensemble size."""
    days = np.array(["2026-02-01", "2026-02-02", "2026-03-20", "2026-03-21", "2026-04-01", "2026-04-02"])
    actual = np.array([100.0] * len(days))
    forecasts = np.array([[101, 104, 95], [101, 104, 95], [103, 97, 110],
                          [103, 97, 110], [101, 103, 90], [101, 103, 90]], dtype=float)
    settings = {"rank_start": "2026-02-01", "rank_end_exclusive": "2026-03-16",
        "tune_start": "2026-03-16", "tune_end_exclusive": "2026-04-01",
        "test_start": "2026-04-01", "test_end_exclusive": "2026-05-01", "ensemble_sizes": [1, 2]}
    result, _, _ = evaluate_group(actual, actual + 5, forecasts, ["a", "b", "c"], days, settings)
    changed = actual.copy()
    changed[-2:] = 500
    altered, _, _ = evaluate_group(changed, actual + 5, forecasts, ["a", "b", "c"], days, settings)
    assert result["selected_candidates"] == altered["selected_candidates"] == ["a", "b"]
    assert result["selected_size"] == altered["selected_size"] == 2
    assert result["test_combination"]["mae_eur_per_mwh"] != altered["test_combination"]["mae_eur_per_mwh"]


def test_ensemble_rejects_overlapping_selection_and_test():
    settings = {"rank_start": "2026-02-01", "rank_end_exclusive": "2026-03-16",
        "tune_start": "2026-03-15", "tune_end_exclusive": "2026-04-01",
        "test_start": "2026-04-01", "test_end_exclusive": "2026-05-01", "ensemble_sizes": [1]}
    with pytest.raises(ValueError, match="nonoverlapping"):
        evaluate_group([1], [1], [[1]], ["a"], ["2026-03-20"], settings)


def test_completed_search_produces_ensemble_evidence_and_test_plot(tmp_path):
    """The artifact workflow consumes all declared matched candidates and emits a later-period report."""
    import json
    import pandas as pd
    from scripts.evaluate_price_ensembles import run

    source = tmp_path / "source"
    source.mkdir()
    config = {"product_groups": {"ida1": {"validation_start": "2026-02-01", "validation_end": "2026-05-01"}},
              "feature_sets": ["price_only"], "candidates": [{"id": "a"}, {"id": "b"}]}
    records = [{"group": "ida1", "feature_set": "price_only", "candidate": name} for name in ("a", "b")]
    (source / "manifest.json").write_text(json.dumps({"status": "complete", "failures": [],
                                                       "config": config, "results": records}))
    for name, value in (("a", 101), ("b", 99)):
        pd.DataFrame({"product_start": ["2026-02-01T12:00Z", "2026-03-20T12:00Z", "2026-04-01T12:00Z"],
                      "actual": [100, 100, 100], "seasonal_naive": [110, 110, 110],
                      "point": [value] * 3}).to_csv(source / f"ida1__price_only__{name}.csv", index=False)
    settings = {"source_search": "source", "output_root": "outputs", "run_id": "test", "timezone": "UTC",
        "rank_start": "2026-02-01", "rank_end_exclusive": "2026-03-16", "tune_start": "2026-03-16",
        "tune_end_exclusive": "2026-04-01", "test_start": "2026-04-01", "test_end_exclusive": "2026-05-01",
        "ensemble_sizes": [1, 2]}
    result = run({"_project_root": str(tmp_path), "price_ensemble_evaluation": settings})
    output = tmp_path / "outputs/test"
    assert result["status"] == "complete"
    assert result["groups"]["ida1"]["selected_size"] == 2
    assert (output / "ida1_daily_mae.png").is_file()
    assert (output / "ida1_test_predictions.csv").is_file()
    assert "not a globally untouched test" in (output / "summary.md").read_text()

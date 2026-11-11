"""Check the user-run holiday diagnostic without training a model."""

import json

import pytest

from scripts.diagnose_price_holiday_winner import best_saved_candidate, feature_variants


def test_select_best_saved_candidate_for_one_cohort(tmp_path):
    """Ignore other cohorts and incomplete candidates while ranking saved scores."""
    (tmp_path / "trials").mkdir()
    arm = {"group": "da_quarter_hourly", "cohort": "full_history", "id": "one", "columns": [0]}
    task = {"id": "one_ridge", "signature": "abc", "arm": arm}
    data = {"signature": "abc", "results": [
        {"trial": 0, "family": "ridge", "status": "failed"},
        {"trial": 1, "family": "ridge", "status": "complete",
         "metrics": {"wape_percent": 8, "mae_eur_per_mwh": 12}}]}
    (tmp_path / "trials" / "one_ridge.json").write_text(json.dumps(data))
    chosen, count = best_saved_candidate(tmp_path, {"tasks": [task]}, "da_quarter_hourly", "full_history")
    assert count == 1
    assert chosen["candidate"]["trial"] == 1
    with pytest.raises(ValueError, match="No completed candidates"):
        best_saved_candidate(tmp_path, {"tasks": [task]}, "da_quarter_hourly", "other")


def test_feature_ablation_retains_existing_holiday_and_other_features():
    """Only the 27 new interactions/adjacent indicators are removed."""
    names = ["own_price", "italian_public_holiday", *[f"holiday_hour_{hour:02d}" for hour in range(24)],
             "day_before_holiday", "day_after_holiday", "weekday_bridge_between_holiday_and_weekend", "gas"]
    variants = feature_variants({"feature_names": names}, {"arm": {"columns": list(range(len(names)))}})
    assert variants["original_holiday_flag_only"] == [0, 1, len(names) - 1]
    assert len(variants["expanded_holidays"]) == len(names)

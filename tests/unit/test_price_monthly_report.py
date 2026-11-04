"""Monthly price-search diagnostics use local months and pooled relative error."""

import pandas as pd
import pytest

from scripts.report_price_search_monthly import monthly_rows


def test_monthly_scores_use_local_delivery_month_and_zero_safe_denominator():
    """A UTC evening product can belong to the next local delivery month."""
    frame = pd.DataFrame({"product_start": ["2026-01-31T23:00Z", "2026-02-01T00:00Z"],
                          "actual": [0., -10.], "point": [1., -7.], "seasonal_naive": [0., -5.]})
    rows = monthly_rows(frame, timezone="Europe/Rome", near_zero_threshold=1.)
    assert len(rows) == 1
    assert rows[0]["month"] == "2026-02"
    assert rows[0]["wape_percent"] == 40.
    assert rows[0]["negative_price_count"] == 1
    assert rows[0]["near_zero_price_count"] == 1


def test_monthly_scores_reject_duplicate_products():
    frame = pd.DataFrame({"product_start": ["2026-01-01T00:00Z"] * 2,
                          "actual": [1., 2.], "point": [1., 2.], "seasonal_naive": [1., 2.]})
    with pytest.raises(ValueError, match="unique"):
        monthly_rows(frame, timezone="UTC", near_zero_threshold=1.)


def test_completed_search_generates_monthly_table_and_figure(tmp_path):
    """The report records every matched candidate and the source manifest hash."""
    import json
    from scripts.report_price_search_monthly import run

    source = tmp_path / "source"
    source.mkdir()
    study = {"product_groups": {"DA": {}}, "feature_sets": ["price_only"],
             "candidates": [{"id": "a"}, {"id": "b"}]}
    records = [{"group": "DA", "feature_set": "price_only", "candidate": candidate,
                "mae_eur_per_mwh": error} for candidate, error in (("a", 2.), ("b", 3.))]
    (source / "manifest.json").write_text(json.dumps({"status": "complete", "failures": [],
                                                       "config": study, "results": records}))
    for candidate, prediction in (("a", [10., 13.]), ("b", [12., 14.])):
        pd.DataFrame({"product_start": ["2026-01-01T00:00Z", "2026-02-01T00:00Z"],
                      "actual": [10., 10.], "point": prediction, "seasonal_naive": [12., 12.]}).to_csv(
                          source / f"DA__price_only__{candidate}.csv", index=False)
    settings = {"source_search": "source", "output_root": "out", "run_id": "test", "timezone": "UTC",
                "near_zero_eur_per_mwh": 5., "target_wape_percent": 4.}
    result = run({"_project_root": str(tmp_path), "price_monthly_report": settings})
    output = tmp_path / "out/test"
    assert result["status"] == "complete"
    assert len(pd.read_csv(output / "monthly_scores.csv")) == 4
    assert (output / "DA_monthly_wape.png").is_file()
    assert "exploratory" in (output / "summary.md").read_text()

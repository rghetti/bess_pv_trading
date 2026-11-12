"""Verify staged reports retain the features needed to interpret ablations."""

import csv
import json

from bess_pv_trading.forecasting.studies.price_staged_reporting import write_stage_reports


def test_ablation_report_names_removed_last_feature_in_a_block(tmp_path):
    """A one-column block removal remains visible after its block label disappears."""
    group = {"feature_names": ["core", "gas"], "block_columns": {"gas": [1]}}
    candidate = {"family": "ridge", "trial": 0, "seed": 2, "parameters": {"alpha": 1}}
    arm = {"group": "da", "cohort": "stage_main", "columns": [0, 1],
           "blocks": ["gas"], "availability_class": "assumed_previous_day"}
    metrics = {"mae_eur_per_mwh": 2.0, "wape_percent": 3.0, "rmse_eur_per_mwh": 4.0,
               "relative_mae_to_reference": 0.8}
    parent = {"status": "complete", "arm": arm, "candidate": candidate, "metrics": metrics}
    stage_path = tmp_path / "stages"
    stage_path.mkdir()
    (stage_path / "combinations.json").write_text(json.dumps({"results": [parent]}))
    reduced = {**parent, "arm": {**arm, "columns": [0], "blocks": []},
               "feature_names": ["core"], "train_rows": 10, "validation_rows": 5}

    write_stage_reports(tmp_path, "ablation", [reduced], {"da": group})

    with (tmp_path / "reports" / "da" / "ablation.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert json.loads(row["removed_features"]) == ["gas"]
    assert json.loads(row["features"]) == ["core"]
    assert row["validation_rows"] == "5"


def test_screen_report_writes_table_and_plot_for_each_product(tmp_path):
    """The screening summary exposes point metrics and a figure without fitting a model."""
    group = {"feature_names": ["core"], "block_columns": {}}
    candidate = {"family": "ridge", "trial": 0, "seed": 2, "parameters": {"alpha": 1}}
    arm = {"group": "ida1", "cohort": "stage_main", "columns": [0],
           "blocks": [], "availability_class": "assumed_previous_day"}
    metrics = {"mae_eur_per_mwh": 2.0, "wape_percent": 3.0, "rmse_eur_per_mwh": 4.0,
               "relative_mae_to_reference": 0.8}
    row = {"status": "complete", "arm": arm, "candidate": candidate, "metrics": metrics,
           "feature_names": ["core"], "train_rows": 10, "validation_rows": 5}

    write_stage_reports(tmp_path, "screen", [row], {"ida1": group})

    assert (tmp_path / "reports" / "ida1" / "screen.png").is_file()
    with (tmp_path / "reports" / "ida1" / "screen.csv").open(newline="") as stream:
        written = next(csv.DictReader(stream))
    assert written["mae_eur_per_mwh"] == "2.0"
    assert json.loads(written["parameters"]) == {"alpha": 1}

"""Guard plotting against mismatched forecast samples."""

import json

import pandas as pd
import pytest

from scripts.plot_matched_price_baselines import run


def test_rejects_mismatched_targets_before_output(tmp_path):
    """A changed actual price cannot be plotted as a matched comparison."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"status": "complete", "failures": []}))
    for name, target in (("reference", 10.0), ("model", 11.0)):
        pd.DataFrame({"product_start": ["2026-01-01T00:00:00Z"], "actual": [target], "point": [10.0]}).to_csv(
            source / f"{name}.csv", index=False)
    settings = {"output_root": "out", "run_id": "study", "timezone": "UTC", "markets": ["IDA1"],
        "studies": [{"label": "period", "run": "source", "groups": [{"market": "IDA1",
            "reference_file": "reference.csv", "model_file": "model.csv"}]}]}
    with pytest.raises(ValueError, match="same|share"):
        run(settings, tmp_path)
    assert not (tmp_path / "out").exists()

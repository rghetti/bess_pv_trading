"""Guard the latest-cleared bias evaluation's source replay boundary."""

import json

import pytest

from scripts.evaluate_latest_price_bias import run


def test_rejects_incomplete_source_without_creating_output(tmp_path):
    """Partial or failed replays cannot be treated as finished forecasts."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"status": "running", "failures": []}))
    config = {"online_price_bias": {"selected_evaluation": "source", "output_root": "results", "run_id": "study"}}
    with pytest.raises(ValueError, match="complete without failures"):
        run(config, tmp_path)
    assert not (tmp_path / "results").exists()

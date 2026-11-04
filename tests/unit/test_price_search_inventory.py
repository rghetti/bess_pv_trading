"""Verify search inventories distinguish missing from recorded convergence data."""

from scripts.inventory_price_searches import inventory_manifest


def test_inventory_distinguishes_unknown_and_recorded_optimizer_warnings():
    """Old fits without warning fields must not be counted as converged."""
    manifest = {"status": "complete", "config": {"product_groups": {"q": {
        "target": "IDA1", "native_minutes": 15, "validation_start": "2026-01-01", "validation_end": "2026-02-01"}}},
        "coverage": [{"group": "q", "shared_days": 30}], "failures": [], "results": [
            {"group": "q"}, {"group": "q", "convergence_warnings": []},
            {"group": "q", "convergence_warnings": ["did not converge"]}]}
    row = inventory_manifest(manifest, run_name="example", source_hash="abc")[0]
    assert row["completed_fits"] == 3
    assert row["warning_status_unknown_fits"] == 1
    assert row["warning_fits"] == 1
    assert row["shared_days"] == 30

"""Verify matched target comparison and transparent partial-run handling."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from bess_pv_trading.reporting import ida3_soc_sweep_comparison as comparison


def _config(target, intermediate):
    """Use identical replay assumptions except the requested policy switch."""
    return {"continuous_optimization": {"run_id": "location", "output_root": "out",
        "ida3_enforce_intermediate_soc": intermediate},
        "state": {"initial_soc_percent": 50, "terminal_soc_percent": target},
        "battery": {"capacity_mwh": 2.0}, "soc_sweep": {"workers": 7}}


def test_config_only_permits_the_requested_midnight_change():
    """A materially different asset cannot masquerade as a paired policy run."""
    old = _config(10, True)
    new = _config(10, False)
    comparison._compatible_config(old, new)
    changed = deepcopy(new)
    changed["battery"]["capacity_mwh"] = 3
    with pytest.raises(ValueError, match="differ beyond"):
        comparison._compatible_config(old, changed)
    with pytest.raises(ValueError, match="does not release"):
        comparison._compatible_config(old, old)


def test_matched_comparison_lists_missing_and_incompatible_pairs(tmp_path, monkeypatch):
    """Only input-matched complete trajectories produce financial differences."""
    old, new = tmp_path / "old", tmp_path / "new"
    for root, intermediate in ((old, True), (new, False)):
        for target in (10, 20):
            destination = root / "runs" / "farm_1" / f"soc_{target}"
            destination.mkdir(parents=True)
            (destination / "resolved_config.json").write_text(json.dumps(_config(target, intermediate)))
    prior = [{"farm": "1", "target_soc_percent": target, "status": "complete",
              "input_fingerprint": f"fingerprint-{target}"} for target in (10, 20)]
    current = deepcopy(prior)
    current[1]["input_fingerprint"] = "different-forecast"
    (old / "manifest.json").write_text(json.dumps({"status": "running", "results": prior}))
    fields = ("auction_revenue_eur", "net_profit_eur", "imbalance_cashflow_eur", "degradation_eur",
              "absolute_imbalance_mwh", "signed_imbalance_mwh", "curtailment_mwh", "pv_realized_energy_mwh",
              "charge_mwh", "discharge_mwh", "initial_soc_percent", "final_soc_percent",
              "mean_midnight_soc_percent", "midnights_at_target", "auction_fallbacks", "auction_solver_seconds")
    def read(root, result):
        offset = 1.0 if root == new else 0.0
        return {key: offset for key in fields}, root
    checks = []
    monkeypatch.setattr(comparison, "_read_trajectory", read)
    monkeypatch.setattr(comparison, "_check_common_inputs", lambda left, right: checks.append((left, right)) or {"products": 2976})
    monkeypatch.setattr(comparison, "_plot_differences", lambda frame, farm, path: path.write_bytes(b"fixture"))
    summary = comparison.write_soc_sweep_comparison(new, old, current, ["1"], [10, 20, 30])
    text = summary.read_text()
    assert "Matched 1 of 3" in text
    assert "Reference batch status when read: running" in text
    assert "20%: prepared-input fingerprints differ" in text
    assert "30%: reference not yet recorded; final-only not yet recorded" in text
    rows = pd.read_csv(new / "baseline_comparison/paired_metrics.csv")
    assert rows.target_soc_percent.tolist() == [10]
    assert rows.delta_net_profit_eur.tolist() == [1.0]
    assert len(checks) == 1
    assert (new / "baseline_comparison/farm_1_summary.md").is_file()
    assert not (old / "baseline_comparison").exists()


def test_no_completed_pairs_still_writes_partial_summary(tmp_path):
    """A running reference does not fail a completed final-only study."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "manifest.json").write_text(json.dumps({"status": "running", "results": []}))
    current = [{"farm": "1", "target_soc_percent": 10, "status": "failed"}]
    summary = comparison.write_soc_sweep_comparison(new, old, current, ["1"], [10, 20])
    assert "Matched 0 of 2" in summary.read_text()
    assert "10%: reference not yet recorded; final-only failed" in summary.read_text()


def test_reference_manifest_retries_partial_json_and_fails_clearly(tmp_path, monkeypatch):
    """A concurrent manifest rewrite is retried without reading an invalid snapshot."""
    path = tmp_path / "manifest.json"
    path.write_text('{"status": "running"}')
    original = type(path).read_text
    attempts = []
    def transient(self, *args, **kwargs):
        attempts.append(1)
        return '{"status":' if len(attempts) < 3 else original(self, *args, **kwargs)
    monkeypatch.setattr(type(path), "read_text", transient)
    monkeypatch.setattr(comparison, "sleep", lambda seconds: None)
    assert comparison._load_reference_manifest(path)["status"] == "running"
    assert len(attempts) == 3
    monkeypatch.setattr(type(path), "read_text", lambda self, *args, **kwargs: '{"status":')
    with pytest.raises(ValueError, match="remained unreadable after retries"):
        comparison._load_reference_manifest(path)

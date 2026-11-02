"""The four-policy summary pairs matching profile/day results and PV-only values."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("extended_participation", ROOT / "scripts/run_extended_participation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_four_policy_summary_pairs_profiles_and_rejects_changed_sources(tmp_path):
    configs, roots = [], []
    day = "2026-01-01"
    for index, policy in enumerate(MODULE.POLICIES):
        root = tmp_path / policy
        roots.append(root)
        config = {"plant": {"farm_id": 1}, "continuous_optimization": {"pv_ensemble": {"planning_policy": policy}},
            "continuous_participation_study": {
            "candidate_auctions": ["DA"], "required_auctions": ["DA"], "profiles": None,
            "sequential_all_auctions_baseline": False, "perfect_pv_baseline": False}}
        configs.append(config)
        _write(root / "json/study_config.json", config)
        _write(root / "json/manifest.json", {"status": "complete", "delivery_days": [day], "completed": ["DA"]})
        _write(root / "json/input_provenance.json", {"plant": {}, "asset_input_settings": {},
            "weather_forecasts": [], "settlement_values_sha256": "s", "pv_ensemble": {},
            "days": {day: {"realized_pv_sha256": "p", "auction_price_sha256": {}, "skipped_auctions": []}}})
        (root / "csv").mkdir()
        pd.DataFrame([{"profile": "DA", "delivery_day": day, "net_profit_eur": 10 + index,
            "absolute_imbalance_mwh": 2, "signed_imbalance_mwh": -1, "curtailment_mwh": 3,
            "initial_energy_mwh": 1, "final_energy_mwh": 0.5}]).to_csv(root / "csv/daily_participation.csv", index=False)
        pd.DataFrame([{"strategy": "DA", "delivery_day": day, "pv_bess_profit_eur": 10 + index, "pv_only_profit_eur": 5,
                       "incremental_profit_eur": 5 + index}]).to_csv(root / "csv/pv_only_comparison_daily.csv", index=False)
    summary = MODULE.summarize(configs, roots, tmp_path / "summary")
    text = summary.read_text()
    assert "DA | 10.00 / 5.00 / 5.00 | 11.00 / 5.00 / 6.00" in text
    assert len(pd.read_csv(tmp_path / "summary/period_comparison.csv")) == 4
    provenance = json.loads((roots[3] / "json/input_provenance.json").read_text())
    provenance["days"][day]["realized_pv_sha256"] = "different"
    _write(roots[3] / "json/input_provenance.json", provenance)
    with pytest.raises(ValueError, match="realized_pv_sha256"):
        MODULE.summarize(configs, roots, tmp_path / "other")


def test_all_farms_launcher_preflights_each_farm(monkeypatch):
    """The check-only option delegates each farm once in the configured order."""
    seen = []
    monkeypatch.setattr(MODULE, "run_farm", lambda farm, args, paths: seen.append((farm, args.check_only, len(paths))))
    monkeypatch.setattr(MODULE.sys, "argv", ["run_extended_participation.py", "--all-farms", "--check-only"])
    MODULE.main()
    assert seen == [(farm, True, 4) for farm in (1, 345, 901, 910)]


@pytest.mark.parametrize("farm_id", [1, 345, 901, 910])
@pytest.mark.parametrize("policy", MODULE.POLICIES)
def test_ten_day_config_resolves_each_farm_and_requires_da(farm_id, policy):
    """All farms share the policy configuration but resolve their own assets."""
    config = MODULE.load_config(ROOT / f"configs/full_experiments/participation_2026_01_ten_days_{policy}.yaml",
                                overrides={"farm_id": farm_id})
    assert config["plant"]["farm_id"] == config["forecast"]["farm_id"] == farm_id
    assert config["continuous_optimization"]["start_date"] == "2026-01-01"
    assert config["continuous_optimization"]["end_date"] == "2026-01-10"
    assert config["solver"]["threads"] == 2
    profiles = MODULE.participation_profiles(config["continuous_participation_study"])
    assert len(profiles) == 8 and all("DA" in profile for profile in profiles)
    assert config["continuous_participation_study"]["matched_pv_only"]


@pytest.mark.parametrize("failed_index", [None, 5])
def test_parallel_launcher_starts_all_farms_and_handles_failure(monkeypatch, capsys, failed_index):
    """Start all policies before waiting and terminate siblings after a failure."""
    launched = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            self.environment = kwargs["env"]
            self.stdout = ["progress\n"]
            self.index = len(launched)
            self.terminated = False
            launched.append(self)

        def poll(self):
            assert len(launched) == 16
            if self.terminated:
                return -15
            if failed_index is None:
                return 0
            return 7 if self.index == failed_index else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return self.poll()

        def kill(self):
            self.terminated = True

    monkeypatch.setattr(MODULE.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(MODULE, "sleep", lambda seconds: None)
    paths = [ROOT / f"{policy}.yaml" for policy in MODULE.POLICIES]
    if failed_index is None:
        MODULE._run_policies_parallel((1, 345, 901, 910), paths)
    else:
        with pytest.raises(MODULE.subprocess.CalledProcessError, match="returned non-zero"):
            MODULE._run_policies_parallel((1, 345, 901, 910), paths)
        assert all(process.terminated for process in launched if process.index != failed_index)
    assert len(launched) == 16
    assert [int(process.command[-1]) for process in launched] == [1] * 4 + [345] * 4 + [901] * 4 + [910] * 4
    assert all(process.environment["OPENBLAS_NUM_THREADS"] == "1" for process in launched)
    assert "[farm 345 envelope] progress" in capsys.readouterr().out


def test_all_farms_runs_one_concurrent_group_before_reports(monkeypatch):
    """Launch every farm together and report only after all policies finish."""
    actions = []
    monkeypatch.setattr(MODULE, "_farm_configs", lambda farm, paths: [])
    monkeypatch.setattr(MODULE, "_run_policies_parallel", lambda farms, paths: actions.append(("run", farms)))
    monkeypatch.setattr(MODULE, "run_farm", lambda farm, args, paths: actions.append(("report", farm)))
    monkeypatch.setattr(MODULE.sys, "argv", ["run_extended_participation.py", "--all-farms"])
    MODULE.main()
    assert actions == [("run", (1, 345, 901, 910)), ("report", 1), ("report", 345),
                       ("report", 901), ("report", 910)]

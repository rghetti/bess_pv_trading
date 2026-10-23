"""Moving completed runs preserves results, namespaces, and navigable references."""
import json
from pathlib import Path

import pytest

from bess_pv_trading.reporting.artifacts import organize_run_outputs, prepare_experiment_output
from battery_case_config import case_config


@pytest.mark.parametrize("scope", ["run", "run/farm/day"])
def test_fresh_output_replaces_old_artifacts_only_in_matching_scope(tmp_path, scope):
    root = tmp_path / "outputs"
    destination, sibling = root / scope, root / "another-run"
    destination.mkdir(parents=True)
    sibling.mkdir()
    (sibling / "keep.csv").write_text("other experiment")
    (destination / "plots").mkdir()
    (destination / "plots/stale.png").write_text("old plot")
    (destination / "checkpoint.json").write_text("old checkpoint")
    prepare_experiment_output(root, destination, overwrite=True)
    assert not list(destination.iterdir())
    assert (sibling / "keep.csv").read_text() == "other experiment"
    (destination / "new.csv").write_text("new result")
    prepare_experiment_output(root, destination, overwrite=True)
    assert not list(destination.iterdir())


def test_resume_preserves_outputs_and_disable_overwrite_rejects_replacement(tmp_path):
    destination = tmp_path / "run"
    destination.mkdir()
    checkpoint = destination / "checkpoint.json"
    checkpoint.write_text("checkpoint")
    prepare_experiment_output(tmp_path, destination, overwrite=True, resume=True)
    assert checkpoint.read_text() == "checkpoint"
    with pytest.raises(FileExistsError, match="overwrite_existing is false"):
        prepare_experiment_output(tmp_path, destination, overwrite=False)
    assert checkpoint.read_text() == "checkpoint"


def test_overwrite_preserves_continuous_lock_inode_and_unlinks_symlinks(tmp_path):
    destination = tmp_path / "run"
    destination.mkdir()
    lock = destination / "run.lock"
    lock.touch()
    inode = lock.stat().st_ino
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep.csv").write_text("input")
    (destination / "shortcut").symlink_to(external, target_is_directory=True)
    prepare_experiment_output(tmp_path, destination, overwrite=True, preserve=("run.lock",))
    assert lock.stat().st_ino == inode
    assert list(destination.iterdir()) == [lock]
    assert (external / "keep.csv").read_text() == "input"


def test_overwrite_rejects_output_root_escape_and_symlink_destination(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    for destination in (root, root / ".." / "outside"):
        with pytest.raises(ValueError, match="strictly below"):
            prepare_experiment_output(root, destination, overwrite=True)
    external = tmp_path / "external"
    external.mkdir()
    (root / "linked-run").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="strictly below"):
        prepare_experiment_output(root, root / "linked-run", overwrite=True)


@pytest.mark.parametrize("relative_root", [False, True])
def test_grouped_run_preserves_payloads_references_and_is_repeatable(tmp_path, monkeypatch, relative_root):
    monkeypatch.chdir(tmp_path)
    root = Path("run") if relative_root else tmp_path / "run"
    source_payloads = {
        "oracle/dispatch.csv": b"header\nrow\n",
        "oracle/dispatch.png": b"plot fixture",
        "soc_evolution.png": b"SOC plot fixture",
        "oracle/models/model.lp": b"model fixture",
        "solution.npz": b"array fixture",
    }
    for name, payload in source_payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    profit = case_config("realized_settlement")["expected"]["single_profit_eur"]
    summary = {"profit_eur": profit, "artifacts": str(root / "oracle"),
        "first_solve": {"model_file": str(root / "oracle/models/model.lp")},
        "input_database": "data/api/market.duckdb"}
    (root / "oracle/summary.json").write_text(json.dumps(summary))
    (root / "manifest.json").write_text(json.dumps({"summary": summary,
        "output_dir": str(root), "artifacts": ["missing.png"]}))

    manifest = organize_run_outputs(root)

    targets = {
        "oracle/dispatch.csv": "csv/oracle/dispatch.csv",
        "oracle/dispatch.png": "plots/oracle/dispatch.png",
        "soc_evolution.png": "plots/soc_evolution.png",
        "oracle/models/model.lp": "models/oracle/model.lp",
        "solution.npz": "arrays/solution.npz",
    }
    for old, new in targets.items():
        assert (root / new).read_bytes() == source_payloads[old]
        assert not (root / old).exists()
    saved_summary = json.loads((root / "json/oracle/summary.json").read_text())
    assert saved_summary["profit_eur"] == profit
    assert saved_summary["input_database"] == summary["input_database"]
    assert saved_summary["artifacts"] == str(root)
    assert saved_summary["first_solve"]["model_file"] == str(root / "models/oracle/model.lp")
    assert manifest["summary"] == saved_summary
    assert manifest["output_dir"] == str(root)
    assert all((root / name).is_file() for name in manifest["artifacts"])
    assert set(manifest["artifacts"]) == {
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file() and path != root / "json/manifest.json"}
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert organize_run_outputs(root) == manifest
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_colliding_names_fail_before_any_file_is_moved(tmp_path):
    (tmp_path / "csv").mkdir()
    (tmp_path / "dispatch.csv").write_text("original")
    (tmp_path / "csv/dispatch.csv").write_text("existing")
    with pytest.raises(FileExistsError):
        organize_run_outputs(tmp_path)
    assert (tmp_path / "dispatch.csv").read_text() == "original"
    assert (tmp_path / "csv/dispatch.csv").read_text() == "existing"


def test_invalid_json_fails_before_any_file_is_moved(tmp_path):
    (tmp_path / "dispatch.csv").write_text("original")
    (tmp_path / "summary.json").write_text("invalid JSON")
    with pytest.raises(json.JSONDecodeError):
        organize_run_outputs(tmp_path)
    assert (tmp_path / "dispatch.csv").is_file()
    assert not (tmp_path / "csv").exists()


def test_symlinks_are_not_followed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    linked = tmp_path / "run"
    linked.mkdir()
    (linked / "input.json").symlink_to(source / "input.json")
    with pytest.raises(ValueError, match="symlinks"):
        organize_run_outputs(linked)

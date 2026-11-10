"""Transfer only scientifically identical inner-fit checkpoints to a fresh snapshot.

The default action reports transferable checkpoints. --apply copies validated
files without overwriting existing work; neither action fits a model.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil

import numpy as np

from bess_pv_trading.forecasting.studies.price_feature_data import digest
from bess_pv_trading.forecasting.studies.price_feature_search import run_lock


def _load(root, descriptor, verified):
    """Verify each immutable prepared array once, then access it without copying."""
    path = root / descriptor["path"]
    if str(path) not in verified:
        if digest(path) != descriptor["sha256"]:
            raise ValueError(f"Prepared array checksum changed: {path}")
        verified.add(str(path))
    return np.load(path, mmap_mode="r", allow_pickle=False)


def same_inner_data(source_root, destination_root, source_group, destination_group, arm, verified):
    """Require identical inner rows, features, target, anchor and reference values."""
    if source_group["feature_names"] != destination_group["feature_names"]:
        return False
    cohort = arm["cohort"]
    old_split = source_group["cohorts"][cohort]["indices"]
    new_split = destination_group["cohorts"][cohort]["indices"]
    old_files = source_group["files"]
    new_files = destination_group["files"]
    columns = arm["columns"]
    # Compare only columns used by this arm; unrelated new features cannot alter its fit.
    for split in ("inner_train", "inner_validation"):
        old_indices = _load(source_root, old_split[split], verified)
        new_indices = _load(destination_root, new_split[split], verified)
        if len(old_indices) != len(new_indices):
            return False
        for name in ("starts", "y", "anchor", "reference"):
            left = _load(source_root, old_files[name], verified)[old_indices]
            right = _load(destination_root, new_files[name], verified)[new_indices]
            if not np.array_equal(left, right, equal_nan=True):
                return False
        left = _load(source_root, old_files["x"], verified)[np.ix_(old_indices, columns)]
        right = _load(destination_root, new_files["x"], verified)[np.ix_(new_indices, columns)]
        if not np.array_equal(left, right, equal_nan=True):
            return False
    return True


def main():
    """Check model/configuration identity and transfer only completed matching tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    if source == destination:
        raise ValueError("Source and destination must differ")
    old = json.loads((source / "prepared.json").read_text())
    new = json.loads((destination / "prepared.json").read_text())
    for field in ("scientific_config_sha256", "implementation_sha256", "software"):
        if old[field] != new[field]:
            raise ValueError(f"Cannot transfer across different {field}")
    old_tasks = {task["id"]: task for task in old["tasks"]}
    transferable, skipped, checked, verified = [], [], {}, set()
    # Lock both searches in stable path order so neither can change while inspected.
    with ExitStack() as stack:
        for path in sorted((source, destination), key=str):
            stack.enter_context(run_lock(path))
        for task in new["tasks"]:
            prior = old_tasks.get(task["id"])
            checkpoint = source / "trials" / f"{task['id']}.json"
            if prior is None or prior["signature"] != task["signature"] or not checkpoint.exists():
                continue
            data = json.loads(checkpoint.read_text())
            if data["signature"] != task["signature"]:
                raise ValueError(f"Invalid checkpoint signature: {checkpoint}")
            arm = task["arm"]
            key = (arm["group"], arm["cohort"], tuple(arm["columns"]))
            if key not in checked:
                checked[key] = same_inner_data(source, destination, old["groups"][arm["group"]],
                    new["groups"][arm["group"]], arm, verified)
            if not checked[key]:
                skipped.append(task["id"])
                continue
            target = destination / "trials" / checkpoint.name
            if target.exists():
                skipped.append(task["id"])
                continue
            transferable.append({"task": task["id"], "saved_fits": len(data["results"])})
            if args.apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checkpoint, target)
        report = {"source": str(source), "destination": str(destination), "applied": args.apply,
            "tasks": len(transferable), "saved_fits": sum(item["saved_fits"] for item in transferable),
            "skipped_tasks": len(skipped), "checks": "identical configurations, software, candidate signatures and inner rows"}
        if args.apply:
            (destination / "checkpoint_transfer.json").write_text(json.dumps({**report,
                "transferred": transferable, "skipped": skipped}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

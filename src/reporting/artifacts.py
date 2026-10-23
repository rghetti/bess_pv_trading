"""Group completed experiment artifacts by type and preserve their references."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any


_DIRECTORY_BY_SUFFIX = {
    **dict.fromkeys((".png", ".svg", ".pdf", ".jpg", ".jpeg"), "plots"),
    ".csv": "csv", ".json": "json",
    **dict.fromkeys((".lp", ".mps", ".ilp", ".sol"), "models"),
    **dict.fromkeys((".npz", ".npy"), "arrays"),
    **dict.fromkeys((".log", ".txt"), "logs"),
    ".parquet": "parquet",
}


def prepare_experiment_output(root: Path, destination: Path, *, overwrite: bool,
                              resume: bool = False, preserve: tuple[str, ...] = ()) -> None:
    """Prepare one run scope under its configured output root.

    A fresh rerun removes all stale artifacts, including plots disabled in the
    new configuration. Resume retains every checkpoint. Continuous callers hold
    their run lock while calling this function and preserve its existing inode.
    """
    root, destination = Path(root), Path(destination)
    resolved_root, resolved_destination = root.resolve(), destination.resolve()
    if (destination.is_symlink() or resolved_destination == resolved_root
            or not resolved_destination.is_relative_to(resolved_root)):
        raise ValueError("Experiment destination must be a directory strictly below its output root")
    if destination.exists() and not destination.is_dir():
        raise ValueError("Experiment destination must be a directory")
    destination.mkdir(parents=True, exist_ok=True)
    if resume:
        return
    entries = [path for path in destination.iterdir() if path.name not in preserve]
    if entries and not overwrite:
        raise FileExistsError(f"Experiment output exists and overwrite_existing is false: {destination}")
    for path in entries:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def _grouped_path(relative: Path) -> Path:
    """Retain strategy namespaces inside each type directory, without nesting models twice."""
    group = _DIRECTORY_BY_SUFFIX.get(relative.suffix.lower(), "other")
    if relative.parts[0] == group:
        return relative
    parts = list(relative.parts)
    if group == "models" and len(parts) > 1 and parts[-2] == "models":
        del parts[-2]
    return Path(group, *parts)


def organize_run_outputs(root: Path) -> dict[str, Any] | None:
    """Finalize or migrate one run into plots/csv/json/models directories.

    Validate collisions and parse JSON before moving anything. Rewrite exact
    output-file references, including exported model paths in nested summaries.
    Input data paths and numerical results are preserved. Repeated calls are
    safe, and a manifest inventories the files that actually exist.
    """
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Run output must be an existing directory, not a symlink")
    entries = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Run output organization does not follow symlinks")
    files = [path for path in entries if path.is_file()]
    destinations = {path: root / _grouped_path(path.relative_to(root)) for path in files}
    if len(set(destinations.values())) != len(files):
        raise FileExistsError("Different output files would have the same grouped destination")
    for source, target in destinations.items():
        if target != source and target.exists():
            raise FileExistsError(f"Grouped artifact already exists: {target}")
        for parent in target.parents:
            if parent == root:
                break
            if parent.exists() and not parent.is_dir():
                raise FileExistsError(f"Grouped artifact directory is occupied: {parent}")

    # JSON can hold root-relative, repository-relative, or absolute output paths.
    references = {}
    target_directories = {parent for path in destinations.values() for parent in path.parents}
    for source, target in destinations.items():
        references[str(source.relative_to(root))] = str(target.relative_to(root))
        references[str(source)] = str(target)
        references[str(source.resolve())] = str(target.resolve())
        if source.is_relative_to(Path.cwd()):
            references[str(source.relative_to(Path.cwd()))] = str(target.relative_to(Path.cwd()))
        elif source.resolve().is_relative_to(Path.cwd()):
            references[str(source.resolve().relative_to(Path.cwd()))] = str(target.resolve().relative_to(Path.cwd()))
        # An old strategy directory is replaced by namespaces within type folders.
        for parent in source.parents:
            if parent == root:
                break
            if parent not in target_directories:
                references[str(parent)] = str(root)
                references[str(parent.resolve())] = str(root.resolve())

    def rewrite(value):
        """Change only exact artifact references, never substrings of source provenance."""
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return references.get(value, value) if isinstance(value, str) else value

    documents = {}
    manifest = None
    for source, target in destinations.items():
        if source.suffix.lower() != ".json":
            continue
        document = rewrite(json.loads(source.read_text()))
        if source.relative_to(root) in (Path("manifest.json"), Path("json/manifest.json")):
            document["artifact_layout"] = "by_file_type"
            document["artifacts"] = sorted(str(path.relative_to(root))
                for path in destinations.values() if path != target)
            manifest = document
        documents[target] = json.dumps(document, indent=2, allow_nan=False) + "\n"

    for source, target in destinations.items():
        if source != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
    for target, document in documents.items():
        target.write_text(document)
    for directory in sorted((path for path in entries if path.is_dir()),
                            key=lambda path: len(path.parts), reverse=True):
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
    return manifest

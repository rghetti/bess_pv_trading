"""Require exact inner-fit data before reusing full-year search checkpoints."""

import numpy as np

from bess_pv_trading.forecasting.studies.price_feature_data import digest
from scripts.transfer_full_year_checkpoints import same_inner_data


def _group(root, matrix):
    """Create minimal prepared arrays with stable split indices and checksums."""
    root.mkdir()
    arrays = {"x": np.asarray(matrix, dtype=np.float32), "starts": np.array([10, 20, 30]),
              "y": np.array([1., 2., 3.]), "anchor": np.array([0., 0., 0.]),
              "reference": np.array([1., 1., 1.]), "inner_train": np.array([0, 1]),
              "inner_validation": np.array([2])}
    files = {}
    for name, array in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, array)
        files[name] = {"path": path.name, "sha256": digest(path)}
    return {"feature_names": ["used", "unused"], "files": files,
        "cohorts": {"price_only": {"indices": {name: files[name] for name in ("inner_train", "inner_validation")}}}}


def test_transfer_checks_only_selected_features_but_all_scored_rows(tmp_path):
    """Later changes to unused features are safe, but changes to used inputs are not."""
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old = _group(old_root, [[1, 10], [2, 20], [3, 30]])
    new = _group(new_root, [[1, 99], [2, 99], [3, 99]])
    arm = {"cohort": "price_only", "columns": [0]}
    assert same_inner_data(old_root, new_root, old, new, arm, set())
    altered = np.load(new_root / "x.npy")
    altered[2, 0] = 999
    np.save(new_root / "x.npy", altered)
    new["files"]["x"]["sha256"] = digest(new_root / "x.npy")
    assert not same_inner_data(old_root, new_root, old, new, arm, set())

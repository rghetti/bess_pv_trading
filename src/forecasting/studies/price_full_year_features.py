"""Add general holiday interactions to separate full-year price-search matrices."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .price_feature_data import digest
from .price_holiday_features import build_holiday_features


def annotate_cohort_coverage(groups, minimum_evaluation_rows):
    """Record training eligibility separately from minimum evaluation row counts."""
    for group in groups.values():
        for cohort in group["cohorts"].values():
            cohort["training_eligible"] = cohort["eligible"]
            counts = cohort["counts"]
            cohort["evaluation_minimum_met"] = min(counts["validation"], counts["test"]) >= minimum_evaluation_rows
    return groups


def append_calendar_features(groups, root, calendar_settings, holiday_settings):
    """Insert shared holiday columns before external blocks and update their indices."""
    root = Path(root)
    # Keep each group's row order unchanged so existing split indices remain valid.
    for group in groups.values():
        x_path = root / group["files"]["x"]["path"]
        starts_path = root / group["files"]["starts"]["path"]
        if digest(x_path) != group["files"]["x"]["sha256"]:
            raise ValueError("Prepared price feature checksum changed")
        if digest(starts_path) != group["files"]["starts"]["sha256"]:
            raise ValueError("Prepared price timestamps checksum changed")
        matrix = np.load(x_path, mmap_mode="r", allow_pickle=False)
        starts = pd.to_datetime(np.load(starts_path, allow_pickle=False), utc=True)
        extra, names = build_holiday_features(starts, calendar_settings, holiday_settings)
        if len(matrix) != len(extra):
            raise ValueError("Holiday features do not align with price rows")
        if set(names) & set(group["feature_names"]):
            raise ValueError("Holiday features duplicate existing names")
        old_width = int(group["base_width"])
        expanded = np.column_stack((matrix[:, :old_width], extra, matrix[:, old_width:])).astype(np.float32)
        temporary = x_path.with_name(x_path.name + ".holiday.tmp.npy")
        np.save(temporary, expanded, allow_pickle=False)
        os.replace(temporary, x_path)
        group["files"]["x"]["sha256"] = digest(x_path)
        group["feature_names"] = (group["feature_names"][:old_width] + names +
                                  group["feature_names"][old_width:])
        group["base_width"] = old_width + len(names)
        group["block_columns"] = {name: [index + len(names) for index in indices]
                                  for name, indices in group["block_columns"].items()}
        group["holiday_feature_names"] = names
    return groups

"""Combine independently sourced prices or fundamentals without hiding provenance."""

from __future__ import annotations

import math

import pandas as pd


def prefer_existing_intervals(primary, fallback, *, identity_columns, value_column="value"):
    """Keep primary rows on overlap and use fallback only for absent native intervals.

    Source snapshot IDs and provider fields stay attached to every selected row.
    Differences on overlapping rows are reported, never silently averaged.
    """
    keys = list(identity_columns)
    if "product_start" not in keys or "product_end" not in keys:
        raise ValueError("Native product start and end are required for fallback")
    for label, frame in (("primary", primary), ("fallback", fallback)):
        if not set(keys).issubset(frame.columns) or frame[keys].isna().any().any():
            raise ValueError(f"{label} records have missing interval identities")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} records have duplicate interval identities")
    # The same product cannot be represented by two different interval lengths.
    start_keys = [key for key in keys if key != "product_end"]
    shared = primary[keys].merge(fallback[keys], on=start_keys, suffixes=("_primary", "_fallback"))
    if not shared.empty and (shared.product_end_primary != shared.product_end_fallback).any():
        raise ValueError("Conflicting native product boundaries between sources")
    joined = fallback.merge(primary[keys + [value_column]], on=keys, how="left", indicator=True,
                            suffixes=("_fallback", "_primary"))
    overlap = joined.loc[joined._merge.eq("both")]
    if overlap.empty:
        max_difference = None
    else:
        max_difference = float((pd.to_numeric(overlap[f"{value_column}_fallback"], errors="coerce") -
                                pd.to_numeric(overlap[f"{value_column}_primary"], errors="coerce")).abs().max())
        if not math.isfinite(max_difference):
            max_difference = None
    # Keep fallback source columns intact; only its matching product keys are removed.
    old_keys = pd.MultiIndex.from_frame(primary[keys])
    new = fallback.loc[~pd.MultiIndex.from_frame(fallback[keys]).isin(old_keys)].copy()
    selected = pd.concat([primary, new], ignore_index=True).sort_values(keys, kind="stable").reset_index(drop=True)
    return selected, {"primary_rows": len(primary), "fallback_rows": len(fallback),
        "overlap_rows": len(overlap), "fallback_rows_used": len(new), "max_overlap_absolute_difference": max_difference}

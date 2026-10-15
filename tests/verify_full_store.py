"""Verify a completed audit against the four known full-store examples.

Usage: ``python tests/verify_full_store.py outputs/qa/<run_id>``. This check is
specific to the supplied Zarr and is separate from the small synthetic tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _covering_event(events: pd.DataFrame, category: str, farm_id: str, when: str) -> pd.Series:
    """Return a grouped episode containing one known UTC quarter-hour."""
    instant = pd.Timestamp(when)
    matches = events.loc[
        events["category"].eq(category)
        & events["farm_id"].eq(farm_id)
        & events["start_utc"].le(instant)
        & events["end_utc"].gt(instant)
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {category} episode for farm {farm_id} at {instant}; got {len(matches)}")
    return matches.iloc[0]


def verify(output_dir: Path) -> None:
    """Check event examples, plots, and exact historical raw-value preservation."""
    project_root = Path(__file__).resolve().parents[1]
    source = xr.open_zarr(
        project_root / "data" / "farms data" / "pv_farms_timeseries_20250101_20260831.zarr"
    )
    context = pd.read_csv(
        output_dir / "historical_context.csv",
        parse_dates=["time_utc"],
        float_precision="round_trip",
    )
    events = pd.read_csv(output_dir / "events.csv", dtype={"farm_id": str})
    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True)
    events["end_utc"] = pd.to_datetime(events["end_utc"], utc=True)
    validation = json.loads((output_dir / "validation.json").read_text(encoding="utf-8"))
    assert all(validation["grid"].values()), validation["grid"]
    assert validation["unexpected_forecast_missing_total"] == 0

    # The graph-ready CSV must contain the exact source numbers, including NaNs.
    for farm_id in source.farm_id.values:
        farm = context.loc[context.farm_id.eq(int(farm_id))]
        assert len(farm) == source.sizes["time"]
        np.testing.assert_array_equal(
            pd.to_datetime(farm.time_utc, utc=True).dt.tz_localize(None).to_numpy(),
            source.time.values,
        )
        for original, exported in (
            ("power", "power_dc_kw"),
            ("ghi", "ghi_w_m2"),
            ("fraction_diffuse", "fraction_diffuse"),
        ):
            raw = source[original].sel(farm_id=farm_id).values
            np.testing.assert_array_equal(
                farm[exported].to_numpy(dtype=raw.dtype),
                raw,
                err_msg=f"Raw {original} mismatch for farm {farm_id}",
            )

    # These examples were identified before implementation and must remain reviewable.
    examples = [
        ("night_positive_power", "345", "2025-01-01T00:15:00Z"),
        ("constant_positive_power", "910", "2025-03-30T12:00:00Z"),
        ("shared_daylight_zero_ghi", "portfolio", "2026-03-26T06:15:00Z"),
        ("power_dip", "901", "2025-03-02T10:30:00Z"),
    ]
    for category, farm_id, instant in examples:
        event = _covering_event(events, category, farm_id, instant)
        assert int(event["duration_quarters"]) >= (52 if category == "constant_positive_power" else 1)
        plot_farm = "1" if farm_id == "portfolio" else farm_id
        date = event["start_utc"].strftime("%Y-%m-%d")
        plot = output_dir / "plots" / plot_farm / date / f"{event['event_id']}.png"
        assert plot.is_file() and plot.stat().st_size > 0, plot
        print(f"{category}: {event['event_id']} ({event['start_utc']}–{event['end_utc']})")

    review = pd.read_csv(output_dir / "review_template.csv")
    assert set(review.event_id) == set(events.event_id)
    assert review["decision"].isna().all() and review["notes"].isna().all()
    print(f"Verified {len(context):,} raw farm intervals and {len(events):,} review episodes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    verify(arguments.output_dir.resolve())

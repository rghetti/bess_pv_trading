"""Inventory ensemble coordinates and measured-date overlap without running a study."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path
from bess_pv_trading.data.ensemble_inventory import inventory_ensemble_sources
from bess_pv_trading.markets import AuctionCalendar


def _ranges(days):
    """Make missing-season boundaries visible in compact inventory prose."""
    if not days:
        return []
    segments, start, previous = [], days[0], days[0]
    for day in days[1:]:
        if pd.Timestamp(day) - pd.Timestamp(previous) != pd.Timedelta(days=1):
            segments.append(start if start == previous else f"{start} through {previous}")
            start = day
        previous = day
    segments.append(start if start == previous else f"{start} through {previous}")
    return segments


def main(argv=None):
    """Read coordinates, report every requested date and save reproducible metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/ensemble_coverage_2026.yaml")
    parser.add_argument("--source", action="append", type=Path,
        help="Append an archive, including a future full-year archive when it arrives")
    parser.add_argument("--measurement-source", type=Path, help="Use a different measured-power Zarr archive")
    parser.add_argument("--overlap-policy", choices=("error", "first", "last"),
        help="Explicit precedence when a new archive overlaps an existing stored cycle")
    parser.add_argument("--output", type=Path, help="Destination for JSON, date CSV and Markdown inventory")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["_project_root"] = str(ROOT)
    settings = config["ensemble_inventory"]
    days = pd.date_range(settings["start_date"], settings["end_date"], freq="D").strftime("%Y-%m-%d").tolist()
    if not days:
        raise ValueError("Ensemble inventory requires a nonempty ordered date range")
    calendar = AuctionCalendar.from_config(config)
    cutoffs = {day: min(event.submission_at for event in calendar.schedule(day)) for day in days}
    sources = [config_path(config, path) for path in settings["sources"]]
    sources.extend(config_path(config, path) for path in args.source or [])
    measured = args.measurement_source or settings["measurement_source"]
    result = inventory_ensemble_sources(sources, delivery_days=days,
        farm_ids=settings.get("farm_ids", list(config["farms"])), timezone=calendar.timezone,
        availability_lag_hours=config["source"]["availability_lag_hours"], first_cutoffs=cutoffs,
        measurement_source=config_path(config, measured),
        required_variables=list(config["forecast_mapping"].values()),
        alignment_modes=tuple(sorted(set(config["forecast_semantics"].values()))),
        overlap_policy=args.overlap_policy or settings.get("overlap_policy", "error"))
    result["calendar_assumption"] = settings["calendar_assumption"]
    result["configured_calendar_validity"] = {"from": str(calendar.valid_from), "to": str(calendar.valid_to)}
    result["studies_started"] = 0
    result["data_downloads"] = 0
    destination = config_path(config, args.output or settings["output_directory"])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ensemble_inventory.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    table = pd.DataFrame(result["days"])
    for column in ("ineligibility_reasons", "source_candidates", "member_ids", "lead_coverage", "missing_or_misaligned_variables"):
        table[column] = table[column].map(json.dumps)
    table.to_csv(destination / "ensemble_delivery_coverage.csv", index=False)
    reasons = Counter(reason for row in result["days"] for reason in row["ineligibility_reasons"])
    lines = ["# Ensemble study data inventory", "",
        f"Requested delivery dates: {days[0]} through {days[-1]} ({len(days)} dates).", "",
        f"Coordinate-eligible delivery dates: {len(result['eligible_delivery_days'])}.", ""]
    lines.extend(f"- {segment}" for segment in _ranges(result["eligible_delivery_days"]))
    lines.extend(["", "Eligibility checks the preceding delivery day's 00 UTC cycle, configured publication lag, "
        "first auction cutoff, forecast lead coverage, requested farms and measured timestamp overlap. "
        "Numerical weather/power values and QC remain separate preflight checks.", "",
        "Source checksums cover consolidated metadata and coordinates. Weather and power payloads are not read or hashed.", "",
        "| Archive | Status | Cycles | Members |", "|---|---|---:|---:|"])
    lines.extend(f"| {Path(row['path']).name} | {row['status']} | {row.get('cycle_count', '')} | {row.get('member_count', '')} |"
                 for row in result["sources"])
    lines.extend(["", "Ineligibility reasons (a date can have several):", ""])
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(reasons.items()))
    lines.extend(["", "Calendar projection assumption: " + settings["calendar_assumption"]["description"], ""])
    lines.extend(f"- [Calendar source]({source})" for source in settings["calendar_assumption"]["sources"])
    lines.extend(["", "No historical optimization, model training, acquisition or environment changes were performed."])
    (destination / "ensemble_inventory.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_directory": str(destination), "requested_days": len(days),
        "coordinate_eligible_days": len(result["eligible_delivery_days"]),
        "eligible_ranges": _ranges(result["eligible_delivery_days"]), "missing_sources": result["missing_sources"],
        "studies_started": 0}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

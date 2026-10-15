"""Run a review-only audit of historical PV data and weather forecast vintages.

Run from this repository with::

    python -m bess_pv_trading.audit --config configs/audit.yaml

The command writes new files below ``outputs/qa`` and never modifies source data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd
import yaml

from .data import (
    build_forecast_checks,
    build_historical_context,
    build_lag_checks,
    load_inputs,
    validate_inputs,
)
from .diagnostics import detect_events, sensitivity_table
from .report import write_outputs


def _read_config(config_path: Path) -> dict:
    """Load YAML and check that the required user choices are well formed."""
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Audit config must be a YAML mapping")

    for section in ("inputs", "window", "solar", "forecast", "diagnostics", "plots", "outputs"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid config section: {section}")
    for key in ("zarr", "metadata_parquet"):
        if not config["inputs"].get(key):
            raise ValueError(f"Missing inputs.{key}")

    start = pd.Timestamp(config["window"]["start"])
    end = pd.Timestamp(config["window"]["end"])
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("window.start and window.end must include UTC timezone information")
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")
    if start > end:
        raise ValueError("window.start must precede window.end")
    if start.minute % 15 or end.minute % 15 or start.second or end.second:
        raise ValueError("Window endpoints must be aligned to 15-minute historical intervals")
    config["window"]["start"] = start
    config["window"]["end"] = end

    latitude = float(config["solar"]["latitude"])
    longitude = float(config["solar"]["longitude"])
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("solar latitude or longitude is outside its geographic range")
    lag = float(config["forecast"]["availability_lag_hours"])
    if lag < 0:
        raise ValueError("forecast.availability_lag_hours must be nonnegative")
    if int(config["forecast"]["max_lag_quarters"]) < 0:
        raise ValueError("forecast.max_lag_quarters must be nonnegative")
    return config


def _project_relative(path_text: str, project_root: Path) -> Path:
    """Resolve an input or output path from the repository, independent of cwd."""
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else project_root / path


def _run_id(config: dict) -> str:
    """Make a safe output directory name or use the configured one."""
    configured = config["outputs"].get("run_id")
    value = str(configured) if configured else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("outputs.run_id must contain only letters, digits, underscores or hyphens")
    return value


def run_audit(config_path: str | Path) -> Path:
    """Execute validation, alignment, diagnostics and report generation in order."""
    config_file = Path(config_path).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    config = _read_config(config_file)
    zarr_path = _project_relative(config["inputs"]["zarr"], project_root)
    metadata_path = _project_relative(config["inputs"]["metadata_parquet"], project_root)
    output_base = _project_relative(config["outputs"]["base_dir"], project_root)
    output_dir = output_base / _run_id(config)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    for source in (zarr_path, metadata_path):
        if not source.exists():
            raise FileNotFoundError(source)

    start = config["window"]["start"]
    end = config["window"]["end"]
    print(f"Loading Zarr and metadata: {zarr_path.name}, {metadata_path.name}", flush=True)
    dataset, metadata = load_inputs(zarr_path, metadata_path)
    validation = validate_inputs(dataset, metadata, start, end)
    print("Aligning historical measurements and available forecast vintages", flush=True)
    context = build_historical_context(
        dataset,
        metadata,
        start,
        end,
        float(config["solar"]["latitude"]),
        float(config["solar"]["longitude"]),
        float(config["forecast"]["availability_lag_hours"]),
    )
    print("Comparing forecasts and finding review episodes", flush=True)
    forecast_checks = build_forecast_checks(dataset, context, start, end)
    lag_checks = build_lag_checks(context, int(config["forecast"]["max_lag_quarters"]))
    events = detect_events(context, config)
    sensitivity = sensitivity_table(context, config)

    output_dir.mkdir(parents=True, exist_ok=False)
    # Save the exact audit choices beside the results for a reproducible rerun.
    saved_config = {**config, "window": {
        **config["window"],
        "start": start.isoformat(),
        "end": end.isoformat(),
    }}
    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(saved_config, stream, sort_keys=False)
    write_outputs(output_dir, context, events, forecast_checks, lag_checks, validation, sensitivity, config)
    print(f"Audit complete: {output_dir}", flush=True)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and return a process status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Audit YAML configuration")
    arguments = parser.parse_args(argv)
    run_audit(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

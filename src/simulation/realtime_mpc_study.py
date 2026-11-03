"""Frozen-ledger and full-continuous MPC comparisons using one event runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from ..config import validate_continuous_optimization
from ..data import config_path
from ..optimization.mpc import MPC_POLICIES, CAUSAL_TRACKER, MpcSettings
from ..reporting.artifacts import prepare_experiment_output
from ..reporting.realtime_mpc import save_mpc_outputs, save_controller_comparison
from .continuous import ContinuousReplay
from .continuous_inputs import prepare_continuous_inputs, _fingerprint
from .synthetic import write_json


def validate_mpc_study(config):
    """Reject ambiguous experiments before opening local data or creating outputs."""
    validate_continuous_optimization(config, require_dates=True)
    MpcSettings.from_config(config)
    study = config.get("realtime_mpc_comparison", {})
    allowed_fields = {"controllers", "mode", "check_only", "representative_days", "frozen_source"}
    if not isinstance(study, dict) or set(study) - allowed_fields:
        raise ValueError("Unknown or invalid realtime_mpc_comparison settings")
    controllers = study.get("controllers", [])
    allowed = MPC_POLICIES | {"track_committed_grid", CAUSAL_TRACKER}
    if (not isinstance(controllers, list) or not controllers or any(not isinstance(c, str) or c not in allowed for c in controllers)
            or len(set(controllers)) != len(controllers) or "track_committed_grid" not in controllers):
        raise ValueError("MPC comparison needs unique supported controllers including track_committed_grid")
    if study.get("mode", "frozen_commitments") not in {"frozen_commitments", "continuous"}:
        raise ValueError("Unknown MPC comparison mode")
    if config["continuous_optimization"]["planning_modes"] != ["integrated"]:
        raise ValueError("MPC comparison v1 uses the existing integrated auction strategy")
    if config["continuous_optimization"].get("perfect_pv_reference", False):
        raise ValueError("MPC comparisons must disable the separate perfect-PV reference")
    for name in ("check_only",):
        if not isinstance(study.get(name, False), bool):
            raise ValueError(f"realtime_mpc_comparison.{name} must be a YAML boolean")
    count = study.get("representative_days", 3)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("representative_days must be a positive integer")
    if config["continuous_optimization"].get("resume", False):
        raise ValueError("Comparison orchestration v1 requires resume:false; individual continuous MPC replay supports checkpoints")
    if study.get("frozen_source") is not None and (not isinstance(study["frozen_source"], str) or not study["frozen_source"]):
        raise ValueError("frozen_source must be a nonempty artifact path or null")


def _compatibility(config, prepared):
    """Pin physical, financial, forecast-source and realized inputs, excluding controllers."""
    return _fingerprint({"plant": asdict(prepared["plant"]), "initial_energy": prepared["initial_energy_mwh"],
        "actual_pv": prepared["full_realized_pv"], "settlement": prepared["imbalance_prices_eur_per_mwh"],
        "days": prepared["days"], "days_evidence": prepared["provenance"]["days"],
        "settings": {name: config.get(name) for name in ("state", "degradation", "transaction_fees", "trading",
            "market", "source", "pv", "policy", "forecast", "inputs", "gme_oracle_experiment")}})


def _freeze_source(config, prepared, replay):
    """Freeze only actual submissions; never substitute planned future orders."""
    submissions = {}
    for day in replay.days:
        submissions[day] = {}
        for event in replay.daily[day]["schedule"]:
            submissions[day][event.auction] = {"cutoff_utc": event.submission_at.isoformat(),
                "publication_utc": event.results_at.isoformat(),
                "quantities": dict(replay.submitted[day][event.auction])}
    ledger = [asdict(trade) for trade in replay.market.accepted_trades]
    return {"schema_version": "frozen-mpc-commitments-v1", "compatibility": _compatibility(config, prepared),
        "input_fingerprint": prepared["input_fingerprint"], "provenance": prepared["provenance"],
        "source_strategy": "integrated_deterministic_pv_auction_price_oracle_grid_tracking",
        "submissions": submissions, "accepted_trades": ledger, "plan_snapshots": replay.plan_snapshots}


def _ledger_signature(rows):
    """Compare contractual fields with one UTC timestamp representation."""
    normalized = deepcopy(rows)
    for row in normalized:
        for name in ("submission_time", "publication_time"):
            row[name] = pd.Timestamp(row[name]).tz_convert("UTC").isoformat()
    return _fingerprint(sorted(normalized, key=lambda row: row["trade_id"]))


def _validate_frozen(config, prepared, frozen):
    """Validate source identity, immutable checksums, quantities, clocks and publications."""
    if frozen.get("schema_version") != "frozen-mpc-commitments-v1" or frozen.get("compatibility") != _compatibility(config, prepared):
        raise ValueError("Frozen commitment source has incompatible physical, financial or data inputs")
    payload = {key: value for key, value in frozen.items() if key != "sha256"}
    if frozen.get("sha256") != _fingerprint(payload):
        raise ValueError("Frozen commitment source checksum mismatch")
    # Accepted rows cross-check every submitted quantity and the pinned actual clearing price.
    trades = {(row["auction"], row["interval_id"]): row for row in frozen["accepted_trades"]}
    if len(trades) != len(frozen["accepted_trades"]):
        raise ValueError("Duplicate frozen ledger products")
    for day, daily in prepared["days_data"].items():
        grid = daily["grid"]
        indices = {product: i for i, product in enumerate(grid.interval_ids)}
        if set(frozen["submissions"][day]) != {event.auction for event in daily["schedule"]}:
            raise ValueError("Frozen source auction participation differs")
        for event in daily["schedule"]:
            row = frozen["submissions"][day][event.auction]
            products = {grid.interval_ids[i] for i in event.eligible_intervals}
            if (pd.Timestamp(row["cutoff_utc"]) != event.submission_at
                    or pd.Timestamp(row["publication_utc"]) != event.results_at
                    or set(row["quantities"]) != products):
                raise ValueError("Frozen source calendar or product eligibility differs")
            for product, quantity in row["quantities"].items():
                trade = trades[event.auction, product]
                if (not np.isfinite(quantity) or quantity != trade["quantity_mwh"]
                        or pd.Timestamp(trade["submission_time"]) != event.submission_at
                        or pd.Timestamp(trade["publication_time"]) != event.results_at
                        or trade["price_eur_per_mwh"] != float(daily["prices"][event.auction][indices[product]])):
                    raise ValueError("Frozen source ledger does not reconcile to pinned actual prices")


def run_realtime_mpc_experiment(config, *, progress=None, prepared_inputs=None):
    """Preflight once, optionally generate one source, and replay independent physical states."""
    validate_mpc_study(config)
    study = config["realtime_mpc_comparison"]
    mode = study.get("mode", "frozen_commitments")
    prepared = (prepare_continuous_inputs(config, progress=progress)
                if prepared_inputs is None else prepared_inputs)
    if prepared["imbalance_prices_eur_per_mwh"] is None:
        raise ValueError("MPC comparison requires complete actual settlement prices")
    if study.get("check_only", False):
        return {"status": "preflight_passed", "mode": mode, "days": prepared["days"],
            "input_fingerprint": prepared["input_fingerprint"], "controllers": study["controllers"],
            "physical_intervals": len(prepared["full_grid"].interval_ids),
            "forecast_cutoffs": len(prepared["forecast_provider"].provenance), "solves_started": 0}
    settings = config["continuous_optimization"]
    if not settings.get("run_id"):
        raise ValueError("MPC comparisons require an explicit recognizable run_id")
    root = config_path(config, settings["output_root"]) / f"farm_{config['farm_id']}"
    destination = root / str(settings["run_id"]) / mode
    # Load and verify a completed common ledger before any scoped output
    # replacement. A failed controller run must not regenerate auction decisions.
    cached_frozen = None
    frozen_path = study.get("frozen_source")
    if mode == "frozen_commitments":
        if frozen_path:
            frozen_path = config_path(config, frozen_path)
        elif (destination / "frozen_commitments.json").is_file():
            frozen_path = destination / "frozen_commitments.json"
        if frozen_path:
            cached_frozen = json.loads(frozen_path.read_text())
            _validate_frozen(config, prepared, cached_frozen)
    prepare_experiment_output(root, destination, overwrite=False, resume=True)
    import fcntl
    with (destination / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        keep_source = (cached_frozen is not None and frozen_path.resolve()
                       == (destination / "frozen_commitments.json").resolve())
        prepare_experiment_output(root, destination,
            overwrite=config.get("experiment_outputs", {}).get("overwrite_existing", True),
            resume=False, preserve=("run.lock", "source") if keep_source else ("run.lock",))
        manifest = {"status": "running", "mode": mode, "input_fingerprint": prepared["input_fingerprint"],
                    "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                    "source_checksums": {str(path.relative_to(Path(__file__).resolve().parents[1])):
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))},
                    "controllers": study["controllers"], "completed_controllers": []}
        write_json(destination / "manifest.json", manifest)
        write_json(destination / "resolved_config.json", config)
        write_json(destination / "input_provenance.json", prepared["provenance"])
        try:
            frozen = None
            if mode == "frozen_commitments":
                if cached_frozen is not None:
                    frozen = cached_frozen
                    manifest["frozen_ledger_reused"] = True
                    if progress:
                        progress("MPC study: reusing the verified immutable auction ledger")
                else:
                    source_config = deepcopy(config)
                    source_config["execution"]["dispatch_policy"] = "track_committed_grid"
                    # Loading members for physical MPC must not enable ensemble auction planning.
                    source_config["continuous_optimization"]["pv_ensemble"]["enabled"] = False
                    if progress:
                        progress("MPC study: generating the shared deterministic auction ledger once")
                    replay = ContinuousReplay(source_config, prepared, "integrated", destination / "source", progress=progress)
                    replay.run()
                    frozen = _freeze_source(config, prepared, replay)
                    # Hash a JSON-normalized payload to make load/save representation identical.
                    write_json(destination / "frozen_commitments.json", frozen)
                    frozen = json.loads((destination / "frozen_commitments.json").read_text())
                    frozen["sha256"] = _fingerprint(frozen)
                _validate_frozen(config, prepared, frozen)
                write_json(destination / "frozen_commitments.json", frozen)
            summaries = {}
            ledgers = {}
            for controller in study["controllers"]:
                selected = deepcopy(config)
                selected["execution"]["dispatch_policy"] = controller
                selected_prepared = dict(prepared)
                if frozen is not None:
                    selected_prepared["frozen_commitments"] = frozen
                if progress:
                    progress(f"MPC study {mode}: {controller}")
                replay = ContinuousReplay(selected, selected_prepared, "integrated",
                    destination / "controllers" / controller, progress=progress)
                summary = replay.run()
                if "mpc_metrics" not in summary:
                    dispatch, _, _ = replay._final_dispatch_and_ledger()
                    summary.update(save_mpc_outputs(selected, replay.destination, dispatch, None, replay.projections))
                summaries[controller] = summary
                ledgers[controller] = [asdict(trade) for trade in replay.market.accepted_trades]
                if frozen is not None and ledgers[controller] != frozen["accepted_trades"]:
                    # Normalize datetime objects before comparing all cash/fee/quantity fields.
                    if _ledger_signature(ledgers[controller]) != _ledger_signature(frozen["accepted_trades"]):
                        raise ValueError("Frozen comparison changed committed quantities, prices or fees")
                manifest["completed_controllers"].append(controller)
                write_json(destination / "manifest.json", manifest)
            comparison = save_controller_comparison(config, destination, summaries, mode)
            if mode == "frozen_commitments":
                for row in comparison:
                    if abs(row["auction_revenue_difference_eur"]) > 1e-6 or abs(row["trading_fee_difference_eur"]) > 1e-6:
                        raise ValueError("Frozen auction revenue or fees differ across controllers")
            manifest.update(status="complete", comparison=comparison)
        except BaseException as error:
            manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise
        finally:
            write_json(destination / "manifest.json", manifest)
    return {"status": "complete", "mode": mode, "output_dir": str(destination), "comparison": comparison}

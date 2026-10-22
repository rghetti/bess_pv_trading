"""Immutable, farm-specific PV parameter artifacts with historical issue cutoffs."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from ..data import utc_timestamp
from .pv_physical import parameter_settings


def digest(value) -> str:
    """Hash a JSON-compatible contract independently of dictionary insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                    allow_nan=False).encode()).hexdigest()


def fixed_contract(settings: dict, night_policy: dict) -> dict:
    """Identify assumptions that a fitted artifact is not allowed to overwrite."""
    return {"dc": settings["dc"], "location": settings["location"],
            "transformer_efficiency": settings["output"]["transformer_efficiency"],
            "ac_limit_mw": settings["output"].get("ac_limit_mw"),
            "axis_tilt": settings["tracker"]["axis_tilt"],
            "axis_azimuth": settings["tracker"]["axis_azimuth"],
            "backtrack": settings["tracker"]["backtrack"],
            "wind_height_m": settings["temperature"]["wind_height_m"],
            "source_azimuth_positive": settings["source_azimuth_positive"],
            "direct_exceeds_ghi": settings["direct_exceeds_ghi"],
            "night_policy": night_policy}


@lru_cache(maxsize=64)
def _json(path: str, mtime_ns: int, size: int) -> dict:
    return json.loads(Path(path).read_text())


def read_json(path: Path) -> dict:
    """Invalidate cached registry reads when an explicitly selected file changes."""
    stat = path.stat()
    return _json(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _equivalent(left, right):
    """Permit float32 metadata roundoff without permitting changed assumptions."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equivalent(left[key],right[key]) for key in left)
    if isinstance(left,(int,float)) and isinstance(right,(int,float)):
        return math.isclose(left,right,rel_tol=1e-7,abs_tol=1e-9)
    return left == right


def resolve_parameters(settings, farm, night_policy, cutoff):
    """Select only a matching farm artifact that existed at the forecast cutoff."""
    registry_path = settings.get("parameter_registry")
    if not registry_path:
        return settings, {"status": "disabled"}
    path = Path(registry_path).resolve()
    registry = read_json(path)
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported PV parameter registry schema")
    if farm.farm_id not in registry["farm_ids"]:
        raise ValueError("PV parameter registry does not contain the requested farm")
    cutoff = utc_timestamp(cutoff)
    entries = [entry for entry in registry["entries"] if entry["farm_id"] == farm.farm_id
               and utc_timestamp(entry["issued_at_utc"]) <= cutoff]
    if not entries:
        return settings, {"status": "fixed_baseline_before_first_fit", "registry": str(path)}
    entry = max(entries, key=lambda item: utc_timestamp(item["issued_at_utc"]))
    artifact_path = path.parent / entry["artifact_path"]
    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("PV parameter artifact checksum mismatch")
    artifact = read_json(artifact_path)
    if artifact["farm_metadata"] != asdict(farm):
        raise ValueError("PV parameter artifact farm metadata mismatch")
    if not _equivalent(artifact["fixed_contract"], fixed_contract(settings, night_policy)):
        raise ValueError("PV parameter artifact fixed physical assumptions mismatch")
    issued = utc_timestamp(artifact["issued_at_utc"])
    if issued != utc_timestamp(entry["issued_at_utc"]) or issued > cutoff:
        raise ValueError("PV parameter artifact was not available at the forecast cutoff")
    if utc_timestamp(artifact["latest_target_available_at_utc"]) > issued:
        raise ValueError("PV parameter artifact contains targets unavailable at issue time")
    fit = artifact["fit"]
    result = parameter_settings(settings, fit["parameters"],
        transposition=fit["transposition_model"], bifacial=fit["bifacial_enabled"])
    result["bifacial"] = deepcopy(fit["bifacial_settings"])
    result["albedo"] = fit["albedo"]
    return result, {"status": "fitted", "artifact": str(artifact_path),
                    "artifact_sha256": entry["sha256"], "method": fit["method"],
                    "issued_at_utc": issued.isoformat(), "registry": str(path)}

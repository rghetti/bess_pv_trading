"""Read YAML safely and reject invalid numerical study settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import yaml


class ConfigurationError(ValueError):
    """An explicit configuration value violates the application contract."""


def load_config(path: str | Path, _ancestors: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a mapping without installing packages or resolving secrets."""
    location = Path(path).resolve()
    if location in _ancestors:
        raise ConfigurationError(f"Circular configuration include: {location.name}")
    with location.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a YAML mapping.")
    if "include" in config:
        included = load_config(location.parent / config.pop("include"), _ancestors | {location})
        config = merge_mapping(included, config)
    validate_config(config)
    return config


def merge_mapping(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply nested configuration overrides without discarding sibling fields."""
    result = dict(base)
    for key, value in overrides.items():
        result[key] = (merge_mapping(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict)
                       else value)
    return result


def validate_config(config: dict[str, Any]) -> None:
    """Check common settings; specialized adapters validate their own inputs."""
    credential_fields = {"token", "securitytoken", "security_token", "api_key", "access_token", "refresh_token",
                         "client_secret", "password", "authorization", "login", "username"}
    def check_credentials(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in credential_fields and child is not None:
                    raise ConfigurationError(f"Use environment credentials instead of YAML field {key}")
                check_credentials(child)
        elif isinstance(value, list):
            for child in value:
                check_credentials(child)
    check_credentials(config)
    solver = config.get("solver", {})
    for key in ("threads", "time_limit_seconds", "time_limit"):
        if key in solver and (not math.isfinite(float(solver[key])) or float(solver[key]) <= 0):
            raise ConfigurationError(f"solver.{key} must be finite and positive")
    if "mip_gap_abs" in solver:
        raise ConfigurationError("Configure the relative MIP gap only; remove solver.mip_gap_abs")
    for key in ("mip_gap",):
        if key in solver and (not math.isfinite(float(solver[key])) or float(solver[key]) < 0):
            raise ConfigurationError(f"solver.{key} must be finite and nonnegative")
    if "threads" in solver and int(solver["threads"]) != float(solver["threads"]):
        raise ConfigurationError("solver.threads must be an integer")
    synthetic = config.get("synthetic", {})
    branches = synthetic.get("branches_per_auction", 1)
    if not isinstance(branches, int) or branches < 1:
        raise ConfigurationError("synthetic.branches_per_auction must be a positive integer")
    if "degradation" in config:
        from .optimization.degradation import degradation_settings_from_config
        section=config["degradation"]
        if section.get("model","throughput") not in {"throughput","constant_throughput","xu_segmented"}:
            raise ConfigurationError("Unknown degradation model")
        # Example templates retain explicit missing values. Execution validates
        # their required inputs; loading a template must not invent coefficients.
        numeric_fields={"cost_eur_per_mwh_throughput","stress_coefficient","stress_exponent","segment_count"}
        if not any(section.get(key,0) is None for key in numeric_fields):
            try:
                degradation_settings_from_config(config)
            except (TypeError,ValueError) as error:
                raise ConfigurationError(str(error)) from None
    for key, value in config.get("battery", {}).items():
        if key.endswith("_efficiency") and value is not None and (not math.isfinite(float(value)) or not 0 < float(value) <= 1):
            raise ConfigurationError(f"battery.{key} must be in (0, 1]")

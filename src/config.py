"""Read YAML safely and reject invalid numerical study settings."""

from __future__ import annotations

from pathlib import Path
from datetime import date
from typing import Any
from copy import deepcopy
import math
import re

import yaml


class ConfigurationError(ValueError):
    """An explicit configuration value violates the application contract."""


def _read_mapping(path: str | Path, _ancestors: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Merge includes before resolving references or validating final overrides."""
    location = Path(path).resolve()
    # Enforce the input contract: Circular configuration include: `location.name`.
    if location in _ancestors:
        raise ConfigurationError(f"Circular configuration include: {location.name}")
    with location.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Enforce the input contract: Configuration must be a YAML mapping.
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a YAML mapping.")
    # When `'include' in config`, set `include` using `config.pop('include')`.
    if "include" in config:
        include = config.pop("include")
        # Enforce the input contract: Configuration include must be a relative YAML filename.
        if not isinstance(include, str):
            raise ConfigurationError("Configuration include must be a relative YAML filename")
        included = _read_mapping(location.parent / include, _ancestors | {location})
        config = merge_mapping(included, config)
    return config


def load_config(path: str | Path, *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load YAML with final CLI overrides; never evaluate code or expand secrets."""
    config = merge_mapping(_read_mapping(path), overrides or {})
    return resolve_config(config)


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve typed references and selected-farm asset defaults, then validate."""
    source = deepcopy(config)
    # When `isinstance(source.get('farms'), dict)`, set `source['farms']` using `{str(key): value for key, value in source['farms'].items()}`.
    if isinstance(source.get("farms"), dict):
        source["farms"] = {str(key): value for key, value in source["farms"].items()}
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9.]*)\}")

    def reference(path: str, ancestors: tuple[str, ...]):
        # Enforce the input contract: Circular configuration reference: `path`.
        if path in ancestors:
            raise ConfigurationError(f"Circular configuration reference: {path}")
        value = source
        # Perform reference for each `part` in `path.split('.')`.
        for part in path.split("."):
            # Enforce the input contract: Unknown configuration reference: `path`.
            if not isinstance(value, dict) or part not in value:
                raise ConfigurationError(f"Unknown configuration reference: {path}")
            value = value[part]
        return visit(value, ancestors + (path,))

    def visit(value: Any, ancestors: tuple[str, ...] = ()):
        # Return `{key: visit(child, ancestors) for key, child in value.items()}` when `isinstance(value, dict)`.
        if isinstance(value, dict):
            return {key: visit(child, ancestors) for key, child in value.items()}
        # Return `[visit(child, ancestors) for child in value]` when `isinstance(value, list)`.
        if isinstance(value, list):
            return [visit(child, ancestors) for child in value]
        # When `isinstance(value, str)`, set `exact` using `pattern.fullmatch(value)`.
        if isinstance(value, str):
            exact = pattern.fullmatch(value)
            # Return `reference(exact[1], ancestors)` when `exact`.
            if exact:
                return reference(exact[1], ancestors)
            def substitute(match):
                result = reference(match[1], ancestors)
                # Enforce the input contract: String reference `match[1]` must have a scalar value.
                if result is None or isinstance(result, (dict, list)):
                    raise ConfigurationError(f"String reference {match[1]} must have a scalar value")
                return str(result)
            return pattern.sub(substitute, value)
        return value

    result = visit(source)
    # Known selected-farm PV settings take precedence over shared assumptions.
    selected_farm = result.get("forecast", {}).get("farm_id", result.get("farm_id"))
    farm_pv = result.get("farms", {}).get(str(selected_farm), {}).get("pv", {})
    # When `farm_pv`, set `shared_pv` using `result.get('_pv_shared_settings', result.get('pv', {}))`.
    if farm_pv:
        shared_pv = result.get("_pv_shared_settings", result.get("pv", {}))
        result["_pv_shared_settings"] = deepcopy(shared_pv)
        result["pv"] = merge_mapping(shared_pv, farm_pv)
    plant = result.get("plant", {})
    # Resolve each automatic BESS dimension independently, preserving numeric
    # study overrides and self-contained numeric settings in saved run configs.
    battery = result.get("battery", {})
    battery_farm_id = plant.get("farm_id", result.get("farm_id"))
    farm_battery = result.get("farms", {}).get(str(battery_farm_id), {}).get("battery", {})
    # Resolve typed references and selected-farm asset defaults, then validate for each `field` in `('capacity_mwh', 'max_power_mw')`.
    for field in ("capacity_mwh", "max_power_mw"):
        # When `battery.get(field) == 'auto'`, set `value` using `farm_battery.get(field) if isinstance(farm_battery, dict) else None`.
        if battery.get(field) == "auto":
            value = farm_battery.get(field) if isinstance(farm_battery, dict) else None
            try:
                valid = (not isinstance(value, bool) and value is not None
                         and math.isfinite(float(value)) and float(value) > 0)
            except (TypeError, ValueError):
                valid = False
            # Enforce the input contract: Automatic battery.`field` requires positive farms.`battery_farm_id`.battery.`field`.
            if not valid:
                raise ConfigurationError(
                    f"Automatic battery.{field} requires positive farms.{battery_farm_id}.battery.{field}")
            battery[field] = float(value)
    # Resolve typed references and selected-farm asset defaults, then validate for each `field` in `('poi_import_mw', 'poi_export_mw')`.
    for field in ("poi_import_mw", "poi_export_mw"):
        # When `plant.get(field) == 'auto'`, set `farm` using `result.get('farms', {}).get(str(plant.get('farm_id')), {})`.
        if plant.get(field) == "auto":
            farm = result.get("farms", {}).get(str(plant.get("farm_id")), {})
            value = farm.get(field)
            # When `value is None and farm.get('ac_capacity_kw') is not None`, set `value` using `float(farm['ac_capacity_kw']) / 1000`.
            if value is None and farm.get("ac_capacity_kw") is not None:
                value = float(farm["ac_capacity_kw"]) / 1000
            # Enforce the input contract: Automatic plant.`field` requires selected-farm capacity or an explicit POI rating.
            if value is None or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ConfigurationError(f"Automatic plant.{field} requires selected-farm capacity or an explicit POI rating")
            plant[field] = float(value)
    pv_output = result.get("pv", {}).get("output", {})
    # When `pv_output.get('ac_limit_mw') == 'auto'`, set `farm_id` using `result.get('forecast', {}).get('farm_id', result.get('farm_id'))`.
    if pv_output.get("ac_limit_mw") == "auto":
        farm_id = result.get("forecast", {}).get("farm_id", result.get("farm_id"))
        farm = result.get("farms", {}).get(str(farm_id), {})
        value = farm.get("ac_capacity_kw")
        # Enforce the input contract: Automatic PV AC limit requires selected-farm AC capacity.
        if value is None or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ConfigurationError("Automatic PV AC limit requires selected-farm AC capacity")
        pv_output["ac_limit_mw"] = float(value) / 1000
    shared_output = result.get("_pv_shared_settings", {}).get("output", {})
    # When `shared_output.get('ac_limit_mw') == 'auto'`, set `shared_output['ac_limit_mw']` using `float(result['farms'][str(selected_farm)]['ac_capacity_kw']) / 1000`.
    if shared_output.get("ac_limit_mw") == "auto":
        shared_output["ac_limit_mw"] = float(result["farms"][str(selected_farm)]["ac_capacity_kw"]) / 1000
    validate_config(result)
    return result


def with_standard_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize saved legacy configs for resume checks without hiding changed settings."""
    standard = _read_mapping(Path(__file__).resolve().parents[1] / "configs/standard.yaml")
    legacy = deepcopy(config)
    # Normalize saved legacy configs for resume checks without hiding changed settings when `isinstance(legacy.get('farms'), dict)`.
    if isinstance(legacy.get("farms"), dict):
        # Normalize saved legacy configs for resume checks without hiding changed settings for each `farm` in `legacy['farms'].values()`.
        for farm in legacy["farms"].values():
            # Update `farm['ac_capacity_kw']` when this condition is met.
            if isinstance(farm, dict) and "dc_capacity_kw" in farm and "ac_capacity_kw" not in farm:
                farm["ac_capacity_kw"] = farm.pop("dc_capacity_kw")
    # When `legacy.get('source', {}).get('capacity_basis') == 'DC'`, set `legacy['source']['capacity_basis']` using `'AC'`.
    if legacy.get("source", {}).get("capacity_basis") == "DC":
        legacy["source"]["capacity_basis"] = "AC"
    # Normalize saved legacy configs for resume checks without hiding changed settings when `'trading' in legacy`.
    if "trading" in legacy:
        legacy["trading"].setdefault("optimize_imbalances", True)
    # Normalize saved legacy configs for resume checks without hiding changed settings when `'synthetic' in legacy`.
    if "synthetic" in legacy:
        legacy.setdefault("execution", {}).setdefault("dispatch_policy", "follow_battery_plan")
        legacy["execution"].setdefault("reoptimize_physical_controls", True)
    # Older direct-cost degradation intentionally omitted replacement economics.
    if "degradation" in legacy:
        # Apply this branch only when its condition is met.
        if not legacy["degradation"].get("replacement_economics") and not legacy["degradation"].get("throughput_lifetime"):
            from dataclasses import asdict
            from .domain import DegradationSettings
            # Normalize saved legacy configs for resume checks without hiding changed settings for each `(field, value)` in `asdict(DegradationSettings()).items()`.
            for field, value in asdict(DegradationSettings()).items():
                # Normalize saved legacy configs for resume checks without hiding changed settings when `field != 'mode'`.
                if field != "mode":
                    legacy["degradation"].setdefault(field, value)
        # Normalize saved legacy configs for resume checks without hiding changed settings for each `field` in `('replacement_economics', 'throughput_lifetime')`.
        for field in ("replacement_economics", "throughput_lifetime"):
            legacy["degradation"].setdefault(field, None)
    return resolve_config(merge_mapping(standard, legacy))


def merge_mapping(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply nested configuration overrides without discarding sibling fields."""
    result = dict(base)
    # Apply nested configuration overrides without discarding sibling fields for each `(key, value)` in `overrides.items()`.
    for key, value in overrides.items():
        # When `key == 'farms' and isinstance(value, dict)`, set `value` using `{str(farm): settings for farm, settings in value.items()}`.
        if key == "farms" and isinstance(value, dict):
            value = {str(farm): settings for farm, settings in value.items()}
            # When `isinstance(result.get(key), dict)`, set `result[key]` using `{str(farm): settings for farm, settings in result[key].items()}`.
            if isinstance(result.get(key), dict):
                result[key] = {str(farm): settings for farm, settings in result[key].items()}
        result[key] = (merge_mapping(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict)
                       else value)
    return result


def validate_continuous_optimization(config: dict[str, Any], *, require_dates: bool = False) -> None:
    """Validate continuous delivery dates and carried-state settings when selected."""
    section = config.get("continuous_optimization", {})
    # Enforce the input contract: continuous_optimization must be a YAML mapping.
    if not isinstance(section, dict):
        raise ConfigurationError("continuous_optimization must be a YAML mapping")
    start, end = (section.get(key) for key in ("start_date", "end_date"))
    # Shared templates leave the workflow inactive until both dates are supplied.
    if not require_dates and start is None and end is None:
        return
    # Enforce the input contract: continuous_optimization requires both start_date and end_date (inclusive delivery dates).
    if start is None or end is None:
        raise ConfigurationError("continuous_optimization requires both start_date and end_date (inclusive delivery dates)")
    try:
        first, last = (date.fromisoformat(str(value)) for value in (start, end))
    except (TypeError, ValueError):
        raise ConfigurationError("continuous_optimization dates must be ISO delivery dates (YYYY-MM-DD)") from None
    # Enforce the input contract: continuous_optimization.end_date must not precede start_date.
    if first > last:
        raise ConfigurationError("continuous_optimization.end_date must not precede start_date")
    modes = section.get("planning_modes")
    allowed_modes = {"integrated", "current_auction_only"}
    # Enforce the input contract: continuous_optimization.planning_modes must be a nonempty unique list of integrated/current_auction_only.
    if (not isinstance(modes, list) or not modes
            or any(not isinstance(mode, str) or mode not in allowed_modes for mode in modes)
            or len(modes) != len(set(modes))):
        raise ConfigurationError("continuous_optimization.planning_modes must be a nonempty unique list of integrated/current_auction_only")
    # Process each `key` in this collection.
    for key in ("perfect_pv_reference", "perfect_pv_reoptimize_at_each_auction", "resume"):
        # Enforce the input contract: continuous_optimization.`key` must be a YAML boolean.
        if not isinstance(section.get(key), bool):
            raise ConfigurationError(f"continuous_optimization.{key} must be a YAML boolean")
    state, battery = config.get("state", {}), config.get("battery", {})
    lower, upper = (float(battery.get(key, default)) for key, default in
                    (("soc_min_percent", 0.0), ("soc_max_percent", 100.0)))
    # Process each `key` in this collection.
    for key in ("initial_soc_percent", "terminal_soc_percent", "terminal_soc_min_percent"):
        value = state.get(key)
        # Skip the current item when `key in {'terminal_soc_percent', 'terminal_soc_min_percent'} and value is None`.
        if key in {"terminal_soc_percent", "terminal_soc_min_percent"} and value is None:
            continue
        try:
            valid = (not isinstance(value, bool) and math.isfinite(float(value))
                     and 0 <= float(value) <= 100 and lower <= float(value) <= upper)
        except (TypeError, ValueError):
            valid = False
        # Enforce the input contract: Continuous optimization state.`key` must lie within battery SOC percentage bounds.
        if not valid:
            raise ConfigurationError(f"Continuous optimization state.{key} must lie within battery SOC percentage bounds")


def validate_config(config: dict[str, Any]) -> None:
    """Check common settings; specialized adapters validate their own inputs."""
    from .transaction_fees import transaction_fees_from_config
    transaction_fees_from_config(config)
    credential_fields = {"token", "securitytoken", "security_token", "api_key", "access_token", "refresh_token",
                         "client_secret", "password", "authorization", "login", "username"}
    def check_credentials(value: Any) -> None:
        # Perform check credentials when `isinstance(value, dict)`.
        if isinstance(value, dict):
            # Perform check credentials for each `(key, child)` in `value.items()`.
            for key, child in value.items():
                # Enforce the input contract: Use environment credentials instead of YAML field `key`.
                if str(key).lower() in credential_fields and child is not None:
                    raise ConfigurationError(f"Use environment credentials instead of YAML field {key}")
                check_credentials(child)
        # Perform check credentials when `isinstance(value, list)`.
        elif isinstance(value, list):
            # Perform check credentials for each `child` in `value`.
            for child in value:
                check_credentials(child)
    check_credentials(config)
    # Enforce the input contract: experiment_outputs.overwrite_existing must be a YAML boolean.
    if not isinstance(config.get("experiment_outputs", {}).get("overwrite_existing", True), bool):
        raise ConfigurationError("experiment_outputs.overwrite_existing must be a YAML boolean")
    forecast_policy = config.get("forecast", {}).get("reoptimization_pv_policy")
    # Enforce the input contract: Unknown forecast.reoptimization_pv_policy.
    if forecast_policy is not None and forecast_policy not in {
            "freeze_initial_forecast_for_unobserved_delivery_intervals",
            "latest_available_forecast_at_each_reoptimization"}:
        raise ConfigurationError("Unknown forecast.reoptimization_pv_policy")
    trading = config.get("trading", {})
    execution = config.get("execution", {})
    # Enforce the input contract: execution.infeasible_auction_policy must be error or previous_plan.
    if execution.get("infeasible_auction_policy", "error") not in {"error", "previous_plan"}:
        raise ConfigurationError("execution.infeasible_auction_policy must be error or previous_plan")
    # Enforce the input contract: execution.save_infeasibility_diagnostics must be a YAML boolean.
    if not isinstance(execution.get("save_infeasibility_diagnostics", False), bool):
        raise ConfigurationError("execution.save_infeasibility_diagnostics must be a YAML boolean")
    from .participation import auction_subset, participation_profiles
    # Enforce the input contract: gme_oracle_experiment.auction_planning_mode must be integrated or current_auction_only.
    if config.get("gme_oracle_experiment", {}).get("auction_planning_mode", "integrated") not in {"integrated", "current_auction_only"}:
        raise ConfigurationError("gme_oracle_experiment.auction_planning_mode must be integrated or current_auction_only")
    participation = config.get("gme_oracle_experiment", {}).get("participating_auctions")
    # Check common settings; specialized adapters validate their own inputs when `participation is not None`.
    if participation is not None:
        auction_subset(participation, "gme_oracle_experiment.participating_auctions")
    # When `'participation_study' in config`, set `study` using `config['participation_study']`.
    if "participation_study" in config:
        study = config["participation_study"]
        # Enforce the input contract: participation_study must be a YAML mapping.
        if not isinstance(study, dict):
            raise ConfigurationError("participation_study must be a YAML mapping")
        participation_profiles(study)
        # Enforce the input contract: Participation study plot/model switches must be YAML booleans.
        if any(not isinstance(study.get(key, default), bool) for key, default in
                (("plots", True), ("per_profile_plots", False), ("save_model_files", False),
                 ("perfect_foresight_reference", True), ("sequential_all_auctions_baseline", False), ("resume", False))):
            raise ConfigurationError("Participation study plot/model switches must be YAML booleans")
    validate_continuous_optimization(config)
    baseline = config.get("reoptimization_baseline", {})
    # Enforce the input contract: reoptimization_baseline accepts enabled, freeze_after_auctions and plots.
    if not isinstance(baseline, dict) or set(baseline) - {"enabled", "freeze_after_auctions", "plots"}:
        raise ConfigurationError("reoptimization_baseline accepts enabled, freeze_after_auctions and plots")
    # Enforce the input contract: reoptimization_baseline.enabled and plots must be YAML booleans.
    if any(not isinstance(baseline.get(key, default), bool)
           for key, default in (("enabled", False), ("plots", True))):
        raise ConfigurationError("reoptimization_baseline.enabled and plots must be YAML booleans")
    freezes = baseline.get("freeze_after_auctions")
    # Enforce the input contract: reoptimization_baseline.freeze_after_auctions must be null or a unique list of auction names.
    if freezes is not None and (not isinstance(freezes, list)
            or any(not isinstance(auction, str) or auction not in {"DA", "IDA1", "IDA2", "IDA3"} for auction in freezes)
            or len(freezes) != len(set(freezes))):
        raise ConfigurationError("reoptimization_baseline.freeze_after_auctions must be null or a unique list of auction names")
    # Enforce the input contract: Remove commercial trading caps; physical grid power uses plant.poi_import_mw and plant.poi_export_mw.
    if {"trade_limit_mwh", "position_limit_mwh", "trade_limit_mwh_per_native_product",
            "position_limit_mwh_per_quarter_hour"} & set(trading):
        raise ConfigurationError("Remove commercial trading caps; physical grid power uses plant.poi_import_mw and plant.poi_export_mw")
    # Enforce the input contract: trading.optimize_imbalances must be a YAML boolean.
    if "optimize_imbalances" in trading and not isinstance(trading["optimize_imbalances"], bool):
        raise ConfigurationError("trading.optimize_imbalances must be a YAML boolean")
    # Enforce the input contract: execution.dispatch_policy must be follow_battery_plan or track_committed_grid.
    if config.get("execution", {}).get("dispatch_policy", "follow_battery_plan") not in {
            "follow_battery_plan", "track_committed_grid"}:
        raise ConfigurationError("execution.dispatch_policy must be follow_battery_plan or track_committed_grid")
    solver = config.get("solver", {})
    # Check common settings; specialized adapters validate their own inputs for each `key` in `('threads', 'time_limit_seconds', 'time_limit')`.
    for key in ("threads", "time_limit_seconds", "time_limit"):
        # Enforce the input contract: solver.`key` must be finite and positive.
        if key in solver and (not math.isfinite(float(solver[key])) or float(solver[key]) <= 0):
            raise ConfigurationError(f"solver.{key} must be finite and positive")
    # Enforce the input contract: Configure the relative MIP gap only; remove solver.mip_gap_abs.
    if "mip_gap_abs" in solver:
        raise ConfigurationError("Configure the relative MIP gap only; remove solver.mip_gap_abs")
    # Check common settings; specialized adapters validate their own inputs for each `key` in `('mip_gap',)`.
    for key in ("mip_gap",):
        # Enforce the input contract: solver.`key` must be finite and nonnegative.
        if key in solver and (not math.isfinite(float(solver[key])) or float(solver[key]) < 0):
            raise ConfigurationError(f"solver.{key} must be finite and nonnegative")
    # Enforce the input contract: solver.threads must be an integer.
    if "threads" in solver and int(solver["threads"]) != float(solver["threads"]):
        raise ConfigurationError("solver.threads must be an integer")
    # Check common settings; specialized adapters validate their own inputs for each `(name, maximum)` in `(('feasibility_tolerance', 0.01), ('integrality_tolerance', 0.1))`.
    for name, maximum in (("feasibility_tolerance", 1e-2), ("integrality_tolerance", 1e-1)):
        value = solver.get(name)
        # Enforce the input contract: solver.`name` is outside the supported Gurobi range.
        if value is not None and (isinstance(value, bool) or not math.isfinite(float(value)) or not 1e-9 <= float(value) <= maximum):
            raise ConfigurationError(f"solver.{name} is outside the supported Gurobi range")
    synthetic = config.get("synthetic", {})
    branches = synthetic.get("branches_per_auction", 1)
    # Enforce the input contract: synthetic.branches_per_auction must be a positive integer.
    if not isinstance(branches, int) or branches < 1:
        raise ConfigurationError("synthetic.branches_per_auction must be a positive integer")
    # Check common settings; specialized adapters validate their own inputs when `'degradation' in config`.
    if "degradation" in config:
        from .optimization.degradation import degradation_settings_from_config
        section=config["degradation"]
        # Enforce the input contract: Unknown degradation model.
        if section.get("model","throughput") not in {"throughput","constant_throughput","xu_segmented"}:
            raise ConfigurationError("Unknown degradation model")
        # Example templates retain explicit missing values. Execution validates
        # their required inputs; loading a template must not invent coefficients.
        numeric_fields={"cost_eur_per_mwh_throughput","stress_coefficient","stress_exponent","segment_count"}
        # Check common settings; specialized adapters validate their own inputs when `section.get('throughput_lifetime') is not None`.
        if section.get("throughput_lifetime") is not None:
            numeric_fields.remove("cost_eur_per_mwh_throughput")
        # Check common settings; specialized adapters validate their own inputs when `not any((section.get(key, 0) is None for key in numeric_fields))`.
        if not any(section.get(key,0) is None for key in numeric_fields):
            try:
                degradation_settings_from_config(config)
            except (TypeError,ValueError) as error:
                raise ConfigurationError(str(error)) from None
    # Check common settings; specialized adapters validate their own inputs for each `(key, value)` in `config.get('battery', {}).items()`.
    for key, value in config.get("battery", {}).items():
        # Enforce the input contract: battery.`key` must be in (0, 1].
        if key.endswith("_efficiency") and value is not None and (not math.isfinite(float(value)) or not 0 < float(value) <= 1):
            raise ConfigurationError(f"battery.{key} must be in (0, 1]")
    battery = config.get("battery", {})
    # Enforce the input contract: Use battery.soc_min_percent and soc_max_percent instead of absolute MWh bounds.
    if {"energy_min_mwh", "energy_max_mwh"} & set(battery):
        raise ConfigurationError("Use battery.soc_min_percent and soc_max_percent instead of absolute MWh bounds")
    # Enforce the input contract: Use one symmetric battery.max_power_mw; remove separate charge/discharge and inverter limits.
    if {"max_charge_mw", "max_discharge_mw", "inverter_charge_mw", "inverter_discharge_mw"} & set(battery):
        raise ConfigurationError("Use one symmetric battery.max_power_mw; remove separate charge/discharge and inverter limits")
    # Apply this branch only when its condition is met.
    if all(battery.get(key) is not None for key in ("capacity_mwh", "max_power_mw")):
        from .domain import battery_parameters_from_config
        try:
            battery_parameters_from_config(battery)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(str(error)) from None

"""Load an initial point-price integrated plan from explicit prepared CSV inputs.

This boundary performs validation only. It neither fetches data nor estimates
missing PV, prices, equipment or historical settlement rules.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..transaction_fees import transaction_fees_from_config
from ..domain import (BatteryParameters, DegradationSettings, MarketState,
                      OptimizationInput, PhysicalState, PlantParameters)
from ..markets import AuctionCalendar
from ..scenarios import synthetic_price_tree


class PreparedInputError(ValueError):
    """A supplied optimization feed is missing, incomplete or unavailable."""


def _utc(value: Any, label: str) -> pd.Timestamp:
    """Require explicit historical clock offsets at this integration boundary."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise PreparedInputError(f"{label} must be a nonmissing timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _timestamps(frame: pd.DataFrame, column: str, cutoff: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    if column not in frame:
        raise PreparedInputError(f"Prepared CSV missing timestamp column {column}")
    values = pd.DatetimeIndex([_utc(value, column) for value in frame[column]])
    if cutoff is not None and (values > cutoff).any():
        raise PreparedInputError(f"Future/unavailable {column} exceeds optimization.decision_cutoff")
    return values


def _read(path: Any, label: str) -> pd.DataFrame:
    if not path or not Path(path).is_file():
        raise PreparedInputError(f"Missing optimization.{label}: supply an existing prepared CSV; no data will be invented")
    frame = pd.read_csv(path)
    if frame.empty:
        raise PreparedInputError(f"Prepared {label} is empty")
    return frame


def _require(settings: Mapping[str, Any], names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if name not in settings or settings[name] is None]
    if missing:
        raise PreparedInputError(f"Missing explicitly configured {label}: {', '.join(missing)}")


def _planning_settlement_price(settings: Mapping[str, Any], cutoff, enabled: bool) -> float:
    """Require a causal settlement assumption only when the optimizer uses it."""
    if not enabled:
        return 0.0  # Unused scenario-array placeholder, not a real settlement price.
    _require(settings, ("settlement_point_price_eur_per_mwh", "settlement_point_price_available_at",
                        "settlement_point_price_status"), "settlement assumption")
    if settings["settlement_point_price_status"] != "study_assumption":
        raise PreparedInputError("Settlement point price must explicitly have status study_assumption")
    if _utc(settings["settlement_point_price_available_at"], "settlement availability") > cutoff:
        raise PreparedInputError("Settlement assumption is unavailable at the decision cutoff")
    price = float(settings["settlement_point_price_eur_per_mwh"])
    if not np.isfinite(price):
        raise PreparedInputError("Settlement point-price assumption must be finite")
    return price


def _terminal_target(settings: Mapping[str, Any]) -> float | None:
    """Require an explicit target, with YAML null selecting unconstrained SOC."""
    if "terminal_target_mwh" not in settings:
        raise PreparedInputError("Missing explicitly configured state.terminal_target_mwh (use null to disable)")
    target = settings["terminal_target_mwh"]
    return None if target is None else float(target)


def terminal_minimum_energy_mwh(settings: Mapping[str, Any], battery: BatteryParameters) -> float | None:
    """Convert an optional terminal SOC floor percentage to stored energy."""
    percentage = settings.get("terminal_soc_min_percent")
    if percentage is None:
        return None
    try:
        value = float(percentage)
    except (TypeError, ValueError):
        raise PreparedInputError("state.terminal_soc_min_percent must be a finite percentage or null") from None
    if not np.isfinite(value) or not 0 <= value <= 100:
        raise PreparedInputError("state.terminal_soc_min_percent must be a finite percentage between 0 and 100")
    energy = battery.capacity_mwh * value / 100.0
    if not battery.energy_min_mwh <= energy <= battery.maximum_energy:
        raise PreparedInputError("state.terminal_soc_min_percent lies outside battery SOC bounds")
    return energy


def _boolean(series: pd.Series, label: str) -> np.ndarray:
    values = series.astype(str).str.lower().str.strip()
    if not values.isin(["true", "false", "1", "0"]).all():
        raise PreparedInputError(f"{label} must contain explicit true/false values without missing entries")
    return values.isin(["true", "1"]).to_numpy()


def load_prepared_inputs(config: Mapping[str, Any]) -> OptimizationInput:
    """Validate one farm/member, full-day PV and point-price forecasts, then plan.

    Only initial pre-DA integrated planning is supported. Every future auction
    uses a supplied point forecast whose source, fit and availability clocks are
    no later than the cutoff. Future observed clearing prices are not accepted.
    The zero-innovation tree is deterministic and still schedules actual auction
    submission/results and pre-interval control events separately.
    """
    settings = config.get("optimization", {})
    _require(settings, ("pv_forecast_path", "price_forecast_path", "decision_cutoff", "delivery_day"), "optimization")
    cutoff = _utc(settings["decision_cutoff"], "optimization.decision_cutoff")
    calendar = AuctionCalendar.from_config(config)
    market_grid = calendar.delivery_grid(str(settings["delivery_day"]))
    grid = market_grid.to_domain()
    schedule = calendar.schedule(str(settings["delivery_day"]))
    if cutoff > schedule[0].submission_at:
        raise PreparedInputError("Prepared planning supports only an initial cutoff no later than the DA submission; historical replay is separate")
    plant_config, battery_config = config.get("plant", {}), config.get("battery", {})
    _require(plant_config, ("farm_id", "poi_import_mw", "poi_export_mw"), "plant equipment")
    _require(battery_config, ("capacity_mwh", "max_power_mw"), "battery equipment")
    try:
        from ..domain import battery_parameters_from_config
        battery = battery_parameters_from_config(battery_config)
        plant = PlantParameters(int(plant_config["farm_id"]), float(plant_config["poi_import_mw"]),
                                float(plant_config["poi_export_mw"]), battery)
    except (TypeError, ValueError) as error:
        raise PreparedInputError(f"Invalid explicit plant/battery configuration: {error}") from error

    # Keep native exported AC forecasts; never repair structural forecast gaps.
    pv = _read(settings["pv_forecast_path"], "pv_forecast_path")
    required_pv = {"farm_id", "member_id", "member_origin", "member_probability", "forecast_cycle_time",
                   "forecast_available_at", "interval_start_utc", "interval_end_utc", "pv_ac_mw",
                   "input_eligibility", "review_status"}
    if missing := required_pv - set(pv):
        raise PreparedInputError(f"PV forecast export missing contract columns: {sorted(missing)}")
    if pv.farm_id.nunique(dropna=False) != 1 or not (pd.to_numeric(pv.farm_id, errors="raise") == plant.farm_id).all():
        raise PreparedInputError("Prepared PV must contain exactly the configured farm")
    if pv.member_id.isna().any() or pv.member_id.nunique() != 1 or pv.member_origin.isna().any() or pv.member_origin.nunique() != 1:
        raise PreparedInputError("Prepared planning requires one explicit deterministic PV member and origin")
    origins = pv.member_origin.astype(str).str.strip()
    if origins.eq("").any() or origins.str.lower().str.contains("realized|oracle|perfect_foresight|observed_power", regex=True).any():
        raise PreparedInputError("PV member origin must describe a usable forecast, not future realized/oracle power")
    if not np.allclose(pd.to_numeric(pv.member_probability, errors="raise"), 1.0, atol=1e-12, rtol=0):
        raise PreparedInputError("The single PV member must carry probability one")
    if not _boolean(pv.input_eligibility, "input_eligibility").all():
        raise PreparedInputError("Invalid PV input_eligibility: gaps/model-invalid intervals cannot be optimized")
    for name in ("review_excluded", "derived_review_excluded"):
        if name in pv and _boolean(pv[name], name).any():
            raise PreparedInputError("PV export contains review exclusions; no automatic override is permitted")
    allowed_reviews = {"no_configured_observation_review_exclusion", "no_review_exclusion", "eligible"}
    if pv.review_status.isna().any() or not pv.review_status.isin(allowed_reviews).all():
        raise PreparedInputError("PV review_status must explicitly declare no review exclusions")
    starts = _timestamps(pv, "interval_start_utc")
    ends = _timestamps(pv, "interval_end_utc")
    source = _timestamps(pv, "forecast_cycle_time", cutoff)
    availability = _timestamps(pv, "forecast_available_at", cutoff)
    if source.nunique() != 1 or availability.nunique() != 1 or (source > availability).any():
        raise PreparedInputError("PV export must retain one causal source vintage and its availability")
    if starts.duplicated().any() or len(starts) != len(market_grid) or set(starts) != set(market_grid.starts):
        raise PreparedInputError("PV must cover the complete local quarter-hour delivery grid exactly once")
    pv = pv.assign(_start=starts, _end=ends).set_index("_start").loc[market_grid.starts]
    if not pd.DatetimeIndex(pv._end).equals(market_grid.ends):
        raise PreparedInputError("PV interval ends do not match the declared market products")
    pv_values = pd.to_numeric(pv.pv_ac_mw, errors="raise").to_numpy(float)
    if not np.isfinite(pv_values).all() or (pv_values < 0).any():
        raise PreparedInputError("PV AC forecast must be finite and nonnegative; missing values are not filled")
    pv_fit = settings.get("pv_model_fit_cutoff")
    if "model_fit_cutoff" in pv:
        _timestamps(pv, "model_fit_cutoff", cutoff)
    elif pv_fit is not None:
        if _utc(pv_fit, "pv_model_fit_cutoff") > cutoff:
            raise PreparedInputError("PV model fit uses future data")
    elif settings.get("pv_model_fit_status") != "unfitted_explicit_assumption":
        raise PreparedInputError("Supply pv_model_fit_cutoff or explicitly declare pv_model_fit_status=unfitted_explicit_assumption")

    # Forecast labels are numerical expectations, never ex-post clearing outcomes.
    prices = _read(settings["price_forecast_path"], "price_forecast_path")
    required_price = {"auction", "product_start", "price_eur_mwh", "available_at", "origin"}
    if missing := required_price - set(prices):
        raise PreparedInputError(f"Price forecast CSV missing columns: {sorted(missing)}")
    if prices.origin.isna().any() or prices.origin.astype(str).str.strip().eq("").any():
        raise PreparedInputError("Price forecasts require explicit origin provenance")
    forbidden = prices.origin.astype(str).str.lower().str.contains("realized|oracle|perfect_foresight|observed_clearing", regex=True)
    if forbidden.any():
        raise PreparedInputError("Future realized/oracle clearing prices cannot be a prepared executable forecast")
    prices["auction"] = prices.auction.replace({"MGP": "DA", "MI-A1": "IDA1", "MI-A2": "IDA2", "MI-A3": "IDA3"})
    if not prices.auction.isin([event.auction for event in schedule]).all():
        raise PreparedInputError("Price CSV has unknown/nonauction products")
    prices["_start"] = _timestamps(prices, "product_start")
    prices["available_at"] = _timestamps(prices, "available_at", cutoff)
    for field, config_field in (("source_time", "price_forecast_source_time"), ("fit_cutoff", "price_forecast_fit_cutoff")):
        if field not in prices:
            _require(settings, (config_field,), "optimization price provenance")
            prices[field] = settings[config_field]
        prices[field] = _timestamps(prices, field, cutoff)
    if (pd.DatetimeIndex(prices.source_time) > pd.DatetimeIndex(prices.available_at)).any():
        raise PreparedInputError("Price source time exceeds its availability")
    if (pd.DatetimeIndex(prices.fit_cutoff) > pd.DatetimeIndex(prices.available_at)).any():
        raise PreparedInputError("Price model fit cutoff exceeds its forecast availability")
    if prices.duplicated(["auction", "_start"]).any():
        raise PreparedInputError("Duplicate price-forecast auction/products")
    prices["price_eur_mwh"] = pd.to_numeric(prices.price_eur_mwh, errors="raise")
    if not np.isfinite(prices.price_eur_mwh).all():
        raise PreparedInputError("Missing price forecasts cannot be filled")
    if "zone" in prices and not prices.zone.eq("NORD").all():
        raise PreparedInputError("Prepared Italy NORD prices must identify the NORD bidding zone")
    paths: dict[str, np.ndarray] = {}
    masks: dict[str, tuple[bool, ...]] = {}
    for event in schedule:
        expected = market_grid.starts[list(event.eligible_intervals)]
        selected = prices.loc[prices.auction == event.auction].set_index("_start")
        if len(selected) != len(expected) or set(selected.index) != set(expected):
            raise PreparedInputError(f"{event.auction} point forecasts must cover its eligible quarter-hour products exactly; IDA3 is noon onward")
        if "product_end" in selected:
            expected_ends = market_grid.ends[list(event.eligible_intervals)]
            if not _timestamps(selected.loc[expected], "product_end").equals(expected_ends):
                raise PreparedInputError(f"{event.auction} price product ends mismatch market grid")
        # Ineligible IDA3 coefficients are never traded; retain predecessor values
        # there instead of pretending a morning IDA3 clearing price was observed.
        previous = paths.get("IDA2", paths.get("IDA1", paths.get("DA", np.zeros(len(market_grid)))))
        paths[event.auction] = previous.copy()
        paths[event.auction][list(event.eligible_intervals)] = selected.loc[expected, "price_eur_mwh"].to_numpy(float)
        masks[event.auction] = tuple(t in event.eligible_intervals for t in range(len(market_grid)))

    settlement = _planning_settlement_price(settings, cutoff,
        config.get("trading", {}).get("optimize_imbalances", True))
    scenarios, tree = synthetic_price_tree(grid, pv_values, schedule, base_prices=paths["DA"],
        stage_spreads={later: paths[later] - paths[earlier] for earlier, later in zip(("DA", "IDA1", "IDA2"), ("IDA1", "IDA2", "IDA3"))},
        residual_scale=0, branch_values=(0,), branch_probabilities=(1,), imbalance_price=settlement, decision_cutoff=cutoff)
    provenance = {"pv_forecast_path": str(settings["pv_forecast_path"]), "price_forecast_path": str(settings["price_forecast_path"]),
        "decision_cutoff": cutoff.isoformat(), "pv_source_cycle": source[0].isoformat(), "pv_available_at": availability[0].isoformat(),
        "pv_member_origin": str(pv.member_origin.iloc[0]), "price_origins": sorted(prices.origin.unique().tolist()),
        "price_policy": "supplied point forecasts frozen at initial cutoff; no residual uncertainty",
        "settlement_policy": ("explicit signed single-price point estimate study assumption; not verified historical Italian settlement"
            if config.get("trading", {}).get("optimize_imbalances", True)
            else "excluded from optimization; actual settlement is ex post"),
        "weather_policy": "one frozen deterministic member", "pv_members": 1,
        "planning_scope": "initial pre-DA integrated planning only; future bids are planned, not accepted",
        "source_basis": "exported AC MW; no extra converter loss applied to PV",
        "ida3_ineligible_coefficients": "copied predecessor placeholders; no morning IDA3 products are tradable"}
    scenarios = replace(scenarios, origin="supplied point-price / deterministic-PV", provenance=provenance)
    state_config, trading, degradation = config.get("state", {}), config.get("trading", {}), config.get("degradation", {})
    _require(state_config, ("initial_energy_mwh", "origin"), "delivery-boundary state")
    if state_config["origin"] != "known_delivery_start_assumption":
        raise PreparedInputError("Initial pre-delivery planning requires an explicit known_delivery_start_assumption")
    if state_config.get("timestamp") and _utc(state_config["timestamp"], "state.timestamp") != pd.Timestamp(grid.starts[0]):
        raise PreparedInputError("state.timestamp must equal delivery-day start; a preceding-day state cannot be reused")
    _require(trading, ("imbalance_penalty_eur_per_mwh", "grid_import_fee_eur_per_mwh"), "trading cost assumptions")
    from .degradation import degradation_settings_from_config
    _require(degradation, ("assumptions",), "degradation assumptions")
    fees = transaction_fees_from_config(config)
    terminal_minimum = terminal_minimum_energy_mwh(state_config, battery)
    return OptimizationInput(grid, plant,
        PhysicalState(grid.starts[0], float(state_config["initial_energy_mwh"]), origin=state_config["origin"],
                      cycle_segment_energy_mwh=state_config.get("cycle_segment_energy_mwh")),
        MarketState(transaction_fee_state=fees.initial_state), scenarios, tree, masks,
        float(trading["imbalance_penalty_eur_per_mwh"]), _terminal_target(state_config),
        degradation_settings_from_config(config),
        float(trading["grid_import_fee_eur_per_mwh"]), imbalance_limit_mwh=trading.get("imbalance_limit_mwh"),
        transaction_fees=fees, auction_submission_times={event.auction: event.submission_at for event in schedule},
        terminal_minimum_mwh=terminal_minimum,
        planned_delivery_balance_resolution_minutes=trading.get("planned_delivery_balance_resolution_minutes"),
        optimize_imbalances=trading.get("optimize_imbalances", True))

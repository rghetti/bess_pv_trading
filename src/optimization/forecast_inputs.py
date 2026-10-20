"""Canonical prepared forecasts, causal price trees and explicit physical state.

This boundary reads prepared CSV/Parquet and saved JSON artifacts only. It does
not acquire API data, fit models, repair missing products or certify vintages.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..domain import (AcceptedTrade, BatteryParameters, DeliveryGrid, InformationNode,
                      InformationTree, MarketState, OptimizationInput, PhysicalState,
                      PlantParameters, ScenarioBundle, battery_parameters_from_config)
from ..forecasting.gaussian_prices import GaussianPriceCalibration, gaussian_price_tree
from ..forecasting.intraday_price_features import ConditionalSpreadPriceForecast
from ..forecasting.linear_price_study import LinearPriceForecast
from ..forecasting.price_oracle import _actual_vector
from ..markets import AuctionCalendar
from ..scenarios import conditional_price_tree
from .degradation import degradation_settings_from_config
from .prepared import PreparedInputError, _boolean, _planning_settlement_price, _require, _terminal_target, _timestamps, _utc


AUCTIONS = ("DA", "IDA1", "IDA2", "IDA3")


def _table(path: Any, label: str) -> pd.DataFrame:
    """Read an explicitly prepared table without accessing provider caches."""
    if not path or not Path(path).is_file():
        raise PreparedInputError(f"Missing prepared {label}: supply an existing CSV or Parquet")
    source = Path(path)
    if source.suffix.lower() not in {".csv", ".parquet"}:
        raise PreparedInputError(f"Prepared {label} must be CSV or Parquet")
    result = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    if result.empty:
        raise PreparedInputError(f"Prepared {label} is empty")
    return result


def _file_provenance(path: Any) -> dict[str, str]:
    """Keep exact file identities at the acquisition-independent boundary."""
    source = Path(path)
    return {"path": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep this stochastic adapter separate from the earlier point CSV mode."""
    result = dict(config.get("optimization", {}).get("forecast_inputs", {}))
    _require(result, ("delivery_day", "decision_cutoff", "price_distribution"), "optimization.forecast_inputs")
    if result["price_distribution"] not in {"gaussian", "point"}:
        raise PreparedInputError("price_distribution must explicitly be gaussian or point")
    for name, default, minimum in (("branches", 2, 1), ("seed", 41, 0), ("maximum_joint_scenarios", 2048, 1)):
        value = result.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise PreparedInputError(f"{name} must be an integer at least {minimum}")
        result[name] = value
    if not isinstance(result.get("allow_retrospective", False), bool):
        raise PreparedInputError("allow_retrospective must be an explicit boolean")
    return result


def _grid_schedule(config: Mapping[str, Any], settings: Mapping[str, Any], known: pd.DataFrame | None):
    """Slice the verified native calendar while retaining original product IDs."""
    calendar = AuctionCalendar.from_config(config)
    full = calendar.delivery_grid(str(settings["delivery_day"]))
    cutoff = _utc(settings["decision_cutoff"], "decision_cutoff")
    boundary = _utc(settings.get("residual_start") or full.starts[0], "residual_start")
    if boundary < cutoff or boundary not in full.starts:
        raise PreparedInputError("residual_start must be an exact delivery boundary no earlier than the decision cutoff")
    offset = full.starts.get_loc(boundary)
    grid = DeliveryGrid(full.product_ids[offset:], tuple(full.starts[offset:].to_pydatetime()),
                        tuple(full.ends[offset:].to_pydatetime()))
    enabled = settings.get("auctions", list(AUCTIONS))
    if (not isinstance(enabled, list) or not enabled or len(set(enabled)) != len(enabled)
            or set(enabled) - set(AUCTIONS)):
        raise PreparedInputError("auctions must list unique DA/IDA1/IDA2/IDA3 market names")
    recorded = {}
    if known is not None and not known.empty:
        _require({key: True for key in known}, ("auction", "published_at"), "known result columns")
        if not known.auction.isin(enabled).all():
            raise PreparedInputError("Known results must belong to explicitly enabled auctions")
        for auction, rows in known.groupby("auction", sort=False):
            times = _timestamps(rows, "published_at", cutoff)
            if times.nunique() != 1:
                raise PreparedInputError("Each known auction curve needs one recorded publication timestamp")
            recorded[str(auction)] = times[0]
    events = []
    for event in calendar.schedule(str(settings["delivery_day"]), recorded_results=recorded):
        eligible = tuple(index - offset for index in event.eligible_intervals if index >= offset)
        if event.auction in enabled and eligible:
            events.append(replace(event, eligible_intervals=eligible))
    return grid, tuple(events), cutoff, full


def _aligned(frame: pd.DataFrame, grid: DeliveryGrid, eligible: tuple[int, ...], label: str) -> pd.DataFrame:
    """Select exact residual products; full-day tables may also contain the past."""
    result = frame.copy()
    result["product_start"] = _timestamps(result, "product_start")
    result["product_end"] = _timestamps(result, "product_end")
    if result.product_start.duplicated().any():
        raise PreparedInputError(f"Duplicate {label} native products")
    expected = pd.DatetimeIndex(grid.starts)[list(eligible)].as_unit("ns")
    if (not result.product_end.eq(result.product_start + pd.Timedelta(minutes=15)).all()
            or not result.product_start.dt.tz_convert("Europe/Rome").dt.date.eq(pd.Timestamp(grid.starts[0]).tz_convert("Europe/Rome").date()).all()):
        raise PreparedInputError(f"{label} must contain only native quarter-hour products of this delivery day")
    if (result.product_start.ge(pd.Timestamp(grid.starts[0])) & ~result.product_start.isin(expected)).any():
        raise PreparedInputError(f"{label} contains products outside its eligible residual grid")
    result = result.loc[result.product_start.isin(expected)].sort_values("product_start").reset_index(drop=True)
    if (not pd.DatetimeIndex(result.product_start).as_unit("ns").equals(expected)
            or not pd.DatetimeIndex(result.product_end).as_unit("ns").equals(pd.DatetimeIndex(grid.ends)[list(eligible)].as_unit("ns"))):
        raise PreparedInputError(f"{label} must cover every eligible native product exactly; no interpolation or filling")
    if "native_interval_minutes" in result and not pd.to_numeric(result.native_interval_minutes).eq(15).all():
        raise PreparedInputError(f"{label} must retain actual native quarter-hour products")
    return result


def _forecast_origin(frame: pd.DataFrame, label: str) -> None:
    """Distinguish usable forecasts from future actual/oracle coefficients."""
    if "origin" not in frame or frame.origin.isna().any():
        raise PreparedInputError(f"{label} needs explicit forecast origin provenance")
    values = frame.origin.astype(str).str.strip()
    if values.eq("").any() or values.str.lower().str.contains("realized|oracle|perfect_foresight|observed_clearing|observed_power", regex=True).any():
        raise PreparedInputError(f"{label} cannot use future realized/oracle observations as forecasts")


def _pv_members(pv: pd.DataFrame, grid: DeliveryGrid, cutoff: pd.Timestamp, settings: Mapping[str, Any], farm_id: int):
    """Validate explicit probabilities and one causal forecast vintage per member."""
    required = {"farm_id", "member_id", "member_origin", "member_probability", "forecast_cycle_time",
                "forecast_available_at", "interval_start_utc", "interval_end_utc", "pv_ac_mw",
                "input_eligibility", "review_status"}
    if required - set(pv) or pv.empty:
        raise PreparedInputError(f"PV member rows require {sorted(required)}")
    if not pd.to_numeric(pv.farm_id, errors="raise").eq(farm_id).all() or pv.member_id.isna().any():
        raise PreparedInputError("PV forecasts must identify exactly the configured farm and explicit members")
    _forecast_origin(pv.rename(columns={"member_origin": "origin"}), "PV members")
    if not _boolean(pv.input_eligibility, "input_eligibility").all():
        raise PreparedInputError("PV input_eligibility must be true for all supplied rows")
    for name in ("review_excluded", "derived_review_excluded"):
        if name in pv and _boolean(pv[name], name).any():
            raise PreparedInputError("PV forecasts contain review exclusions")
    allowed = {"no_configured_observation_review_exclusion", "no_review_exclusion", "eligible"}
    if not pv.review_status.isin(allowed).all():
        raise PreparedInputError("PV review_status must explicitly declare no review exclusion")
    frames, identifiers, weights, vintages = [], [], [], []
    for member, rows in pv.groupby("member_id", sort=False):
        if not str(member).strip() or rows.member_origin.nunique() != 1:
            raise PreparedInputError("PV member ID and origin must be explicit and constant within a member")
        probability = pd.to_numeric(rows.member_probability, errors="raise").to_numpy(float)
        if not np.isfinite(probability).all() or (probability <= 0).any() or not np.equal(probability, probability[0]).all():
            raise PreparedInputError("Each PV member needs one positive constant probability")
        source = _timestamps(rows, "forecast_cycle_time", cutoff)
        available = _timestamps(rows, "forecast_available_at", cutoff)
        if source.nunique() != 1 or available.nunique() != 1 or (source > available).any():
            raise PreparedInputError("PV member source/availability clocks must describe one causal vintage")
        if "model_fit_cutoff" in rows:
            fits = _timestamps(rows, "model_fit_cutoff", cutoff)
            if (fits > available).any():
                raise PreparedInputError("PV model fitting must precede forecast availability")
        elif settings.get("pv_model_fit_status") != "unfitted_explicit_assumption":
            raise PreparedInputError("PV rows need model_fit_cutoff or pv_model_fit_status=unfitted_explicit_assumption")
        aligned = _aligned(rows.rename(columns={"interval_start_utc": "product_start", "interval_end_utc": "product_end"}),
                           grid, tuple(range(len(grid.starts))), f"PV member {member}")
        values = pd.to_numeric(aligned.pv_ac_mw, errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise PreparedInputError("PV AC forecasts must be finite and nonnegative")
        identifiers.append(str(member))
        frames.append(values)
        weights.append(float(probability[0]))
        vintages.append({"member_id": str(member), "origin": str(rows.member_origin.iloc[0]),
                         "source_cycle": source[0].isoformat(), "available_at": available[0].isoformat()})
    if len(set(identifiers)) != len(identifiers) or not np.isclose(sum(weights), 1, rtol=0, atol=1e-10):
        raise PreparedInputError("PV member identifiers must be unique and probabilities must sum to one")
    return np.stack(frames), tuple(identifiers), np.asarray(weights), vintages


def _prices(grid, schedule, cutoff, specs, known_rows, settings):
    """Freeze current predictors, rebuilding latent ancestor prices per branch."""
    known, means, callbacks, calibrations, available, fits, audit = {}, {}, {}, {}, {}, {}, {}
    expected = {event.auction for event in schedule}
    if set(specs) - expected:
        raise PreparedInputError("Price model inputs contain an auction outside the enabled schedule")
    for event in schedule:
        auction, eligible = event.auction, event.eligible_intervals
        rows = None if known_rows is None else known_rows.loc[known_rows.auction.eq(auction)]
        if auction == "IDA3" and rows is not None and not rows.empty:
            if (_timestamps(rows, "product_start").tz_convert("Europe/Rome").hour < 12).any():
                raise PreparedInputError("IDA3 actual labels exist only from noon onward")
        if rows is not None and not rows.empty:
            if auction in specs:
                raise PreparedInputError(f"Known {auction} must be supplied once as actual results, without a forecast model")
            aligned = _aligned(rows, grid, eligible, f"known {auction}")
            if "zone" not in aligned or not aligned.zone.eq("NORD").all():
                raise PreparedInputError("Known auction results must identify zone NORD")
            known[auction], _ = _actual_vector(aligned, grid, auction, eligible)
            audit[auction] = {"kind": "known_actual_result", "published_at": event.results_at.isoformat()}
            continue
        if event.results_at <= cutoff:
            raise PreparedInputError(f"Previously published {auction} requires its actual known result curve")
        if auction not in specs:
            raise PreparedInputError(f"Missing explicitly prepared forecast for {auction}")
        specification = specs[auction]
        if sum(name in specification for name in ("point_model", "spread_model", "mean_curve")) != 1:
            raise PreparedInputError(f"{auction} must specify exactly one mean_curve, point_model or spread_model")
        if specification.get("native_interval_minutes") != 15:
            raise PreparedInputError(f"{auction} model scope must explicitly declare native_interval_minutes=15")
        model = specification.get("point_model", specification.get("spread_model"))
        spread = isinstance(model, ConditionalSpreadPriceForecast)
        if model is None:
            if "mean_curve" not in specification:
                raise PreparedInputError(f"{auction} needs a supplied mean curve or saved point/spread model")
            frame = _aligned(specification["mean_curve"], grid, eligible, f"{auction} mean curve")
            _forecast_origin(frame, f"{auction} mean curve")
            clock = "available_at"
            fit = _timestamps(frame, "fit_cutoff", cutoff).max()
            values = pd.to_numeric(frame.mean_price_eur_mwh, errors="raise").to_numpy(float)
            if not np.isfinite(values).all():
                raise PreparedInputError(f"{auction} mean curve must be finite")
            means[auction] = values
        else:
            if not isinstance(model, (LinearPriceForecast, ConditionalSpreadPriceForecast)):
                raise PreparedInputError("Price point/spread models must use the persisted forecast artifact contracts")
            if model.training_max_delivery_day >= str(settings["delivery_day"]):
                raise PreparedInputError("Saved price models must train on strictly earlier delivery days")
            if spread and model.auction != auction:
                raise PreparedInputError("Conditional spread model auction does not match its forecast equation")
            if not spread and auction != "DA":
                raise PreparedInputError("Intraday fitted equations require an explicit conditional spread model")
            if "features" not in specification:
                raise PreparedInputError(f"{auction} fitted model requires a target-free future features table")
            frame = _aligned(specification["features"], grid, eligible, f"{auction} future features")
            if any(name in frame and frame[name].notna().any() for name in ("actual", "target", "price_eur_mwh", "actual_price_eur_mwh")):
                raise PreparedInputError("Future feature tables must be target-free")
            clock = "static_feature_available_at" if spread else "feature_available_at"
            fit = _utc(model.fit_cutoff, f"{auction} model fit_cutoff")
            if not settings.get("allow_retrospective", False) and not model.operational_eligible:
                raise PreparedInputError("Unverified saved point models require explicit allow_retrospective=True")
            if spread:
                frame["scenario_grid_index"] = list(eligible)
                callbacks[auction] = lambda history, fitted=model, frozen=frame.copy(deep=True): fitted.predict_absolute(frozen, history)
            else:
                means[auction] = model.predict(frame)
        if "zone" not in frame or not frame.zone.eq("NORD").all():
            raise PreparedInputError("Price forecast features and means must explicitly identify zone NORD")
        if "auction" in frame and not frame.auction.eq(auction).all():
            raise PreparedInputError("Price forecast auction column differs from its configured equation")
        availability = _timestamps(frame, clock, cutoff)
        source = _timestamps(frame, "source_time", cutoff)
        if fit > cutoff or (source > availability).any() or (model is None and (fit > availability).any()):
            raise PreparedInputError(f"{auction} uses unavailable model fitting or source data")
        if not settings.get("allow_retrospective", False):
            if "historical_feature_eligible" not in frame or not _boolean(frame.historical_feature_eligible, "historical_feature_eligible").all():
                raise PreparedInputError("Unverified historical price features require explicit allow_retrospective=True")
        available[auction], fits[auction] = availability.max(), fit
        if settings["price_distribution"] == "gaussian":
            calibration = specification.get("calibration")
            if not isinstance(calibration, GaussianPriceCalibration):
                raise PreparedInputError(f"{auction} needs its saved scoped Gaussian calibration")
            calibrations[auction] = calibration
        audit[auction] = {"kind": "conditional_spread" if spread else "saved_point" if model else "supplied_absolute_mean",
                          "static_input_available_at": availability.max().isoformat(), "model_fit_cutoff": fit.isoformat(),
                          "future_ancestor_policy": "rebuilt from branch history" if spread else "fixed supplied absolute mean",
                          "intended_forecast_cutoffs": sorted(frame.forecast_cutoff.astype(str).unique().tolist()) if "forecast_cutoff" in frame else []}
    return known, means, callbacks, calibrations, available, fits, audit


@dataclass
class _JointHistory:
    """One currently indistinguishable group of joint price/PV leaves."""
    node: str
    leaves: tuple[int, ...]
    bids: dict[str, str]
    controls: dict[int, str]
    curtailments: dict[int, str]


def _joint_pv_tree(price_bundle, grid, pv, weights, member_ids, schedule, cutoff, delay_minutes):
    """Overlay completed PV observations without revealing latent member labels."""
    price_count, intervals = price_bundle.pv_ac_mw.shape
    members = len(pv)
    probabilities = np.repeat(weights, price_count) * np.tile(price_bundle.probabilities, members)
    pv_values = np.repeat(pv, price_count, axis=0)
    prices = {auction: np.tile(values, (members, 1)) for auction, values in price_bundle.auction_prices_eur_per_mwh.items()}
    known = tuple(price_bundle.provenance.get("known_prices", ()))
    root_observations = tuple(f"auction:{auction}" for auction in known)
    publication = {event.auction: event.results_at for event in schedule}
    nodes = [InformationNode("root", None, cutoff.to_pydatetime(), observations=root_observations,
                            observation_available_at={key: publication[key.split(":")[1]].to_pydatetime() for key in root_observations})]
    active = [_JointHistory("root", tuple(range(len(probabilities))), {}, {}, {})]
    events = []
    for event in schedule:
        if event.auction not in known:
            events.append((event.results_at, 1, "result", event.auction))
            if event.submission_at >= cutoff:
                events.append((event.submission_at, 2, "bid", event.auction))
    for index, (start, end) in enumerate(zip(grid.starts, grid.ends)):
        events.extend(((pd.Timestamp(end) + pd.Timedelta(minutes=delay_minutes), 0, "pv", index),
                       (pd.Timestamp(start), 3, "control", index), (pd.Timestamp(end), 4, "curtailment", index)))
    events.sort(key=lambda event: (event[0], event[1]))
    for event_at, _, kind, payload in events:
        following = []
        for history in active:
            groups = {}
            for leaf in history.leaves:
                key = tuple(prices[payload][leaf]) if kind == "result" else float(pv_values[leaf, payload]) if kind == "pv" else None
                groups.setdefault(key, []).append(leaf)
            mass = probabilities[list(history.leaves)].sum()
            for group in groups.values():
                node_id = f"joint_n{len(nodes)}"
                bids, controls, curtailments = dict(history.bids), dict(history.controls), dict(history.curtailments)
                observed = (f"auction:{payload}",) if kind == "result" else (f"pv:{grid.interval_ids[payload]}",) if kind == "pv" else ()
                if kind == "bid":
                    bids[payload] = node_id
                elif kind == "control":
                    controls[payload] = node_id
                elif kind == "curtailment":
                    curtailments[payload] = node_id
                nodes.append(InformationNode(node_id, history.node, pd.Timestamp(event_at).to_pydatetime(),
                    float(probabilities[group].sum() / mass), observed,
                    {key: pd.Timestamp(event_at).to_pydatetime() for key in observed}))
                following.append(_JointHistory(node_id, tuple(group), bids, controls, curtailments))
        active = following
    leaves, controls, curtailments = [None] * len(probabilities), [None] * len(probabilities), [None] * len(probabilities)
    open_auctions = tuple(event.auction for event in schedule if event.auction not in known and event.submission_at >= cutoff)
    bids = {auction: [None] * len(probabilities) for auction in open_auctions}
    terminal = pd.Timestamp(events[-1][0])
    for history in active:
        mass = probabilities[list(history.leaves)].sum()
        for index in history.leaves:
            node = f"joint_leaf_{index}"
            nodes.append(InformationNode(node, history.node, terminal.to_pydatetime(), float(probabilities[index] / mass)))
            leaves[index] = node
            controls[index] = tuple(history.controls[t] for t in range(intervals))
            curtailments[index] = tuple(history.curtailments[t] for t in range(intervals))
            for auction in bids:
                bids[auction][index] = history.bids[auction]
    tree = InformationTree(tuple(nodes), tuple(leaves), {key: tuple(value) for key, value in bids.items()},
        tuple(controls), tuple(curtailments), "pre_interval_battery; end_interval_curtailment; completed_PV_observations")
    bundle = ScenarioBundle(tuple(f"{member}:price_{index}" for member in member_ids for index in range(price_count)),
        probabilities, pv_values, prices, np.tile(price_bundle.imbalance_prices_eur_per_mwh, (members, 1)),
        None if price_bundle.imbalance_negative_prices_eur_per_mwh is None else np.tile(price_bundle.imbalance_negative_prices_eur_per_mwh, (members, 1)),
        origin="prepared forecast prices / explicit probabilistic PV members", provenance={**price_bundle.provenance,
            "pv_members": members, "pv_member_ids": list(member_ids), "pv_member_probabilities": weights.tolist(),
            "price_paths_per_pv_member": price_count, "price_pv_coupling": "independent product measure; common price draws across members",
            "pv_member_revelation": "latent until completed-interval PV observations distinguish histories",
            "pv_observation_delay_minutes": delay_minutes, "execution_convention": tree.execution_convention})
    return bundle, tree


def build_forecast_inputs(config: Mapping[str, Any], pv_forecasts: pd.DataFrame,
                          auction_inputs: Mapping[str, Mapping[str, Any]], *,
                          known_results: pd.DataFrame | None = None,
                          accepted_trades: pd.DataFrame | None = None) -> OptimizationInput:
    """Assemble a solver-ready stochastic plan using explicit prepared primitives."""
    settings = _settings(config)
    grid, schedule, cutoff, full = _grid_schedule(config, settings, known_results)
    plant_cfg, battery_cfg = config.get("plant", {}), config.get("battery", {})
    _require(plant_cfg, ("farm_id", "poi_import_mw", "poi_export_mw"), "plant equipment")
    _require(battery_cfg, ("capacity_mwh", "max_power_mw"), "battery equipment")
    plant = PlantParameters(int(plant_cfg["farm_id"]), float(plant_cfg["poi_import_mw"]), float(plant_cfg["poi_export_mw"]), battery_parameters_from_config(battery_cfg))
    pv, members, weights, vintages = _pv_members(pv_forecasts, grid, cutoff, settings, plant.farm_id)
    known, means, callbacks, calibrations, available, fits, audit = _prices(grid, schedule, cutoff, auction_inputs, known_results, settings)
    if (any(event.submission_at < cutoff < event.results_at and event.auction not in known for event in schedule)
            and "pending_position_mwh" not in settings):
        raise PreparedInputError("Between submission and results, supply explicit pending_position_mwh for guaranteed committed quantities awaiting clearing price, including explicit zeros if no quantity was submitted")
    unknown_count = sum(event.auction not in known for event in schedule)
    estimated = len(members) * (settings["branches"] ** unknown_count if settings["price_distribution"] == "gaussian" else 1)
    if estimated > settings["maximum_joint_scenarios"]:
        raise PreparedInputError(f"Joint tree could contain {estimated} leaves; reduce branches/members or explicitly raise maximum_joint_scenarios")
    settlement = _planning_settlement_price(settings, cutoff,
        config.get("trading", {}).get("optimize_imbalances", True))
    if settings["price_distribution"] == "gaussian":
        prices, _ = gaussian_price_tree(grid, pv[0], schedule, mean_prices=means, conditional_means=callbacks,
            calibrations=calibrations, feature_available_at=available, model_fit_cutoffs=fits,
            known_prices=known, decision_cutoff=cutoff, branches=settings["branches"], seed=settings["seed"],
            imbalance_price=settlement, allow_retrospective=settings.get("allow_retrospective", False))
    else:
        def absolute(auction, history):
            event = next(event for event in schedule if event.auction == auction)
            values = callbacks[auction](history) if auction in callbacks else means[auction]
            result = np.zeros(len(grid.starts))
            result[list(event.eligible_intervals)] = values
            return result
        prices, _ = conditional_price_tree(grid, pv[0], schedule, history_mean=absolute,
            innovation_sampler=lambda auction, count, random: np.zeros((1, count)), branch_probabilities=(1.,),
            known_prices=known, decision_cutoff=cutoff, seed=settings["seed"], imbalance_price=settlement,
            provenance={"operational_eligible": False, "price_distribution": "explicit point means; zero innovations",
                        "auction_eligible_product_indices": {event.auction: list(event.eligible_intervals) for event in schedule}})
    delay = float(settings.get("pv_observation_delay_minutes", 0))
    if not np.isfinite(delay) or delay < 0:
        raise PreparedInputError("PV observation delay must be finite and nonnegative")
    scenarios, tree = _joint_pv_tree(prices, grid, pv, weights, members, schedule, cutoff, delay)
    scenarios = replace(scenarios, provenance={**scenarios.provenance, "forecast_input_audit": audit,
        "pv_vintages": vintages, "decision_cutoff": cutoff.isoformat(),
        "execution_policy": "Every submitted quantity executes at the published clearing price; no acceptance/rejection uncertainty",
        "committed_unpriced_positions": "pending_position_mwh are fixed executed quantities awaiting price publication; record actual cash exactly once in the external ledger",
        "objective_scope": "Incremental new trading and dispatch profit; already committed cash belongs to the external realized ledger",
        "operational_eligible": False, "historical_deployment_status": "prepared study boundary; source vintage claims require independent verification",
        "settlement_policy": ("explicit signed single-price study assumption; historical settlement semantics unverified"
            if config.get("trading", {}).get("optimize_imbalances", True)
            else "excluded from optimization; actual settlement is ex post")})
    state_cfg, trading, degradation = config.get("state", {}), config.get("trading", {}), config.get("degradation", {})
    _require(state_cfg, ("timestamp", "initial_energy_mwh", "origin"), "explicit residual physical state")
    if _utc(state_cfg["timestamp"], "state.timestamp") != pd.Timestamp(grid.starts[0]):
        raise PreparedInputError("Physical state timestamp must equal the residual grid start")
    if grid.starts[0] == full.starts[0] and state_cfg["origin"] != "known_delivery_start_assumption":
        raise PreparedInputError("Delivery-start planning requires known_delivery_start_assumption")
    if grid.starts[0] != full.starts[0] and state_cfg["origin"] == "known_delivery_start_assumption":
        raise PreparedInputError("Residual reoptimization must supply carried/measured physical state")
    state = PhysicalState(grid.starts[0], float(state_cfg["initial_energy_mwh"]),
        float(state_cfg.get("cumulative_throughput_mwh", 0)), float(state_cfg.get("cumulative_degradation_eur", 0)),
        str(state_cfg["origin"]), state_cfg.get("cycle_segment_energy_mwh"))
    market = MarketState()
    if accepted_trades is not None and not accepted_trades.empty:
        required = ("auction", "submission_time", "publication_time", "interval_id", "quantity_mwh", "price_eur_per_mwh", "trade_id")
        _require({name: True for name in accepted_trades}, required, "accepted trade columns")
        for row in accepted_trades.to_dict("records"):
            publication = _utc(row["publication_time"], "trade publication")
            if publication > cutoff or row["auction"] not in known:
                raise PreparedInputError("Accepted trades require published known auction results by the decision cutoff")
            event = next(event for event in schedule if event.auction == row["auction"])
            if publication != event.results_at:
                raise PreparedInputError("Accepted trade publication must equal its known auction result publication")
            if _utc(row["submission_time"], "trade submission") > event.gate_at:
                raise PreparedInputError("Accepted trade submission exceeds the actual auction gate")
            if row["interval_id"] not in full.product_ids:
                raise PreparedInputError("Accepted trade interval_id does not belong to the delivery day")
            stamp = pd.Timestamp(row["interval_id"])
            if row["auction"] == "IDA3" and stamp.tz_convert("Europe/Rome").hour < 12:
                raise PreparedInputError("An accepted IDA3 trade cannot target a nonexistent morning product")
            if row["interval_id"] in grid.interval_ids:
                actual = known[row["auction"]][grid.interval_ids.index(row["interval_id"])]
                if not np.isclose(float(row["price_eur_per_mwh"]), actual, rtol=0, atol=1e-9):
                    raise PreparedInputError("Accepted auction trade price differs from its known clearing curve")
            market = market.accept(AcceptedTrade(str(row["auction"]), _utc(row["submission_time"], "trade submission").to_pydatetime(),
                publication.to_pydatetime(), str(row["interval_id"]), float(row["quantity_mwh"]), float(row["price_eur_per_mwh"]), str(row["trade_id"])))
    _require(trading, ("imbalance_penalty_eur_per_mwh", "grid_import_fee_eur_per_mwh"), "trading costs")
    _require(degradation, ("assumptions",), "degradation assumptions")
    masks = {event.auction: tuple(t in event.eligible_intervals for t in range(len(grid.starts)))
             for event in schedule if event.auction in tree.bid_nodes}
    return OptimizationInput(grid, plant, state, market, scenarios, tree, masks,
        float(trading["imbalance_penalty_eur_per_mwh"]),
        _terminal_target(state_cfg), degradation_settings_from_config(config), float(trading["grid_import_fee_eur_per_mwh"]),
        planned_delivery_balance_resolution_minutes=trading.get("planned_delivery_balance_resolution_minutes"),
        optimize_imbalances=trading.get("optimize_imbalances", True),
        imbalance_limit_mwh=trading.get("imbalance_limit_mwh"), pending_position_mwh=settings.get("pending_position_mwh"))


def load_forecast_inputs(config: Mapping[str, Any]) -> OptimizationInput:
    """Load canonical prepared tables/artifacts and return the existing MILP input."""
    settings = _settings(config)
    _require(settings, ("pv_forecast_path", "auction_inputs"), "prepared forecast files")
    provenance = {"pv_forecast": _file_provenance(settings["pv_forecast_path"])} if settings["pv_forecast_path"] and Path(settings["pv_forecast_path"]).is_file() else {}
    pv = _table(settings["pv_forecast_path"], "PV member forecasts")
    specs = {}
    for auction, declared in settings["auction_inputs"].items():
        values = dict(declared)
        for name, constructor in (("point_model", LinearPriceForecast.load), ("spread_model", ConditionalSpreadPriceForecast.load),
                                  ("calibration", GaussianPriceCalibration.load)):
            key = f"{name}_path"
            if key in values:
                if not values[key] or not Path(values[key]).is_file():
                    raise PreparedInputError(f"Missing prepared {auction} {name} artifact: supply an existing JSON file")
                values[name] = constructor(values.pop(key))
                provenance[f"{auction}_{name}"] = _file_provenance(declared[key])
        for name in ("features", "mean_curve"):
            key = f"{name}_path"
            if key in values:
                path = values.pop(key)
                values[name] = _table(path, f"{auction} {name}")
                provenance[f"{auction}_{name}"] = _file_provenance(path)
        specs[auction] = values
    extra = {}
    for name in ("known_results", "accepted_trades"):
        key = f"{name}_path"
        if settings.get(key):
            extra[name] = _table(settings[key], name)
            provenance[name] = _file_provenance(settings[key])
    result = build_forecast_inputs(config, pv, specs, **extra)
    return replace(result, scenarios=replace(result.scenarios,
        provenance={**result.scenarios.provenance, "prepared_file_sha256": provenance,
                    "acquisition_policy": "prepared artifacts only; acquisition uses the separate DuckDB catalog"}))

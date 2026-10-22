"""Reproducible single-day synthetic auction replay and comparable policies."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, datetime, timezone
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..domain import (AcceptedTrade, BatteryParameters,
    DeliveryGrid, MarketState, OptimizationInput, PhysicalState, PlantParameters,
    SolverSettings, battery_parameters_from_config)
from ..markets import AUCTION_DISPATCH_SCOPE, AuctionCalendar
from ..optimization import build_model, solve_model, validate_solution
from ..optimization.degradation import degradation_settings_from_config
from ..optimization.prepared import terminal_minimum_energy_mwh
from ..reporting.artifacts import organize_run_outputs
from ..reporting.optimization_timing import optimization_timing_report, solve_timing
from ..scenarios import synthetic_price_tree
from ..transaction_fees import account_trade_fees, transaction_fees_from_config
from .events import Event, EventEngine, EventKind
from .replay import execute_interval
from .settlement import expected_realized_kpis, settle_realized_dispatch


class InfeasibleAuctionPlan(RuntimeError):
    """An infeasible model is distinct from solver errors or missing incumbents."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def json_default(value: Any) -> Any:
    """Serialize evidence arrays and clocks without serializing private objects."""
    if isinstance(value, (date, datetime, pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    """Write a human-readable numerical artifact, rejecting nonfinite numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")


def synthetic_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Create declared test equipment and one deterministic synthetic AC trace."""
    study = config["synthetic"]
    calendar = AuctionCalendar.from_config(config)
    market_grid = calendar.delivery_grid(study["delivery_day"])
    grid = market_grid.to_domain()
    schedule = calendar.schedule(study["delivery_day"])
    local = market_grid.starts.tz_convert(calendar.timezone)
    hour = np.asarray(local.hour + local.minute / 60 + 0.125)
    phase = (hour - study["sunrise_hour"]) / (study["sunset_hour"] - study["sunrise_hour"])
    pv = study["pv_peak_mw"] * np.maximum(0, np.sin(np.pi * np.clip(phase, 0, 1)))
    base = study["base_price_eur_per_mwh"] + study["daily_price_amplitude"] * np.cos(2*np.pi*(hour-19)/24)
    battery = battery_parameters_from_config(config["battery"])
    plant = PlantParameters(config["plant"]["farm_id"], config["plant"]["poi_import_mw"],
                            config["plant"]["poi_export_mw"], battery)
    kwargs = dict(base_prices=base, stage_spreads=study["stage_spreads"],
        residual_scale=study["residual_scale"], branch_values=study["branch_values"],
        branch_probabilities=study["branch_probabilities"],
        imbalance_price=study["imbalance_price_eur_per_mwh"], seed=study["seed"])
    if study["branches_per_auction"] != len(study["branch_values"]):
        raise ValueError("branches_per_auction differs from declared branch_values")
    scenarios, tree = synthetic_price_tree(grid, pv, schedule, **kwargs)
    realized = int(np.random.default_rng(study["seed"]).choice(len(scenarios.scenario_ids), p=scenarios.probabilities))
    actual_prices = {auction: values[realized].copy()
                     for auction, values in scenarios.auction_prices_eur_per_mwh.items()}
    return dict(calendar=calendar, grid=grid, schedule=schedule, pv=pv, base=base,
                plant=plant, tree=tree, scenarios=scenarios, realized=realized,
                actual_prices=actual_prices, kwargs=kwargs)


def _replay_strategy(config: dict[str, Any], fixture: dict[str, Any], strategy: str,
                     destination: Path) -> dict[str, Any]:
    """Submit only the actionable auction, then carry its ledger and physical state."""
    if strategy not in {"oracle", "deterministic_sequential", "deterministic_integrated", "stochastic_integrated"}:
        raise ValueError(f"Unsupported strategy {strategy}")
    started = perf_counter()
    full_grid, plant = fixture["grid"], fixture["plant"]
    if destination.exists():
        raise FileExistsError(f"Replay output exists; choose a fresh destination: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    schedule, actual_prices = fixture["schedule"], fixture["actual_prices"]
    planning_mode = fixture.get("auction_planning_mode", "integrated")
    current_only = planning_mode == "current_auction_only" or strategy == "deterministic_sequential"
    if planning_mode not in {"integrated", "current_auction_only"}:
        raise ValueError(f"Unsupported auction planning mode {planning_mode}")
    state = PhysicalState(full_grid.starts[0], config["state"]["initial_energy_mwh"], origin=config["state"]["origin"],
        cycle_segment_energy_mwh=config["state"].get("cycle_segment_energy_mwh"))
    fees = transaction_fees_from_config(config)
    market = MarketState(transaction_fee_state=fees.initial_state)
    fee_records: dict[str, dict] = {}
    known: dict[str, np.ndarray] = {}
    submitted: dict[str, dict[str, float]] = {}
    pending: dict[int, tuple[float, float, float]] = {}
    records: dict[int, dict[str, Any]] = {}
    solves: list[dict[str, Any]] = []
    timing_records: list[dict[str, Any]] = []
    infeasible_decisions: list[dict[str, Any]] = []
    publications: list[dict[str, Any]] = []
    plan_snapshots: dict[str, list[dict[str, Any]]] = {}
    forecast_input_snapshots: list[dict[str, Any]] = []
    control_plan: dict[int, tuple[float, float, float]] = {}
    control_plan_details: dict[int, dict] = {}
    execution_details: dict[int, dict] = {}
    grid_targets: dict[int, float] = {}
    current_forecast: np.ndarray | None = None
    pv_forecast_metadata = {
        "pv_forecast_policy": fixture.get("pv_forecast_policy", "fixed_fixture_forecast"),
        "pv_forecast_vintage_utc": (pd.Timestamp(fixture["pv_forecast_vintage"]).isoformat()
            if fixture.get("pv_forecast_vintage") is not None else None),
        "pv_forecast_available_at_utc": (pd.Timestamp(fixture["forecast_available_at"]).isoformat()
            if fixture.get("forecast_available_at") is not None else None),
    }
    aging = degradation_settings_from_config(config)
    settings = SolverSettings(**config["solver"])
    trading = config["trading"]
    dispatch_policy = config.get("execution", {}).get("dispatch_policy", "follow_battery_plan")
    # An auction decision can replace commands only from its first delivery
    # product onward. Before first participation, execute zero-position feedback.
    first_participation_interval = (max(min(schedule[0].eligible_intervals),
        int(pd.DatetimeIndex(full_grid.starts).searchsorted(schedule[0].submission_at)))
        if schedule else len(full_grid.interval_ids))

    def plan(cutoff: pd.Timestamp, current_auction: str | None = None, *,
             trigger: str = "physical_control", published_auction: str | None = None):
        """Use published information and a known boundary for committed commands."""
        nonlocal current_forecast, pv_forecast_metadata
        if current_forecast is None:
            raise ValueError("No published PV forecast at the decision cutoff")
        provider = fixture.get("pv_forecast_provider")
        if provider is not None:
            # Refresh only unobserved intervals, including an unfinished interval
            # needed to project the next boundary. Executed history stays fixed.
            update_first = int(pd.DatetimeIndex(full_grid.starts).searchsorted(state.timestamp))
            update_starts = pd.DatetimeIndex(full_grid.starts[update_first:])
            update = provider(cutoff, update_starts)
            if update.available_at > cutoff or update.forecast_vintage > cutoff:
                raise ValueError("PV forecast update is unavailable at the optimization cutoff")
            if update.farm_id != plant.farm_id or len(update.members) != 1:
                raise ValueError("PV forecast update must contain this farm and one explicit member")
            if not pd.DatetimeIndex(update.values.index).equals(update_starts):
                raise ValueError("PV forecast update does not match the remaining execution grid")
            values = pd.to_numeric(update.values["pv_ac_mw"], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("PV forecast update must be finite and nonnegative")
            updated = current_forecast.copy()
            updated[update_first:] = values
            current_forecast = updated
            pv_forecast_metadata = {**pv_forecast_metadata,
                "pv_forecast_vintage_utc": update.forecast_vintage.isoformat(),
                "pv_forecast_available_at_utc": update.available_at.isoformat(),
                "pv_forecast_refresh_cutoff_utc": cutoff.isoformat(),
                "pv_forecast_updated_from_utc": update_starts[0].isoformat(),
                "pv_forecast_updated_intervals": len(update_starts),
                "pv_forecast_missing_filled_night_zero_intervals": update.provenance.get(
                    "forecast_missing_filled_night_zero_intervals", 0)}
        first = int(pd.DatetimeIndex(full_grid.starts).searchsorted(cutoff))
        if cutoff < pd.Timestamp(full_grid.starts[0]):
            first = 0
        auction_delivery_start = None
        if current_auction is not None:
            auction_event = next(event for event in schedule if event.auction == current_auction)
            delivery_first = min(auction_event.eligible_intervals)
            auction_delivery_start = pd.Timestamp(full_grid.starts[delivery_first]).isoformat()
            first = max(first, delivery_first)
        if first >= len(full_grid.interval_ids):
            raise ValueError("No remaining delivery intervals")
        boundary = state
        projected_import_fee = 0.0
        locked_prefix_projection = []
        # Project to the first tradable delivery boundary using retained commands
        # and cutoff-available PV. IDA3 cannot replace the pre-noon dispatch plan;
        # neither its noon SOC nor its wear may use future realized telemetry.
        projection_first = int(pd.DatetimeIndex(full_grid.starts).searchsorted(state.timestamp))
        for prior in range(projection_first, first):
            if prior in pending:
                commands = pending[prior]
                detail = execution_details[prior]
                target = grid_targets[prior]
            elif prior in control_plan:
                commands = control_plan[prior]
                detail = control_plan_details[prior]
                target = sum(offer.get(full_grid.interval_ids[prior], 0.0)
                    for offer in submitted.values()) / full_grid.delta_hours[prior]
            elif prior < first_participation_interval:
                commands = (0.0, 0.0, 0.0)
                detail = {"auction": "pre_participation"}
                target = 0.0
            else:
                raise ValueError("No retained dispatch plan covers the locked delivery prefix")
            projection_policy = ("track_committed_grid" if detail.get("auction")
                == "pre_participation" else dispatch_policy)
            boundary, projected = execute_interval(state=boundary, start=full_grid.starts[prior],
                end=full_grid.ends[prior], pv_ac_mw=float(current_forecast[prior]),
                requested_charge_mw=commands[0], requested_discharge_mw=commands[1],
                requested_curtailment_mw=commands[2], plant=plant,
                degradation_settings=aging, dispatch_policy=projection_policy, grid_target_mw=target)
            projected_import_fee += max(0.0, -projected["grid_mw"]) * full_grid.delta_hours[prior] * trading["grid_import_fee_eur_per_mwh"]
            locked_prefix_projection.append({"interval_id": full_grid.interval_ids[prior],
                "start": str(full_grid.starts[prior]), "end": str(full_grid.ends[prior]),
                "plan_source_auction": detail.get("auction"), **projected})
            boundary = replace(boundary, origin=("forecast_projected_boundary_under_feedback"
                if projection_policy == "track_committed_grid" else "projected_boundary_of_committed_command"))
        residual = DeliveryGrid(full_grid.interval_ids[first:], full_grid.starts[first:], full_grid.ends[first:])
        # Save information supplied to every attempt, including failed solves;
        # a fallback's retained old plan is not the newly available PV input.
        if trigger == "auction_submission":
            update_first = int(pd.DatetimeIndex(full_grid.starts).searchsorted(state.timestamp))
            forecast_input_snapshots.extend({"auction": current_auction, "cutoff_utc": cutoff.isoformat(),
                "interval_start_utc": pd.Timestamp(full_grid.starts[t]).isoformat(),
                "pv_ac_forecast_mw": float(current_forecast[t]), **pv_forecast_metadata}
                for t in range(update_first, len(full_grid.interval_ids)))
        kwargs = dict(fixture["kwargs"])
        kwargs.update(base_prices=fixture["base"][first:],
            known_prices={key: value[first:] for key, value in known.items()}, decision_cutoff=cutoff)
        selected_schedule = schedule
        if strategy != "stochastic_integrated":
            scenario_settings = config.get("synthetic", {})
            values = scenario_settings.get("branch_values", (0.0,))
            probabilities = scenario_settings.get("branch_probabilities", (1.0,))
            mean_innovation = float(np.dot(values, probabilities))
            kwargs.update(branch_values=(mean_innovation,), branch_probabilities=(1.0,))
        if current_only:
            selected_schedule = tuple(event for event in schedule if event.auction == current_auction)
        if strategy == "oracle":
            kwargs["branch_values"] = (0.0,)
            if current_only:
                # Use this auction's absolute oracle curve, independent of all
                # later curves. Earlier commitments enter through the ledger.
                kwargs.update(base_prices=actual_prices[current_auction][first:] if current_auction else np.zeros(len(residual.interval_ids)),
                    stage_spreads={}, known_prices={})
            else:
                auction_order = [event.auction for event in schedule]
                first_auction = auction_order[0]
                kwargs["base_prices"] = actual_prices[first_auction][first:]
                kwargs["stage_spreads"] = {current: (actual_prices[current]-actual_prices[previous])[first:]
                           for previous,current in zip(auction_order,auction_order[1:])}
        bundle, tree = synthetic_price_tree(residual, current_forecast[first:], selected_schedule, **kwargs)
        masks = {a: tuple((t+first) in next(e.eligible_intervals for e in schedule if e.auction==a)
                         for t in range(len(residual.interval_ids))) for a in tree.bid_nodes}
        # A submitted fixed-quantity offer is already committed before clearing.
        # Carry its quantity while its price is still unpublished; only publication
        # moves it into the accepted ledger and realizes its contractual cash.
        pending_positions = tuple(sum(offer.get(product,0.0)
                                    for auction,offer in submitted.items() if auction not in known)
                                  for product in residual.interval_ids)
        inputs = OptimizationInput(residual, plant, boundary, market, bundle, tree, masks,
            trading["imbalance_penalty_eur_per_mwh"], config["state"]["terminal_target_mwh"],
            aging, trading["grid_import_fee_eur_per_mwh"],
            imbalance_limit_mwh=trading.get("imbalance_limit_mwh"),
            planned_delivery_balance_resolution_minutes=trading.get("planned_delivery_balance_resolution_minutes"),
            optimize_imbalances=trading.get("optimize_imbalances", True),
            pending_position_mwh=pending_positions,
            decision_resolution_minutes=config.get("decision_resolution_minutes"),
            market_timezone=config.get("market", {}).get("timezone", fixture["calendar"].timezone),
            native_interval_minutes_by_auction={auction: minutes for auction, minutes
                in config.get("native_interval_minutes_by_auction", {}).items()
                if auction in masks}, transaction_fees=fees,
            auction_submission_times={event.auction: event.submission_at for event in schedule},
            terminal_minimum_mwh=terminal_minimum_energy_mwh(config["state"], plant.battery))
        built = build_model(inputs)
        planning_diagnostics = {"auction_planning_mode": "current_auction_only" if current_only else "integrated",
            "auction_dispatch_scope": AUCTION_DISPATCH_SCOPE,
            "auction_delivery_start_utc": auction_delivery_start,
            "physical_planning_intervals": len(residual.interval_ids),
            "locked_prefix_intervals": len(locked_prefix_projection),
            "locked_prefix_projection": locked_prefix_projection,
            "projected_boundary_energy_mwh": boundary.energy_mwh,
            "modeled_auctions": list(masks),
            "trade_variable_counts_by_auction": {auction: sum(key[0] == auction for key in built.variables["trades"])
                for auction in masks}, "trade_variable_count": len(built.variables["trades"]),
            "fixed_commitment_auctions": list(submitted)}
        try:
            model_settings = config["solver"]
            model_file = None
            if config.get("outputs", {}).get("save_model_files", False):
                model_directory = destination / "models"
                model_directory.mkdir(parents=True, exist_ok=True)
                model_file = model_directory / f"model_{len(solves) + 1:03d}_{cutoff.strftime('%Y%m%dT%H%M%SZ')}_{trigger}.lp"
                model_settings = {**model_settings, "export_path": str(model_file)}
            if config.get("execution", {}).get("save_infeasibility_diagnostics", False):
                model_settings = {**model_settings, "iis_path": str(destination / "models" /
                    f"infeasible_{cutoff.strftime('%Y%m%dT%H%M%SZ')}_{trigger}.ilp")}
            result = solve_model(built, SolverSettings(**model_settings))
        finally:
            built.model.dispose()
        # Save native solver time for every attempt, including infeasible solves
        # that retain an earlier plan. Simulation wall time is a separate metric.
        timing_records.append(solve_timing(result.diagnostics, category="replay",
            strategy=strategy, auction=current_auction, cutoff_utc=cutoff.isoformat(),
            trigger=trigger, status=result.status))
        write_json(destination / "optimization_timing.json", optimization_timing_report(timing_records))
        if not result.has_incumbent:
            if result.status == "infeasible":
                raise InfeasibleAuctionPlan(f"{strategy} at {cutoff}: {result.status}: {result.message}",
                    {"cutoff": cutoff.isoformat(), "current_auction": current_auction,
                     "solve_trigger": trigger, "status": result.status,
                     **pv_forecast_metadata, **planning_diagnostics, **result.diagnostics})
            raise RuntimeError(f"{strategy} at {cutoff}: {result.status}: {result.message}")
        validation = validate_solution(inputs, result)
        if not validation.valid:
            raise RuntimeError(f"Independent solution validation failed: {validation.violations}")
        solves.append({"cutoff":cutoff.isoformat(), "current_auction":current_auction,
            **pv_forecast_metadata,
            "planning_start_utc": pd.Timestamp(residual.starts[0]).isoformat(),
            "planning_end_utc": pd.Timestamp(residual.ends[-1]).isoformat(),
            "solve_trigger":trigger, "published_auction":published_auction,
            "model_file":str(model_file) if model_file is not None else None,
            "known_auctions":list(known),
            "known_prices_fixed":{auction:bool(np.all(bundle.auction_prices_eur_per_mwh[auction] == curve[first:]))
                                  for auction,curve in known.items() if auction in bundle.auction_prices_eur_per_mwh},
            **planning_diagnostics,
            "state_origin":boundary.origin,"state_timestamp":str(boundary.timestamp),
            "pending_position_abs_mwh":sum(abs(value) for value in pending_positions),
            "pre_horizon_degradation_eur": boundary.cumulative_degradation_eur,
            "pre_horizon_import_fee_eur": sum(record.get("import_fee_eur", 0.0) for record in records.values()) + projected_import_fee,
            "status":result.status,"objective_eur":result.objective_eur,
            "expected_auction_revenue_eur": float(np.dot(
                inputs.scenarios.probabilities, result.components["auction_revenue"])),
            "expected_transaction_fees_eur": float(np.dot(
                inputs.scenarios.probabilities, result.components["transaction_fees"])),
            "expected_transaction_fees_by_auction_eur": {auction: sum(
                float(probability) * item["by_auction"].get(auction, {}).get("transaction_fee_eur", 0.0)
                for probability, item in zip(inputs.scenarios.probabilities,
                    result.diagnostics.get("transaction_fees_by_scenario", []))) for auction in masks},
            "validation":asdict(validation),**result.diagnostics})
        return inputs, result

    def publish_forecast(event: Event) -> None:
        nonlocal current_forecast
        current_forecast = fixture["pv"].copy()

    def submit(event: Event) -> None:
        nonlocal market
        auction = event.payload.auction
        if auction in submitted:
            raise RuntimeError("An auction cannot submit and incur transaction fees twice")
        try:
            inputs, result = plan(event.at, auction, trigger="auction_submission")
        except InfeasibleAuctionPlan as error:
            if config.get("execution", {}).get("infeasible_auction_policy", "error") != "previous_plan" or not plan_snapshots:
                raise
            # Preserve the last deterministic trading/dispatch plan. Previously
            # unsubmitted orders are submitted now, with the same fee accounting.
            previous_auction = next(reversed(plan_snapshots))
            previous = plan_snapshots[previous_auction]
            trade_column = f"planned_trade_{auction}_mwh"
            if any(row.get("snapshot_scenario_count") != 1 or trade_column not in row for row in previous):
                raise RuntimeError("Previous-plan fallback requires a complete deterministic auction snapshot") from error
            by_product = {row["interval_id"]: row for row in previous}
            products = [full_grid.interval_ids[index] for index in event.payload.eligible_intervals]
            if any(product not in by_product for product in products):
                raise RuntimeError("Previous-plan fallback is missing eligible auction products") from error
            submitted[auction] = {product: (0.0 if current_only else float(by_product[product][trade_column])) for product in products}
            next_fee_state, charged, _ = account_trade_fees(fees,
                market.transaction_fee_state, [{"auction": auction, "submission_time": event.at,
                    "interval_id": product, "quantity_mwh": quantity, "trade_id": f"{auction}:{product}"}
                    for product, quantity in submitted[auction].items()])
            market = replace(market, transaction_fee_state=next_fee_state)
            fee_records.update({row["trade_id"]: row for row in charged})
            snapshot = []
            for prior in previous:
                index = full_grid.interval_ids.index(prior["interval_id"])
                if full_grid.starts[index] < event.at or index < min(event.payload.eligible_intervals):
                    continue
                quantity = submitted[auction].get(prior["interval_id"], 0.0)
                row = {**prior, "auction": auction, "optimizer_cutoff_utc": event.at.isoformat(),
                    "plan_status": "previous_plan_after_infeasible_reoptimization",
                    "plan_source_auction": previous_auction,
                    "plan_source_cutoff_utc": prior["optimizer_cutoff_utc"],
                    "auction_trade_mwh": quantity, "auction_gross_traded_mwh": abs(quantity),
                    "auction_transaction_fee_eur": fee_records.get(f"{auction}:{prior['interval_id']}", {}).get("transaction_fee_eur", 0.0)}
                snapshot.append(row)
                control_plan[index] = tuple(row[name] for name in ("charge_mw", "discharge_mw", "curtailment_mw"))
                control_plan_details[index] = row
            plan_snapshots[auction] = snapshot
            infeasible_decisions.append({**error.diagnostics, "fallback_policy": "previous_plan",
                "fallback_order_policy": "zero_new_trades" if current_only else "previous_planned_trades",
                "plan_source_auction": previous_auction,
                "submitted_gross_mwh": sum(abs(quantity) for quantity in submitted[auction].values())})
            return
        owner = inputs.tree.bid_nodes[auction][0]
        if len(set(inputs.tree.bid_nodes[auction])) != 1:
            raise RuntimeError("Actionable bid is not shared at the current information cutoff")
        submitted[auction] = {product:quantity for (a,node,product),quantity in result.trades.items()
                              if a==auction and node==owner}
        # Guaranteed full execution makes quantities and fees fixed at submission,
        # even while MCP cashflow is unpublished. Future contingent bids stay free.
        next_fee_state, charged, _ = account_trade_fees(fees,
            market.transaction_fee_state, [{"auction": auction, "submission_time": event.at,
                "interval_id": product, "quantity_mwh": quantity, "trade_id": f"{auction}:{product}"}
                for product, quantity in submitted[auction].items()])
        market = replace(market, transaction_fee_state=next_fee_state)
        fee_records.update({row["trade_id"]: row for row in charged})
        snapshot = []
        for local_t, product in enumerate(inputs.grid.interval_ids):
            t = full_grid.interval_ids.index(product)
            owner = inputs.tree.bid_nodes[auction][0]
            trade_mwh = float(result.trades.get((auction, owner, product), 0.0))
            # Validation has rejected material bound violations; floor tolerated
            # negative roundoff before handing commands to physical execution.
            commands = {name: max(0.0, float(result.arrays[name][0, local_t]))
                for name in ("charge_mw", "discharge_mw", "curtailment_mw")}
            row = {"auction": auction, "interval_id": product,
                **pv_forecast_metadata,
                "interval_start_utc": str(full_grid.starts[t]),
                "interval_end_utc": str(full_grid.ends[t]),
                **commands,
                "net_battery_dispatch_mw": commands["discharge_mw"] - commands["charge_mw"],
                "pv_ac_forecast_mw": float(inputs.scenarios.pv_ac_mw[0, local_t]),
                "pv_injection_forecast_mw": float(inputs.scenarios.pv_ac_mw[0, local_t]
                    - commands["curtailment_mw"]),
                "expected_grid_mw": float(result.arrays["grid_mw"][0, local_t]),
                "auction_trade_mwh": trade_mwh,
                "auction_gross_traded_mwh": abs(trade_mwh),
                "auction_transaction_fee_eur": fee_records.get(f"{auction}:{product}", {}).get("transaction_fee_eur", 0.0),
                "snapshot_scenario_count": len(inputs.scenarios.scenario_ids),
                "planned_net_position_mwh": float(result.arrays["net_position_mwh"][0, local_t]),
                "soc_start_mwh": float(result.arrays["energy_mwh"][0, local_t]),
                "soc_end_mwh": float(result.arrays["energy_mwh"][0, local_t + 1]),
                "optimizer_cutoff_utc": event.at.isoformat()}
            # Retain the complete deterministic trading plan, including fixed
            # earlier commitments, so frozen-plan replay never borrows later bids.
            for planned_event in schedule:
                planned_auction = planned_event.auction
                if planned_auction in inputs.tree.bid_nodes:
                    planned_owner = inputs.tree.bid_nodes[planned_auction][0]
                    quantity = result.trades.get((planned_auction, planned_owner, product), 0.0)
                else:
                    quantity = submitted.get(planned_auction, {}).get(product, 0.0)
                row[f"planned_trade_{planned_auction}_mwh"] = float(quantity)
            snapshot.append(row)
            control_plan[t] = (row["charge_mw"], row["discharge_mw"], row["curtailment_mw"])
            control_plan_details[t] = row
        plan_snapshots[auction] = snapshot

    def publish_results(event: Event) -> None:
        nonlocal market
        auction = event.payload.auction
        if auction not in submitted:
            raise RuntimeError("Results event has no previously submitted offer")
        if auction in known:
            raise RuntimeError("Auction publication cannot book a committed quantity twice")
        previous_cash = market.cashflow_eur
        known[auction] = actual_prices[auction].copy()
        for product, quantity in submitted[auction].items():
            t = full_grid.interval_ids.index(product)
            market = market.accept(AcceptedTrade(auction, event.payload.submission_at.to_pydatetime(),
                event.at.to_pydatetime(), product, quantity, float(actual_prices[auction][t]),
                f"{auction}:{product}", transaction_fee_eur=fee_records[f"{auction}:{product}"]["transaction_fee_eur"]))
        # Publication settles fixed offers and updates information. The next
        # actionable auction/control decision uses it without a diagnostic solve.
        publications.append(dict(auction=auction, published_at=event.at.isoformat(),
            known_auctions=sorted(known),
            accepted_contractual_cashflow_eur=market.cashflow_eur,
            publication_booked_trades=len(submitted[auction]),
            publication_quantity_mwh=sum(submitted[auction].values()),
            publication_booked_transaction_fee_eur=0.0,
            publication_booked_cashflow_eur=market.cashflow_eur-previous_cash))

    def control(event: Event) -> None:
        t = event.payload
        if schedule and config.get("execution", {}).get("reoptimize_physical_controls", True):
            _, result = plan(event.at)
            pending[t] = tuple(max(0.0, float(result.arrays[name][0,0]))
                            for name in ("charge_mw","discharge_mw","curtailment_mw"))
        elif t in control_plan:
            pending[t] = control_plan[t]
        elif t < first_participation_interval:
            # No market position exists yet. Physical grid feedback stores PV
            # within the SOC box and curtails the rest, with zero net delivery.
            pending[t] = (0.0, 0.0, 0.0)
            projected, _ = execute_interval(state=state, start=full_grid.starts[t], end=full_grid.ends[t],
                pv_ac_mw=float(current_forecast[t]), requested_charge_mw=0.0,
                requested_discharge_mw=0.0, requested_curtailment_mw=0.0, plant=plant,
                degradation_settings=aging, dispatch_policy="track_committed_grid", grid_target_mw=0.0)
            control_plan_details[t] = {"auction": "pre_participation",
                "optimizer_cutoff_utc": None, "soc_end_mwh": projected.energy_mwh,
                "pv_ac_forecast_mw": float(current_forecast[t]), **pv_forecast_metadata}
        else:
            raise ValueError(f"No auction optimization plan covers delivery interval {full_grid.interval_ids[t]}")
        # Submitted offers are fixed commitments even while their prices are unknown.
        position = sum(offer.get(full_grid.interval_ids[t], 0.0) for offer in submitted.values())
        grid_targets[t] = position / full_grid.delta_hours[t]
        execution_details[t] = dict(control_plan_details.get(t, {}))

    def observation(event: Event) -> None:
        nonlocal state
        t = event.payload
        charge, discharge, curtailment = pending[t]
        realized_pv = fixture.get("realized_pv", fixture["pv"])
        detail = execution_details[t]
        interval_policy = "track_committed_grid" if detail.get("auction") == "pre_participation" else dispatch_policy
        state, record = execute_interval(state=state, start=full_grid.starts[t], end=full_grid.ends[t],
            pv_ac_mw=float(realized_pv[t]), requested_charge_mw=charge,
            requested_discharge_mw=discharge, requested_curtailment_mw=curtailment,
            plant=plant, degradation_settings=aging, dispatch_policy=interval_policy,
            grid_target_mw=grid_targets[t])
        records[t] = {"interval_id": full_grid.interval_ids[t],"start":str(full_grid.starts[t]),
            "end":str(full_grid.ends[t]),"observation_available_at":event.at.isoformat(),**record,
            "active_plan_auction": detail.get("auction"),
            "plan_cutoff_utc": detail.get("optimizer_cutoff_utc"),
            "planned_energy_end_mwh": detail.get("soc_end_mwh"),
            "pv_ac_forecast_mw": detail.get("pv_ac_forecast_mw"),
            "pv_forecast_vintage_utc": detail.get("pv_forecast_vintage_utc"),
            "pv_forecast_available_at_utc": detail.get("pv_forecast_available_at_utc")}

    def settle(event: Event) -> None:
        t = event.payload
        record = records[t]
        position = float(market.positions(full_grid)[t])
        imbalance = record["grid_mw"]*full_grid.delta_hours[t]-position
        record.update(net_position_mwh=position,imbalance_mwh=imbalance,
            imbalance_cashflow_eur=imbalance*trading.get("imbalance_price_eur_per_mwh",
                config.get("synthetic", {}).get("imbalance_price_eur_per_mwh", 0.0)),
            imbalance_penalty_eur=(abs(imbalance)*trading["imbalance_penalty_eur_per_mwh"]
                if trading.get("optimize_imbalances", True) else 0.0),
            import_fee_eur=max(0,-record["grid_mw"])*full_grid.delta_hours[t]*trading["grid_import_fee_eur_per_mwh"])

    forecast_at = fixture.get("forecast_available_at",
        min([pd.Timestamp(full_grid.starts[0]), *(event.submission_at for event in schedule)])-pd.Timedelta(hours=1))
    events = [Event(pd.Timestamp(forecast_at), EventKind.FORECAST_PUBLICATION)]
    for event in schedule:
        events.extend([Event(event.submission_at,EventKind.AUCTION_SUBMISSION,payload=event),
                       Event(event.results_at,EventKind.AUCTION_RESULTS,payload=event)])
    for t,(start,end) in enumerate(zip(full_grid.starts,full_grid.ends)):
        events.extend([Event(pd.Timestamp(start),EventKind.PHYSICAL_CONTROL,t,t),
            Event(pd.Timestamp(end),EventKind.OBSERVATION_AVAILABLE,t,t),
            Event(pd.Timestamp(end),EventKind.SETTLEMENT,t,t)])
    engine = EventEngine(events)
    engine.run({EventKind.FORECAST_PUBLICATION:publish_forecast,EventKind.AUCTION_SUBMISSION:submit,
        EventKind.AUCTION_RESULTS:publish_results,EventKind.PHYSICAL_CONTROL:control,
        EventKind.OBSERVATION_AVAILABLE:observation,EventKind.SETTLEMENT:settle})
    frame = pd.DataFrame([records[t] for t in sorted(records)])
    # Recompute fees independently of the solver and compare with the executed
    # ledger and carried monthly state before publishing any profit metrics.
    checked_state, checked_records, fee_summary = account_trade_fees(fees, fees.initial_state,
        [asdict(trade) for trade in market.accepted_trades])
    reference = {row["trade_id"]: row for row in checked_records}
    fee_errors = {"ledger_fee_eur": max((abs(trade.transaction_fee_eur - reference[trade.trade_id]["transaction_fee_eur"])
        for trade in market.accepted_trades), default=0.0),
        "monthly_volume_mwh": max((abs(checked_state.monthly_volume_mwh.get(month, 0.0)
                - market.transaction_fee_state.monthly_volume_mwh.get(month, 0.0))
            for month in set(checked_state.monthly_volume_mwh) | set(market.transaction_fee_state.monthly_volume_mwh)), default=0.0),
        "cumulative_fee_eur": abs(checked_state.cumulative_fees_eur - market.transaction_fee_state.cumulative_fees_eur)}
    fee_summary["validation"] = {"valid": max(fee_errors.values()) <= 1e-6, "max_errors": fee_errors}
    if not fee_summary["validation"]["valid"]:
        raise RuntimeError(f"Independent executed transaction-fee accounting failed: {fee_errors}")
    for name in ("transaction_fee_eur", "gross_traded_mwh"):
        by_product = {}
        for row in checked_records:
            by_product[row["interval_id"]] = by_product.get(row["interval_id"], 0.0) + row[name]
        frame[name] = frame.interval_id.map(by_product).fillna(0.0)
    frame, realized_settlement = settle_realized_dispatch(frame, auction_revenue_eur=market.cashflow_eur,
        imbalance_prices=fixture.get("realized_imbalance_prices_eur_per_mwh"),
        imbalance_negative_prices=fixture.get("realized_imbalance_negative_prices_eur_per_mwh"),
        transaction_fees_eur=fee_summary["transaction_fee_eur"])
    frame.to_csv(destination/"dispatch.csv",index=False)
    plan_rows = [row for auction in [event.auction for event in schedule]
                 for row in plan_snapshots.get(auction, [])]
    plan_columns = ["auction", "interval_id", "optimizer_cutoff_utc", "interval_start_utc", "interval_end_utc",
        "pv_ac_forecast_mw", "pv_forecast_vintage_utc", "pv_forecast_available_at_utc", "charge_mw",
        "discharge_mw", "curtailment_mw", "expected_grid_mw", "soc_start_mwh", "soc_end_mwh"]
    pd.DataFrame(plan_rows, columns=None if plan_rows else plan_columns).to_csv(
        destination/"expected_dispatch_plans.csv", index=False)
    ledger_rows = [{**asdict(trade), **reference[trade.trade_id],
        "cashflow_eur":trade.cashflow_eur} for trade in market.accepted_trades]
    ledger_columns = [*AcceptedTrade.__dataclass_fields__, "cashflow_eur", "gross_traded_mwh", "purchased_mwh", "sold_mwh"]
    pd.DataFrame(ledger_rows, columns=None if ledger_rows else ledger_columns).to_csv(destination/"ledger.csv",index=False)
    write_json(destination/"event_trace.json",engine.trace)
    write_json(destination/"solves.json",solves)
    pd.DataFrame(forecast_input_snapshots, columns=None if forecast_input_snapshots else
        ["auction", "cutoff_utc", "interval_start_utc", "pv_ac_forecast_mw"]).to_csv(
        destination/"pv_forecast_inputs.csv", index=False)
    write_json(destination/"price_publications.json",publications)
    write_json(destination/"infeasible_auction_decisions.json",infeasible_decisions)
    auction_cash = {a:sum(trade.cashflow_eur for trade in market.accepted_trades if trade.auction==a)
                    for a in actual_prices}
    components = {"auction_revenue_eur":market.cashflow_eur,
        "transaction_fee_eur": fee_summary["transaction_fee_eur"],
        **{key:float(frame[key].sum()) for key in
            ("imbalance_cashflow_eur","imbalance_penalty_eur","degradation_eur","import_fee_eur")}}
    profit = components["auction_revenue_eur"]+components["imbalance_cashflow_eur"]-components["imbalance_penalty_eur"]-components["degradation_eur"]-components["import_fee_eur"]-components["transaction_fee_eur"]
    first_solve = solves[0] if solves else {}
    initial_expected_profit = (first_solve["objective_eur"]
        - first_solve["pre_horizon_degradation_eur"] - first_solve["pre_horizon_import_fee_eur"]) if solves else None
    summary = {"strategy":strategy,"executable":strategy!="oracle","components":components,
        "optimization_timing": optimization_timing_report(timing_records),
        "auction_dispatch_scope": AUCTION_DISPATCH_SCOPE,
        "auction_planning_mode": "current_auction_only" if current_only else "integrated",
        **realized_settlement,
        "transaction_fees": fee_summary,
        "initial_expected_transaction_fees_eur": first_solve.get("expected_transaction_fees_eur", 0.0),
        "realized_minus_expected_transaction_fees_eur": fee_summary["transaction_fee_eur"] - first_solve.get("expected_transaction_fees_eur", 0.0),
        "optimization_imbalances_enabled": trading.get("optimize_imbalances", True),
        "execution_dispatch_policy": dispatch_policy,
        "pv_forecast_policy": fixture.get("pv_forecast_policy", "fixed_fixture_forecast"),
        "max_feedback_adjustment_mw": float(frame.feedback_adjustment_mw.max()),
        "realized_operating_profit_status": "legacy_assumed_settlement_and_model_penalty_not_final_cash_profit",
        "auction_cashflows_eur":auction_cash,"realized_operating_profit_eur":profit,
        "initial_expected_profit_eur":initial_expected_profit,"final_state":asdict(state),
        "initial_residual_objective_eur": first_solve.get("objective_eur"),
        "initial_expected_prefix_cost_eur": (first_solve["pre_horizon_degradation_eur"] + first_solve["pre_horizon_import_fee_eur"]) if solves else None,
        "pre_participation_intervals": int(frame.active_plan_auction.eq("pre_participation").sum()),
        "terminal_constraint_enabled":config["state"]["terminal_target_mwh"] is not None,
        "terminal_error_mwh":(None if config["state"]["terminal_target_mwh"] is None
            else abs(state.energy_mwh-config["state"]["terminal_target_mwh"])),
        "max_safety_adjustment_mw":float(frame.safety_adjustment_mw.max()),
        "solves":len(solves)+len(infeasible_decisions),"runtime_seconds":perf_counter()-started,
        "auction_decision_solves":sum(item["solve_trigger"] == "auction_submission" for item in solves)+len(infeasible_decisions),
        "successful_auction_decision_solves":sum(item["solve_trigger"] == "auction_submission" for item in solves),
        "infeasible_auction_decision_count":len(infeasible_decisions),
        "infeasible_auction_decisions":infeasible_decisions,
        "infeasible_auction_policy":config.get("execution", {}).get("infeasible_auction_policy", "error"),
        "physical_control_solves":sum(item["solve_trigger"] == "physical_control" for item in solves),
        "price_publication_events":len(publications),
        "max_validation_error":max((max(item["validation"]["max_errors"].values()) for item in solves), default=0.0),
        "first_solve":first_solve or None,"artifacts":str(destination),
        "execution_assumptions":["pre-interval commands; final AC averages revealed at interval end",
            "idealized interval-average safety response", fixture.get("execution_pv_assumption", "optimization PV forecast reused as execution"),
            "retained commands and cutoff-available PV project battery state to the auction delivery boundary; no earlier commands are replaced",
            "independent delivery-day study with known delivery-start energy"]}
    summary.update(expected_realized_kpis(
        initial_expected_profit_eur=initial_expected_profit,
        initial_expected_auction_revenue_eur=first_solve.get("expected_auction_revenue_eur", 0.0),
        realized_profit_after_imbalance_eur=summary["realized_profit_after_imbalance_eur"],
        auction_revenue_eur=market.cashflow_eur))
    write_json(destination / "optimization_timing.json", summary["optimization_timing"])
    write_json(destination/"summary.json",summary)
    if config["outputs"].get("plots",True):
        from ..reporting.plots import plot_dispatch
        plot_dispatch(frame,actual_prices,summary,destination/"dispatch.png",
                      timezone=fixture["calendar"].timezone)
    return summary


def run_synthetic(config: dict[str, Any]) -> dict[str, Any]:
    """Run all configured strategies on the same synthetic realization."""
    fixture = synthetic_inputs(config)
    run_id = config["outputs"].get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path(config["outputs"]["root"])/run_id/f"farm_{fixture['plant'].farm_id}"/config["synthetic"]["delivery_day"]
    if destination.exists():
        raise FileExistsError(f"Output exists; choose a fresh run_id: {destination}")
    summaries = [_replay_strategy(config,fixture,strategy,destination/strategy)
                 for strategy in config["synthetic"]["strategies"]]
    timing = optimization_timing_report(row for summary in summaries
        for row in summary["optimization_timing"]["solves"])
    write_json(destination / "optimization_timing.json", timing)
    revision = subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip()
    kind=("synthetic stochastic-price / deterministic-PV" if len(fixture["scenarios"].scenario_ids)>1
          else "synthetic point-price / deterministic-PV")
    manifest = {"run_id":run_id,"kind":kind,
        "config":config,"code_revision":revision,"farm_id":fixture["plant"].farm_id,
        "delivery_day":config["synthetic"]["delivery_day"],"intervals":len(fixture["grid"].interval_ids),
        "scenario_count":len(fixture["scenarios"].scenario_ids),"tree_nodes":len(fixture["tree"].nodes),
        "probabilities":fixture["scenarios"].probabilities,"scenario_provenance":fixture["scenarios"].provenance,
        "actual_leaf":fixture["realized"],"boundary_state_assumption":config["state"],
        "raw_dataset_used":False,"strategies":summaries,"output_dir":str(destination)}
    write_json(destination/"manifest.json",manifest)
    return organize_run_outputs(destination)

"""Reproducible single-day synthetic auction replay and comparable policies."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
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
from ..markets import AuctionCalendar
from ..optimization import build_model, solve_model, validate_solution
from ..optimization.degradation import advance_degradation_state, degradation_settings_from_config
from ..scenarios import synthetic_price_tree
from .events import Event, EventEngine, EventKind
from .replay import execute_interval


def json_default(value: Any) -> Any:
    """Serialize evidence arrays and clocks without serializing private objects."""
    if isinstance(value, (datetime, pd.Timestamp, Path)):
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
    state = PhysicalState(full_grid.starts[0], config["state"]["initial_energy_mwh"], origin=config["state"]["origin"],
        cycle_segment_energy_mwh=config["state"].get("cycle_segment_energy_mwh"))
    market = MarketState()
    known: dict[str, np.ndarray] = {}
    submitted: dict[str, dict[str, float]] = {}
    pending: dict[int, tuple[float, float, float]] = {}
    records: dict[int, dict[str, Any]] = {}
    solves: list[dict[str, Any]] = []
    publications: list[dict[str, Any]] = []
    plan_snapshots: dict[str, list[dict[str, Any]]] = {}
    control_plan: dict[int, tuple[float, float, float]] = {}
    current_forecast: np.ndarray | None = None
    aging = degradation_settings_from_config(config)
    settings = SolverSettings(**config["solver"])
    trading = config["trading"]

    def plan(cutoff: pd.Timestamp, current_auction: str | None = None, *,
             trigger: str = "physical_control", published_auction: str | None = None):
        """Use published information and a known boundary for committed commands."""
        if current_forecast is None:
            raise ValueError("No published PV forecast at the decision cutoff")
        first = int(pd.DatetimeIndex(full_grid.starts).searchsorted(cutoff))
        if cutoff < pd.Timestamp(full_grid.starts[0]):
            first = 0
        if first >= len(full_grid.interval_ids):
            raise ValueError("No remaining delivery intervals")
        boundary = state
        # Between grid points the current command is already committed. Its battery
        # trajectory provides the next boundary without reading final PV telemetry.
        if state.timestamp != full_grid.starts[first]:
            prior = first - 1
            charge, discharge, _ = pending[prior]
            duration = full_grid.delta_hours[prior]
            battery = plant.battery
            energy = state.energy_mwh * (1-battery.self_discharge_per_hour)**duration
            energy += duration*(battery.eta_charge*charge-discharge/battery.eta_discharge)
            boundary, _ = advance_degradation_state(state=state, battery=battery,
                settings=aging, end=full_grid.starts[first], energy_mwh=energy,
                charge_mw=charge, discharge_mw=discharge,
                origin="projected_boundary_of_committed_command")
        residual = DeliveryGrid(full_grid.interval_ids[first:], full_grid.starts[first:], full_grid.ends[first:])
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
        if strategy == "deterministic_sequential":
            selected_schedule = tuple(event for event in schedule if event.auction == current_auction)
        if strategy == "oracle":
            kwargs["branch_values"] = (0.0,)
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
            trading["trade_limit_mwh"], trading["position_limit_mwh"],
            trading["imbalance_penalty_eur_per_mwh"], config["state"]["terminal_target_mwh"],
            aging, trading["grid_import_fee_eur_per_mwh"],
            imbalance_limit_mwh=trading.get("imbalance_limit_mwh"),
            planned_delivery_balance_resolution_minutes=trading.get("planned_delivery_balance_resolution_minutes"),
            pending_position_mwh=pending_positions,
            decision_resolution_minutes=config.get("decision_resolution_minutes"),
            market_timezone=config.get("market", {}).get("timezone", fixture["calendar"].timezone),
            native_interval_minutes_by_auction={auction: minutes for auction, minutes
                in config.get("native_interval_minutes_by_auction", {}).items()
                if auction in masks})
        built = build_model(inputs)
        try:
            model_settings = config["solver"]
            model_file = None
            if config.get("outputs", {}).get("save_model_files", False):
                model_directory = destination / "models"
                model_directory.mkdir(parents=True, exist_ok=True)
                model_file = model_directory / f"model_{len(solves) + 1:03d}_{cutoff.strftime('%Y%m%dT%H%M%SZ')}_{trigger}.lp"
                model_settings = {**model_settings, "export_path": str(model_file)}
            result = solve_model(built, SolverSettings(**model_settings))
        finally:
            built.model.dispose()
        if not result.has_incumbent:
            raise RuntimeError(f"{strategy} at {cutoff}: {result.status}: {result.message}")
        validation = validate_solution(inputs, result)
        if not validation.valid:
            raise RuntimeError(f"Independent solution validation failed: {validation.violations}")
        solves.append({"cutoff":cutoff.isoformat(), "current_auction":current_auction,
            "solve_trigger":trigger, "published_auction":published_auction,
            "model_file":str(model_file) if model_file is not None else None,
            "known_auctions":list(known),
            "known_prices_fixed":{auction:bool(np.all(bundle.auction_prices_eur_per_mwh[auction] == curve[first:]))
                                  for auction,curve in known.items()},
            "state_origin":boundary.origin,"state_timestamp":str(boundary.timestamp),
            "pending_position_abs_mwh":sum(abs(value) for value in pending_positions),
            "status":result.status,"objective_eur":result.objective_eur,
            "validation":asdict(validation),**result.diagnostics})
        return inputs, result

    def publish_forecast(event: Event) -> None:
        nonlocal current_forecast
        current_forecast = fixture["pv"].copy()

    def submit(event: Event) -> None:
        auction = event.payload.auction
        inputs, result = plan(event.at, auction, trigger="auction_submission")
        owner = inputs.tree.bid_nodes[auction][0]
        if len(set(inputs.tree.bid_nodes[auction])) != 1:
            raise RuntimeError("Actionable bid is not shared at the current information cutoff")
        submitted[auction] = {product:quantity for (a,node,product),quantity in result.trades.items()
                              if a==auction and node==owner}
        snapshot = []
        for local_t, product in enumerate(inputs.grid.interval_ids):
            t = full_grid.interval_ids.index(product)
            owner = inputs.tree.bid_nodes[auction][0]
            trade_mwh = float(result.trades.get((auction, owner, product), 0.0))
            row = {"auction": auction, "interval_id": product,
                "interval_start_utc": str(full_grid.starts[t]),
                "charge_mw": float(result.arrays["charge_mw"][0, local_t]),
                "discharge_mw": float(result.arrays["discharge_mw"][0, local_t]),
                "net_battery_dispatch_mw": float(result.arrays["discharge_mw"][0, local_t]
                    - result.arrays["charge_mw"][0, local_t]),
                "pv_ac_forecast_mw": float(inputs.scenarios.pv_ac_mw[0, local_t]),
                "expected_grid_mw": float(result.arrays["grid_mw"][0, local_t]),
                "auction_trade_mwh": trade_mwh,
                "curtailment_mw": float(result.arrays["curtailment_mw"][0, local_t]),
                "soc_start_mwh": float(result.arrays["energy_mwh"][0, local_t]),
                "soc_end_mwh": float(result.arrays["energy_mwh"][0, local_t + 1]),
                "optimizer_cutoff_utc": event.at.isoformat()}
            snapshot.append(row)
            control_plan[t] = (row["charge_mw"], row["discharge_mw"], row["curtailment_mw"])
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
                f"{auction}:{product}"))
        # Publication settles fixed offers and updates information. The next
        # actionable auction/control decision uses it without a diagnostic solve.
        publications.append(dict(auction=auction, published_at=event.at.isoformat(),
            known_auctions=sorted(known),
            accepted_contractual_cashflow_eur=market.cashflow_eur,
            publication_booked_trades=len(submitted[auction]),
            publication_quantity_mwh=sum(submitted[auction].values()),
            publication_booked_cashflow_eur=market.cashflow_eur-previous_cash))

    def control(event: Event) -> None:
        t = event.payload
        if config.get("execution", {}).get("reoptimize_physical_controls", True):
            _, result = plan(event.at)
            pending[t] = tuple(float(result.arrays[name][0,0])
                            for name in ("charge_mw","discharge_mw","curtailment_mw"))
        elif t in control_plan:
            pending[t] = control_plan[t]
        else:
            raise ValueError(f"No auction optimization plan covers delivery interval {full_grid.interval_ids[t]}")

    def observation(event: Event) -> None:
        nonlocal state
        t = event.payload
        charge, discharge, curtailment = pending[t]
        realized_pv = fixture.get("realized_pv", fixture["pv"])
        state, record = execute_interval(state=state, start=full_grid.starts[t], end=full_grid.ends[t],
            pv_ac_mw=float(realized_pv[t]), requested_charge_mw=charge,
            requested_discharge_mw=discharge, requested_curtailment_mw=curtailment,
            plant=plant, degradation_settings=aging)
        records[t] = {"interval_id": full_grid.interval_ids[t],"start":str(full_grid.starts[t]),
            "end":str(full_grid.ends[t]),"observation_available_at":event.at.isoformat(),**record}

    def settle(event: Event) -> None:
        t = event.payload
        record = records[t]
        position = float(market.positions(full_grid)[t])
        imbalance = record["grid_mw"]*full_grid.delta_hours[t]-position
        record.update(net_position_mwh=position,imbalance_mwh=imbalance,
            imbalance_cashflow_eur=imbalance*trading.get("imbalance_price_eur_per_mwh",
                config.get("synthetic", {}).get("imbalance_price_eur_per_mwh", 0.0)),
            imbalance_penalty_eur=abs(imbalance)*trading["imbalance_penalty_eur_per_mwh"],
            import_fee_eur=max(0,-record["grid_mw"])*full_grid.delta_hours[t]*trading["grid_import_fee_eur_per_mwh"])

    forecast_at = fixture.get("forecast_available_at", schedule[0].submission_at-pd.Timedelta(hours=1))
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
    frame.to_csv(destination/"dispatch.csv",index=False)
    plan_rows = [row for auction in [event.auction for event in schedule]
                 for row in plan_snapshots.get(auction, [])]
    if plan_rows:
        pd.DataFrame(plan_rows).to_csv(destination/"expected_dispatch_plans.csv", index=False)
    pd.DataFrame([{**asdict(trade),"cashflow_eur":trade.cashflow_eur} for trade in market.accepted_trades]).to_csv(destination/"ledger.csv",index=False)
    write_json(destination/"event_trace.json",engine.trace)
    write_json(destination/"solves.json",solves)
    write_json(destination/"price_publications.json",publications)
    auction_cash = {a:sum(trade.cashflow_eur for trade in market.accepted_trades if trade.auction==a)
                    for a in actual_prices}
    components = {"auction_revenue_eur":market.cashflow_eur,
        **{key:float(frame[key].sum()) for key in
            ("imbalance_cashflow_eur","imbalance_penalty_eur","degradation_eur","import_fee_eur")}}
    profit = components["auction_revenue_eur"]+components["imbalance_cashflow_eur"]-components["imbalance_penalty_eur"]-components["degradation_eur"]-components["import_fee_eur"]
    summary = {"strategy":strategy,"executable":strategy!="oracle","components":components,
        "auction_cashflows_eur":auction_cash,"realized_operating_profit_eur":profit,
        "initial_expected_profit_eur":solves[0]["objective_eur"],"final_state":asdict(state),
        "terminal_constraint_enabled":config["state"]["terminal_target_mwh"] is not None,
        "terminal_error_mwh":(None if config["state"]["terminal_target_mwh"] is None
            else abs(state.energy_mwh-config["state"]["terminal_target_mwh"])),
        "max_safety_adjustment_mw":float(frame.safety_adjustment_mw.max()),
        "solves":len(solves),"runtime_seconds":perf_counter()-started,
        "auction_decision_solves":sum(item["solve_trigger"] == "auction_submission" for item in solves),
        "physical_control_solves":sum(item["solve_trigger"] == "physical_control" for item in solves),
        "price_publication_events":len(publications),
        "max_validation_error":max(max(item["validation"]["max_errors"].values()) for item in solves),
        "first_solve":solves[0],"artifacts":str(destination),
        "execution_assumptions":["pre-interval commands; final AC averages revealed at interval end",
            "idealized interval-average safety response", fixture.get("execution_pv_assumption", "optimization PV forecast reused as execution"),
            "committed commands project battery state to next boundary for intrainterval auction cutoffs",
            "independent delivery-day study with known delivery-start energy"]}
    write_json(destination/"summary.json",summary)
    if config["outputs"].get("plots",True):
        from ..reporting.plots import plot_dispatch
        plot_dispatch(frame,actual_prices,summary,destination/"dispatch.png")
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
    return manifest

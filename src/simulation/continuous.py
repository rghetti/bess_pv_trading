"""Causal continuous execution with one delivery-day optimization at each gate."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd

from ..config import validate_continuous_optimization
from ..data import config_path
from ..domain import AcceptedTrade, DeliveryGrid, MarketState, OptimizationInput, PhysicalState, SolverSettings
from ..optimization import build_model, solve_model, validate_solution
from ..optimization.degradation import degradation_settings_from_config
from ..reporting.optimization_timing import optimization_timing_report, solve_timing
from ..reporting.artifacts import prepare_experiment_output
from ..scenarios import synthetic_price_tree
from ..transaction_fees import TransactionFeeState, account_trade_fees, transaction_fees_from_config
from .continuous_inputs import prepare_continuous_inputs
from .events import Event, EventEngine, EventKind
from .replay import execute_interval
from .synthetic import json_default, write_json


CONTINUOUS_POLICY_VERSION = "continuous-retained-plan-v2"


@dataclass(frozen=True)
class ContinuousAuction:
    """Auction names repeat across days; each event belongs to a delivery book."""
    delivery_day: str
    event: object


def _atomic_json(path, value):
    """Replace a complete checkpoint atomically on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, default=json_default, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _configuration_fingerprint(config):
    """Resume changes no physical, financial or input-selection assumptions."""
    comparable = deepcopy(config)
    comparable.get("continuous_optimization", {}).pop("resume", None)
    return hashlib.sha256(json.dumps(comparable, sort_keys=True, default=json_default).encode()).hexdigest()


def _physical_state(value):
    """Restore immutable physical state, including optional Xu segment history."""
    values = dict(value)
    values["timestamp"] = pd.Timestamp(values["timestamp"]).to_pydatetime()
    if values.get("cycle_segment_energy_mwh") is not None:
        values["cycle_segment_energy_mwh"] = tuple(values["cycle_segment_energy_mwh"])
    return PhysicalState(**values)


def _accepted_trade(value):
    """Restore contractual clocks without changing quantity or price."""
    values = {key: value[key] for key in AcceptedTrade.__dataclass_fields__}
    for key in ("submission_time", "publication_time"):
        values[key] = pd.Timestamp(values[key]).to_pydatetime()
    return AcceptedTrade(**values)


class ContinuousReplay:
    """One actual state and ledger, with date-scoped plans and shadow projections."""

    def __init__(self, config, prepared, strategy, destination, resume=False, progress=None):
        if strategy not in {"integrated", "current_auction_only", "perfect_pv"}:
            raise ValueError(f"Unsupported continuous strategy {strategy}")
        self.config, self.prepared, self.strategy = config, prepared, strategy
        self.progress = progress
        self.destination = Path(destination)
        self.current_only = strategy == "current_auction_only"
        self.perfect = strategy == "perfect_pv"
        self.perfect_reoptimizes = config["continuous_optimization"].get(
            "perfect_pv_reoptimize_at_each_auction", False)
        self.grid, self.plant = prepared["full_grid"], prepared["plant"]
        self.days, self.daily = tuple(prepared["days"]), prepared["days_data"]
        self.starts = pd.DatetimeIndex(self.grid.starts)
        self.index = {product: index for index, product in enumerate(self.grid.interval_ids)}
        self.day_for_product = {product: day for day in self.days for product in self.daily[day]["grid"].interval_ids}
        self.day_offsets = {day: self.index[self.daily[day]["grid"].interval_ids[0]] for day in self.days}
        self.fees = transaction_fees_from_config(config)
        self.aging = degradation_settings_from_config(config)
        self.trading = config["trading"]
        self.dispatch_policy = config["execution"].get("dispatch_policy", "track_committed_grid")
        self.config_fingerprint = _configuration_fingerprint(config)
        self.identity = {"policy_version": CONTINUOUS_POLICY_VERSION, "strategy": strategy,
            "config_fingerprint": self.config_fingerprint, "input_fingerprint": prepared["input_fingerprint"]}
        if self.destination.exists() and not resume:
            raise FileExistsError(f"Continuous strategy output exists: {self.destination}")
        self.destination.mkdir(parents=True, exist_ok=resume)
        for kind in ("json", "csv", "models"):
            (self.destination / kind).mkdir(exist_ok=True)
        metadata = self.destination / "json/replay_identity.json"
        if metadata.exists() and json.loads(metadata.read_text()) != self.identity:
            raise ValueError("Cannot resume continuous replay with different configuration, inputs or policy")
        if resume and any(self.destination.iterdir()) and not metadata.exists():
            # Empty type folders are created above, so require identity only if
            # actual artifacts from an earlier invocation are present.
            if any(path.is_file() for path in self.destination.rglob("*")):
                raise ValueError("Continuous output has no validated resume identity")
        _atomic_json(metadata, self.identity)
        self.state = PhysicalState(self.grid.starts[0], prepared["initial_energy_mwh"],
            origin="known_delivery_start_assumption",
            cycle_segment_energy_mwh=config["state"].get("cycle_segment_energy_mwh"))
        self.market = MarketState(transaction_fee_state=self.fees.initial_state)
        self.submitted = {day: {} for day in self.days}
        self.known = {day: [] for day in self.days}
        self.fee_records, self.control_plan, self.pending, self.records = {}, {}, {}, {}
        self.plan_snapshots = {day: {} for day in self.days}
        self.solves, self.projections, self.event_trace, self.publications = [], [], [], []
        self.forecast_inputs, self.initial_expectations, self.initial_revenues = [], {}, {}
        self.cursor = None
        checkpoint = self.destination / "json/checkpoint.json"
        if resume and checkpoint.is_file():
            self._restore(json.loads(checkpoint.read_text()))
        attempt_path = self.destination / "json/optimization_attempts.json"
        self.attempts = json.loads(attempt_path.read_text()) if resume and attempt_path.exists() else []
        active_attempts = {row.get("attempt_id") for row in self.solves}
        for row in self.attempts:
            if row.get("attempt_id") not in active_attempts:
                row["rolled_back_for_resume"] = True
        self.events = self._events()

    def _events(self):
        """Interleave current-day IDA3, next-day trading and physical execution."""
        events = []
        for day in self.days:
            for auction in self.daily[day]["schedule"]:
                instance = ContinuousAuction(day, auction)
                events.append(Event(auction.submission_at, EventKind.AUCTION_SUBMISSION, len(events), instance))
                events.append(Event(auction.results_at, EventKind.AUCTION_RESULTS, len(events), instance))
        for index, (start, end) in enumerate(zip(self.grid.starts, self.grid.ends)):
            events.append(Event(pd.Timestamp(start), EventKind.PHYSICAL_CONTROL, len(events), index))
            events.append(Event(pd.Timestamp(end), EventKind.OBSERVATION_AVAILABLE, len(events), index))
            events.append(Event(pd.Timestamp(end), EventKind.SETTLEMENT, len(events), index))
        return sorted(events)

    @staticmethod
    def _event_key(event):
        return [event.at.isoformat(), int(event.kind), event.sequence]

    def _restore(self, checkpoint):
        """Restore actual and outstanding state together; projections stay virtual."""
        if checkpoint["identity"] != self.identity:
            raise ValueError("Continuous checkpoint configuration/input fingerprint mismatch")
        self.state = _physical_state(checkpoint["state"])
        fees = TransactionFeeState(**checkpoint["transaction_fee_state"])
        self.market = MarketState(tuple(_accepted_trade(row) for row in checkpoint["accepted_trades"]), fees)
        for name in ("submitted", "known", "fee_records", "control_plan", "pending", "records",
                     "plan_snapshots", "solves", "projections", "event_trace", "publications",
                     "forecast_inputs", "initial_expectations", "initial_revenues"):
            setattr(self, name, checkpoint[name])
        self.cursor = checkpoint["event_cursor"]

    def save_checkpoint(self, event):
        """Commit a midnight after observation, next control and settlement events."""
        checkpoint = {"identity": self.identity, "state": asdict(self.state),
            "accepted_trades": [asdict(trade) for trade in self.market.accepted_trades],
            "transaction_fee_state": asdict(self.market.transaction_fee_state),
            "event_cursor": self._event_key(event)}
        checkpoint.update({name: getattr(self, name) for name in (
            "submitted", "known", "fee_records", "control_plan", "pending", "records", "plan_snapshots",
            "solves", "projections", "event_trace", "publications", "forecast_inputs",
            "initial_expectations", "initial_revenues")})
        _atomic_json(self.destination / "json/checkpoint.json", checkpoint)

    def _position(self, product):
        """Unpriced submitted quantities are already binding commitments."""
        day = self.day_for_product[product]
        return sum(offer.get(product, 0.0) for offer in self.submitted[day].values())

    def _commands(self, product):
        """Return the retained physical policy and immutable grid target."""
        index = self.index[product]
        row = self.control_plan.get(product)
        if row is None:
            day = self.day_for_product[product]
            local = index - self.day_offsets[day]
            row = {"auction": "pre_participation", "pv_ac_forecast_mw": float(
                self.prepared["full_realized_pv"][index] if self.perfect else self.daily[day]["initial_forecast"][local])}
            commands = (0.0, 0.0, 0.0)
        else:
            commands = tuple(float(row[name]) for name in ("charge_mw", "discharge_mw", "curtailment_mw"))
        policy = "track_committed_grid" if row["auction"] == "pre_participation" else self.dispatch_policy
        return commands, row, policy, self._position(product) / self.grid.delta_hours[index]

    def _forecast(self, cutoff, starts, auction):
        """Operational trajectories never inspect future measured production."""
        if self.perfect:
            indices = self.starts.get_indexer(starts)
            return np.asarray(self.prepared["full_realized_pv"])[indices].copy(), {
                "pv_forecast_policy": "perfect_pv_reference",
                "pv_forecast_vintage_utc": None, "pv_forecast_available_at_utc": None}
        bundle = self.prepared["forecast_provider"](cutoff, starts)
        if bundle.available_at > cutoff or bundle.forecast_vintage > cutoff:
            raise ValueError("Continuous forecast is unavailable at the submission cutoff")
        if bundle.farm_id != self.plant.farm_id or len(bundle.members) != 1:
            raise ValueError("Continuous forecasting requires this farm and one deterministic member")
        if not pd.DatetimeIndex(bundle.values.index).equals(starts):
            raise ValueError("Continuous forecast does not cover the requested projection/planning grid")
        values = bundle.values["pv_ac_mw"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Continuous PV forecast must be finite nonnegative AC power")
        model_settings = self.config["pv"].get("forecast_replacement", {})
        model = model_settings.get("models_by_farm", {}).get(str(self.plant.farm_id),
            model_settings.get("models_by_farm", {}).get(self.plant.farm_id))
        model_provenance = {"pv_forecast_model": "pvlib", "pv_forecast_model_status": "baseline"}
        if model_settings.get("enabled") and model is not None:
            artifact = config_path(self.config, model["artifact_path"])
            available_at = pd.Timestamp(model["available_at_utc"])
            if available_at.tzinfo is None:
                raise ValueError("PV replacement artifact availability must be timezone-aware")
            retrospective = available_at.tz_convert("UTC") > pd.Timestamp(cutoff).tz_convert("UTC")
            if not artifact.is_file():
                model_provenance["pv_forecast_model_status"] = "fallback_artifact_missing"
            elif retrospective and not model_settings.get("allow_retrospective_artifacts", False):
                model_provenance["pv_forecast_model_status"] = "fallback_artifact_not_yet_available"
            else:
                from ..forecasting.pv_neural_data import feature_frame, transform_features
                model_frame = bundle.values.copy(deep=True)
                model_frame["interval_start_utc"] = starts
                local = starts.tz_convert(self.prepared["calendar"].timezone)
                model_frame["delivery_day"] = local.strftime("%Y-%m-%d")
                model_frame["auction"] = str(auction)
                model_frame["forecast_cycle_utc"] = pd.Timestamp(bundle.forecast_vintage)
                if "lead_hours" not in model_frame:
                    midpoint=starts+pd.Timedelta(minutes=float(self.config['source']['historical_interval_minutes'])/2)
                    model_frame["lead_hours"] = ((midpoint-pd.Timestamp(bundle.forecast_vintage)).total_seconds()/3600)
                model_frame["daylight"] = (~model_frame.get("nighttime_zero", pd.Series(
                    model_frame["solar_zenith_deg"].to_numpy() >= 90, index=model_frame.index)).astype(bool))
                model_frame["measured_ac_mw"] = np.nan
                model_frame["eligible"] = False
                model_frame["split"] = "prediction"
                if model["kind"] == "weather_bundle":
                    from ..forecasting.pv_forecast_bundle import predict_bundle
                    replacement = predict_bundle(model_frame,artifact.parent,farm_id=self.plant.farm_id)
                elif model["kind"] == "neural":
                    from ..forecasting.pv_neural_predict import predict_neural_frame
                    replacement = predict_neural_frame(model_frame, artifact, farm_id=self.plant.farm_id)
                elif model["kind"] == "linear":
                    details = json.loads(artifact.read_text())
                    selected = details["selected"][model["family"]]
                    preprocessor_path = config_path(self.config, model["preprocessors_path"])
                    preprocessors = json.loads(preprocessor_path.read_text())
                    features = feature_frame(model_frame, selected["features"],
                        self.config["pv_neural"]["delivery_timezone"])
                    x = transform_features(features, preprocessors[selected["features"]])
                    rating = float(model["ac_capacity_mw"])
                    if rating <= 0:
                        raise ValueError("Linear PV replacement requires a positive configured AC rating")
                    replacement = np.clip(x @ np.asarray(selected["coefficients"]) +
                        float(selected["intercept"]), 0, 1) * rating
                else:
                    raise ValueError(f"Unknown PV forecast replacement kind: {model['kind']}")
                replacement = np.asarray(replacement, dtype=float)
                if replacement.shape != values.shape or not np.isfinite(replacement).all() or (replacement < 0).any():
                    raise ValueError("PV replacement must return finite nonnegative AC power on the requested grid")
                if "nighttime_zero" in model_frame:
                    replacement[model_frame["nighttime_zero"].to_numpy(dtype=bool)] = 0.0
                bundle.values.loc[:, "pv_ac_mw"] = replacement
                values = replacement
                model_provenance = {"pv_forecast_model": model["name"],
                    "pv_forecast_model_status": "retrospective_artifact" if retrospective else "artifact",
                    "pv_forecast_model_kind": model["kind"], "pv_forecast_model_artifact": str(artifact),
                    "pv_forecast_model_available_at_utc": available_at.isoformat(),
                    "pv_forecast_model_retrospective": bool(retrospective)}
        return values, {"pv_forecast_policy": "latest_available_forecast_at_each_reoptimization",
            "pv_forecast_vintage_utc": bundle.forecast_vintage.isoformat(),
            "pv_forecast_available_at_utc": bundle.available_at.isoformat(), **model_provenance}

    def _planning_inputs(self, instance, cutoff):
        """Project retained operation, then construct only the target day's MILP."""
        day, event = instance.delivery_day, instance.event
        daily, offset = self.daily[day], self.day_offsets[day]
        day_grid = daily["grid"]
        first = max(min(event.eligible_intervals), int(pd.DatetimeIndex(day_grid.starts).searchsorted(cutoff)))
        first_global, stop_global = offset + first, offset + len(day_grid.interval_ids)
        update_first = int(self.starts.searchsorted(self.state.timestamp))
        if self.state.timestamp > cutoff and not (day == self.days[0] and cutoff < pd.Timestamp(self.grid.starts[0])):
            raise ValueError("Actual state lies beyond the auction information cutoff")
        update_starts = self.starts[update_first:stop_global]
        forecast, provenance = self._forecast(cutoff, update_starts, event.auction)
        by_product = dict(zip(self.grid.interval_ids[update_first:stop_global], map(float, forecast)))
        self.forecast_inputs.extend({"delivery_day": day, "auction": event.auction,
            "cutoff_utc": cutoff.isoformat(), "interval_start_utc": str(start),
            "pv_ac_forecast_mw": value, "forecast_role": "locked_projection" if self.index[product] < first_global else "target_delivery",
            **provenance} for product, start, value in zip(
                self.grid.interval_ids[update_first:stop_global], update_starts, forecast))
        boundary, projection = self.state, []
        for index in range(update_first, first_global):
            product = self.grid.interval_ids[index]
            if product in self.pending:
                pending = self.pending[product]
                commands, row, policy, target = tuple(pending["commands"]), pending["detail"], pending["policy"], pending["target"]
            else:
                commands, row, policy, target = self._commands(product)
            boundary, record = execute_interval(state=boundary, start=self.grid.starts[index],
                end=self.grid.ends[index], pv_ac_mw=by_product[product],
                requested_charge_mw=commands[0], requested_discharge_mw=commands[1],
                requested_curtailment_mw=commands[2], plant=self.plant,
                degradation_settings=self.aging, dispatch_policy=policy, grid_target_mw=target)
            projection.append({"interval_id": product, "start": str(self.grid.starts[index]),
                "end": str(self.grid.ends[index]), "plan_source_auction": row.get("auction"),
                "plan_source_delivery_day": self.day_for_product[product], **record})
        if projection:
            boundary = replace(boundary, origin="forecast_projected_continuous_boundary")
        grid = DeliveryGrid(day_grid.interval_ids[first:], day_grid.starts[first:], day_grid.ends[first:])
        schedule = tuple(daily["schedule"])
        prices = daily["prices"]
        if self.current_only:
            selected = tuple(item for item in schedule if item.auction == event.auction)
            base, spreads, known = prices[event.auction][first:], {}, {}
        else:
            selected = schedule
            names = [item.auction for item in schedule]
            base = prices[names[0]][first:]
            spreads = {current: (prices[current] - prices[previous])[first:]
                for previous, current in zip(names, names[1:])}
            known = {auction: prices[auction][first:] for auction in self.known[day]}
        pv = np.asarray([by_product[product] for product in grid.interval_ids])
        bundle, tree = synthetic_price_tree(grid, pv, selected, base_prices=base,
            stage_spreads=spreads, known_prices=known, residual_scale=0.0,
            branch_values=(0.0,), branch_probabilities=(1.0,), decision_cutoff=cutoff,
            imbalance_price=float(self.trading.get("imbalance_price_eur_per_mwh", 0.0)),
            seed=int(self.config["solver"].get("seed", 0)))
        masks = {auction: tuple(first + index in next(item.eligible_intervals for item in schedule
            if item.auction == auction) for index in range(len(grid.interval_ids))) for auction in tree.bid_nodes}
        target_products = set(day_grid.interval_ids)
        market = MarketState(tuple(trade for trade in self.market.accepted_trades if trade.interval_id in target_products),
            self.market.transaction_fee_state)
        pending_positions = tuple(sum(offer.get(product, 0.0) for auction, offer in self.submitted[day].items()
            if auction not in self.known[day]) for product in grid.interval_ids)
        terminal = self.prepared["terminal_energy_mwh"] if day == self.days[-1] else None
        terminal_minimum = self.prepared["terminal_minimum_energy_mwh"]
        native = daily.get("native_interval_minutes_by_auction")
        if native is None:
            from ..data.gme import native_price_resolution_minutes
            native = {auction: native_price_resolution_minutes(
                self.config["gme_oracle_experiment"]["auction_datasets"][auction]["dataset"],
                pd.Timestamp(day).date()) for auction in masks}
        inputs = OptimizationInput(grid=grid, plant=self.plant, state=boundary, market=market,
            scenarios=bundle, tree=tree, auction_eligibility=masks,
            imbalance_penalty_eur_per_mwh=self.trading["imbalance_penalty_eur_per_mwh"],
            terminal_target_mwh=terminal, terminal_minimum_mwh=terminal_minimum,
            degradation=self.aging,
            grid_import_fee_eur_per_mwh=self.trading["grid_import_fee_eur_per_mwh"],
            optimize_imbalances=self.trading.get("optimize_imbalances", True),
            imbalance_limit_mwh=self.trading.get("imbalance_limit_mwh"),
            planned_delivery_balance_resolution_minutes=self.trading.get("planned_delivery_balance_resolution_minutes"),
            pending_position_mwh=pending_positions,
            decision_resolution_minutes=int(self.config["gme_oracle_experiment"]["decision_resolution_minutes"]),
            market_timezone=self.prepared["calendar"].timezone,
            native_interval_minutes_by_auction={auction: native[auction] for auction in masks},
            transaction_fees=self.fees,
            auction_submission_times={item.auction: item.submission_at for item in schedule})
        diagnostic = {"delivery_day": day, "current_auction": event.auction, "cutoff": cutoff.isoformat(),
            "cutoff_utc": cutoff.isoformat(), "strategy": self.strategy,
            "auction_planning_mode": "current_auction_only" if self.current_only else "integrated",
            "observed_state_timestamp": str(self.state.timestamp), "observed_energy_mwh": self.state.energy_mwh,
            "observed_state_origin": self.state.origin,
            "initial_state_is_supplied_boundary": bool(self.state.timestamp > cutoff),
            "planning_start_utc": str(grid.starts[0]), "planning_end_utc": str(grid.ends[-1]),
            "physical_planning_intervals": len(grid.interval_ids), "terminal_target_mwh": terminal,
            "terminal_minimum_mwh": terminal_minimum,
            "projected_boundary_energy_mwh": boundary.energy_mwh, "locked_prefix_projection": projection,
            "locked_prefix_intervals": len(projection), "modeled_auctions": list(masks),
            "known_auctions": list(self.known[day]), "fixed_commitment_auctions": list(self.submitted[day]),
            "pending_position_abs_mwh": sum(abs(value) for value in pending_positions), **provenance}
        self.projections.append({"delivery_day": day, "auction": event.auction,
            "cutoff_utc": cutoff.isoformat(), "planning_start_utc": str(grid.starts[0]),
            "boundary_kind": "ida3_noon" if first else "delivery_midnight",
            "predicted_initial_energy_mwh": boundary.energy_mwh,
            "actual_initial_energy_mwh": None, "prediction_error_mwh": None, **provenance})
        return inputs, diagnostic

    def _save_attempts(self):
        """Keep actual solver work even when a partial day is replayed on resume."""
        _atomic_json(self.destination / "json/optimization_attempts.json", self.attempts)
        records = [solve_timing(row, category="perfect_pv_reference" if self.perfect else "replay",
            strategy=self.strategy, delivery_day=row["delivery_day"], auction=row["current_auction"],
            cutoff_utc=row["cutoff"], status=row["status"],
            rolled_back_for_resume=row.get("rolled_back_for_resume", False)) for row in self.attempts]
        _atomic_json(self.destination / "json/optimization_timing.json", optimization_timing_report(records))

    def _submit(self, event):
        """Submit one immutable order and scope every change to its delivery book."""
        instance, cutoff = event.payload, event.at
        day, auction = instance.delivery_day, instance.event.auction
        if auction in self.submitted[day]:
            raise RuntimeError(f"Duplicate continuous submission {day}/{auction}")
        # Perfect information needs only the first solve of this delivery day.
        # Later gates submit its saved orders without replacing physical commands.
        if self.perfect and not self.perfect_reoptimizes and self.plan_snapshots[day]:
            source, snapshot = next(iter(self.plan_snapshots[day].items()))
            by_product = {row["interval_id"]: row for row in snapshot}
            products = [self.daily[day]["grid"].interval_ids[index]
                for index in instance.event.eligible_intervals]
            column = f"planned_trade_{auction}_mwh"
            if any(product not in by_product or column not in by_product[product] for product in products):
                raise RuntimeError("Perfect-PV initial plan has no complete retained auction order")
            quantities = {product: float(by_product[product][column]) for product in products}
            self._commit_quantities(day, auction, cutoff, quantities)
            return {"optimization_performed": False,
                "order_policy": "retained_initial_perfect_pv_plan", "plan_source_auction": source,
                "plan_source_delivery_day": day,
                "plan_source_cutoff_utc": snapshot[0]["optimizer_cutoff_utc"],
                "submitted_product_count": len(quantities)}
        inputs, diagnostic = self._planning_inputs(instance, cutoff)
        built = build_model(inputs)
        diagnostic.update(trade_variable_count=len(built.variables["trades"]),
            trade_variable_counts_by_auction={name: sum(key[0] == name for key in built.variables["trades"])
                for name in inputs.auction_eligibility})
        settings = dict(self.config["solver"])
        name = f"{day}_{auction}_{cutoff.strftime('%Y%m%dT%H%M%SZ')}"
        if self.config["experiment_outputs"].get("save_model_files", True):
            settings["export_path"] = str(self.destination / "models" / f"{name}.lp")
        if self.config["execution"].get("save_infeasibility_diagnostics", False):
            settings["iis_path"] = str(self.destination / "models" / f"{name}_infeasible.ilp")
        try:
            result = solve_model(built, SolverSettings(**settings))
        finally:
            built.model.dispose()
        diagnostic.update(result.diagnostics)
        diagnostic.update(status=result.status, objective_eur=result.objective_eur,
            attempt_id=len(self.attempts), fallback_applied=False)
        self.attempts.append(diagnostic)
        self._save_attempts()
        if not result.has_incumbent:
            snapshots = self.plan_snapshots[day]
            if (result.status != "infeasible" or
                    self.config["execution"].get("infeasible_auction_policy", "error") != "previous_plan" or not snapshots):
                raise RuntimeError(f"Continuous {self.strategy} {day}/{auction}: {result.status}: {result.message}")
            previous = next(reversed(snapshots))
            retained = snapshots[previous]
            by_product = {row["interval_id"]: row for row in retained}
            products = [self.daily[day]["grid"].interval_ids[index] for index in instance.event.eligible_intervals]
            column = f"planned_trade_{auction}_mwh"
            if any(product not in by_product or column not in by_product[product] for product in products):
                raise RuntimeError("Continuous fallback has no complete target-day prior order plan")
            quantities = {product: 0.0 if self.current_only else float(by_product[product][column]) for product in products}
            self._commit_quantities(day, auction, cutoff, quantities)
            first = pd.Timestamp(inputs.grid.starts[0])
            snapshot = []
            for prior in retained:
                if pd.Timestamp(prior["interval_start_utc"]) < first:
                    continue
                row = {**prior, "auction": auction, "optimizer_cutoff_utc": cutoff.isoformat(),
                    "plan_status": "previous_plan_after_infeasible_reoptimization",
                    "plan_source_auction": previous, "plan_source_delivery_day": day,
                    "auction_trade_mwh": quantities.get(prior["interval_id"], 0.0)}
                self.control_plan[row["interval_id"]] = row
                snapshot.append(row)
            self.plan_snapshots[day][auction] = snapshot
            diagnostic.update(fallback_applied=True, fallback_policy="previous_plan",
                fallback_order_policy="zero_new_trades" if self.current_only else "previous_planned_trades",
                plan_source_auction=previous, plan_source_delivery_day=day)
            self.solves.append(diagnostic)
            self._save_attempts()
            return
        # Gurobi may return numerical residuals at its configured tolerances.
        # Keep model bounds intact, while checking continuous and binary
        # residuals against their distinct solver tolerances.
        solver_feasibility_tolerance = float(result.diagnostics.get("feasibility_tolerance") or 1e-6)
        solver_integrality_tolerance = float(result.diagnostics.get("integrality_tolerance") or 1e-5)
        validation_tolerance = max(1e-6, 2.0 * solver_feasibility_tolerance)
        validation_integrality_tolerance = solver_integrality_tolerance
        validation = validate_solution(inputs, result, tolerance=validation_tolerance,
            integrality_tolerance=validation_integrality_tolerance)
        diagnostic["solution_validation_tolerance"] = validation_tolerance
        diagnostic["solution_validation_integrality_tolerance"] = validation_integrality_tolerance
        if not validation.valid:
            raise RuntimeError(f"Continuous solution validation failed: {validation.violations}")
        diagnostic["validation"] = asdict(validation)
        diagnostic["expected_auction_revenue_eur"] = float(np.dot(
            inputs.scenarios.probabilities, result.components["auction_revenue"]))
        self.solves.append(diagnostic)
        if day not in self.initial_expectations:
            self.initial_expectations[day] = float(result.objective_eur)
            self.initial_revenues[day] = diagnostic["expected_auction_revenue_eur"]
        owners = inputs.tree.bid_nodes[auction]
        if len(set(owners)) != 1:
            raise RuntimeError("Current continuous auction order is not shared at its cutoff")
        owner = owners[0]
        quantities = {product: float(quantity) for (name, node, product), quantity in result.trades.items()
            if name == auction and node == owner}
        self._commit_quantities(day, auction, cutoff, quantities)
        snapshot = []
        for index, product in enumerate(inputs.grid.interval_ids):
            commands = {name: max(0.0, float(result.arrays[name][0, index]))
                for name in ("charge_mw", "discharge_mw", "curtailment_mw")}
            row = {"delivery_day": day, "auction": auction, "interval_id": product,
                "interval_start_utc": str(inputs.grid.starts[index]), "interval_end_utc": str(inputs.grid.ends[index]),
                "optimizer_cutoff_utc": cutoff.isoformat(), "plan_status": "optimized",
                "pv_forecast_vintage_utc": diagnostic["pv_forecast_vintage_utc"],
                "pv_forecast_available_at_utc": diagnostic["pv_forecast_available_at_utc"],
                "pv_ac_forecast_mw": float(inputs.scenarios.pv_ac_mw[0, index]),
                **commands, "net_battery_dispatch_mw": commands["discharge_mw"] - commands["charge_mw"],
                "expected_grid_mw": float(result.arrays["grid_mw"][0, index]),
                "planned_net_position_mwh": float(result.arrays["net_position_mwh"][0, index]),
                "soc_start_mwh": float(result.arrays["energy_mwh"][0, index]),
                "soc_end_mwh": float(result.arrays["energy_mwh"][0, index + 1]),
                "auction_trade_mwh": quantities.get(product, 0.0), "snapshot_scenario_count": 1}
            for planned in self.daily[day]["schedule"]:
                if planned.auction in inputs.tree.bid_nodes:
                    node = inputs.tree.bid_nodes[planned.auction][0]
                    quantity = result.trades.get((planned.auction, node, product), 0.0)
                else:
                    quantity = self.submitted[day].get(planned.auction, {}).get(product, 0.0)
                row[f"planned_trade_{planned.auction}_mwh"] = float(quantity)
            self.control_plan[product] = row
            snapshot.append(row)
        self.plan_snapshots[day][auction] = snapshot
        self._save_attempts()

    def _commit_quantities(self, day, auction, cutoff, quantities):
        """Charge fees once at guaranteed submission, before price publication."""
        self.submitted[day][auction] = quantities
        state, charged, _ = account_trade_fees(self.fees, self.market.transaction_fee_state,
            [{"auction": auction, "submission_time": cutoff, "interval_id": product,
              "quantity_mwh": quantity, "trade_id": f"{day}:{auction}:{product}"}
             for product, quantity in quantities.items()])
        self.market = replace(self.market, transaction_fee_state=state)
        self.fee_records.update({row["trade_id"]: row for row in charged})

    def _publish(self, event):
        """Book each fixed quantity at its actual clearing price exactly once."""
        day, auction = event.payload.delivery_day, event.payload.event.auction
        if auction not in self.submitted[day] or auction in self.known[day]:
            raise RuntimeError(f"Invalid publication state for {day}/{auction}")
        previous = self.market.cashflow_eur
        grid = self.daily[day]["grid"]
        indices = {product: index for index, product in enumerate(grid.interval_ids)}
        for product, quantity in self.submitted[day][auction].items():
            trade_id = f"{day}:{auction}:{product}"
            self.market = self.market.accept(AcceptedTrade(auction,
                event.payload.event.submission_at.to_pydatetime(), event.at.to_pydatetime(),
                product, quantity, float(self.daily[day]["prices"][auction][indices[product]]), trade_id,
                transaction_fee_eur=self.fee_records[trade_id]["transaction_fee_eur"]))
        self.known[day].append(auction)
        self.publications.append({"delivery_day": day, "auction": auction,
            "published_at": event.at.isoformat(), "booked_cashflow_eur": self.market.cashflow_eur - previous,
            "booked_trades": len(self.submitted[day][auction])})

    def _update_projection_actuals(self):
        """Resolve predictions only when their physical boundary is reached."""
        for prediction in self.projections:
            if prediction["actual_initial_energy_mwh"] is None and pd.Timestamp(prediction["planning_start_utc"]) == self.state.timestamp:
                prediction["actual_initial_energy_mwh"] = self.state.energy_mwh
                prediction["prediction_error_mwh"] = self.state.energy_mwh - prediction["predicted_initial_energy_mwh"]

    def _control(self, event):
        """Issue retained commands; no optimization occurs at physical boundaries."""
        self._update_projection_actuals()
        product = self.grid.interval_ids[event.payload]
        commands, detail, policy, target = self._commands(product)
        if detail["auction"] == "pre_participation":
            projected, _ = execute_interval(state=self.state, start=self.grid.starts[event.payload],
                end=self.grid.ends[event.payload], pv_ac_mw=float(detail["pv_ac_forecast_mw"]),
                requested_charge_mw=commands[0], requested_discharge_mw=commands[1], requested_curtailment_mw=commands[2],
                plant=self.plant, degradation_settings=self.aging, dispatch_policy=policy, grid_target_mw=target)
            detail = {**detail, "soc_end_mwh": projected.energy_mwh}
        self.pending[product] = {"commands": commands, "detail": dict(detail), "policy": policy, "target": target}

    def _observe(self, event):
        """Reveal interval-average PV only to execution and subsequent observations."""
        index = event.payload
        product = self.grid.interval_ids[index]
        pending = self.pending[product]
        charge, discharge, curtail = pending["commands"]
        self.state, record = execute_interval(state=self.state,
            start=self.grid.starts[index], end=self.grid.ends[index],
            pv_ac_mw=float(self.prepared["full_realized_pv"][index]),
            requested_charge_mw=charge, requested_discharge_mw=discharge, requested_curtailment_mw=curtail,
            plant=self.plant, degradation_settings=self.aging, dispatch_policy=pending["policy"],
            grid_target_mw=pending["target"])
        detail = pending["detail"]
        self.records[product] = {"interval_id": product, "start": str(self.grid.starts[index]),
            "end": str(self.grid.ends[index]), "delivery_day": self.day_for_product[product],
            **record, "active_plan_auction": detail["auction"],
            "active_plan_delivery_day": detail.get("delivery_day", self.day_for_product[product]),
            "plan_cutoff_utc": detail.get("optimizer_cutoff_utc"),
            "pv_ac_forecast_mw": detail.get("pv_ac_forecast_mw"),
            "pv_forecast_vintage_utc": detail.get("pv_forecast_vintage_utc"),
            "pv_forecast_available_at_utc": detail.get("pv_forecast_available_at_utc"),
            "planned_energy_end_mwh": detail.get("soc_end_mwh")}
        self._update_projection_actuals()

    def _settle(self, event):
        """Preserve each quarter-hour's realized settlement and operating costs."""
        index = event.payload
        product = self.grid.interval_ids[index]
        record = self.records[product]
        position = self._position(product)
        imbalance = record["grid_mw"] * self.grid.delta_hours[index] - position
        curve = self.prepared["imbalance_prices_eur_per_mwh"]
        if curve is None:
            raise ValueError("Continuous profit comparisons require settlement.enabled: true and actual settlement prices")
        record.update(net_position_mwh=position, imbalance_mwh=imbalance,
            imbalance_price_eur_mwh=float(curve[index]), imbalance_cashflow_eur=imbalance * float(curve[index]),
            imbalance_penalty_eur=(abs(imbalance) * self.trading["imbalance_penalty_eur_per_mwh"]
                if self.trading.get("optimize_imbalances", True) else 0.0),
            import_fee_eur=max(0.0, -record["grid_mw"]) * self.grid.delta_hours[index] * self.trading["grid_import_fee_eur_per_mwh"])
        self.pending.pop(product)

    def _handle(self, event):
        """Preserve event priorities; checkpoint after a complete midnight handoff."""
        handlers = {EventKind.AUCTION_SUBMISSION: self._submit, EventKind.AUCTION_RESULTS: self._publish,
            EventKind.PHYSICAL_CONTROL: self._control, EventKind.OBSERVATION_AVAILABLE: self._observe,
            EventKind.SETTLEMENT: self._settle}
        detail = handlers[event.kind](event)
        if event.kind == EventKind.AUCTION_SUBMISSION:
            detail = {"auction": event.payload.event.auction,
                "optimization_performed": True, **(detail or {})}
        self.event_trace.append({"at": event.at.isoformat(), "kind": event.kind.name, "sequence": event.sequence,
            "delivery_day": event.payload.delivery_day if isinstance(event.payload, ContinuousAuction)
                else self.day_for_product[self.grid.interval_ids[event.payload]], **(detail or {})})
        if event.kind == EventKind.SETTLEMENT and event.at.tz_convert(self.prepared["calendar"].timezone).time().isoformat() == "00:00:00":
            self.save_checkpoint(event)
            if self.progress:
                day = self.day_for_product[self.grid.interval_ids[event.payload]]
                self.progress(f"Continuous {self.strategy}: completed {day}; "
                    f"SOC {100 * self.state.energy_mwh / self.plant.battery.capacity_mwh:.2f}%; "
                    f"{len(self.solves)} auction solves")

    def _final_dispatch_and_ledger(self):
        """Reconcile the global ledger and allocate increments to delivery products."""
        if len(self.records) != len(self.grid.interval_ids):
            raise RuntimeError("Continuous replay did not execute the complete delivery window")
        checked_state, checked, fee_summary = account_trade_fees(self.fees, self.fees.initial_state,
            [asdict(trade) for trade in self.market.accepted_trades])
        reference = {row["trade_id"]: row for row in checked}
        if abs(checked_state.cumulative_fees_eur - self.market.transaction_fee_state.cumulative_fees_eur) > 1e-6:
            raise RuntimeError("Continuous fees do not reconcile to the immutable executed ledger")
        for month, volume in self.market.transaction_fee_state.monthly_volume_mwh.items():
            if abs(volume - checked_state.monthly_volume_mwh.get(month, 0.0)) > 1e-6:
                raise RuntimeError("Continuous monthly operator volume does not reconcile")
        ledger = [{**asdict(trade), **reference[trade.trade_id], "cashflow_eur": trade.cashflow_eur,
            "delivery_day": self.day_for_product[trade.interval_id]} for trade in self.market.accepted_trades]
        revenue, fees = {}, {}
        for row in ledger:
            product = row["interval_id"]
            revenue[product] = revenue.get(product, 0.0) + row["cashflow_eur"]
            fees[product] = fees.get(product, 0.0) + row["transaction_fee_eur"]
        dispatch = pd.DataFrame([self.records[product] for product in self.grid.interval_ids])
        dispatch["auction_revenue_eur"] = dispatch.interval_id.map(revenue).fillna(0.0)
        dispatch["transaction_fee_eur"] = dispatch.interval_id.map(fees).fillna(0.0)
        return dispatch, ledger, fee_summary

    def run(self):
        """Run or resume one continuous trajectory and finalize its typed outputs."""
        started = perf_counter()
        events = self.events
        if self.cursor is not None:
            key = tuple(self.cursor)
            if not any(tuple(self._event_key(event)) == key for event in events):
                raise ValueError("Continuous checkpoint cursor is not in this replay's event calendar")
            events = [event for event in events if tuple(self._event_key(event)) > key]
        engine = EventEngine(events)
        engine.run({kind: self._handle for kind in (EventKind.AUCTION_SUBMISSION, EventKind.AUCTION_RESULTS,
            EventKind.PHYSICAL_CONTROL, EventKind.OBSERVATION_AVAILABLE, EventKind.SETTLEMENT)})
        dispatch, ledger, fees = self._final_dispatch_and_ledger()
        plans = [row for day in self.days for snapshot in self.plan_snapshots[day].values() for row in snapshot]
        self._save_attempts()
        write_json(self.destination / "json/event_trace.json", self.event_trace)
        write_json(self.destination / "json/price_publications.json", self.publications)
        write_json(self.destination / "json/study_config.json", self.config)
        pd.DataFrame(self.forecast_inputs).to_csv(self.destination / "csv/pv_forecast_inputs.csv", index=False)
        from ..reporting.continuous_outputs import save_continuous_strategy_outputs
        summary = save_continuous_strategy_outputs(self.config, self.destination, self.strategy,
            dispatch, pd.DataFrame(plans), ledger, self.solves, self.projections,
            summary_extra={"initial_expected_profit_by_day": self.initial_expectations,
                "initial_expected_auction_revenue_by_day": self.initial_revenues,
                "optimization_attempts": self.attempts, "transaction_fees": fees,
                "auction_planning_mode": "current_auction_only" if self.current_only else "integrated",
                "perfect_pv_reoptimize_at_each_auction": self.perfect_reoptimizes if self.perfect else None,
                "retained_auction_submission_count": sum(row.get("optimization_performed") is False
                    for row in self.event_trace),
                "policy_version": CONTINUOUS_POLICY_VERSION, "input_fingerprint": self.prepared["input_fingerprint"],
                "final_state": asdict(self.state), "final_fee_state": asdict(self.market.transaction_fee_state),
                "terminal_target_mwh": self.prepared["terminal_energy_mwh"],
                "terminal_minimum_mwh": self.prepared["terminal_minimum_energy_mwh"],
                "terminal_error_mwh": (None if self.prepared["terminal_energy_mwh"] is None else
                    self.state.energy_mwh - self.prepared["terminal_energy_mwh"]),
                "replay_wall_seconds_this_invocation": perf_counter() - started,
                "comparison_scope": "Continuous physical execution, retained-plan boundary projection, one delivery-day planning horizon; no intermediate terminal target or salvage value"})
        write_json(self.destination / "json/manifest.json", {"status": "complete", **self.identity,
            "output_dir": str(self.destination), "days": list(self.days), "summary": summary})
        return summary


def replay_continuous_strategy(config, prepared, strategy, destination, resume=False, progress=None):
    """Reuse one replay implementation for causal policies and perfect PV."""
    return ContinuousReplay(config, prepared, strategy, destination, resume=resume, progress=progress).run()


def run_continuous_experiment(config, *, progress=None):
    """Preflight local inputs once and run independent continuous trajectories."""
    validate_continuous_optimization(config, require_dates=True)
    settings = config["continuous_optimization"]
    if progress:
        progress(f"Continuous optimization: {settings['start_date']} through {settings['end_date']} delivery; offline preflight")
    prepared = prepare_continuous_inputs(config, progress=progress)
    if prepared["imbalance_prices_eur_per_mwh"] is None:
        raise ValueError("Continuous comparisons require actual settlement; set settlement.enabled: true")
    root = config_path(config, settings["output_root"])
    run_id = str(settings.get("run_id") or datetime.now(timezone.utc).strftime("continuous_%Y%m%dT%H%M%S%fZ"))
    destination = root / run_id
    resume = bool(settings.get("resume", False))
    # Validate the run scope before touching its lock; clearing happens only
    # after obtaining that lock, so an active study cannot be overwritten.
    prepare_experiment_output(root, destination, overwrite=False, resume=True)
    import fcntl
    with (destination / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is already running continuous study {run_id}") from None
        prepare_experiment_output(root, destination,
            overwrite=config.get("experiment_outputs", {}).get("overwrite_existing", True),
            resume=resume, preserve=("run.lock",))
        identity = {"policy_version": CONTINUOUS_POLICY_VERSION,
            "config_fingerprint": _configuration_fingerprint(config), "input_fingerprint": prepared["input_fingerprint"]}
        manifest_path = destination / "json/manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if any(previous.get(key) != value for key, value in identity.items()):
                raise ValueError("Cannot resume continuous study with different configuration, inputs or policy")
        write_json(destination / "json/study_config.json", config)
        write_json(destination / "json/input_provenance.json", prepared["provenance"])
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
        manifest = {"status": "running", **identity, "run_id": run_id, "output_dir": str(destination),
            "delivery_days": list(prepared["days"]), "code_revision": revision,
            "completed_strategies": [], "initial_soc_percent": config["state"]["initial_soc_percent"],
            "final_planned_soc_percent": config["state"]["terminal_soc_percent"],
            "terminal_soc_min_percent": config["state"].get("terminal_soc_min_percent"),
            "terminal_target_policy": "study_end_only_with_daily_minimum_floor",
            "delivery_timezone": prepared["calendar"].timezone}
        _atomic_json(manifest_path, manifest)
        strategies = list(settings["planning_modes"])
        if settings.get("perfect_pv_reference", True):
            strategies.append("perfect_pv")
        summaries = {}
        try:
            for strategy in strategies:
                if progress:
                    progress(f"Continuous trajectory {strategy}: {len(prepared['days'])} delivery days")
                summaries[strategy] = replay_continuous_strategy(config, prepared, strategy,
                    destination / "strategies" / strategy, resume=resume, progress=progress)
                manifest["completed_strategies"].append(strategy)
                _atomic_json(manifest_path, manifest)
            from ..reporting.continuous_outputs import save_continuous_comparison
            comparison = save_continuous_comparison(config, destination, summaries)
            manifest.update(status="complete", comparison=comparison,
                strategy_summaries={key: {"output_dir": value["output_dir"], "total_gain_eur": value["total_gain_eur"]}
                    for key, value in summaries.items()})
            _atomic_json(manifest_path, manifest)
        except BaseException as error:
            manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error) or type(error).__name__)
            _atomic_json(manifest_path, manifest)
            raise
    return {"status": "complete", "run_id": run_id, "output_dir": str(destination),
        "delivery_days": len(prepared["days"]), "strategies": manifest["strategy_summaries"]}

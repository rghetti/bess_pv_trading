"""Physical MPC integration for the existing chronological auction runner."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from ..domain import DeliveryGrid, SolverSettings
from ..optimization.mpc import (CAUSAL_TRACKER, MpcAction, MpcControllerState,
    MpcInput, MpcSettings, OraclePrices, RESEARCH_MPC_POLICIES, solve_mpc)
from ..reporting.mpc_history import archive_completed_history, validate_mpc_history_settings
from .energy_price_inputs import (settlement_forecasts, transform_mpc_scenarios,
    conditional_terminal_value)
from .replay import execute_interval


class RealtimeControl:
    """Retain issued actions and common plans without owning any market ledger."""

    def __init__(self, replay):
        self.replay = replay
        self.settings = MpcSettings.from_config(replay.config)
        self.state = MpcControllerState()
        self.horizon_history_mode = validate_mpc_history_settings(replay.config)
        self.history_archives = []
        self.scenario_artifact_cache = {}

    def archive_completed(self, cutoff):
        """Spill finished delivery books while retaining the next midnight action."""
        if self.horizon_history_mode == "daily_parquet":
            completed = [day for day, daily in self.replay.daily.items()
                         if pd.Timestamp(daily["grid"].ends[-1]) <= pd.Timestamp(cutoff)]
            self.history_archives = archive_completed_history(self.replay.destination, self.state,
                                                               self.history_archives, completed)

    def forecast(self, cutoff, starts, context):
        """Always request eligible forecasts, including under an upper PV oracle."""
        return self.replay._forecast(cutoff, starts, context, causal=True)

    def planning_commitments(self, day, grid, cutoff):
        """Supplement binding positions with the eligible IDA2 forecast of IDA3 orders."""
        replay = self.replay
        submitted = np.array([replay._position(key) for key in grid.interval_ids])
        provisional = np.zeros(len(grid.interval_ids))
        evidence = {"mode": self.settings.provisional_ida3_mode,
                    "provisional_ida3_active": False, "provisional_orders_are_binding": False}
        # The order forecast stops immediately when IDA3 becomes binding, even
        # when publication has not happened. Other auction plans never enter here.
        if self.settings.provisional_ida3_mode == "submitted_only" or "IDA3" in replay.submitted[day]:
            return submitted, provisional, evidence
        participating = replay.config.get("gme_oracle_experiment", {}).get("participating_auctions")
        if participating is None:
            participating = ["DA", "IDA1", "IDA2", "IDA3"]
        ida3 = next((event for event in replay.daily[day]["schedule"] if event.auction == "IDA3"), None)
        snapshot = replay.plan_snapshots[day].get("IDA2")
        if ("IDA3" not in participating or "IDA2" not in replay.submitted[day]
                or ida3 is None or not ida3.available or not snapshot
                or pd.Timestamp(cutoff) >= pd.Timestamp(ida3.submission_at)):
            return submitted, provisional, evidence
        eligible = {replay.daily[day]["grid"].interval_ids[t] for t in ida3.eligible_intervals}
        rows = {row["interval_id"]: row for row in snapshot}
        source_cutoffs = []
        # Only IDA3-eligible delivery products receive the forecast signed trade.
        for t, product in enumerate(grid.interval_ids):
            if product not in eligible:
                continue
            row = rows.get(product, {})
            if "planned_trade_IDA3_mwh" not in row:
                raise ValueError("IDA2 snapshot lacks the eligible provisional IDA3 order")
            source_cutoff = pd.Timestamp(row["optimizer_cutoff_utc"])
            if source_cutoff > pd.Timestamp(cutoff):
                raise ValueError("Provisional IDA3 plan is unavailable at the MPC cutoff")
            provisional[t] = float(row["planned_trade_IDA3_mwh"])
            source_cutoffs.append(source_cutoff)
        evidence.update(provisional_ida3_active=bool(source_cutoffs), source_auction="IDA2",
            source_cutoff_utc=max(source_cutoffs).isoformat() if source_cutoffs else None,
            provisional_ida3_total_mwh=float(provisional.sum()))
        return submitted, provisional, evidence

    def issue(self, index, cutoff):
        """Solve with the selected current-PV information and issue only its first action."""
        replay = self.replay
        product = replay.grid.interval_ids[index]
        day = replay.day_for_product[product]
        daily = replay.daily[day]
        local = index - replay.day_offsets[day]
        grid = DeliveryGrid(daily["grid"].interval_ids[local:], daily["grid"].starts[local:], daily["grid"].ends[local:])
        deterministic, provenance = self.forecast(cutoff, pd.DatetimeIndex(grid.starts), "realtime_mpc")
        commands, detail, _, target = replay._commands(product)
        # The optional information-matched tracker uses forecast feedback once, at issuance.
        if replay.dispatch_policy == CAUSAL_TRACKER:
            _, forecast_action = execute_interval(state=replay.state, start=grid.starts[0], end=grid.ends[0],
                pv_ac_mw=float(deterministic[0]), requested_charge_mw=commands[0],
                requested_discharge_mw=commands[1], requested_curtailment_mw=commands[2],
                plant=replay.plant, degradation_settings=replay.aging,
                dispatch_policy="track_committed_grid", grid_target_mw=target)
            fraction = (forecast_action["pv_injection_mw"] / deterministic[0]) if deterministic[0] else 1.0
            action = MpcAction(product, cutoff.isoformat(), forecast_action["charge_mw"],
                               forecast_action["discharge_mw"], float(fraction))
            self._pending(action, {**provenance, "pv_ac_forecast_mw": float(deterministic[0])}, None, None)
            return
        values, ids = deterministic[None, :], ("deterministic",)
        if replay.dispatch_policy.startswith("ensemble_mpc"):
            from .continuous_inputs import ensemble_at_cutoff

            weather, members, info = ensemble_at_cutoff(daily,
                replay.prepared["provenance"]["pv_ensemble"][day], cutoff, grid.starts)
            weather_ids = tuple(f"weather_{member}" for member in members)
            values = np.vstack((values, weather)) if self.settings.ensemble_include_deterministic_member else weather
            ids = ids + weather_ids if self.settings.ensemble_include_deterministic_member else weather_ids
            provenance.update(ensemble_vintage_utc=info["forecast_vintage_utc"],
                              ensemble_available_at_utc=info["available_at_utc"])
        weights = self.settings.scenario_weights if replay.dispatch_policy.startswith("ensemble_mpc") else None
        if weights is not None and set(weights) != set(ids):
            raise ValueError("MPC scenario_weights must identify every represented scenario exactly")
        probabilities = np.full(len(ids), 1 / len(ids)) if weights is None else np.array([weights[key] for key in ids])
        commitments, provisional_ida3, commitment_evidence = self.planning_commitments(day, grid, cutoff)
        planning_positions = commitments + provisional_ida3
        provenance["planning_commitments"] = commitment_evidence
        prices = (settlement_forecasts(replay, daily, cutoff, pd.DatetimeIndex(grid.starts), len(ids))
            if replay.dispatch_policy in RESEARCH_MPC_POLICIES else None)
        law, prices, _, scenario_evidence = transform_mpc_scenarios(replay.config, grid, cutoff,
            replay.plant, values, ids, probabilities, planning_positions, prices,
            artifact_cache=self.scenario_artifact_cache)
        values, ids, probabilities = law.pv_ac_mw, law.scenario_ids, law.weights
        provenance["scenario_design"] = scenario_evidence
        forecast_values = values.copy()
        current_pv = None
        if self.settings.current_pv_information == "realized_interval_average":
            # User-selected retrospective information: no later actual PV is copied.
            current_pv = float(replay.prepared["full_realized_pv"][index])
            values = values.copy()
            values[:, 0] = current_pv
        # Only this marked pathway retrieves future settlement prices for the optimizer.
        oracle = None
        if replay.dispatch_policy.endswith("price_oracle"):
            end = index + len(grid.interval_ids)
            negative = replay.prepared.get("imbalance_negative_prices_eur_per_mwh")
            oracle = OraclePrices(replay.prepared["imbalance_prices_eur_per_mwh"][index:end],
                None if negative is None else negative[index:end],
                {"noncausal": True, "source": "actual_NORD_settlement",
                 "settlement": {**replay.prepared["provenance"].get("settlement", {}),
                     "status": "explicitly_disclosed_noncausal_mpc_oracle",
                     "supplied_to_optimizer": True}})
        inputs = MpcInput(cutoff, grid, replay.plant, replay.state,
            commitments, values, ids, probabilities,
            replay.aging, policy=replay.dispatch_policy, settings=self.settings,
            import_fee_eur_per_mwh=replay.trading["grid_import_fee_eur_per_mwh"],
            hard_terminal_target_mwh=(replay.prepared["terminal_energy_mwh"] if day == replay.days[-1] else None),
            hard_terminal_minimum_mwh=replay.prepared["terminal_minimum_energy_mwh"],
            forecast_provenance=provenance, oracle=oracle, current_realized_pv_mw=current_pv,
            price_forecast=prices,
            terminal_value=conditional_terminal_value(replay, daily, cutoff, grid.ends[-1]),
            provisional_ida3_mwh=provisional_ida3)
        result = solve_mpc(inputs, SolverSettings(**replay.config["solver"]))
        diagnostic = {**result.diagnostics, "delivery_day": day, "interval_id": product,
                      "current_pv_observation_source": ("initial_frozen_forecast_fallback"
                          if current_pv is not None and product in daily.get("pv_provenance", {}).get("realized_power_fallback_interval_ids", [])
                          else "effective_realized_interval_average" if current_pv is not None else "forecast"),
                      "solve_id": f"{replay.dispatch_policy}:{cutoff.isoformat()}"}
        self.state.solves.append(diagnostic)
        if replay.progress and (local % 16 == 0 or result.plan is None):
            replay.progress(f"MPC {replay.dispatch_policy}: {cutoff.isoformat()}; "
                f"{len(grid.interval_ids)} intervals, {len(ids)} scenarios; {result.status}; "
                f"Gurobi {diagnostic.get('runtime_seconds', 0):.3f}s")
        if result.plan is None:
            replay.pending[product] = {"commands": commands, "detail": {**detail,
                "mpc_fallback": True, "mpc_failure_reason": diagnostic.get("failure_reason", result.status),
                "current_realized_pv_mw": current_pv,
                **provenance}, "policy": "track_committed_grid", "target": target}
            self.state.outstanding_action = dict(replay.pending[product])
            self.state.issued_actions.append({"interval_id": product, "issued_at": cutoff.isoformat(),
                "dispatch_policy": "track_committed_grid", "grid_target_mw": target,
                "charge_mw": commands[0], "discharge_mw": commands[1],
                "pv_utilization_fraction": None, "fallback_reason": diagnostic.get("failure_reason", result.status),
                "solve_id": diagnostic["solve_id"], **provenance})
            return
        plan = result.plan
        current_plan = {}
        # The first action is common. Adaptive rows after it are a labelled
        # representative shadow path; every conditional command is saved below.
        for t, action in enumerate(plan.actions):
            row = {**asdict(action), "solve_id": diagnostic["solve_id"], "delivery_day": day,
                "interval_start_utc": str(grid.starts[t]), "interval_end_utc": str(grid.ends[t]),
                "energy_start_mwh": float(plan.energy_mwh[t]), "energy_end_mwh": float(plan.energy_mwh[t + 1]),
                "submitted_commitment_mwh": float(commitments[t]),
                "provisional_ida3_mwh": float(provisional_ida3[t]),
                "planning_position_mwh": float(planning_positions[t]),
                "expected_absolute_imbalance_mwh": float(plan.expected_absolute_imbalance_by_interval_mwh[t]),
                "expected_shortfall_mwh": float(plan.expected_shortfall_by_interval_mwh[t]),
                "future_plan_role": ("representative_shadow_path" if plan.scenario_actions is not None
                    else "common_open_loop_controls"),
                "representative_scenario_id": ids[plan.representative_path_index], **provenance}
            current_plan[action.interval_id] = row
            self.state.plans.append(row)
            for s, scenario in enumerate(ids):
                self.state.scenarios.append({"solve_id": diagnostic["solve_id"], "interval_id": action.interval_id,
                    "delivery_day": day,
                    "scenario_id": scenario, "weight": float(probabilities[s]), "pv_ac_mw": float(values[s, t]),
                    "weather_forecast_pv_ac_mw": float(forecast_values[s, t]),
                    "pv_input_role": "realized_current_interval" if t == 0 and current_pv is not None else "forecast",
                    "grid_mw": float(plan.grid_mw[s, t]), "imbalance_mwh": float(plan.imbalance_mwh[s, t]),
                    "submitted_commitment_mwh": float(commitments[t]),
                    "provisional_ida3_mwh": float(provisional_ida3[t]),
                    "planning_position_mwh": float(planning_positions[t]),
                    **({"charge_mw": float(plan.scenario_actions[s, t, 0]),
                        "discharge_mw": float(plan.scenario_actions[s, t, 1]),
                        "pv_utilization_fraction": float(plan.scenario_actions[s, t, 2]),
                        "energy_start_mwh": float(plan.scenario_energy_mwh[s, t]),
                        "energy_end_mwh": float(plan.scenario_energy_mwh[s, t + 1]),
                        "control_node": plan.control_nodes[s][t]}
                       if plan.scenario_actions is not None else {}),
                    **({"forecast_positive_price_eur_per_mwh": float(prices.positive_eur_per_mwh[s, t]),
                        "forecast_negative_price_eur_per_mwh": float((prices.positive_eur_per_mwh
                            if prices.negative_eur_per_mwh is None else prices.negative_eur_per_mwh)[s, t])}
                       if prices is not None else {})})
        self.state.latest_plan = current_plan
        self._pending(plan.actions[0], {**provenance, "pv_ac_forecast_mw": float(deterministic[0])}, current_plan[product], diagnostic)

    def _pending(self, action, provenance, planned, diagnostic):
        """Freeze the physical action independently of subsequent auction events."""
        replay = self.replay
        detail = {"auction": "realtime_mpc", "optimizer_cutoff_utc": action.issued_at, **provenance,
                  "pv_utilization_fraction": action.pv_utilization_fraction,
                  "mpc_solve_id": None if diagnostic is None else diagnostic["solve_id"],
                  "current_realized_pv_mw": None if diagnostic is None else diagnostic.get("current_realized_pv_mw"),
                  "soc_end_mwh": None if planned is None else planned["energy_end_mwh"],
                  "expected_absolute_imbalance_mwh": None if planned is None else planned["expected_absolute_imbalance_mwh"],
                  "expected_shortfall_mwh": None if planned is None else planned["expected_shortfall_mwh"]}
        replay.pending[action.interval_id] = {"commands": [action.charge_mw, action.discharge_mw, 0.0],
            "detail": detail, "policy": "follow_battery_plan", "target": replay._position(action.interval_id)
                / replay.grid.delta_hours[replay.index[action.interval_id]]}
        self.state.outstanding_action = asdict(action)
        self.state.issued_actions.append({**asdict(action), "dispatch_policy": "follow_battery_plan",
            "grid_target_mw": replay.pending[action.interval_id]["target"],
            "solve_id": detail["mpc_solve_id"], **provenance})

    def projection_commands(self, product, pv):
        """Replay a locked action or the last MPC plan, never solve hypothetical MPCs."""
        replay = self.replay
        if product in replay.pending:
            pending = replay.pending[product]
            commands = list(pending["commands"])
            detail = pending["detail"]
            fraction = detail.get("pv_utilization_fraction")
            if fraction is not None:
                commands[2] = (1 - fraction) * pv
            return commands, detail, pending["policy"], pending["target"]
        row = self.state.latest_plan.get(product)
        target = replay._position(product) / replay.grid.delta_hours[replay.index[product]]
        if row is not None:
            return (row["charge_mw"], row["discharge_mw"], (1 - row["pv_utilization_fraction"]) * pv), {
                "auction": "realtime_mpc", "mpc_solve_id": row["solve_id"],
                "future_plan_role": row.get("future_plan_role", "common_open_loop_controls"),
                "representative_scenario_id": row.get("representative_scenario_id")}, "follow_battery_plan", target
        # Coverage gaps are explicit forecast-only tracking rollouts, not invented commands.
        commands, detail, _, _ = replay._commands(product)
        return commands, {**detail, "projection_fallback": "MPC plan does not cover interval"}, "track_committed_grid", target

    def projection_pv(self, product, deterministic_forecast):
        """Use a previously disclosed current PV scalar; never read future actuals."""
        pending = self.replay.pending.get(product, {})
        known = pending.get("detail", {}).get("current_realized_pv_mw")
        if known is not None:
            return float(known), "disclosed_current_interval_realized_pv"
        return deterministic_forecast, "eligible_deterministic_forecast"

"""Matched-day comparison of configurable auction-participation subsets."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import config_path
from ..data.gme import native_price_resolution_minutes
from ..markets import AuctionCalendar
from ..participation import AUCTION_ORDER, auction_subset, participation_profiles, profile_id
from .gme_oracle import json_load, run_gme_oracle_experiment
from .synthetic import json_default, write_json

SEQUENTIAL_PROFILE = "Sequential (all auctions)"


def profile_input_checksums(output):
    """Fingerprint common time grid, initial weather/PV and actual market curves."""
    pv = pd.read_csv(output / "csv/pv_forecast_and_realized.csv")
    checksums = {"delivery_grid": hashlib.sha256("\n".join(pv.interval_id.astype(str)).encode()).hexdigest()}
    for column in ("pv_realized_execution_ac_mw", "pv_initial_forecast_ac_mw"):
        checksums[column] = hashlib.sha256(pv[column].to_numpy(dtype="<f8").tobytes()).hexdigest()
    prices = pd.read_csv(output / "csv/auction_prices_expanded.csv")
    if list(prices.interval_id) != list(pv.interval_id):
        raise ValueError("Participation price coefficients do not match the PV delivery grid")
    for auction in AUCTION_ORDER:
        if auction in prices:
            values = prices[auction].to_numpy(dtype="<f8")
            payload = np.isfinite(values).tobytes() + np.nan_to_num(values).tobytes()
            checksums[f"actual_prices_{auction}"] = hashlib.sha256(payload).hexdigest()
    forecast_path = output / "csv/oracle/pv_forecast_inputs.csv"
    if forecast_path.exists():
        forecasts = pd.read_csv(forecast_path)
        for auction, frame in forecasts.groupby("auction"):
            columns = ["cutoff_utc", "interval_start_utc", "pv_ac_forecast_mw",
                "pv_forecast_vintage_utc", "pv_forecast_available_at_utc"]
            frame = frame[[name for name in columns if name in frame]].sort_values("interval_start_utc")
            checksums[f"pv_inputs_{auction}"] = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    settlement = output / "csv/imbalance_prices.csv"
    if settlement.exists():
        checksums["settlement_inputs"] = hashlib.sha256(settlement.read_bytes()).hexdigest()
    saved_config = output / "json/study_config.json"
    if saved_config.exists():
        configuration = json_load(saved_config)
        common = {key: configuration.get(key) for key in
            ("inputs", "source", "pv", "battery", "state", "plant", "degradation", "transaction_fees", "trading", "market", "solver")}
        checksums["common_settings"] = hashlib.sha256(json.dumps(common, sort_keys=True, default=json_default).encode()).hexdigest()
    return checksums


def summarize_sequential_days(sequential, integrated, days, auctions):
    """Pair the sequential baseline with integrated all-auction results by day."""
    if sequential.duplicated("study_date").any() or set(sequential.study_date) != set(days):
        raise ValueError("Sequential baseline requires every identical requested delivery day")
    if not np.isfinite(sequential.net_profit_eur).all():
        raise ValueError("Sequential baseline requires fully priced realized profits")
    full = integrated.loc[integrated.profile == profile_id(auctions)].set_index("study_date")
    if set(full.index) != set(days):
        raise ValueError("Sequential baseline requires an integrated all-auction profile on every day")
    sequential = sequential.copy()
    sequential["integrated_all_auctions_profit_eur"] = sequential.study_date.map(full.net_profit_eur)
    sequential["integrated_minus_sequential_profit_eur"] = sequential.integrated_all_auctions_profit_eur - sequential.net_profit_eur
    sequential["sequential_minus_integrated_profit_eur"] = -sequential.integrated_minus_sequential_profit_eur
    summary = {"label": SEQUENTIAL_PROFILE, "participating_auctions": list(auctions), "delivery_days": len(sequential),
        "auction_planning_mode": "current_auction_only",
        "mean_net_profit_eur": float(sequential.net_profit_eur.mean()),
        "total_net_profit_eur": float(sequential.net_profit_eur.sum()),
        "std_daily_profit_eur": float(sequential.net_profit_eur.std(ddof=0)),
        "mean_final_energy_mwh": float(sequential.final_energy_mwh.mean()),
        "mean_final_soc_percent": float(sequential.final_soc_percent.mean()),
        "mean_integrated_minus_sequential_profit_eur": float(sequential.integrated_minus_sequential_profit_eur.mean()),
        "total_integrated_minus_sequential_profit_eur": float(sequential.integrated_minus_sequential_profit_eur.sum()),
        "std_integrated_minus_sequential_profit_eur": float(sequential.integrated_minus_sequential_profit_eur.std(ddof=0)),
        "total_infeasible_auction_decisions": int(sequential.infeasible_auction_decisions.sum())}
    for column in ("degradation_eur", "curtailment_mwh", "absolute_imbalance_mwh", "transaction_fee_eur", "auction_solves"):
        summary[f"mean_{column}"] = float(sequential[column].mean())
    return sequential, summary


def summarize_participation_days(daily, profiles, delivery_days, tolerance):
    """Rank profiles only on complete identical delivery-day sets, retaining SOC."""
    expected_profiles = [profile_id(profile) for profile in profiles]
    expected = {(profile, day) for profile in expected_profiles for day in delivery_days}
    observed = set(zip(daily.profile, daily.study_date))
    if observed != expected or daily.duplicated(["profile", "study_date"]).any():
        raise ValueError("Participation ranking requires each profile on every identical requested delivery day")
    if not np.isfinite(daily.net_profit_eur).all():
        raise ValueError("Participation ranking requires fully priced realized profits; no zero settlement substitution")
    reference = max(profiles, key=len)
    reference_name = profile_id(reference)
    reference_profit = daily.loc[daily.profile == reference_name].set_index("study_date").net_profit_eur
    daily = daily.copy()
    daily["profit_minus_reference_eur"] = daily.net_profit_eur - daily.study_date.map(reference_profit)
    day_max = daily.groupby("study_date").net_profit_eur.transform("max")
    daily["is_daily_best_within_tolerance"] = day_max - daily.net_profit_eur <= tolerance
    rows = []
    for profile in profiles:
        name = profile_id(profile)
        frame = daily.loc[daily.profile == name]
        rows.append({"profile": name, "participating_auctions": list(profile),
            "selected_auction_count": len(profile), "delivery_days": len(frame),
            "mean_net_profit_eur": float(frame.net_profit_eur.mean()),
            "total_net_profit_eur": float(frame.net_profit_eur.sum()),
            "std_daily_profit_eur": float(frame.net_profit_eur.std(ddof=0)),
            "median_daily_profit_eur": float(frame.net_profit_eur.median()),
            "p10_daily_profit_eur": float(frame.net_profit_eur.quantile(0.1)),
            "p90_daily_profit_eur": float(frame.net_profit_eur.quantile(0.9)),
            "min_daily_profit_eur": float(frame.net_profit_eur.min()),
            "max_daily_profit_eur": float(frame.net_profit_eur.max()),
            "negative_profit_days": int(frame.net_profit_eur.lt(0).sum()),
            "zero_active_auction_days": int(frame.active_auction_count.eq(0).sum()),
            "infeasible_reoptimization_days": int(frame.get("infeasible_auction_decisions", pd.Series(0, index=frame.index)).gt(0).sum()),
            "total_infeasible_auction_decisions": int(frame.get("infeasible_auction_decisions", pd.Series(0, index=frame.index)).sum()),
            "daily_wins_including_ties": int(frame.is_daily_best_within_tolerance.sum()),
            "mean_profit_minus_reference_eur": float(frame.profit_minus_reference_eur.mean()),
            "reference_profile": reference_name,
            **{f"mean_{column}": float(frame[column].mean()) for column in (
                "final_energy_mwh", "final_soc_percent", "degradation_eur",
                "curtailment_mwh", "absolute_imbalance_mwh", "transaction_fee_eur",
                "auction_solves", "active_auction_count")}})
    ranking = pd.DataFrame(rows).sort_values(
        ["mean_net_profit_eur", "min_daily_profit_eur", "profile"],
        ascending=[False, False, True]).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return daily, ranking


def _solve_all_auction_reference(config, source_output, destination):
    """Solve just the missing full-information reference with the common actual PV."""
    from ..domain import PlantParameters, battery_parameters_from_config
    from .gme_oracle import (_load_auction_prices, _soc_energy_targets,
                             _solve_perfect_information_schedule)

    reference = deepcopy(config)
    study = reference["gme_oracle_experiment"]
    study["participating_auctions"] = list(auction_subset(list(study["auction_datasets"]),
        "perfect-foresight reference auctions"))
    calendar = AuctionCalendar.from_config(reference)
    grid = calendar.delivery_grid(study["delivery_day"]).to_domain()
    events = calendar.schedule(study["delivery_day"])
    prices, provenance, skipped = _load_auction_prices(reference, calendar, grid, events)
    schedule = tuple(event for event in events if event.auction in prices)
    if not schedule:
        raise ValueError("No available auction for the perfect-foresight reference")
    pv = pd.read_csv(source_output / "csv/pv_forecast_and_realized.csv")
    if list(pv.interval_id) != list(grid.interval_ids):
        raise ValueError("Perfect-foresight reference must use the identical delivery grid")
    battery = battery_parameters_from_config(reference["battery"])
    plant = PlantParameters(**{key: reference["plant"][key] for key in
        ("farm_id", "poi_import_mw", "poi_export_mw")}, battery=battery)
    initial, terminal = _soc_energy_targets(reference["state"], battery)
    reference["state"].update(initial_energy_mwh=initial, terminal_target_mwh=terminal)
    for kind in ("csv", "json"):
        (destination / kind).mkdir(parents=True, exist_ok=True)
    if reference["experiment_outputs"].get("save_model_files", False):
        (destination / "models").mkdir()
    perfect, diagnostics = _solve_perfect_information_schedule(reference, grid, plant,
        schedule, prices, pv.pv_realized_execution_ac_mw.to_numpy(float), initial, destination)
    diagnostics.update(active_auctions=list(prices), participating_auctions=study["participating_auctions"],
        price_provenance=provenance, skipped_auctions=skipped,
        common_pv_source=str(source_output), additional_optimizer_solves=1)
    perfect.to_csv(destination / "csv/perfect_pv_counterfactual_schedule.csv", index=False)
    write_json(destination / "json/perfect_pv_counterfactual.json", diagnostics)
    write_json(destination / "json/study_config.json", reference)
    return destination


def collect_perfect_foresight_reference(config, experiments, destination):
    """Reuse one all-auction ObjVal per day, solving only when that profile is absent."""
    if not config["participation_study"].get("perfect_foresight_reference", True):
        return None, None
    all_auctions = auction_subset(list(config["gme_oracle_experiment"]["auction_datasets"]),
        "perfect-foresight reference auctions")
    rows = []
    for day in sorted(str(day) for day in config["participation_study"]["delivery_days"]):
        day_runs = [row for row in experiments if row["study_date"] == day]
        if not day_runs:
            raise ValueError("Perfect-foresight reference requires every requested delivery day")
        full_run = next((row for row in day_runs if row["profile"] == profile_id(all_auctions)), None)
        extra_solves = 0
        if full_run is not None:
            output = Path(full_run["output_dir"])
            active = json_load(output / "json/kpis.json")["active_auctions"]
        else:
            child = json_load(Path(day_runs[0]["output_dir"]) / "json/study_config.json")
            child["experiment_outputs"]["save_model_files"] = config["participation_study"].get("save_model_files", False)
            output = _solve_all_auction_reference(child, Path(day_runs[0]["output_dir"]),
                destination / "references/perfect_foresight" / f"farm_{config['plant']['farm_id']}" / day)
            active = json_load(output / "json/perfect_pv_counterfactual.json")["active_auctions"]
            extra_solves = 1
        diagnostics = json_load(output / "json/perfect_pv_counterfactual.json")
        perfect = pd.read_csv(output / "csv/perfect_pv_counterfactual_schedule.csv")
        source_pv = pd.read_csv(Path(day_runs[0]["output_dir"]) / "csv/pv_forecast_and_realized.csv")
        if (list(perfect.interval_id) != list(source_pv.interval_id)
                or not np.array_equal(perfect.pv_ac_mw, source_pv.pv_realized_execution_ac_mw)):
            raise ValueError("Perfect-foresight reference must use the identical effective realized PV and grid")
        objective = float(diagnostics["objective_eur"])
        if not diagnostics["validation"]["valid"] or not np.isfinite(objective):
            raise ValueError("Perfect-foresight reference requires a validated finite ObjVal")
        rows.append({"study_date": day, "objective_eur": objective,
            "active_auctions": active, "selected_auctions": list(all_auctions),
            "final_energy_mwh": float(perfect.soc_end_mwh.iloc[-1]),
            "final_soc_percent": float(perfect.soc_end_mwh.iloc[-1]) / config["battery"]["capacity_mwh"] * 100,
            "model_status": diagnostics["status"], "model_mip_gap": diagnostics.get("mip_gap"),
            "objective_basis": "Gurobi ObjVal; configured costs and terminal SOC, effective realized PV and actual auction prices",
            "additional_optimizer_solves": extra_solves, "source_output_dir": str(output)})
    daily = pd.DataFrame(rows)
    summary = {"label": "Perfect foresight (all auctions)", "selected_auctions": list(all_auctions),
        "delivery_days": len(daily), "mean_objective_eur": float(daily.objective_eur.mean()),
        "total_objective_eur": float(daily.objective_eur.sum()),
        "std_daily_objective_eur": float(daily.objective_eur.std(ddof=0)),
        "min_daily_objective_eur": float(daily.objective_eur.min()),
        "max_daily_objective_eur": float(daily.objective_eur.max()),
        "mean_final_energy_mwh": float(daily.final_energy_mwh.mean()),
        "mean_final_soc_percent": float(daily.final_soc_percent.mean()),
        "additional_optimizer_solves": int(daily.additional_optimizer_solves.sum()),
        "objective_basis": rows[0]["objective_basis"],
        "comparison_scope": "Full-information model objective reference; participation profiles use settled feedback profits, whose imbalance settlements and realized terminal inventory can differ"}
    daily.to_csv(destination / "csv/perfect_foresight_daily.csv", index=False)
    write_json(destination / "json/perfect_foresight_reference.json", {**summary, "daily": rows})
    return daily, summary


def participation_comparison_table(ranking, perfect_foresight=None, sequential=None):
    """Place the sequential baseline next to integrated all-auction planning."""
    chart = ranking.copy()
    if sequential is not None:
        anchor = profile_id(sequential["participating_auctions"])
        matching = np.flatnonzero(chart.profile.eq(anchor))
        if not len(matching):
            raise ValueError("Sequential chart requires its integrated all-auction row")
        position = int(matching[0]) + 1
        chart = pd.concat([chart.iloc[:position], pd.DataFrame([{**sequential, "profile": sequential["label"]}]),
            chart.iloc[position:]], ignore_index=True)
    if perfect_foresight is not None:
        reference_row = {"profile": perfect_foresight["label"],
            "participating_auctions": perfect_foresight["selected_auctions"],
            "mean_net_profit_eur": perfect_foresight["mean_objective_eur"],
            "std_daily_profit_eur": perfect_foresight["std_daily_objective_eur"],
            "mean_final_energy_mwh": perfect_foresight["mean_final_energy_mwh"]}
        chart = pd.concat([pd.DataFrame([reference_row]), chart], ignore_index=True)
    return chart


def _plot_participation(ranking, path, perfect_foresight=None, sequential=None):
    """Compare average cash profit, chosen auctions and ending stored energy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart = participation_comparison_table(ranking, perfect_foresight, sequential)
    colors = ["goldenrod" if perfect_foresight is not None and index == 0 else
        "tab:orange" if name == SEQUENTIAL_PROFILE else "tab:blue" for index, name in enumerate(chart.profile)]
    fig, axes = plt.subplots(1, 3, figsize=(15, max(5, len(chart) * 0.43 + 2)),
        sharey=True, gridspec_kw={"width_ratios": [3, 1.5, 2]}, layout="constrained")
    positions = np.arange(len(chart))
    axes[0].barh(positions, chart.mean_net_profit_eur,
        xerr=chart.std_daily_profit_eur, capsize=3, color=colors, alpha=0.8)
    axes[0].set_yticks(positions, chart.profile)
    axes[0].set_xlabel("Mean daily profit / reference objective (EUR)\nWhiskers = daily SD"
        if perfect_foresight is not None else "Mean daily realized profit (EUR); whiskers = daily SD")
    axes[0].set_title("Ranked on the same delivery days")
    choices = np.array([[auction in profile for auction in AUCTION_ORDER]
                        for profile in chart.participating_auctions], dtype=float)
    axes[1].imshow(choices, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1,
        extent=(-0.5, len(AUCTION_ORDER) - 0.5, len(chart) - 0.5, -0.5))
    axes[1].set_xticks(np.arange(len(AUCTION_ORDER)), AUCTION_ORDER, rotation=45)
    axes[1].set_title("Selected auctions")
    axes[2].barh(positions, chart.mean_final_energy_mwh,
        color=["tab:purple" if color == "tab:blue" else color for color in colors], alpha=0.8)
    axes[2].set_xlabel("Mean final stored energy (MWh), unvalued")
    axes[2].set_title("Terminal inventory context")
    axes[0].set_ylim(len(chart) - 0.5, -0.5)
    if perfect_foresight is not None:
        for axis in axes:
            axis.axhline(0.5, color="0.45", linestyle="--", linewidth=0.8)
        axes[0].get_yticklabels()[0].set_color("darkgoldenrod")
        axes[0].get_yticklabels()[0].set_fontweight("bold")
    for axis in (axes[0], axes[2]):
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle(f"Auction participation: independent-day price-oracle study ({int(ranking.delivery_days.iloc[0])} days)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_participation_study(config, *, progress=None):
    """Run the same assets and initial state for each profile/day, entirely offline."""
    settings = config["participation_study"]
    profiles = participation_profiles(settings)
    cases = [(profile_id(profile), profile, "integrated") for profile in sorted(profiles, key=len, reverse=True)]
    all_auctions = auction_subset(list(config["gme_oracle_experiment"]["auction_datasets"]), "sequential baseline auctions")
    if settings.get("sequential_all_auctions_baseline", False):
        if all_auctions not in profiles:
            raise ValueError("Sequential all-auction baseline requires the integrated all-auction participation profile")
        cases.append((SEQUENTIAL_PROFILE, all_auctions, "current_auction_only"))
    days = settings.get("delivery_days")
    if not isinstance(days, list) or not days or len(days) != len(set(map(str, days))):
        raise ValueError("participation_study.delivery_days must be a nonempty unique list")
    days = sorted(date.fromisoformat(str(day)).isoformat() for day in days)
    tolerance = float(settings.get("profit_tolerance_eur", 1e-6))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("participation_study.profit_tolerance_eur must be finite and nonnegative")
    root = config_path(config, settings["output_root"])
    run_id = str(settings.get("run_id") or datetime.now(timezone.utc).strftime("participation_%Y%m%dT%H%M%S%fZ"))
    destination = root / run_id
    resume = settings.get("resume", False) and destination.exists()
    if resume:
        previous_config = json_load(destination / "json/study_config.json")
        comparable = deepcopy(config)
        previous_config["participation_study"].pop("resume", None)
        comparable["participation_study"].pop("resume", None)
        from ..config import with_standard_defaults
        previous_config = with_standard_defaults(previous_config)
        comparable = with_standard_defaults(comparable)
        if json.loads(json.dumps(previous_config, default=json_default)) != json.loads(json.dumps(comparable, default=json_default)):
            raise ValueError("Cannot resume a participation study with different configuration")
        previous_manifest = json_load(destination / "json/manifest.json")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    for kind in ("json", "csv"):
        (destination / kind).mkdir(exist_ok=resume)
    if settings.get("plots", True):
        (destination / "plots").mkdir(exist_ok=resume)
    write_json(destination / "json/study_config.json", config)
    manifest = {"status": "running", "run_id": run_id, "output_dir": str(destination),
        "delivery_days": days, "profiles": [list(profile) for profile in profiles],
        "profile_count": len(profiles), "requested_experiments": len(days) * len(cases),
        "sequential_all_auctions_baseline": settings.get("sequential_all_auctions_baseline", False),
        "initial_state_convention": "Independent delivery days; identical configured initial SOC and initial operator fee state per profile/day, no state carry across days",
        "initial_forecast_convention": "Common initialization cutoff before the earliest candidate gate and delivery start; later updates only at selected auction decisions",
        "ranking_metric": "mean realized full-day profit after actual settlement, executed wear, import and gross transaction fees; no terminal-energy salvage value",
        "completed_experiments": 0, "experiments": []}
    path = destination / "json/manifest.json"
    if resume:
        manifest = previous_manifest
        manifest.setdefault("resumptions", []).append({"at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_status": manifest["status"], "previous_error": manifest.get("error"),
            "completed_experiments": manifest["completed_experiments"]})
        manifest.update(status="running", ranking_published=False)
        manifest.pop("error", None)
        if progress:
            progress(f"Resuming {manifest['completed_experiments']} completed cases after checking configuration and saved inputs")
    write_json(path, manifest)
    rows = list(manifest["experiments"])
    completed = {(row["profile"], row["study_date"]): row for row in rows}
    expected = {(name, day) for name, _, _ in cases for day in days}
    if len(completed) != len(rows) or set(completed) - expected:
        raise ValueError("Resume checkpoint has duplicate or unexpected profile/day cases")
    try:
        for day in days:
            shared = deepcopy(config)
            shared["market"].update(valid_from=days[0], valid_to=days[-1])
            calendar = AuctionCalendar.from_config(shared)
            candidate_names = set(settings.get("candidate_auctions", ["DA", "IDA1", "IDA2", "IDA3"]))
            events = [event for event in calendar.schedule(day) if event.auction in candidate_names]
            initialization = min([calendar.delivery_grid(day).starts[0], *(event.submission_at for event in events)])
            shared["gme_oracle_experiment"]["delivery_day"] = day
            for specification in shared["gme_oracle_experiment"]["auction_datasets"].values():
                specification["native_interval_minutes"] = native_price_resolution_minutes(
                    specification["dataset"], date.fromisoformat(day))
            shared["forecast"]["initialization_cutoff"] = initialization.isoformat()
            shared["forecast"]["initial_decision_cutoff"] = initialization.isoformat()
            shared["forecast_run"].update(delivery_day=day, decision_cutoff=initialization.isoformat())
            common_checksums = {}
            for name, profile, mode in cases:
                output_name = "sequential_all_auctions" if mode == "current_auction_only" else name
                if (name, day) in completed:
                    row = completed[name, day]
                    output = Path(row["output_dir"])
                    if json_load(output / "json/manifest.json")["status"] != "complete":
                        raise ValueError("Resume checkpoint references an incomplete child experiment")
                    checksums = profile_input_checksums(output)
                    if checksums != row["input_checksums"] or json_load(output / "json/kpis.json")["total_gain_eur"] != row["net_profit_eur"]:
                        raise ValueError("Resume checkpoint input or profit verification failed")
                    for label, checksum in checksums.items():
                        if label in common_checksums and common_checksums[label] != checksum:
                            raise ValueError("Resume checkpoint has inconsistent shared inputs")
                        common_checksums[label] = checksum
                    continue
                child = deepcopy(shared)
                child["gme_oracle_experiment"].update(participating_auctions=list(profile),
                    study_id=f"{run_id}_{output_name}", auction_planning_mode=mode)
                child["experiment_outputs"].update(root=str(destination / "runs"), run_id=output_name,
                    plots=settings.get("per_profile_plots", False),
                    save_model_files=settings.get("save_model_files", False))
                if progress:
                    progress(f"[{len(rows) + 1}/{len(days) * len(cases)}] {day} {name}")
                incomplete = destination / "runs" / output_name / f"farm_{config['plant']['farm_id']}" / day
                if resume and incomplete.exists():
                    retained = destination / "diagnostics/incomplete_runs" / output_name / f"{day}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                    retained.parent.mkdir(parents=True, exist_ok=True)
                    incomplete.rename(retained)
                result = run_gme_oracle_experiment(child)
                output = Path(result["output_dir"])
                kpis = json_load(output / "json/kpis.json")
                checksums = profile_input_checksums(output)
                for label, checksum in checksums.items():
                    if label in common_checksums and common_checksums[label] != checksum:
                        raise ValueError(f"Participation profiles received different shared input {label}; comparison aborted")
                    common_checksums[label] = checksum
                if kpis["total_gain_eur"] is None:
                    raise ValueError("Participation ranking needs actual imbalance settlement, not an unpriced profit")
                row = {"profile": name, "study_date": day, "net_profit_eur": kpis["total_gain_eur"],
                    "auction_planning_mode": mode,
                    "auction_revenue_eur": kpis["auction_revenue_eur"],
                    "degradation_eur": kpis["degradation_cost_eur"],
                    "transaction_fee_eur": kpis["transaction_fee_eur"],
                    "curtailment_mwh": kpis["curtailment_mwh"],
                    "absolute_imbalance_mwh": kpis["absolute_imbalance_mwh"],
                    "final_energy_mwh": kpis["final_energy_mwh"], "final_soc_percent": kpis["final_soc_percent"],
                    "auction_solves": kpis["auction_decision_solves"],
                    "infeasible_auction_decisions": kpis.get("infeasible_auction_decision_count", 0),
                    "active_auction_count": len(kpis["active_auctions"]),
                    "active_auctions": kpis["active_auctions"], "skipped_auctions": kpis["skipped_auctions"],
                    "realized_pv_sha256": checksums["pv_realized_execution_ac_mw"],
                    "input_checksums": checksums, "initialization_cutoff_utc": initialization.isoformat(),
                    "output_dir": str(output)}
                rows.append(row)
                manifest["experiments"].append(row)
                manifest["completed_experiments"] = len(rows)
                pd.DataFrame(rows).to_csv(destination / "csv/daily_results.csv", index=False)
                write_json(path, manifest)
        all_daily = pd.DataFrame(rows)
        integrated = all_daily.loc[all_daily.auction_planning_mode == "integrated"]
        daily, ranking = summarize_participation_days(integrated, profiles, days, tolerance)
        sequential_summary = None
        if settings.get("sequential_all_auctions_baseline", False):
            sequential, sequential_summary = summarize_sequential_days(
                all_daily.loc[all_daily.auction_planning_mode == "current_auction_only"], daily, days, all_auctions)
            sequential.to_csv(destination / "csv/sequential_daily_results.csv", index=False)
            write_json(destination / "json/sequential_baseline.json", sequential_summary)
        perfect_daily, perfect_reference = collect_perfect_foresight_reference(config,
            [row for row in rows if row["profile"] != SEQUENTIAL_PROFILE], destination)
        if perfect_reference is not None:
            daily["perfect_foresight_objective_eur"] = daily.study_date.map(perfect_daily.set_index("study_date").objective_eur)
            daily["perfect_foresight_minus_profit_eur"] = daily.perfect_foresight_objective_eur - daily.net_profit_eur
            ranking["mean_perfect_foresight_minus_profit_eur"] = perfect_reference["mean_objective_eur"] - ranking.mean_net_profit_eur
        daily.to_csv(destination / "csv/daily_results.csv", index=False)
        ranking.to_csv(destination / "csv/profile_ranking.csv", index=False)
        participation_comparison_table(ranking, perfect_reference, sequential_summary).to_csv(
            destination / "csv/participation_comparison.csv", index=False)
        daily.loc[daily.is_daily_best_within_tolerance].to_csv(destination / "csv/daily_best_profiles.csv", index=False)
        if settings.get("plots", True):
            _plot_participation(ranking, destination / "plots/participation_profiles.png", perfect_reference, sequential_summary)
        manifest.update(status="complete", ranking_published=True, best_mean_profile=ranking.iloc[0].profile,
            best_mean_net_profit_eur=float(ranking.iloc[0].mean_net_profit_eur),
            perfect_foresight_reference=perfect_reference,
            sequential_baseline=sequential_summary,
            ranking=ranking.to_dict("records"),
            ranking_scope="Descriptive ranking on the configured common dates; not a selected or validated optimal policy for unseen dates")
        write_json(path, manifest)
    except Exception as error:
        manifest.update(status="failed", error=str(error), ranking_published=False)
        write_json(path, manifest)
        raise
    return {key: manifest[key] for key in ("status", "run_id", "output_dir", "profile_count",
        "delivery_days", "completed_experiments", "best_mean_profile", "best_mean_net_profit_eur")}

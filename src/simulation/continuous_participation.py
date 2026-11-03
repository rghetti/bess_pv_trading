"""Compare auction subsets as independent, uninterrupted physical trajectories."""
from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd

from ..data import config_path
from ..participation import auction_subset, participation_profiles, profile_id
from ..reporting.artifacts import prepare_experiment_output
from .continuous import _atomic_json, _configuration_fingerprint, replay_continuous_strategy
from .continuous_inputs import prepare_continuous_inputs
from .synthetic import write_json


_worker_context = None


def _initialize_profile_worker(config, prepared, pv_config, pv_prepared, destination, resume, fingerprint):
    """Give each independent worker its own copy of the prepared inputs."""
    global _worker_context
    _worker_context = (config, prepared, pv_config, pv_prepared, destination, resume, fingerprint)


def _run_profile_case(case):
    """Replay one profile and its matched PV-only trajectory in the same worker."""
    name, profile, strategy = case
    config, prepared, pv_config, pv_prepared, destination, resume, fingerprint = _worker_context
    def progress(message):
        print(f"[{name}] {message}", flush=True)
    case_fingerprint = hashlib.sha256(f"{fingerprint}:{name}".encode()).hexdigest()
    summary = replay_continuous_strategy(config, _profile_inputs(prepared, profile), strategy,
        destination / "strategies" / name, resume=resume, progress=progress,
        config_fingerprint=case_fingerprint)
    pv_summary = None
    if pv_config is not None:
        pv_fingerprint = hashlib.sha256(f"{fingerprint}:{name}:pv_only".encode()).hexdigest()
        pv_summary = replay_continuous_strategy(pv_config, _profile_inputs(pv_prepared, profile),
            strategy, destination / "pv_only" / name, resume=resume, progress=progress,
            config_fingerprint=pv_fingerprint)
    return summary, pv_summary


def _profile_inputs(prepared, profile):
    """Select gates without changing the common data, grids or initial state."""
    result = dict(prepared)
    result["days_data"] = {}
    for day, daily in prepared["days_data"].items():
        selected = dict(daily)
        selected["schedule"] = tuple(event for event in daily["schedule"] if event.auction in profile)
        result["days_data"][day] = selected
    return result


def _save_comparison(destination, summaries, profiles, days, plots):
    """Rank whole-month cash profits and preserve paired daily distributions."""
    frames = []
    for name, summary in summaries.items():
        frame = pd.DataFrame(summary["daily_kpis"]).copy()
        if frame.delivery_day.duplicated().any() or set(frame.delivery_day) != set(days):
            raise ValueError(f"Incomplete or duplicate delivery days for {name}")
        if not np.isfinite(frame.net_profit_eur.to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite realized profits for {name}")
        frame["profile"] = name
        frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    names = [profile_id(profile) for profile in profiles]
    rows = []
    for name, group in daily.groupby("profile", sort=False):
        summary = summaries[name]
        rows.append({"profile": name, "kind": "participation" if name in names else "baseline",
            "delivery_days": len(group), "total_net_profit_eur": float(group.net_profit_eur.sum()),
            "mean_daily_profit_eur": float(group.net_profit_eur.mean()),
            "std_daily_profit_eur": float(group.net_profit_eur.std(ddof=0)),
            "final_energy_mwh": summary["final_energy_mwh"],
            "final_soc_percent": summary["final_soc_percent"],
            "fallback_count": summary["fallback_count"]})
    ranking = pd.DataFrame(rows).sort_values("total_net_profit_eur", ascending=False)
    (destination / "csv").mkdir(exist_ok=True)
    daily.to_csv(destination / "csv/daily_participation.csv", index=False)
    ranking.to_csv(destination / "csv/monthly_participation.csv", index=False)
    if plots:
        import matplotlib.pyplot as plt

        (destination / "plots").mkdir(exist_ok=True)
        ordered = ranking.profile.tolist()
        fig, axis = plt.subplots(figsize=(12, max(5, len(ordered) * 0.35)), layout="constrained")
        axis.barh(ordered, ranking.total_net_profit_eur, color=["tab:blue" if x in names else "tab:orange" for x in ordered])
        axis.invert_yaxis()
        axis.set(xlabel="Realized monthly net profit (EUR)", title="Continuous auction participation")
        fig.savefig(destination / "plots/monthly_profit_comparison.png", dpi=140)
        plt.close(fig)
        # A common set of bins makes the daily profit distributions directly comparable.
        bins = np.histogram_bin_edges(daily.net_profit_eur.to_numpy(dtype=float), bins="auto")
        fig, axes = plt.subplots(len(ordered), 1, sharex=True, figsize=(10, max(6, len(ordered) * 1.5)), layout="constrained", squeeze=False)
        for axis, name in zip(axes[:, 0], ordered):
            axis.hist(daily.loc[daily.profile.eq(name), "net_profit_eur"], bins=bins,
                      color="tab:blue" if name in names else "tab:orange")
            axis.set_ylabel(name)
        axes[-1, 0].set_xlabel("Realized daily net profit (EUR)")
        fig.suptitle("Daily profit distributions across the same delivery month")
        fig.savefig(destination / "plots/daily_profit_histograms.png", dpi=140)
        plt.close(fig)
    return ranking.to_dict("records")


def run_continuous_participation(config, *, progress=None):
    """Preflight once and replay each profile with its own SOC and market ledger."""
    settings = config["continuous_optimization"]
    study = config["continuous_participation_study"]
    profiles = participation_profiles(study)
    all_auctions = auction_subset(list(config["gme_oracle_experiment"]["auction_datasets"]),
                                 "continuous participation auctions")
    sequential_enabled = study.get("sequential_all_auctions_baseline", True)
    perfect_enabled = study.get("perfect_pv_baseline", True)
    if (sequential_enabled or perfect_enabled) and all_auctions not in profiles:
        raise ValueError("Continuous participation requires the all-auction profile")
    prepared = prepare_continuous_inputs(config, progress=progress)
    if prepared["imbalance_prices_eur_per_mwh"] is None:
        raise ValueError("Continuous participation requires actual imbalance settlement")
    root = config_path(config, settings["output_root"]) / f"farm_{config['plant']['farm_id']}"
    run_id = settings.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Continuous participation requires an explicit run_id")
    destination = root / run_id
    resume = bool(settings.get("resume", False))
    prepare_experiment_output(root, destination, overwrite=False, resume=True)
    with (destination / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is running continuous participation {run_id}") from None
        prepare_experiment_output(root, destination,
            overwrite=config.get("experiment_outputs", {}).get("overwrite_existing", True),
            resume=resume, preserve=("run.lock",))
        (destination / "json").mkdir(exist_ok=True)
        fingerprint = _configuration_fingerprint(config)
        identity = {"config_fingerprint": fingerprint, "input_fingerprint": prepared["input_fingerprint"],
                    "policy": "continuous-participation-v1"}
        manifest_path = destination / "json/manifest.json"
        if manifest_path.exists() and any(json.loads(manifest_path.read_text()).get(k) != v for k, v in identity.items()):
            raise ValueError("Cannot resume continuous participation with different configuration or inputs")
        write_json(destination / "json/study_config.json", config)
        write_json(destination / "json/input_provenance.json", prepared["provenance"])
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=False).stdout.strip()
        manifest = {**identity, "status": "running", "run_id": run_id, "code_revision": revision,
                    "delivery_days": list(prepared["days"]), "completed": []}
        _atomic_json(manifest_path, manifest)
        cases = [(profile_id(profile), profile, "integrated") for profile in profiles]
        if sequential_enabled:
            cases.append(("sequential_all_auctions", all_auctions, "current_auction_only"))
        if perfect_enabled:
            cases.append(("perfect_pv_all_auctions", all_auctions, "perfect_pv"))
        summaries = {}
        pv_summaries = {}
        pv_config = pv_prepared = None
        if study.get("matched_pv_only", False):
            from .continuous_pv_only import pv_only_inputs
            pv_config, pv_prepared = pv_only_inputs(config, prepared)
        try:
            workers = int(os.environ.get("BESS_PARTICIPATION_WORKERS", "1"))
            if workers < 1:
                raise ValueError("BESS_PARTICIPATION_WORKERS must be positive")

            def announce(index, name):
                """Keep the original profile counter for serial and parallel runs."""
                if progress:
                    # Count auction subsets separately from the two reference trajectories.
                    if index <= len(profiles):
                        counter = f"{index}/{len(profiles)}"
                    else:
                        counter = f"baseline {index - len(profiles)}/{len(cases) - len(profiles)}"
                    progress(f"Starting {name} [{counter}]: {len(prepared['days'])} delivery days")

            if workers > 1:
                # Spawn independent processes: Gurobi environments must not be shared across threads.
                context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                        initializer=_initialize_profile_worker,
                        initargs=(config, prepared, pv_config, pv_prepared, destination, resume, fingerprint)) as executor:
                    pending = {}
                    next_case = iter(enumerate(cases, start=1))
                    def submit_next():
                        """Start a profile only when a worker is available."""
                        item = next(next_case, None)
                        if item is not None:
                            index, case = item
                            announce(index, case[0])
                            pending[executor.submit(_run_profile_case, case)] = case[0]
                    for _ in range(min(workers, len(cases))):
                        submit_next()
                    try:
                        while pending:
                            done, _ = wait(pending, return_when=FIRST_COMPLETED)
                            # Check every completed future before scheduling more work.
                            finished = [(pending.pop(future), future.result()) for future in done]
                            for name, (summary, pv_summary) in finished:
                                summaries[name] = summary
                                if pv_summary is not None:
                                    pv_summaries[name] = pv_summary
                                manifest["completed"].append(name)
                                _atomic_json(manifest_path, manifest)
                                submit_next()
                    except BaseException:
                        # Python 3.14 can terminate running workers promptly on failure.
                        if hasattr(executor, "terminate_workers"):
                            executor.terminate_workers()
                        raise
                # Preserve the configured profile order in all aggregate outputs.
                summaries = {name: summaries[name] for name, _, _ in cases}
                pv_summaries = {name: pv_summaries[name] for name, _, _ in cases} if pv_summaries else {}
            else:
                for index, (name, profile, strategy) in enumerate(cases, start=1):
                    announce(index, name)
                    case_fingerprint = hashlib.sha256(f"{fingerprint}:{name}".encode()).hexdigest()
                    summaries[name] = replay_continuous_strategy(config, _profile_inputs(prepared, profile), strategy,
                        destination / "strategies" / name, resume=resume, progress=progress,
                        config_fingerprint=case_fingerprint)
                    if study.get("matched_pv_only", False):
                        pv_fingerprint = hashlib.sha256(f"{fingerprint}:{name}:pv_only".encode()).hexdigest()
                        pv_summaries[name] = replay_continuous_strategy(pv_config,
                            _profile_inputs(pv_prepared, profile), strategy,
                            destination / "pv_only" / name, resume=resume, progress=progress,
                            config_fingerprint=pv_fingerprint)
                    manifest["completed"].append(name)
                    _atomic_json(manifest_path, manifest)
            ranking = _save_comparison(destination, summaries, profiles, prepared["days"], study.get("plots", True))
            if pv_summaries:
                from .continuous_pv_only import save_pv_only_comparison
                save_pv_only_comparison(destination, summaries, pv_summaries)
            manifest.update(status="complete", comparison=ranking,
                comparison_scope="Continuous realized cash profit; each profile carries its own physical and fee state. Final SOC may differ.")
            _atomic_json(manifest_path, manifest)
        except BaseException as error:
            manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            _atomic_json(manifest_path, manifest)
            raise
    return {"status": "complete", "run_id": run_id, "output_dir": str(destination),
            "delivery_days": len(prepared["days"]), "profiles": len(profiles)}

"""Run four matched continuous participation policies or inspect inputs offline."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from time import sleep

import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.numerical_tolerances import continuous_tolerances
from bess_pv_trading.participation import participation_profiles, profile_id
from bess_pv_trading.simulation.continuous_inputs import prepare_continuous_inputs

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("deterministic", "envelope", "ensemble_mean", "ensemble_stochastic")


def _json(path):
    """Read one pinned study artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def _common_config(config):
    """Keep all model and execution settings except the intended PV policy."""
    common = deepcopy(config)
    settings = common["continuous_optimization"]
    for key in ("pv_ensemble", "run_id", "resume"):
        settings.pop(key, None)
    return common


def summarize(configs, roots, output):
    """Compare completed fixed-profile results against matching PV-only replays."""
    inputs = [_json(root / "json/input_provenance.json") for root in roots]
    manifests = [_json(root / "json/manifest.json") for root in roots]
    saved_configs = [_json(root / "json/study_config.json") for root in roots]
    if any(_common_config(config) != _common_config(saved_configs[0]) for config in saved_configs):
        raise ValueError("Study configurations differ beyond the intended ensemble policy")
    if any(_common_config(saved) != _common_config(current)
           or saved["continuous_optimization"]["pv_ensemble"] != current["continuous_optimization"]["pv_ensemble"]
           for saved, current in zip(saved_configs, configs)):
        raise ValueError("Saved study configuration differs from the requested configuration")
    if any(saved["continuous_optimization"]["pv_ensemble"].get("planning_policy") != policy
           for saved, policy in zip(saved_configs, POLICIES)):
        raise ValueError("Saved study policy does not match its comparison slot")
    if any(item.get("status") != "complete" for item in manifests):
        raise ValueError("All four studies must be complete")
    days = manifests[0]["delivery_days"]
    for item, manifest in zip(inputs, manifests):
        if manifest["delivery_days"] != days or list(item["days"]) != days:
            raise ValueError("Study dates do not match")
        for key in ("plant", "asset_input_settings", "weather_forecasts", "settlement_values_sha256", "pv_ensemble"):
            if item[key] != inputs[0][key]:
                raise ValueError(f"Study input {key} differs")
        for day in days:
            for key in ("realized_pv_sha256", "auction_price_sha256", "skipped_auctions"):
                if item["days"][day][key] != inputs[0]["days"][day][key]:
                    raise ValueError(f"Study {key} differs on {day}")
    names = [profile_id(profile) for profile in participation_profiles(configs[0]["continuous_participation_study"])]
    settings = configs[0]["continuous_participation_study"]
    if settings.get("sequential_all_auctions_baseline", True):
        names.append("sequential_all_auctions")
    if settings.get("perfect_pv_baseline", True):
        names.append("perfect_pv_all_auctions")
    frames = []
    for policy, root, manifest in zip(POLICIES, roots, manifests):
        if len(manifest["completed"]) != len(names) or set(manifest["completed"]) != set(names):
            raise ValueError(f"Incomplete profiles for {policy}")
        daily = pd.read_csv(root / "csv/daily_participation.csv", dtype={"delivery_day": str, "profile": str})
        expected = {(name, day) for name in names for day in days}
        if daily.duplicated(["profile", "delivery_day"]).any() or set(zip(daily.profile, daily.delivery_day)) != expected:
            raise ValueError(f"Unmatched daily results for {policy}")
        paired = pd.read_csv(root / "csv/pv_only_comparison_daily.csv", dtype={"delivery_day": str, "strategy": str})
        daily = daily.merge(paired, left_on=["profile", "delivery_day"], right_on=["strategy", "delivery_day"],
                            validate="one_to_one", how="left")
        if daily.pv_only_profit_eur.isna().any():
            raise ValueError(f"Missing PV-only comparison for {policy}")
        tolerances = continuous_tolerances(configs[0])
        if not tolerances.close(daily.net_profit_eur, daily.pv_bess_profit_eur, monetary=True).all():
            raise ValueError(f"PV-only pairing disagrees with realized {policy} profit")
        if not tolerances.close(daily.incremental_profit_eur,
                daily.net_profit_eur - daily.pv_only_profit_eur, monetary=True).all():
            raise ValueError(f"PV-only incremental profit does not reconcile for {policy}")
        diagnostics = []
        for name in names:
            path = root / "strategies" / name / "csv/ensemble_diagnostics_daily.csv"
            if path.exists():
                gap = pd.read_csv(path, dtype={"delivery_day": str})[["delivery_day", "mean_min_gap_mwh"]]
                gap["profile"] = name
                diagnostics.append(gap)
        if diagnostics:
            daily = daily.merge(pd.concat(diagnostics, ignore_index=True),
                on=["profile", "delivery_day"], how="left", validate="one_to_one")
        else:
            daily["mean_min_gap_mwh"] = float("nan")
        daily["policy"] = policy
        frames.append(daily)
    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values(["profile", "policy", "delivery_day"]).reset_index(drop=True)
    monthly = daily.groupby(["profile", "policy"], sort=False).agg(
        profit_eur=("net_profit_eur", "sum"), absolute_imbalance_mwh=("absolute_imbalance_mwh", "sum"),
        signed_imbalance_mwh=("signed_imbalance_mwh", "sum"), curtailment_mwh=("curtailment_mwh", "sum"),
        pv_only_profit_eur=("pv_only_profit_eur", "sum"), incremental_profit_eur=("incremental_profit_eur", "sum"),
        initial_energy_mwh=("initial_energy_mwh", "first"), final_energy_mwh=("final_energy_mwh", "last")).reset_index()
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_comparison.csv", index=False)
    monthly.to_csv(output / "period_comparison.csv", index=False)
    lines = [f"# Farm {configs[0]['plant']['farm_id']}: continuous participation policy comparison", "",
        f"Local delivery dates: {days[0]} to {days[-1]} ({len(days)} days).", "",
        "Each profit cell shows PV+BESS / PV-only / incremental profit in EUR.", "",
        "| Participation / strategy | Deterministic | Envelope | Ensemble mean | Ensemble stochastic |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for name in names:
        cells = [name]
        for policy in POLICIES:
            selected = monthly.loc[monthly.profile.eq(name) & monthly.policy.eq(policy)]
            if len(selected) != 1:
                raise ValueError(f"Missing matched policy for {name}/{policy}")
            row = selected.iloc[0]
            cells.append(" / ".join(f"{row[column]:,.2f}" for column in
                ("profit_eur", "pv_only_profit_eur", "incremental_profit_eur")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Each physical cell shows absolute imbalance / signed imbalance / curtailment / initial to final stored energy, all in MWh.", "",
        "| Participation / strategy | Deterministic | Envelope | Ensemble mean | Ensemble stochastic |",
        "| --- | ---: | ---: | ---: | ---: |"])
    for name in names:
        cells = [name]
        for policy in POLICIES:
            row = monthly.loc[monthly.profile.eq(name) & monthly.policy.eq(policy)].iloc[0]
            cells.append(f"{row.absolute_imbalance_mwh:,.3f} / {row.signed_imbalance_mwh:,.3f} / "
                         f"{row.curtailment_mwh:,.3f} / {row.initial_energy_mwh:,.3f} → {row.final_energy_mwh:,.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    envelope = saved_configs[1]["continuous_optimization"]["pv_ensemble"]
    stochastic = saved_configs[3]["continuous_optimization"]["pv_ensemble"]
    lines += ["", "Profit uses actual auction prices and actual imbalance settlement, less execution costs. The planning price curve is a perfect-price oracle. The PV-only run uses the same forecast and participation policy. Differences exclude any value for stored energy.",
        "", "The ensemble uses the preceding day's 00 UTC weather cycle with assumed availability. "
        f"Envelope includes the updating deterministic member: {envelope.get('include_deterministic_member', True)}; "
        f"stochastic includes it: {stochastic.get('include_deterministic_member', True)}. "
        f"Stochastic planning penalty: {stochastic.get('planning_imbalance_penalty_eur_per_mwh')} EUR/MWh; "
        f"CVaR confidence: {stochastic.get('cvar_confidence', 0.95)}, "
        f"CVaR weight: {stochastic.get('cvar_weight_eur_per_mwh', 0.0)} EUR/MWh. "
        "Realized imbalance prices are not used in planning. Each profile carries its own state across days.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    # Plot daily values for each profile without generating per-day images.
    import matplotlib.pyplot as plt
    for name in names:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 11), layout="constrained")
        for policy in POLICIES:
            data = daily.loc[daily.profile.eq(name) & daily.policy.eq(policy)].sort_values("delivery_day")
            dates = pd.to_datetime(data.delivery_day)
            for axis, column in zip(axes, ("net_profit_eur", "absolute_imbalance_mwh", "curtailment_mwh", "mean_min_gap_mwh")):
                axis.plot(dates, data[column], marker=".", label=policy)
        for axis, label in zip(axes, ("Profit (EUR)", "Absolute imbalance (MWh)", "Curtailment (MWh)", "Mean-min PV gap (MWh)")):
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        axes[0].legend()
        fig.suptitle(name)
        fig.savefig(output / f"daily_metrics_{name}.png", dpi=140)
        plt.close(fig)
    return output / "summary.md"


def _run_policies_parallel(farm_ids, paths):
    """Run one process per farm and forecast policy; stop the group on failure."""
    processes = []
    readers = []
    environment = os.environ.copy()
    # Limit numerical-library thread pools; Gurobi keeps its configured thread count.
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
        environment[key] = "1"

    def forward_output(process, farm_id, label):
        """Tag each child line so overlapping progress remains attributable."""
        for line in process.stdout:
            print(f"[farm {farm_id} {label}] {line.rstrip()}", flush=True)

    try:
        # Start all farm-policy combinations before waiting for any one of them.
        for farm_id in farm_ids:
            for policy, path in zip(POLICIES, paths):
                command = [sys.executable, "-u", "-m", "bess_pv_trading", "experiment-continuous",
                           "--config", str(path), "--farm-id", str(farm_id)]
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                processes.append((farm_id, policy, process, command))
                reader = Thread(target=forward_output, args=(process, farm_id, policy), daemon=True)
                reader.start()
                readers.append(reader)
        pending = list(processes)
        while pending:
            for entry in pending[:]:
                farm_id, policy, process, command = entry
                code = process.poll()
                if code is None:
                    continue
                pending.remove(entry)
                if code:
                    raise subprocess.CalledProcessError(code, command, output=f"Farm {farm_id} policy {policy} failed")
            if pending:
                sleep(0.2)
    finally:
        # Also clean up children after interruption or a failed process launch.
        for _, _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, process, _ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for reader in readers:
            reader.join()


def main():
    """Preflight inputs or run all selected farms and policies concurrently."""
    parser = argparse.ArgumentParser(description=__doc__)
    farms = parser.add_mutually_exclusive_group(required=True)
    farms.add_argument("--farm-id", type=int, choices=(1, 345, 901, 910))
    farms.add_argument("--all-farms", action="store_true", help="Process farms 1, 345, 901 and 910 concurrently")
    parser.add_argument("--period", choices=("ten_days", "smoke"), default="ten_days")
    parser.add_argument("--check-only", action="store_true", help="Preflight local data only; no solve or output write")
    parser.add_argument("--report-only", action="store_true", help="Summarize existing completed runs only")
    args = parser.parse_args()
    if args.check_only and args.report_only:
        parser.error("--check-only and --report-only are mutually exclusive")
    paths = [ROOT / f"configs/full_experiments/participation_2026_01_{args.period}_{policy}.yaml" for policy in POLICIES]
    farm_ids = (1, 345, 901, 910) if args.all_farms else (args.farm_id,)
    if not args.check_only and not args.report_only:
        # Validate all configurations before starting the concurrent process group.
        for farm_id in farm_ids:
            _farm_configs(farm_id, paths)
        _run_policies_parallel(farm_ids, paths)
    for farm_id in farm_ids:
        run_farm(farm_id, args, paths)


def _farm_configs(farm_id, paths):
    """Resolve each policy and reject mismatched common study settings."""
    configs = [load_config(path, overrides={"farm_id": farm_id}) for path in paths]
    if any(_common_config(config) != _common_config(configs[0]) for config in configs):
        raise ValueError("Study configurations differ beyond the intended ensemble policy")
    return configs


def run_farm(farm_id, args, paths):
    """Preflight or report one farm after all selected policies finish."""
    print(f"Farm {farm_id}: {args.period}", flush=True)
    configs = _farm_configs(farm_id, paths)
    if args.check_only:
        # Inputs are identical across policies; preflight pins all prices and 50 members once.
        prepared = prepare_continuous_inputs(configs[0], progress=lambda message: print(message, flush=True))
        if prepared["imbalance_prices_eur_per_mwh"] is None:
            raise ValueError("Actual imbalance settlement data are required")
        profiles = participation_profiles(configs[0]["continuous_participation_study"])
        fallback = sum(item["pv_provenance"].get("realized_power_missing_fallback_intervals", 0)
                       for item in prepared["days_data"].values())
        print(f"Preflight OK: {len(prepared['days'])} days, {len(profiles)} DA-containing profiles, "
              f"50 weather members, {fallback} realized PV fallback intervals, "
              f"input fingerprint {prepared['input_fingerprint']}", flush=True)
        return
    roots = [ROOT / config["continuous_optimization"]["output_root"] / f"farm_{farm_id}" /
             config["continuous_optimization"]["run_id"] for config in configs]
    output = ROOT / "outputs/continuous_optimization/comparisons" / f"farm_{farm_id}" / f"2026_01_participation_{args.period}_four_policies"
    print(f"Comparison saved to {summarize(configs, roots, output)}", flush=True)


if __name__ == "__main__":
    main()

"""Compare paired SOC sweeps without changing either trajectory's results."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from time import sleep

import pandas as pd

from .ida3_horizon_comparison import _check_common_inputs
from .ida3_soc_sweep import _read_trajectory, _trajectory_root


def _load_reference_manifest(path):
    """Retry short reads while the older runner rewrites its manifest in place."""
    for attempt in range(5):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as error:
            if attempt == 4:
                raise ValueError(f"Reference manifest remained unreadable after retries: {path}") from error
            sleep(0.1)


def _index_results(results):
    """Index a manifest snapshot, rejecting ambiguous duplicate trajectories."""
    indexed = {}
    for result in results:
        key = (str(result["farm"]), int(result["target_soc_percent"]))
        if key in indexed:
            raise ValueError(f"Duplicate sweep trajectory {key}")
        indexed[key] = result
    return indexed


def _compatible_config(previous, current):
    """Verify both runs differ only in output location and intermediate policy."""
    left, right = deepcopy(previous), deepcopy(current)
    left_options = left["continuous_optimization"]
    right_options = right["continuous_optimization"]
    if left_options.get("ida3_enforce_intermediate_soc", True) is not True:
        raise ValueError("Reference run does not enforce the intermediate midnight")
    if right_options.get("ida3_enforce_intermediate_soc", True) is not False:
        raise ValueError("New run does not release the intermediate midnight")
    # Study metadata and file destinations do not affect replay decisions.
    for config in (left, right):
        config.pop("soc_sweep", None)
        options = config["continuous_optimization"]
        for key in ("output_root", "run_id", "ida3_enforce_intermediate_soc"):
            options.pop(key, None)
    if left != right:
        raise ValueError("Sweep configurations differ beyond the intended midnight constraint")


def _plot_differences(frame, farm, path):
    """Plot matched per-target differences, keeping one small figure per farm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = frame.loc[frame.farm == farm].sort_values("target_soc_percent")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout="constrained")
    measures = [("delta_net_profit_eur", "Profit change (€)"),
                ("delta_auction_revenue_eur", "Auction cash change (€)"),
                ("delta_absolute_imbalance_mwh", "Absolute imbalance change (MWh)"),
                ("delta_curtailment_mwh", "Curtailment change (MWh)")]
    for axis, (column, label) in zip(axes.flat, measures):
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.plot(data.target_soc_percent, data[column], marker="o", markersize=3)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("End-of-horizon target SOC (%)")
    figure.suptitle(f"Farm {farm}: final-only minus both-midnight target")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_soc_sweep_comparison(batch, baseline_batch, results, expected_farms, expected_targets):
    """Compare complete pairs from manifest snapshots, and identify every missing pair."""
    batch, baseline_batch = Path(batch), Path(baseline_batch)
    baseline_manifest = _load_reference_manifest(baseline_batch / "manifest.json")
    old = _index_results(baseline_manifest.get("results", []))
    new = _index_results(results)
    expected = [(str(farm), int(target)) for farm in expected_farms for target in expected_targets]
    records, missing, mismatched, evidence = [], [], [], {}
    fields = ("auction_revenue_eur", "net_profit_eur", "imbalance_cashflow_eur", "degradation_eur",
              "absolute_imbalance_mwh", "signed_imbalance_mwh", "curtailment_mwh", "pv_realized_energy_mwh",
              "charge_mwh", "discharge_mwh", "initial_soc_percent", "final_soc_percent",
              "mean_midnight_soc_percent", "midnights_at_target", "auction_fallbacks", "auction_solver_seconds")
    for farm, target in expected:
        left, right = old.get((farm, target)), new.get((farm, target))
        if left is None or right is None or left.get("status") != "complete" or right.get("status") != "complete":
            missing.append((farm, target, (left or {}).get("status", "not yet recorded"),
                            (right or {}).get("status", "not yet recorded")))
            continue
        old_dir, old_root = _trajectory_root(baseline_batch, farm, target)
        new_dir, new_root = _trajectory_root(batch, farm, target)
        old_config = json.loads((old_dir / "resolved_config.json").read_text())
        new_config = json.loads((new_dir / "resolved_config.json").read_text())
        try:
            _compatible_config(old_config, new_config)
            # The complete fingerprint includes target, pinned forecast/model
            # provenance and prices. Reject rather than silently compare mismatch.
            if not left.get("input_fingerprint") or left["input_fingerprint"] != right.get("input_fingerprint"):
                raise ValueError("prepared-input fingerprints differ (including forecast/model provenance)")
            evidence[f"{farm}/{target}"] = _check_common_inputs(old_root, new_root)
        except ValueError as error:
            # A bad pair must not hide other usable matches in a long sweep.
            mismatched.append((farm, target, str(error)))
            continue
        previous, _ = _read_trajectory(baseline_batch, left)
        current, _ = _read_trajectory(batch, right)
        row = {"farm": farm, "target_soc_percent": target}
        for field in fields:
            row[f"both_{field}"] = previous[field]
            row[f"final_only_{field}"] = current[field]
            row[f"delta_{field}"] = current[field] - previous[field]
        records.append(row)

    # Keep the original standalone reports untouched, and write comparison
    # artifacts separately so either run can be refreshed independently.
    output = batch / "baseline_comparison"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "paired_metrics.csv", index=False)
    new_manifest_path = batch / "manifest.json"
    new_manifest = json.loads(new_manifest_path.read_text()) if new_manifest_path.is_file() else {}
    old_checksums = baseline_manifest.get("source_checksums", {})
    new_checksums = new_manifest.get("source_checksums", {})
    changed_sources = sorted(name for name in set(old_checksums) | set(new_checksums)
                             if old_checksums.get(name) != new_checksums.get(name))
    check_record = {"paired_inputs": evidence, "reference_revision": baseline_manifest.get("git_revision"),
                    "final_only_revision": new_manifest.get("git_revision"),
                    "changed_source_files": changed_sources}
    (output / "input_checks.json").write_text(json.dumps(check_record, indent=2) + "\n")
    lines = ["# January SOC sweep: effect of releasing the intermediate midnight", "",
        "Every difference is final-only minus both-midnight target for the same farm and target. "
        "The two formulations retain the final 50% or other selected target and use greedy tracking. "
        "The tracker does not enforce the target on actual midnight SOC.", "",
        "Inputs are matched by complete prepared-input fingerprint and separately checked on every January "
        "realized PV, settlement-price and auction-price product. Forecast selection is retrospective and "
        "the auction-price curves are an oracle. Profit includes modeled degradation but does not value "
        "final inventory; differing actual final SOC limits the financial comparison.", "",
        f"Matched {len(records)} of {len(expected)} expected farm/target pairs. "
        f"Reference batch status when read: {baseline_manifest.get('status', 'unknown')}. "
        f"Recorded source files with differing checksums: {len(changed_sources)}.", "",
        "[All paired numeric results](paired_metrics.csv) · [Input checks](input_checks.json)", ""]
    if missing:
        lines.extend(["## Pairs not compared", ""])
        lines.extend(f"- Farm {farm}, {target}%: reference {left}; final-only {right}."
                     for farm, target, left, right in missing)
        lines.append("")
    if mismatched:
        lines.extend(["## Pairs excluded for incompatible inputs or configuration", ""])
        lines.extend(f"- Farm {farm}, {target}%: {reason}." for farm, target, reason in mismatched)
        lines.append("")
    if not frame.empty:
        lines.extend(["## Per-farm comparisons", ""])
        for farm in map(str, expected_farms):
            farm_frame = frame.loc[frame.farm == farm].sort_values("target_soc_percent")
            if farm_frame.empty:
                continue
            figure = output / f"farm_{farm}_comparison.png"
            _plot_differences(frame, farm, figure)
            report = [f"# Farm {farm}: releasing the intermediate midnight", "",
                "Differences are final-only minus both-midnight target. Profit excludes final-inventory value; "
                "forecast selection and auction-price information are retrospective.", "",
                "| Target (%) | Both profit (€) | Final-only profit (€) | Δ profit (€) | Δ auction cash (€) | "
                "Δ settlement (€) | Δ wear (€) | Δ imbalance (MWh) | Δ curtailment (MWh) | "
                "Both final SOC (%) | Final-only final SOC (%) | Both / final-only fallbacks |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for item in farm_frame.to_dict("records"):
                report.append(f"| {item['target_soc_percent']} | {item['both_net_profit_eur']:,.2f} | "
                    f"{item['final_only_net_profit_eur']:,.2f} | {item['delta_net_profit_eur']:+,.2f} | "
                    f"{item['delta_auction_revenue_eur']:+,.2f} | {item['delta_imbalance_cashflow_eur']:+,.2f} | "
                    f"{item['delta_degradation_eur']:+,.2f} | {item['delta_absolute_imbalance_mwh']:+,.3f} | "
                    f"{item['delta_curtailment_mwh']:+,.3f} | {item['both_final_soc_percent']:.2f} | "
                    f"{item['final_only_final_soc_percent']:.2f} | "
                    f"{item['both_auction_fallbacks']} / {item['final_only_auction_fallbacks']} |")
            report.extend(["", f"![Farm {farm} changes](farm_{farm}_comparison.png)", "",
                           "[Comparison summary](summary.md)", ""])
            (output / f"farm_{farm}_summary.md").write_text("\n".join(report))
            lines.append(f"- [Farm {farm}](farm_{farm}_summary.md)")
        lines.append("")
    path = output / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path

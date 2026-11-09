"""Compare completed K25 batches on matched farms, days and input snapshots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import numpy as np


KEYS = ["farm_id", "window_id", "case_id", "controller", "delivery_day"]
COMBINATION = KEYS[:-1]
SUM_METRICS = ["absolute_imbalance_mwh", "pv_curtailment_mwh", "realized_operating_profit_eur",
               "degradation_eur", "mpc_fallbacks"]
REQUIRED_METRICS = SUM_METRICS + ["soc_final_percent"]


def load_batch(path):
    """Require a fully reported batch and pin each trajectory's input identity."""
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Batch is not complete: {path}")
    jobs = manifest["plan"]["jobs"]
    results = {(item["job_id"], item["controller"]): item for item in manifest["results"]}
    if len(results) != len(manifest["results"]):
        raise ValueError(f"Duplicate trajectory results in {path}")
    expected = {}
    targets = {}
    for job in jobs:
        farm = str(job["farm_id"])
        config_path = path / "runs" / f"farm_{farm}" / job["job_id"] / "continuous" / "resolved_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        target = config["state"]["terminal_soc_percent"]
        if target is None:
            raise ValueError(f"Missing terminal SOC target in {config_path}")
        for controller in job["controllers"]:
            key = (farm, job["window_id"], job["case_id"], controller)
            result = results.get((job["job_id"], controller))
            if key in expected or result is None or result.get("status") != "complete" or not result.get("input_fingerprint"):
                raise ValueError(f"Missing, duplicate or incomplete trajectory {key} in {path}")
            expected[key] = {"fingerprint": result["input_fingerprint"], "days": set(job["days"])}
            targets[key] = float(target)
            if not math.isfinite(targets[key]):
                raise ValueError(f"Invalid terminal SOC target in {config_path}")
    if len(results) != len(expected):
        raise ValueError(f"Unexpected trajectory results in {path}")
    frame = pd.read_csv(path / "daily_metrics.csv", dtype={key: str for key in KEYS})
    missing = set(KEYS + REQUIRED_METRICS) - set(frame.columns)
    if missing or frame.duplicated(KEYS).any():
        raise ValueError(f"Incomplete or duplicate daily metrics in {path}: {sorted(missing)}")
    frame[REQUIRED_METRICS] = frame[REQUIRED_METRICS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[REQUIRED_METRICS].to_numpy()).all():
        raise ValueError(f"Missing or nonfinite numerical daily metrics in {path}")
    actual = {tuple(values) for values in frame[COMBINATION].itertuples(index=False, name=None)}
    if actual != set(expected):
        raise ValueError(f"Daily metrics do not match planned trajectories in {path}")
    for key, group in frame.groupby(COMBINATION, sort=False):
        if set(group["delivery_day"]) != expected[key]["days"]:
            raise ValueError(f"Incomplete delivery days for {key} in {path}")
    return {"path": path, "manifest": manifest, "expected": expected, "targets": targets, "daily": frame}


def compare_batches(baseline, candidate, soc_tolerance_percent=0.01):
    """Match every delivery day, reject changed input fingerprints and aggregate metrics."""
    if not math.isfinite(soc_tolerance_percent) or soc_tolerance_percent < 0:
        raise ValueError("SOC tolerance must be finite and nonnegative")
    old = load_batch(baseline)
    new = load_batch(candidate)
    if set(old["expected"]) != set(new["expected"]):
        raise ValueError("Batches have different farm, case or controller combinations")
    for key in old["expected"]:
        if old["expected"][key] != new["expected"][key]:
            raise ValueError(f"Input fingerprint or delivery days differ for {key}")
        if old["targets"][key] != new["targets"][key]:
            raise ValueError(f"Configured terminal SOC targets differ for {key}")
    left = old["daily"][KEYS + REQUIRED_METRICS]
    right = new["daily"][KEYS + REQUIRED_METRICS]
    paired = left.merge(right, on=KEYS, how="outer", suffixes=("_old", "_new"), indicator=True,
                        validate="one_to_one")
    if not paired["_merge"].eq("both").all():
        raise ValueError("Batches have unmatched delivery dates")
    paired = paired.drop(columns="_merge").sort_values(KEYS)
    records = []
    for key, frame in paired.groupby(COMBINATION, sort=True):
        target = new["targets"][key]
        row = dict(zip(COMBINATION, key))
        row.update(days=len(frame), target_soc_percent=target)
        for metric in SUM_METRICS:
            for suffix in ("old", "new"):
                row[f"{metric}_{suffix}"] = float(frame[f"{metric}_{suffix}"].sum())
            row[f"{metric}_change"] = row[f"{metric}_new"] - row[f"{metric}_old"]
        for suffix in ("old", "new"):
            soc = frame[f"soc_final_percent_{suffix}"]
            row[f"mean_end_soc_percent_{suffix}"] = float(soc.mean())
            row[f"days_at_target_{suffix}"] = int((soc - target).abs().le(soc_tolerance_percent).sum())
        records.append(row)
    return old, new, paired, pd.DataFrame(records)


def save_soc_plots(paired, summary, output):
    """Plot each farm's matched daily end SOC; one tracker is shown per farm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for farm, data in paired.groupby("farm_id", sort=True):
        combinations = []
        tracker_included = False
        for (case, controller), rows in data.groupby(["case_id", "controller"], sort=True):
            if controller == "track_committed_grid":
                if tracker_included:
                    continue
                tracker_included = True
                label = "Grid tracker"
            else:
                label = case
            combinations.append((label, rows))
        figure, axes = plt.subplots(len(combinations), 1, figsize=(12, 2.8 * len(combinations)),
                                    sharex=True, squeeze=False)
        for axis, (label, rows) in zip(axes[:, 0], combinations):
            dates = pd.to_datetime(rows["delivery_day"])
            target = summary.loc[summary.farm_id == farm, "target_soc_percent"].iloc[0]
            axis.plot(dates, rows["soc_final_percent_old"], label="Previous", alpha=0.75)
            axis.plot(dates, rows["soc_final_percent_new"], label="New", alpha=0.75)
            axis.axhline(target, color="black", linestyle=":", label=f"Target {target:g}%")
            axis.set(title=label, ylabel="End SOC (%)", ylim=(0, 100))
            axis.grid(alpha=0.2)
            axis.legend(loc="best")
        figure.suptitle(f"Farm {farm}: daily end-of-day SOC")
        figure.tight_layout()
        figure.savefig(output / f"farm_{farm}_daily_soc.png", dpi=150)
        plt.close(figure)


def write_comparison(old, new, paired, summary, output, tolerance):
    """Write matched daily data, aggregate metrics, a Markdown table and SOC plots."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paired.to_csv(output / "matched_daily_metrics.csv", index=False)
    summary.to_csv(output / "comparison.csv", index=False)
    lines = ["# K25 formulation comparison", "",
        f"Previous batch: `{old['path']}`  ", f"New batch: `{new['path']}`  ",
        f"A day counts as at target within {tolerance:g} percentage points of its configured SOC.", "",
        "Both auction planning and MPC settings changed; these are full-policy differences, not isolated MPC effects.",
        "Grid tracking is reused across cases; do not sum its repeated rows. The reviewed PV data include synthetic labels.", "",
        "| Farm | Case | Controller | Days | Imbalance MWh old → new | Curtailment MWh old → new | Profit € old → new | Mean end SOC % old → new | Days at target old → new | MPC fallbacks old → new |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['farm_id']} | {row['case_id']} | {row['controller']} | {row['days']} | "
            f"{row['absolute_imbalance_mwh_old']:.2f} → {row['absolute_imbalance_mwh_new']:.2f} | "
            f"{row['pv_curtailment_mwh_old']:.2f} → {row['pv_curtailment_mwh_new']:.2f} | "
            f"{row['realized_operating_profit_eur_old']:.2f} → {row['realized_operating_profit_eur_new']:.2f} | "
            f"{row['mean_end_soc_percent_old']:.2f} → {row['mean_end_soc_percent_new']:.2f} | "
            f"{row['days_at_target_old']} → {row['days_at_target_new']} | "
            f"{int(row['mpc_fallbacks_old'])} → {int(row['mpc_fallbacks_new'])} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_soc_plots(paired, summary, output)


def main(argv=None):
    """Read completed reports only; no forecast preflight or optimization is run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New output directory; defaults to candidate/formulation_comparison")
    parser.add_argument("--soc-tolerance-percent", type=float, default=0.01)
    args = parser.parse_args(argv)
    old, new, paired, summary = compare_batches(args.baseline, args.candidate, args.soc_tolerance_percent)
    output = args.output or new["path"] / "formulation_comparison"
    write_comparison(old, new, paired, summary, output, args.soc_tolerance_percent)
    print(f"Comparison written to {output / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

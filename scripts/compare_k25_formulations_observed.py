"""Compare completed K25 batches after checking their observed replay inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_k25_batches import KEYS, load_batch


def _matched(left, right, keys, label):
    """Require a one-to-one match before comparing any aligned numerical values."""
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError(f"Duplicate {label} keys")
    data = left.merge(right, on=keys, how="outer", suffixes=("_old", "_new"),
                      indicator=True, validate="one_to_one")
    if not data["_merge"].eq("both").all():
        raise ValueError(f"Different {label} keys")
    return data


def compare_observed_inputs(old_dispatch, new_dispatch, old_trades, new_trades):
    """Check physical PV and actual prices; disclose changes to recorded PV forecasts."""
    aligned = _matched(old_dispatch, new_dispatch, ["interval_id"], "dispatch interval")
    for field in ("pv_ac_mw", "imbalance_price_eur_mwh"):
        old = pd.to_numeric(aligned[f"{field}_old"], errors="raise").to_numpy()
        new = pd.to_numeric(aligned[f"{field}_new"], errors="raise").to_numpy()
        if not np.isfinite(old).all() or not np.isfinite(new).all() or not np.allclose(old, new, rtol=0, atol=1e-9):
            raise ValueError(f"Different realized {field} values")
    # Forecasts can depend on the issued controller action. Do not treat
    # a changed forecast as identical input evidence; report each difference.
    prior = pd.to_numeric(aligned.pv_ac_forecast_mw_old, errors="raise")
    current = pd.to_numeric(aligned.pv_ac_forecast_mw_new, errors="raise")
    if ((prior.notna() & ~np.isfinite(prior)).any() or (current.notna() & ~np.isfinite(current)).any()):
        raise ValueError("Nonfinite recorded PV forecast")
    missing_mismatch = prior.isna() != current.isna()
    delta = (prior - current).abs()
    vintage_changes = (aligned.pv_forecast_vintage_utc_old.fillna("")
                       != aligned.pv_forecast_vintage_utc_new.fillna("")).sum()
    trades = _matched(old_trades, new_trades, ["auction", "interval_id"], "auction price")
    old_prices = pd.to_numeric(trades.price_eur_per_mwh_old, errors="raise").to_numpy()
    new_prices = pd.to_numeric(trades.price_eur_per_mwh_new, errors="raise").to_numpy()
    if not np.isfinite(old_prices).all() or not np.isfinite(new_prices).all() or not np.allclose(
            old_prices, new_prices, rtol=0, atol=1e-9):
        raise ValueError("Different realized auction clearing prices")
    return {"intervals": len(aligned), "auction_products": len(trades),
        "forecast_values_changed": int((delta.gt(1e-9) | missing_mismatch).sum()),
        "maximum_forecast_difference_mw": float(delta.max()) if delta.notna().any() else None,
        "forecast_vintages_changed": int(vintage_changes)}


def _path(batch, result):
    """Resolve the controller output within its saved batch directory."""
    return (batch / "runs" / f"farm_{result['farm_id']}" / result["job_id"] /
            "continuous" / "controllers" / result["controller"] / "csv")


def compare_batches_observed(old, new, output):
    """Compare matched trajectories, without claiming their distinct fingerprints are equal."""
    previous = load_batch(old)
    current = load_batch(new)
    if set(previous["expected"]) != set(current["expected"]):
        raise ValueError("Batches contain different trajectory combinations")
    for key, value in previous["expected"].items():
        if value["days"] != current["expected"][key]["days"] or previous["targets"][key] != current["targets"][key]:
            raise ValueError(f"Delivery dates or configured SOC target differ for {key}")
    old_results = {(r["job_id"], r["controller"]): r for r in previous["manifest"]["results"]}
    new_results = {(r["job_id"], r["controller"]): r for r in current["manifest"]["results"]}
    if set(old_results) != set(new_results):
        raise ValueError("Batch result identities differ")
    evidence = []
    seen = set()
    for key, old_result in old_results.items():
        new_result = new_results[key]
        if old_result.get("reused_from_job_id") or new_result.get("reused_from_job_id"):
            if old_result.get("reused_from_job_id") != new_result.get("reused_from_job_id"):
                raise ValueError("Tracker reuse differs between batches")
            continue
        old_path, new_path = _path(previous["path"], old_result), _path(current["path"], new_result)
        columns = ["interval_id", "pv_ac_mw", "imbalance_price_eur_mwh", "pv_ac_forecast_mw", "pv_forecast_vintage_utc"]
        price_columns = ["auction", "interval_id", "price_eur_per_mwh"]
        verified = compare_observed_inputs(pd.read_csv(old_path / "realized_dispatch.csv", usecols=columns),
            pd.read_csv(new_path / "realized_dispatch.csv", usecols=columns),
            pd.read_csv(old_path / "accepted_trades.csv", usecols=price_columns),
            pd.read_csv(new_path / "accepted_trades.csv", usecols=price_columns))
        evidence.append({"farm": old_result["farm_id"], "job_id": old_result["job_id"],
            "controller": old_result["controller"], **verified})
        seen.add(key)
    # Match full daily results first. Reused trackers are later displayed once.
    paired = _matched(previous["daily"], current["daily"], KEYS, "daily trajectory")
    paired = paired.drop(columns="_merge")
    groups = []
    for key, daily in paired.groupby(KEYS[:-1], sort=True):
        farm, window, case, controller = key
        job_id = next(j["job_id"] for j in previous["manifest"]["plan"]["jobs"]
                      if str(j["farm_id"]) == farm and j["window_id"] == window and j["case_id"] == case)
        if (job_id, controller) not in seen:
            continue
        target = previous["targets"][key]
        row = {"farm": farm, "case": case, "controller": controller, "days": len(daily)}
        for metric in ("absolute_imbalance_mwh", "pv_curtailment_mwh", "realized_operating_profit_eur",
                       "degradation_eur", "mpc_fallbacks"):
            row[f"{metric}_old"] = float(daily[f"{metric}_old"].sum())
            row[f"{metric}_new"] = float(daily[f"{metric}_new"].sum())
        for period in ("old", "new"):
            end = daily[f"soc_final_percent_{period}"]
            row[f"mean_end_soc_{period}"] = float(end.mean())
            row[f"days_at_target_{period}"] = int((end - target).abs().le(0.01).sum())
        # Settlement cash is recorded in each controller's final summary.
        for period, batch in (("old", previous["path"]), ("new", current["path"])):
            summary_path = batch / "runs" / f"farm_{farm}" / job_id / "continuous" / f"summary__{controller}.json"
            summary = json.loads(summary_path.read_text())
            row[f"imbalance_cashflow_eur_{period}"] = summary["components"]["imbalance_cashflow_eur"]
            row[f"pv_mwh_{period}"] = summary["pv_realized_energy_mwh"]
        groups.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(groups).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(evidence).to_csv(output / "input_checks.csv", index=False)
    paired.to_csv(output / "matched_daily_metrics.csv", index=False)
    lines = ["# K25 formulation comparison using observed inputs", "",
        f"Previous: `{previous['path']}`  ", f"New: `{current['path']}`", "",
        "Each row covers the same delivery dates and uses matching observed PV, actual settlement prices "
        "and clearing prices at matched auction products. The batch input fingerprints are different; these "
        "checks do not prove that every upstream input or assumption is identical.", "",
        "Multiple settings can differ between batches. Differences describe the entire policy change "
        "and cannot be attributed to any single setting without a controlled comparison. Consult each "
        "batch's resolved configurations and formulation summaries.", "",
        "## Saved formulation settings", "",
        "| Farm | Case | Auction terminal scope old → new | MPC terminal mode old → new | MPC objective old → new |",
        "| --- | --- | --- | --- | --- |"]
    inspected_configs = []
    for job in previous["manifest"]["plan"]["jobs"]:
        farm, case, job_id = str(job["farm_id"]), job["case_id"], job["job_id"]
        suffix = Path("runs") / f"farm_{farm}" / job_id / "continuous/resolved_config.json"
        old_config = json.loads((previous["path"] / suffix).read_text())
        new_config = json.loads((current["path"] / suffix).read_text())
        inspected_configs.extend((old_config, new_config))
        scopes = [cfg.get("continuous_optimization", {}).get("terminal_soc_scope", "final_replay_day")
                  for cfg in (old_config, new_config)]
        modes = [cfg.get("realtime_mpc", {}).get("terminal_inventory_mode", "not specified")
                 for cfg in (old_config, new_config)]
        objectives = [cfg.get("realtime_mpc", {}).get("planning_objective", "imbalance_and_costs")
                      for cfg in (old_config, new_config)]
        lines.append(f"| {farm} | {case} | {scopes[0]} → {scopes[1]} | {modes[0]} → {modes[1]} | "
                     f"{objectives[0]} → {objectives[1]} |")
    lines.extend(["", "## Physical results", "",
        "| Farm | Case | Controller | Days | Imbalance old → new (MWh) | Change | Curtailment old → new (MWh) | Change | PV production (MWh) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for r in groups:
        x,y=r["absolute_imbalance_mwh_old"],r["absolute_imbalance_mwh_new"]
        a,b=r["pv_curtailment_mwh_old"],r["pv_curtailment_mwh_new"]
        imbalance_change="n/a" if x == 0 else f"{100 * (y-x)/x:+.1f}%"
        curtailment_change="n/a" if a == 0 else f"{100 * (b-a)/a:+.1f}%"
        lines.append(f"| {r['farm']} | {r['case']} | {r['controller']} | {r['days']} | {x:.2f} → {y:.2f} | "
                     f"{imbalance_change} | {a:.2f} → {b:.2f} | {curtailment_change} | {r['pv_mwh_new']:.2f} |")
    lines.extend(["", "## Financial and battery results", "",
        "| Farm | Case | Controller | Profit old → new (€) | Settlement old → new (€) | Degradation old → new (€) | Mean daily end SOC old → new | Days at target old → new | MPC fallbacks old → new |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for r in groups:
        lines.append(f"| {r['farm']} | {r['case']} | {r['controller']} | "
            f"{r['realized_operating_profit_eur_old']:,.2f} → {r['realized_operating_profit_eur_new']:,.2f} | "
            f"{r['imbalance_cashflow_eur_old']:,.2f} → {r['imbalance_cashflow_eur_new']:,.2f} | "
            f"{r['degradation_eur_old']:,.2f} → {r['degradation_eur_new']:,.2f} | "
            f"{r['mean_end_soc_old']:.2f}% → {r['mean_end_soc_new']:.2f}% | "
            f"{r['days_at_target_old']} → {r['days_at_target_new']} | "
            f"{int(r['mpc_fallbacks_old'])} → {int(r['mpc_fallbacks_new'])} |")
    lines.extend(["", "## Input checks and limitations", "",
        "- Tracker trajectories reused across cases are shown once. Actual final SOC may differ between "
        "controllers, and profit has no adjustment for remaining battery energy.",
        "- Net realized imbalance and settlement do not measure the forecast's standalone accuracy."])
    if any(cfg.get("forecast", {}).get("curated_realized", {}).get("enabled") for cfg in inspected_configs):
        lines.append("- Curated PV replay values can include synthetic counterfactual labels.")
    if any(cfg.get("pv", {}).get("forecast_replacement", {}).get("allow_retrospective_artifacts")
           for cfg in inspected_configs):
        lines.append("- The configuration permits retrospectively selected PV model artifacts.")
    price_oracle = any(job.get("auction_price_information") == "perfect_foresight_actual_clearing_curves"
                       for job in previous["manifest"]["plan"]["jobs"])
    if price_oracle:
        lines.append("- Auction optimization uses actual clearing-price curves as a retrospective price oracle.")
    lines.extend(["",
        "| Farm | Case | Controller | Matched intervals | Matched auction products | Forecast values changed | Max forecast difference (MW) | Forecast vintages changed |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for r in evidence:
        difference = "n/a" if r["maximum_forecast_difference_mw"] is None else f"{r['maximum_forecast_difference_mw']:.8f}"
        lines.append(f"| {r['farm']} | {r['job_id']} | {r['controller']} | {r['intervals']} | "
            f"{r['auction_products']} | {r['forecast_values_changed']} | {difference} | {r['forecast_vintages_changed']} |")
    lines.extend(["", "[Comparison data](comparison.csv) · [Input checks](input_checks.csv) · "
                  "[Matched daily data](matched_daily_metrics.csv)", ""])
    path = output / "comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None):
    """Read and compare finished results only; no optimization is run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.current / "formulation_comparison_observed"
    print(f"Comparison written to {compare_batches_observed(args.previous, args.current, output)}")


if __name__ == "__main__":
    main()

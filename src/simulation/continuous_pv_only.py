"""Matched forecast-based PV-only continuous replay and paired cash accounting."""
from copy import deepcopy
from dataclasses import replace

import pandas as pd

from ..domain import battery_parameters_from_config


def pv_only_inputs(config, prepared):
    """Share pinned market and weather inputs while removing the physical battery."""
    child = deepcopy(config)
    child["battery"].update(capacity_mwh=0.0, max_power_mw=0.0)
    child["state"].update(terminal_soc_percent=None, terminal_soc_min_percent=None,
                          cycle_segment_energy_mwh=None)
    child["degradation"] = {"model": "throughput", "cost_eur_per_mwh_throughput": 0.0}
    source = dict(prepared)
    source["plant"] = replace(prepared["plant"], battery=battery_parameters_from_config(child["battery"]))
    source.update(initial_energy_mwh=0.0, terminal_energy_mwh=None, terminal_minimum_energy_mwh=None)
    return child, source


def save_pv_only_comparison(destination, battery_summaries, pv_summaries):
    """Pair each trajectory with its own state-independent no-battery replay."""
    if set(battery_summaries) != set(pv_summaries):
        raise ValueError("PV-only comparison requires one matched counterfactual per strategy")
    records = []
    for name, battery in battery_summaries.items():
        baseline = pv_summaries[name]
        left = pd.DataFrame(battery["daily_kpis"])
        right = pd.DataFrame(baseline["daily_kpis"])
        merged = left.merge(right, on="delivery_day", suffixes=("_pv_bess", "_pv_only"),
                            validate="one_to_one", how="outer", indicator=True)
        if not merged._merge.eq("both").all():
            raise ValueError(f"PV-only dates do not match strategy {name}")
        for row in merged.to_dict("records"):
            records.append({"strategy": name, "delivery_day": row["delivery_day"],
                "pv_bess_profit_eur": row["net_profit_eur_pv_bess"],
                "pv_only_profit_eur": row["net_profit_eur_pv_only"],
                "incremental_profit_eur": row["net_profit_eur_pv_bess"] - row["net_profit_eur_pv_only"],
                "pv_bess_initial_energy_mwh": row["initial_energy_mwh_pv_bess"],
                "pv_bess_final_energy_mwh": row["final_energy_mwh_pv_bess"],
                "pv_only_validation_warning_count": row.get("validation_warning_count_pv_only", 0),
                "pv_only_initial_energy_mwh": row["initial_energy_mwh_pv_only"],
                "pv_only_final_energy_mwh": row["final_energy_mwh_pv_only"]})
    daily = pd.DataFrame(records)
    (destination / "csv").mkdir(exist_ok=True)
    daily.to_csv(destination / "csv/pv_only_comparison_daily.csv", index=False)
    rows = []
    for name, frame in daily.groupby("strategy", sort=False):
        for month, month_frame in frame.groupby(frame.delivery_day.str[:7], sort=True):
            rows.append({"strategy": name, "delivery_month": month,
                "pv_bess_profit_eur": month_frame.pv_bess_profit_eur.sum(),
                "pv_only_profit_eur": month_frame.pv_only_profit_eur.sum(),
                "incremental_profit_eur": month_frame.incremental_profit_eur.sum(),
                "pv_bess_initial_energy_mwh": month_frame.pv_bess_initial_energy_mwh.iloc[0],
                "pv_bess_final_energy_mwh": month_frame.pv_bess_final_energy_mwh.iloc[-1],
                "pv_only_initial_energy_mwh": month_frame.pv_only_initial_energy_mwh.iloc[0],
                "pv_only_final_energy_mwh": month_frame.pv_only_final_energy_mwh.iloc[-1]})
    pd.DataFrame(rows).to_csv(destination / "csv/pv_only_comparison_monthly.csv", index=False)

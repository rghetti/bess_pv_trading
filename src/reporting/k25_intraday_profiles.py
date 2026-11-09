"""Summarize realized imbalance, curtailment and commitments by local time."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def prepare_interval_profile(dispatch, timezone, battery_capacity_mwh):
    """Convert executed intervals to local quarter-hours and MWh quantities."""
    required = {"start", "delivery_day", "duration_hours", "curtailment_mw", "imbalance_mwh",
                "net_position_mwh", "grid_mw", "charge_mw", "discharge_mw", "energy_start_mwh"}
    if not required.issubset(dispatch.columns) or dispatch.empty:
        raise ValueError(f"Missing intraday dispatch fields: {sorted(required - set(dispatch.columns))}")
    data = dispatch[list(required)].copy()
    utc = pd.to_datetime(data.start, utc=True, errors="raise")
    if utc.duplicated().any():
        raise ValueError("Duplicate UTC intervals in dispatch")
    local = utc.dt.tz_convert(timezone)
    if not local.dt.strftime("%Y-%m-%d").eq(data.delivery_day.astype(str)).all():
        raise ValueError("Local interval dates do not match delivery days")
    if (local.dt.minute.mod(15).ne(0) | local.dt.second.ne(0) | local.dt.microsecond.ne(0)).any():
        raise ValueError("Dispatch interval starts are not aligned to quarter-hours")
    numeric = ["duration_hours", "curtailment_mw", "imbalance_mwh", "net_position_mwh", "grid_mw",
               "charge_mw", "discharge_mw", "energy_start_mwh"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="raise")
    if not data[numeric].map(math.isfinite).all().all():
        raise ValueError("Nonfinite intraday dispatch data")
    if not data.duration_hours.between(0.25 - 1e-9, 0.25 + 1e-9).all():
        raise ValueError("Intraday profiles require 15-minute settlement intervals")
    if (data.curtailment_mw < -1e-7).any():
        raise ValueError("Negative physical curtailment")
    if not math.isfinite(battery_capacity_mwh) or battery_capacity_mwh <= 0:
        raise ValueError("Battery capacity must be positive and finite")
    if not (data.grid_mw * data.duration_hours - data.net_position_mwh - data.imbalance_mwh).abs().le(1e-7).all():
        raise ValueError("Realized grid energy, commitments and imbalance do not reconcile")
    # Preserve repeated autumn clock hours as distinct UTC observations, and
    # average the available samples rather than inventing spring DST intervals.
    data["month"] = local.dt.strftime("%Y-%m")
    data["slot"] = local.dt.hour * 4 + local.dt.minute // 15
    data["hour"] = local.dt.hour
    data["absolute_imbalance_mwh"] = data.imbalance_mwh.abs()
    data["curtailment_mwh"] = data.curtailment_mw.clip(lower=0) * data.duration_hours
    data["grid_energy_mwh"] = data.grid_mw * data.duration_hours
    data["charge_energy_mwh"] = data.charge_mw * data.duration_hours
    data["discharge_energy_mwh"] = data.discharge_mw * data.duration_hours
    data["soc_start_percent"] = 100 * data.energy_start_mwh / battery_capacity_mwh
    data["absolute_net_position_mwh"] = data.net_position_mwh.abs()
    return data


def aggregate_intraday(data, grouping):
    """Calculate observation-weighted averages and accumulated MWh without netting imbalance."""
    return data.groupby(grouping, sort=True, dropna=False).agg(
        samples=("absolute_imbalance_mwh", "size"),
        avg_absolute_imbalance_mwh=("absolute_imbalance_mwh", "mean"),
        total_absolute_imbalance_mwh=("absolute_imbalance_mwh", "sum"),
        avg_curtailment_mwh=("curtailment_mwh", "mean"),
        total_curtailment_mwh=("curtailment_mwh", "sum"),
        avg_net_position_mwh=("net_position_mwh", "mean"),
        avg_absolute_net_position_mwh=("absolute_net_position_mwh", "mean"),
        avg_grid_energy_mwh=("grid_energy_mwh", "mean"),
        avg_charge_energy_mwh=("charge_energy_mwh", "mean"),
        avg_discharge_energy_mwh=("discharge_energy_mwh", "mean"),
        avg_soc_start_percent=("soc_start_percent", "mean"),
    ).reset_index()


def prepare_price_samples(dispatch, ledger, intervals, farm, label):
    """Align actual auction and settlement prices with their local delivery products."""
    times = intervals[["month", "slot"]].copy()
    times["interval_id"] = dispatch["interval_id"].to_numpy()
    auctions = ledger[["interval_id", "auction", "price_eur_per_mwh"]].merge(
        times, on="interval_id", how="left", validate="many_to_one")
    if auctions[["month", "slot"]].isna().any().any() or auctions.duplicated(["interval_id", "auction"]).any():
        raise ValueError("Auction price records do not align uniquely with delivery products")
    auctions = auctions.rename(columns={"auction": "curve"})
    settlement = times.copy()
    settlement["price_eur_per_mwh"] = pd.to_numeric(dispatch.imbalance_price_eur_mwh, errors="raise").to_numpy()
    # A separate shortage price, when present, must not be silently replaced by the surplus price.
    if "imbalance_negative_price_eur_mwh" in dispatch:
        negative = pd.to_numeric(dispatch.imbalance_negative_price_eur_mwh, errors="raise").to_numpy()
        if not pd.Series(negative).equals(pd.Series(settlement.price_eur_per_mwh.to_numpy())):
            shortage = settlement.copy()
            shortage["price_eur_per_mwh"] = negative
            shortage["curve"] = "Imbalance shortfall"
            settlement["curve"] = "Imbalance surplus"
            settlement = pd.concat([settlement, shortage], ignore_index=True)
        else:
            settlement["curve"] = "Imbalance"
    else:
        settlement["curve"] = "Imbalance"
    samples = pd.concat([auctions, settlement], ignore_index=True)
    samples["price_eur_per_mwh"] = pd.to_numeric(samples.price_eur_per_mwh, errors="raise")
    if not samples.price_eur_per_mwh.map(math.isfinite).all():
        raise ValueError("Nonfinite realized market price")
    samples["farm"] = farm
    samples["label"] = label
    return samples


def aggregate_monthly_prices(samples):
    """Avoid double-counting shared prices and distinguish genuinely different case curves."""
    output = []
    for farm, farm_samples in samples.groupby("farm", sort=True):
        records = farm_samples.copy()
        identity = ["interval_id", "curve"]
        ranges = records.groupby(identity).price_eur_per_mwh.agg(["min", "max"])
        if (ranges["max"] - ranges["min"]).gt(1e-9).any():
            # If cases use different prices, plot each controller's curve separately.
            records["curve"] = records.curve + " (" + records.label + ")"
        records = records.drop_duplicates(["interval_id", "curve"])
        grouped = records.groupby(["month", "slot", "curve"], sort=True).price_eur_per_mwh.agg(
            samples="size", avg_price_eur_mwh="mean").reset_index()
        grouped["farm"] = farm
        grouped["local_time"] = grouped.slot.map(lambda slot: f"{slot // 4:02d}:{15 * (slot % 4):02d}")
        output.append(grouped)
    return pd.concat(output, ignore_index=True)


def _plot_months(output, profiles, prices, timezone):
    """Show all trajectories separately for each farm and delivery month."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for (farm, month), group in profiles.groupby(["farm", "month"], sort=True):
        figure, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True, layout="constrained")
        for label, selected in group.groupby("label", sort=False):
            selected = selected.sort_values("slot")
            x = selected.slot / 4
            axes[0].plot(x, selected.avg_absolute_imbalance_mwh, label=label)
            axes[1].plot(x, selected.avg_curtailment_mwh, label=label)
            axes[2].plot(x, selected.avg_net_position_mwh, label=label)
            axes[3].plot(x, selected.avg_soc_start_percent, label=label)
        month_prices = prices[(prices.farm == farm) & (prices.month == month)]
        for curve, selected in month_prices.groupby("curve", sort=True):
            # Reindex to show genuine unavailable products as gaps in the line.
            values = selected.set_index("slot").avg_price_eur_mwh.reindex(range(96))
            axes[4].plot(values.index / 4, values, label=curve)
        axes[2].axhline(0, color="black", linewidth=0.8)
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(loc="best")
        axes[0].set_ylabel("Absolute imbalance\nMWh / interval")
        axes[1].set_ylabel("Curtailment\nMWh / interval")
        axes[2].set_ylabel("Net commitment\nMWh / interval (+ sale)")
        axes[3].set_ylabel("SOC at interval\nstart (%)")
        axes[4].set_ylabel("Realized price\n€ / MWh")
        axes[4].set_xlabel(f"Quarter-hour start, {timezone}")
        axes[4].set_xticks(range(0, 25, 3))
        figure.suptitle(f"Farm {farm} · {month}: mean by quarter-hour")
        directory = output / f"farm_{farm}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{month}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return paths


def write_intraday_profiles(batch, rows, *, plots=True):
    """Write monthly quarter-hour plots and year-period hourly tables for a completed batch."""
    batch = Path(batch)
    frames = []
    prices = []
    timezones = {row["config"]["market"]["timezone"] for row in rows}
    if len(timezones) != 1:
        raise ValueError("Intraday comparisons require one market timezone")
    timezone = timezones.pop()
    for row in rows:
        path = row["csv_dir"] / "realized_dispatch.csv"
        columns = ["interval_id", "start", "delivery_day", "duration_hours", "curtailment_mw", "imbalance_mwh",
                   "net_position_mwh", "grid_mw", "charge_mw", "discharge_mw", "energy_start_mwh",
                   "imbalance_price_eur_mwh"]
        header = pd.read_csv(path, nrows=0).columns
        if "imbalance_negative_price_eur_mwh" in header:
            columns.append("imbalance_negative_price_eur_mwh")
        dispatch = pd.read_csv(path, usecols=columns)
        ledger = pd.read_csv(row["csv_dir"] / "accepted_trades.csv",
                             usecols=["interval_id", "quantity_mwh", "auction", "price_eur_per_mwh"])
        commitments = ledger.groupby("interval_id").quantity_mwh.sum()
        recorded = dispatch.set_index("interval_id").net_position_mwh
        if not recorded.index.is_unique or set(recorded.index) != set(commitments.index):
            raise ValueError(f"Commitment ledger intervals do not match dispatch for {path}")
        difference = (recorded - commitments).abs()
        if not difference.le(1e-7).all():
            raise ValueError(f"Net commitments do not reconcile to the auction ledger for {path}")
        data = prepare_interval_profile(dispatch, timezone, row["config"]["battery"]["capacity_mwh"])
        summary = row["summary"]
        if not math.isclose(data.absolute_imbalance_mwh.sum(), summary["absolute_imbalance_mwh"],
                            rel_tol=1e-9, abs_tol=1e-5):
            raise ValueError(f"Intraday imbalance does not reconcile for {path}")
        if not math.isclose(data.curtailment_mwh.sum(), summary["curtailment_mwh"], rel_tol=1e-9, abs_tol=1e-5):
            raise ValueError(f"Intraday curtailment does not reconcile for {path}")
        data["farm"] = row["farm"]
        data["case"] = row["case"]
        data["controller"] = row["controller"]
        data["label"] = row["label"]
        frames.append(data)
        prices.append(prepare_price_samples(dispatch, ledger, data, row["farm"], row["label"]))
    all_data = pd.concat(frames, ignore_index=True)
    identity = ["farm", "case", "controller", "label"]
    monthly = aggregate_intraday(all_data, identity + ["month", "slot"])
    monthly["local_time"] = monthly.slot.map(lambda slot: f"{slot // 4:02d}:{15 * (slot % 4):02d}")
    hourly = aggregate_intraday(all_data, identity + ["hour"])
    totals = hourly.groupby(identity, dropna=False).total_absolute_imbalance_mwh.transform("sum")
    hourly["imbalance_share_percent"] = (hourly.total_absolute_imbalance_mwh / totals.where(totals != 0)) * 100
    output = batch / "intraday_analysis"
    output.mkdir(exist_ok=True)
    monthly.to_csv(output / "monthly_quarter_hour_profiles.csv", index=False)
    hourly.to_csv(output / "hourly_profiles.csv", index=False)
    price_profiles = aggregate_monthly_prices(pd.concat(prices, ignore_index=True))
    price_profiles.to_csv(output / "monthly_price_profiles.csv", index=False)
    paths = _plot_months(output, monthly, price_profiles, timezone) if plots else []
    lines = ["# Intraday imbalance, curtailment and final commitments", "",
        f"Local time is {timezone}; labels are interval start times. Each plotted point is the average MWh "
        "per observed 15-minute interval in that calendar month. Absolute imbalance is averaged before "
        "aggregation; curtailment is curtailed MW multiplied by interval duration.", "",
        "The final net commitment combines all auction purchases and sales: **positive means net sale, "
        "negative means net purchase**. The absolute commitment column is the mean magnitude before "
        "averaging, which avoids hiding purchases and sales that cancel. It is not gross traded volume.", "",
        "SOC is actual battery energy at the start of each interval divided by configured capacity. "
        "Charging and discharging are measured at the AC side and included in the tables and data files.", "",
        "Prices are realized auction clearing and imbalance settlement prices, in euros per MWh. Each "
        "curve includes recorded eligible products, including zero-quantity submissions; missing products "
        "are left as gaps. The price CSV includes observation counts and does not duplicate shared prices "
        "across controller trajectories.", "",
        "Spring clock-change intervals absent from a delivery day are not invented. Repeated autumn clock "
        "intervals are counted as distinct UTC observations. CSV sample counts expose the number of observations.", "",
        "## Hourly tables across the replay period", "",
        "Imbalance and curtailment columns are total MWh across all days in the hour. The percentage is the "
        "share of that trajectory's total absolute imbalance. Commitment columns are average MWh per "
        "15-minute interval, rather than four-quarter-hour sums. Grid exchange uses positive for export "
        "and negative for import; battery charge is the AC-side charging energy.", "",
        "| Farm | Controller | Peak imbalance hour | Imbalance MWh (share) | Mean net commitment at imbalance peak (MWh / 15 min) | Peak absolute commitment hour | Mean absolute commitment at peak (MWh / 15 min) |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |"]
    for (farm, label), selected in hourly.groupby(["farm", "label"], sort=False):
        peak = selected.loc[selected.total_absolute_imbalance_mwh.idxmax()]
        commitment = selected.loc[selected.avg_absolute_net_position_mwh.idxmax()]
        share = "n/a" if pd.isna(peak.imbalance_share_percent) else f"{peak.imbalance_share_percent:.1f}%"
        peak_hour, commitment_hour = int(peak.hour), int(commitment.hour)
        lines.append(f"| {farm} | {label} | {peak_hour:02d}:00–{peak_hour + 1:02d}:00 | "
            f"{peak.total_absolute_imbalance_mwh:.2f} ({share}) | {peak.avg_net_position_mwh:.4f} | "
            f"{commitment_hour:02d}:00–{commitment_hour + 1:02d}:00 | {commitment.avg_absolute_net_position_mwh:.4f} |")
    lines.append("")
    for (farm, label), selected in hourly.groupby(["farm", "label"], sort=False):
        lines.extend([f"### Farm {farm} — {label}", "",
            "| Local hour | Absolute imbalance (MWh) | Share | Curtailment (MWh) | Mean net commitment (MWh / 15 min) | Mean absolute commitment (MWh / 15 min) | Mean grid exchange (MWh / 15 min) | Mean battery charge (MWh / 15 min) | Mean battery discharge (MWh / 15 min) | Mean starting SOC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for item in selected.sort_values("hour").to_dict("records"):
            share = "n/a" if pd.isna(item["imbalance_share_percent"]) else f"{item['imbalance_share_percent']:.2f}%"
            hour = int(item["hour"])
            lines.append(f"| {hour:02d}:00–{hour + 1:02d}:00 | {item['total_absolute_imbalance_mwh']:.3f} | "
                f"{share} | {item['total_curtailment_mwh']:.3f} | {item['avg_net_position_mwh']:.4f} | "
                f"{item['avg_absolute_net_position_mwh']:.4f} | {item['avg_grid_energy_mwh']:.4f} | "
                f"{item['avg_charge_energy_mwh']:.4f} | {item['avg_discharge_energy_mwh']:.4f} | "
                f"{item['avg_soc_start_percent']:.2f}% |")
        lines.append("")
    lines.extend(["## Monthly plots", ""])
    if paths:
        lines.extend(f"- [Farm {path.parent.name.removeprefix('farm_')} · {path.stem}]({path.relative_to(output)})"
                     for path in paths)
    else:
        lines.append("Plots were disabled for this report.")
    lines.extend(["", "[Monthly quarter-hour data](monthly_quarter_hour_profiles.csv) · "
                  "[Monthly price data](monthly_price_profiles.csv) · [Hourly data](hourly_profiles.csv)", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output / "summary.md"

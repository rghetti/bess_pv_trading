"""Explain continuous auction revenue and the full-participation policy gap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..numerical_tolerances import continuous_tolerances

AUCTIONS = ("DA", "IDA1", "IDA2", "IDA3")


def _native_products(trades, dispatch, config, da_prices):
    """Rebuild auction products and position changes from immutable accepted trades."""
    market = config["market"]
    local_zone = market["timezone"]
    buffer = pd.Timedelta(minutes=float(market["decision_buffer_minutes"]))
    dispatch = dispatch.copy()
    dispatch["interval_id"] = dispatch.interval_id.astype(str)
    dispatch["start"] = pd.to_datetime(dispatch.start, utc=True)
    interval_start = dict(zip(dispatch.interval_id, dispatch.start))
    if trades.empty:
        trades = pd.DataFrame(columns=["auction", "delivery_day", "interval_id", "quantity_mwh",
                                       "price_eur_per_mwh", "cashflow_eur", "submission_time"])
    else:
        trades = trades.copy()
        trades["interval_id"] = trades.interval_id.astype(str)
        trades["submission_time"] = pd.to_datetime(trades.submission_time, utc=True)
        trades["product_start_utc"] = trades.interval_id.map(interval_start)
        if trades.product_start_utc.isna().any():
            raise ValueError("Submitted auction trade references an interval outside realized dispatch")
        # Check the immutable submission timestamp against the configured eligible gate.
        expected = []
        for row in trades.itertuples():
            auction = row.auction
            event_day = pd.Timestamp(row.delivery_day).date() + pd.Timedelta(
                days=int(market["gate_day_offsets"][auction]))
            local_gate = pd.Timestamp(f"{event_day} {market['gates'][auction]}", tz=local_zone)
            expected.append((local_gate.tz_convert("UTC") - buffer).to_pydatetime())
        if not np.array_equal(trades.submission_time.to_numpy(), pd.to_datetime(expected, utc=True).to_numpy()):
            bad = trades.loc[trades.submission_time.ne(pd.to_datetime(expected, utc=True))].head(3)
            raise ValueError(f"Submitted trades do not match their actual eligible auction gate: {bad.to_dict('records')}")
        id3_start = market.get("delivery_start_times", {}).get("IDA3")
        if id3_start:
            for row in trades.loc[trades.auction.eq("IDA3")].itertuples():
                boundary = pd.Timestamp(f"{row.delivery_day} {id3_start}", tz=local_zone).tz_convert("UTC")
                if row.product_start_utc < boundary:
                    raise ValueError("IDA3 ledger contains a product before its eligible delivery window")
        signed_cash = trades.quantity_mwh * trades.price_eur_per_mwh
        tolerances = continuous_tolerances(config)
        if not tolerances.close(signed_cash, trades.cashflow_eur, monetary=True).all():
            raise ValueError("Immutable trade quantity and auction cash flow do not reconcile")
        if not trades.auction.isin(AUCTIONS).all():
            raise ValueError("Unknown auction in immutable trade ledger")
        trades["product_start_utc"] = pd.to_datetime(trades.product_start_utc, utc=True)

    # Retain quarter-hour position increments for a common cumulative position path.
    wide = trades.pivot_table(index=["delivery_day", "interval_id"], columns="auction",
        values="quantity_mwh", aggfunc="sum", fill_value=0.0).reindex(columns=AUCTIONS, fill_value=0.0)
    interval_table = dispatch[["delivery_day", "interval_id", "start"]].copy()
    position = interval_table.set_index(["delivery_day", "interval_id"])
    for auction in AUCTIONS:
        position[f"adjustment_{auction}"] = wide[auction].reindex(position.index, fill_value=0.0)
    position["position_before_DA_mwh"] = 0.0
    cumulative = np.zeros(len(position), dtype=float)
    for auction in AUCTIONS:
        position[f"position_before_{auction}_mwh"] = cumulative
        cumulative = cumulative + position[f"adjustment_{auction}"].to_numpy(float)
        position[f"position_after_{auction}_mwh"] = cumulative.copy()
    position["final_position_mwh"] = cumulative
    position = position.reset_index()
    # Pre-index position snapshots so hourly and quarter-hour products are O(1) lookups.
    position["hour_start_utc"] = position.start.dt.floor("h")
    hourly_position = position.groupby(["delivery_day", "hour_start_utc"], sort=False).agg(
        **{field: (field, "sum") for auction in AUCTIONS for field in (
            f"position_before_{auction}_mwh", f"position_after_{auction}_mwh")},
        final_position_mwh=("final_position_mwh", "sum")).to_dict("index")
    quarter_position = position.set_index(["delivery_day", "start"]).to_dict("index")

    # DA products aggregate to the actual hourly delivery product; IDAs remain quarter-hourly.
    product_rows = []
    da_resolutions = {}
    if not trades.empty:
        da_trades = trades.loc[trades.auction.eq("DA")].copy()
        if not da_trades.empty:
            da_trades["hour"] = da_trades.product_start_utc.dt.floor("h")
            for day, rows in da_trades.groupby("delivery_day", sort=False):
                da_resolutions[day] = 60 if rows.groupby("hour").price_eur_per_mwh.nunique().max() == 1 else 15

    def reference_price(start):
        """Look up an observed DA price at quarter-hour or its containing hour."""
        stamp = pd.Timestamp(start)
        return da_prices.get(stamp, da_prices.get(stamp.floor("h"), np.nan))

    all_days = sorted(dispatch.delivery_day.astype(str).unique())
    for auction in AUCTIONS:
        active = trades.loc[trades.auction.eq(auction)].copy() if not trades.empty else trades.copy()
        active["native_resolution_minutes"] = active.delivery_day.map(da_resolutions).fillna(15).astype(int) if auction == "DA" else 15
        if auction == "DA":
            active["native_product_start_utc"] = active.apply(
                lambda row: row.product_start_utc.floor("h") if da_resolutions.get(row.delivery_day, 15) == 60
                else row.product_start_utc, axis=1)
        else:
            active["native_product_start_utc"] = active.product_start_utc
        if active.empty and auction == "DA" and da_prices:
            # An omitted DA participation profile still gets a reference valuation
            # when the same run contains the actual DA price curve.
            for day, day_position in position.groupby("delivery_day", sort=False):
                starts = pd.DatetimeIndex(day_position.start)
                quarter_prices = {stamp: reference_price(stamp) for stamp in starts}
                hourly_groups = day_position.assign(hour_start_utc=day_position.start.dt.floor("h")).groupby(
                    "hour_start_utc", sort=False)
                hourly_resolution = all(len(group) == 4 and
                    len({quarter_prices[stamp] for stamp in group.start}) == 1 for _, group in hourly_groups)
                if hourly_resolution:
                    products_to_value = [(stamp, hourly_position[(day, stamp)], quarter_prices[group.start.iloc[0]])
                        for stamp, group in hourly_groups]
                else:
                    products_to_value = [(row.start, quarter_position[(day, row.start)], quarter_prices[row.start])
                        for row in day_position.itertuples()]
                for product_start, stats, price in products_to_value:
                    product_rows.append({"delivery_day": day, "auction": "DA",
                        "native_resolution_minutes": 60 if hourly_resolution else 15,
                        "product_start_utc": product_start, "submission_time_utc": pd.NaT,
                        "signed_submitted_mwh": 0.0, "gross_traded_mwh": 0.0,
                        "position_increasing_mwh": 0.0, "position_reducing_mwh": 0.0,
                        "cumulative_position_before_mwh": 0.0, "cumulative_position_after_mwh": 0.0,
                        "auction_price_eur_per_mwh": np.nan,
                        "da_reference_price_eur_per_mwh": float(price), "trade_cashflow_eur": 0.0,
                        "da_referenced_spread_eur": 0.0,
                        "final_position_da_value_eur": float(stats["final_position_mwh"] * price),
                        "final_cumulative_position_mwh": float(stats["final_position_mwh"]),
                        "da_reference_available": bool(np.isfinite(price)), "participated": False})
            continue
        if active.empty:
            continue
        grouping = ["delivery_day", "native_product_start_utc"]
        grouped = active.groupby(grouping, sort=True)
        products = grouped.agg(signed_submitted_mwh=("quantity_mwh", "sum"),
            gross_traded_mwh=("quantity_mwh", lambda x: float(np.abs(x).sum())),
            auction_price_eur_per_mwh=("price_eur_per_mwh", "first"),
            trade_cashflow_eur=("cashflow_eur", "sum"),
            submission_time=("submission_time", "first")).reset_index()
        for row in products.itertuples():
            resolution = da_resolutions.get(row.delivery_day, 15) if auction == "DA" else 15
            position_key = (row.delivery_day, row.native_product_start_utc)
            stats = hourly_position.get(position_key, {}) if resolution == 60 else quarter_position.get(position_key, {})
            before = float(stats.get(f"position_before_{auction}_mwh", 0.0))
            after = float(stats.get(f"position_after_{auction}_mwh", 0.0))
            if auction == "DA":
                da_price = row.auction_price_eur_per_mwh
            else:
                da_price = reference_price(row.native_product_start_utc)
            spread = ((row.auction_price_eur_per_mwh - da_price) * row.signed_submitted_mwh
                      if auction != "DA" and np.isfinite(da_price) else (0.0 if auction == "DA" else np.nan))
            product_rows.append({"delivery_day": row.delivery_day, "auction": auction,
                "native_resolution_minutes": int(resolution),
                "product_start_utc": row.native_product_start_utc,
                "submission_time_utc": row.submission_time,
                "signed_submitted_mwh": float(row.signed_submitted_mwh),
                "gross_traded_mwh": float(row.gross_traded_mwh),
                "position_increasing_mwh": float(max(row.signed_submitted_mwh, 0.0)),
                "position_reducing_mwh": float(max(-row.signed_submitted_mwh, 0.0)),
                "cumulative_position_before_mwh": before, "cumulative_position_after_mwh": after,
                "auction_price_eur_per_mwh": float(row.auction_price_eur_per_mwh),
                "da_reference_price_eur_per_mwh": da_price,
                "trade_cashflow_eur": float(row.trade_cashflow_eur),
                "da_referenced_spread_eur": spread,
                "final_position_da_value_eur": (float(stats.get("final_position_mwh", 0.0) * da_price)
                    if auction == "DA" and np.isfinite(da_price) else np.nan),
                "final_cumulative_position_mwh": float(stats.get("final_position_mwh", 0.0)),
                "da_reference_available": bool(np.isfinite(da_price)), "participated": True})
    # Include zero-quantity DA products so the DA valuation covers the entire final position.
    existing_da_products = {(row["delivery_day"], pd.Timestamp(row["product_start_utc"]))
        for row in product_rows if row["auction"] == "DA"}
    if da_prices:
        for day, day_position in position.groupby("delivery_day", sort=False):
            daily_groups = day_position.assign(hour_start_utc=day_position.start.dt.floor("h")).groupby(
                "hour_start_utc", sort=False)
            hourly_resolution = all(len(group) == 4 and
                len({reference_price(stamp) for stamp in group.start}) == 1 for _, group in daily_groups)
            if hourly_resolution:
                selected_products = [(stamp, hourly_position[(day, stamp)], reference_price(group.start.iloc[0]))
                    for stamp, group in daily_groups]
            else:
                selected_products = [(row.start, quarter_position[(day, row.start)], reference_price(row.start))
                    for row in day_position.itertuples()]
            for product_start, stats, price in selected_products:
                if (day, pd.Timestamp(product_start)) in existing_da_products:
                    continue
                product_rows.append({"delivery_day": day, "auction": "DA",
                    "native_resolution_minutes": 60 if hourly_resolution else 15,
                    "product_start_utc": product_start, "submission_time_utc": pd.NaT,
                    "signed_submitted_mwh": 0.0, "gross_traded_mwh": 0.0,
                    "position_increasing_mwh": 0.0, "position_reducing_mwh": 0.0,
                    "cumulative_position_before_mwh": 0.0, "cumulative_position_after_mwh": 0.0,
                    "auction_price_eur_per_mwh": np.nan,
                    "da_reference_price_eur_per_mwh": float(price), "trade_cashflow_eur": 0.0,
                    "da_referenced_spread_eur": 0.0,
                    "final_position_da_value_eur": float(stats["final_position_mwh"] * price),
                    "final_cumulative_position_mwh": float(stats["final_position_mwh"]),
                    "da_reference_available": bool(np.isfinite(price)), "participated": False})
    return pd.DataFrame(product_rows), position.reset_index(drop=True), da_prices


def _extract_da_reference(root, profile, case):
    """Use actual DA trade prices from this profile, or its same-run full-auction case."""
    candidates = [root / case / profile / "csv/accepted_trades.csv",
                  root / case / "DA+IDA1+IDA2+IDA3/csv/accepted_trades.csv",
                  root / "strategies/DA+IDA1+IDA2+IDA3/csv/accepted_trades.csv"]
    for path in candidates:
        if not path.is_file():
            continue
        table = pd.read_csv(path, dtype={"interval_id": str, "delivery_day": str})
        table = table.loc[table.auction.eq("DA")]
        if table.empty:
            continue
        table["start"] = pd.to_datetime(table.interval_id, utc=True)
        return dict(zip(table.start, table.price_eur_per_mwh))
    return {}


def _financial_daily(strategy_directory, label):
    """Read realized financial components without recomputing settlement."""
    path = strategy_directory / "csv/realized_dispatch.csv"
    frame = pd.read_csv(path, dtype={"delivery_day": str})
    columns = ["auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur",
        "import_fee_eur", "transaction_fee_eur", "imbalance_penalty_eur", "net_profit_eur"]
    if "imbalance_penalty_eur" not in frame:
        frame["imbalance_penalty_eur"] = 0.0
    daily = frame.groupby("delivery_day", sort=True)[columns].sum().reset_index()
    daily["case"] = label
    tolerances = continuous_tolerances(json.loads((strategy_directory / "json/study_config.json").read_text()))
    reconstructed = (daily.auction_revenue_eur + daily.imbalance_cashflow_eur - daily.degradation_eur
        - daily.import_fee_eur - daily.transaction_fee_eur - daily.imbalance_penalty_eur)
    if not tolerances.close(reconstructed, daily.net_profit_eur, monetary=True).all():
        raise ValueError(f"{label} realized profit does not reconcile to its ledger components")
    return daily


def _profile_products(root, profile, case, strategy):
    """Build trade products for one BESS or independent PV-only replay."""
    directory = profile if strategy == "integrated" else "sequential_all_auctions"
    base = root / case / directory
    ledger_path = base / "csv/accepted_trades.csv"
    dispatch_path = base / "csv/realized_dispatch.csv"
    config_path = base / "json/study_config.json"
    if not all(path.is_file() for path in (ledger_path, dispatch_path, config_path)):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    config = json.loads(config_path.read_text())
    ledger = pd.read_csv(ledger_path, dtype={"interval_id": str, "delivery_day": str})
    dispatch = pd.read_csv(dispatch_path, dtype={"interval_id": str, "delivery_day": str})
    da_prices = _extract_da_reference(root, profile, case)
    products, positions, _ = _native_products(ledger, dispatch, config, da_prices)
    products["profile"] = profile
    products["case"] = case
    positions["profile"] = profile
    positions["case"] = case
    finances = _financial_daily(base, case)
    finances["strategy"] = strategy
    finances["profile"] = profile
    return products, positions, finances


def _paired_comparison(products, positions, finances, profile, case):
    """Pair integrated and sequential auction changes on common native products."""
    selection = products.loc[products.profile.eq(profile) & products.case.eq(case)]
    integrated = selection.loc[selection.strategy.eq("integrated")] if "strategy" in selection else selection.iloc[0:0]
    sequential = selection.loc[selection.strategy.eq("sequential_all_auctions")] if "strategy" in selection else selection.iloc[0:0]
    keys = ["delivery_day", "auction", "product_start_utc", "native_resolution_minutes"]
    fields = ["signed_submitted_mwh", "gross_traded_mwh", "cumulative_position_before_mwh",
        "cumulative_position_after_mwh", "final_cumulative_position_mwh", "trade_cashflow_eur",
        "da_referenced_spread_eur", "final_position_da_value_eur"]
    if integrated.empty or sequential.empty:
        return pd.DataFrame()
    left = integrated[keys + fields].rename(columns={column: f"integrated_{column}" for column in fields})
    right = sequential[keys + fields].rename(columns={column: f"sequential_{column}" for column in fields})
    paired = left.merge(right, on=keys, how="outer")
    for field in fields:
        paired[f"difference_{field}"] = paired[f"integrated_{field}"].fillna(0.0) - paired[f"sequential_{field}"].fillna(0.0)
    return paired


def _plots(output, products, positions, daily_pairs, profile, config):
    """Save month-scale revenue, position and auction contribution figures."""
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    selected = products.loc[products.profile.eq(profile)]
    if selected.empty:
        return
    strategies = set(selected.strategy)
    days = sorted(selected.delivery_day.unique())
    for case in sorted(selected.case.unique()):
        frame = selected.loc[selected.case.eq(case)]
        totals = frame.groupby(["strategy", "auction"], sort=False).agg(
            cash=("trade_cashflow_eur", "sum"), spread=("da_referenced_spread_eur", "sum")).reset_index()
        order = [name for name in ("integrated", "sequential_all_auctions") if name in strategies]
        figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
        base_values = [float(frame.loc[(frame.strategy.eq(name)) & frame.auction.eq("DA"), "final_position_da_value_eur"].sum()) for name in order]
        axis.bar(order, base_values, label="Final commitment valued at DA", color="tab:blue")
        bottom = np.asarray(base_values)
        for auction in AUCTIONS[1:]:
            values = [float(totals.loc[(totals.strategy.eq(name)) & totals.auction.eq(auction), "spread"].sum()) for name in order]
            axis.bar(order, values, bottom=bottom, label=f"{auction} DA-referenced spread")
            bottom += values
        axis.set_ylabel("Auction revenue (EUR)")
        axis.set_title(f"{profile} {case}: revenue by DA value and intraday spreads")
        axis.legend()
        figure.savefig(output / f"stacked_revenue_{case}.png", dpi=140)
        plt.close(figure)

        # Show first, middle and last selected days to keep the report compact.
        representative = sorted(set((days[0], days[len(days) // 2], days[-1])))
        figure, axes = plt.subplots(len(representative), 1, figsize=(11, 3 * len(representative)), layout="constrained", squeeze=False)
        for axis, day in zip(axes[:, 0], representative):
            for strategy in order:
                state = positions.loc[(positions.case.eq(case)) & positions.profile.eq(profile)
                    & positions.strategy.eq(strategy) & positions.delivery_day.eq(day)].sort_values("start")
                axis.step(pd.to_datetime(state.start, utc=True), state.final_position_mwh,
                    where="post", label=strategy)
            axis.set_title(day)
            axis.set_ylabel("Committed MWh")
            axis.grid(alpha=0.2)
        axes[0, 0].legend()
        figure.savefig(output / f"cumulative_positions_{case}.png", dpi=140)
        plt.close(figure)

        auction = totals.pivot(index="auction", columns="strategy", values="spread").reindex(AUCTIONS)
        auction.plot(kind="bar", figsize=(9, 5), ylabel="DA-referenced spread revenue (EUR)",
                     title=f"{profile} {case}: auction spread contribution", grid=True)
        plt.tight_layout()
        plt.savefig(output / f"spread_by_auction_{case}.png", dpi=140)
        plt.close()
        gross = frame.groupby(["strategy", "auction"], sort=False).agg(
            purchases_mwh=("signed_submitted_mwh", lambda x: float((-x.clip(upper=0)).sum())),
            sales_mwh=("signed_submitted_mwh", lambda x: float(x.clip(lower=0).sum())),
            net_position_mwh=("signed_submitted_mwh", "sum")).reset_index()
        gross.to_csv(output / f"auction_gross_net_{case}.csv", index=False)
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
        auctions = list(AUCTIONS)
        width = 0.35
        x = np.arange(len(auctions))
        for offset, strategy in enumerate(order):
            values = gross.loc[gross.strategy.eq(strategy)].set_index("auction").reindex(auctions).fillna(0.0)
            axes[0].bar(x + (offset - (len(order) - 1) / 2) * width, values.purchases_mwh,
                width=width, label=f"{strategy} purchases")
            axes[0].bar(x + (offset - (len(order) - 1) / 2) * width, values.sales_mwh,
                width=width, bottom=values.purchases_mwh, alpha=0.45, label=f"{strategy} sales")
            axes[1].bar(x + (offset - (len(order) - 1) / 2) * width, values.net_position_mwh,
                width=width, label=strategy)
        axes[0].set_ylabel("Gross traded volume (MWh)")
        axes[0].legend(fontsize="small", ncol=2)
        axes[1].set_ylabel("Net signed submissions (MWh)")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].legend()
        for axis in axes:
            axis.set_xticks(x, auctions)
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle(f"{profile} {case}: auction-level buys, sales and net positions")
        figure.savefig(output / f"auction_gross_sales_positions_{case}.png", dpi=140)
        plt.close(figure)

    # The waterfall uses additive cash-profit effects and labels cost reductions as gains.
    if not daily_pairs.empty:
        full = daily_pairs.groupby("case", sort=False).sum(numeric_only=True)
        for case, values in full.iterrows():
            if not {"integrated", "sequential_all_auctions"}.issubset(set(
                    products.loc[(products.profile.eq(profile)) & products.case.eq(case), "strategy"])):
                continue
            selected = products.loc[(products.profile.eq(profile)) & products.case.eq(case)]
            spread = selected.loc[selected.strategy.eq("integrated")].groupby("auction").da_referenced_spread_eur.sum(min_count=1)
            sequential_spread = selected.loc[selected.strategy.eq("sequential_all_auctions")].groupby("auction").da_referenced_spread_eur.sum(min_count=1)
            components = {"DA commitment value": float(selected.loc[(selected.strategy.eq("integrated"))
                & selected.auction.eq("DA"), "final_position_da_value_eur"].sum()
                - selected.loc[(selected.strategy.eq("sequential_all_auctions")) & selected.auction.eq("DA"),
                    "final_position_da_value_eur"].sum())}
            for auction in AUCTIONS[1:]:
                components[f"{auction} spread"] = float(spread.get(auction, np.nan) - sequential_spread.get(auction, np.nan))
            components["Imbalance settlement"] = float(values.get("difference_imbalance_cashflow_eur", 0.0))
            components["Degradation savings"] = -float(values.get("difference_degradation_eur", 0.0))
            components["Import cost savings"] = -float(values.get("difference_import_fee_eur", 0.0))
            components["Transaction fee savings"] = -float(values.get("difference_transaction_fee_eur", 0.0))
            components["Imbalance penalty savings"] = -float(values.get("difference_imbalance_penalty_eur", 0.0))
            components = {key: value for key, value in components.items() if np.isfinite(value)}
            if not components:
                continue
            expected_profit_difference = float(values.get("difference_net_profit_eur", np.nan))
            if not continuous_tolerances(config).close(
                    sum(components.values()), expected_profit_difference, monetary=True):
                raise ValueError(f"Profit difference waterfall does not reconcile for {profile}/{case}")
            figure, axis = plt.subplots(figsize=(11, 5), layout="constrained")
            names = list(components)
            vals = np.asarray(list(components.values()), dtype=float)
            running = 0.0
            for index, (name, value) in enumerate(zip(names, vals)):
                bottom = min(running, running + value)
                axis.bar(index, abs(value), bottom=bottom,
                    color="tab:green" if value >= 0 else "tab:red", width=0.75)
                axis.plot([index - 0.4, index + 0.4], [running + value, running + value],
                    color="0.4", linewidth=0.6)
                running += value
            axis.bar(len(names), expected_profit_difference,
                color="tab:blue" if expected_profit_difference >= 0 else "tab:orange", width=0.75)
            names.append("Net profit Δ")
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_ylabel("Integrated minus sequential profit contribution (EUR)")
            axis.set_title(f"{profile} {case}: additive profit difference components")
            axis.set_xticks(range(len(names)), names)
            axis.tick_params(axis="x", rotation=25)
            figure.savefig(output / f"profit_difference_waterfall_{case}.png", dpi=140)
            plt.close(figure)


def generate_continuous_trade_attribution(run_root, *, representative_days=None):
    """Write native-product ledgers, reconciliations, paired summaries and plots."""
    root = Path(run_root)
    config_path = root / "json/study_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Participation study config not found: {config_path}")
    config = json.loads(config_path.read_text())
    profile_names = sorted(path.name for path in (root / "strategies").iterdir()
        if path.is_dir() and path.name not in {"sequential_all_auctions", "perfect_pv_all_auctions"})
    if "DA+IDA1+IDA2+IDA3" not in profile_names:
        raise ValueError("Full-auction integrated participation profile is required for paired reporting")
    output = root / "trade_attribution"
    products_all, positions_all, finances_all = [], [], []
    strategy_cases = [("strategies", "integrated"), ("strategies", "sequential_all_auctions"),
                      ("pv_only", "integrated"), ("pv_only", "sequential_all_auctions")]
    for case, strategy in strategy_cases:
        for profile in profile_names if strategy == "integrated" else ["DA+IDA1+IDA2+IDA3"]:
            base = root / case / profile
            if not (base / "csv/accepted_trades.csv").is_file():
                continue
            products, positions, finances = _profile_products(root, profile, case, strategy)
            products["strategy"] = strategy
            positions["strategy"] = strategy
            finances["strategy"] = strategy
            finances["profile"] = profile
            products_all.append(products)
            positions_all.append(positions)
            finances_all.append(finances)
    products = pd.concat(products_all, ignore_index=True)
    positions = pd.concat(positions_all, ignore_index=True)
    finances = pd.concat(finances_all, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    products.to_csv(output / "auction_products.csv", index=False)
    positions.to_csv(output / "quarter_hour_positions.csv", index=False)
    finances.to_csv(output / "realized_financials_daily.csv", index=False)
    if not products.empty:
        product_daily = products.groupby(["case", "strategy", "profile", "delivery_day", "auction"],
            sort=False, dropna=False).agg(signed_submitted_mwh=("signed_submitted_mwh", "sum"),
            gross_traded_mwh=("gross_traded_mwh", "sum"), position_increasing_mwh=("position_increasing_mwh", "sum"),
            position_reducing_mwh=("position_reducing_mwh", "sum"), trade_cashflow_eur=("trade_cashflow_eur", "sum"),
            da_referenced_spread_eur=("da_referenced_spread_eur", "sum"),
            final_position_da_value_eur=("final_position_da_value_eur", "sum"),
            da_reference_available=("da_reference_available", "all")).reset_index()
        spread_daily = products.groupby(["case", "strategy", "profile", "delivery_day", "auction"],
            sort=False, dropna=False).da_referenced_spread_eur.sum(min_count=1).rename(
                "da_referenced_spread_eur").reset_index()
        product_daily = product_daily.drop(columns="da_referenced_spread_eur").merge(spread_daily,
            on=["case", "strategy", "profile", "delivery_day", "auction"], validate="one_to_one")
    else:
        product_daily = pd.DataFrame()
    product_daily.to_csv(output / "auction_revenue_daily.csv", index=False)
    if not product_daily.empty:
        product_daily["purchased_mwh"] = ((product_daily.gross_traded_mwh - product_daily.signed_submitted_mwh) / 2).clip(lower=0)
        product_daily["sold_mwh"] = ((product_daily.gross_traded_mwh + product_daily.signed_submitted_mwh) / 2).clip(lower=0)
        product_daily[["case", "strategy", "profile", "delivery_day", "auction", "purchased_mwh",
            "sold_mwh", "signed_submitted_mwh", "gross_traded_mwh"]].to_csv(
                output / "auction_trade_volumes_daily.csv", index=False)
    if not product_daily.empty:
        product_monthly = product_daily.assign(delivery_month=product_daily.delivery_day.str[:7]).groupby(
            ["case", "strategy", "profile", "delivery_month", "auction"], sort=False, dropna=False).agg(
            signed_submitted_mwh=("signed_submitted_mwh", "sum"), gross_traded_mwh=("gross_traded_mwh", "sum"),
            position_increasing_mwh=("position_increasing_mwh", "sum"), position_reducing_mwh=("position_reducing_mwh", "sum"),
            trade_cashflow_eur=("trade_cashflow_eur", "sum"), da_referenced_spread_eur=("da_referenced_spread_eur", "sum"),
            final_position_da_value_eur=("final_position_da_value_eur", "sum"),
            da_reference_available=("da_reference_available", "all")).reset_index()
        spread_monthly = product_daily.assign(delivery_month=product_daily.delivery_day.str[:7]).groupby(
            ["case", "strategy", "profile", "delivery_month", "auction"], sort=False,
            dropna=False).da_referenced_spread_eur.sum(min_count=1).rename(
                "da_referenced_spread_eur").reset_index()
        product_monthly = product_monthly.drop(columns="da_referenced_spread_eur").merge(spread_monthly,
            on=["case", "strategy", "profile", "delivery_month", "auction"], validate="one_to_one")
    else:
        product_monthly = pd.DataFrame()
    product_monthly.to_csv(output / "auction_revenue_monthly.csv", index=False)
    if not product_monthly.empty:
        product_monthly["purchased_mwh"] = ((product_monthly.gross_traded_mwh - product_monthly.signed_submitted_mwh) / 2).clip(lower=0)
        product_monthly["sold_mwh"] = ((product_monthly.gross_traded_mwh + product_monthly.signed_submitted_mwh) / 2).clip(lower=0)
        product_monthly[["case", "strategy", "profile", "delivery_month", "auction", "purchased_mwh",
            "sold_mwh", "signed_submitted_mwh", "gross_traded_mwh"]].to_csv(
                output / "auction_trade_volumes_monthly.csv", index=False)
    if not product_daily.empty:
        decomposition_daily = product_daily.groupby(["case", "strategy", "profile", "delivery_day"],
            sort=False, dropna=False).agg(auction_ledger_revenue_eur=("trade_cashflow_eur", "sum"),
            da_reference_value_eur=("final_position_da_value_eur", lambda values: values.sum(min_count=1)),
            intraday_spread_revenue_eur=("da_referenced_spread_eur", lambda values: values.sum(min_count=1)),
            da_reference_available=("da_reference_available", "all")).reset_index()
        decomposition_daily["decomposed_auction_revenue_eur"] = (
            decomposition_daily.da_reference_value_eur + decomposition_daily.intraday_spread_revenue_eur)
        decomposition_daily["reconciliation_error_eur"] = (
            decomposition_daily.decomposed_auction_revenue_eur - decomposition_daily.auction_ledger_revenue_eur)
        available = decomposition_daily.da_reference_available
        if available.any() and not continuous_tolerances(config).close(
                decomposition_daily.loc[available, "decomposed_auction_revenue_eur"],
                decomposition_daily.loc[available, "auction_ledger_revenue_eur"], monetary=True).all():
            raise ValueError("Daily DA-value and intraday-spread decomposition does not reconcile")
        decomposition_daily.to_csv(output / "revenue_decomposition_daily.csv", index=False)
        decomposition_monthly = decomposition_daily.assign(delivery_month=decomposition_daily.delivery_day.str[:7]).groupby(
            ["case", "strategy", "profile", "delivery_month"], sort=False, dropna=False).agg(
                auction_ledger_revenue_eur=("auction_ledger_revenue_eur", "sum"),
                da_reference_value_eur=("da_reference_value_eur", lambda values: values.sum(min_count=1)),
                intraday_spread_revenue_eur=("intraday_spread_revenue_eur", lambda values: values.sum(min_count=1)),
                da_reference_available=("da_reference_available", "all")).reset_index()
        decomposition_monthly["decomposed_auction_revenue_eur"] = (
            decomposition_monthly.da_reference_value_eur + decomposition_monthly.intraday_spread_revenue_eur)
        decomposition_monthly["reconciliation_error_eur"] = (
            decomposition_monthly.decomposed_auction_revenue_eur - decomposition_monthly.auction_ledger_revenue_eur)
        decomposition_monthly.to_csv(output / "revenue_decomposition_monthly.csv", index=False)

    # Quantify buy/sell orders that offset across auctions on each delivery product.
    if not positions.empty:
        positions["purchased_mwh"] = sum((-positions[f"adjustment_{auction}"]).clip(lower=0) for auction in AUCTIONS)
        positions["sold_mwh"] = sum(positions[f"adjustment_{auction}"].clip(lower=0) for auction in AUCTIONS)
        positions["cross_auction_offsetting_mwh"] = positions[[f"adjustment_{auction}" for auction in AUCTIONS]].apply(
            lambda row: min(float(row.clip(lower=0).sum()), float((-row.clip(upper=0)).sum())), axis=1)
        trade_daily = positions.groupby(["case", "strategy", "profile", "delivery_day"], sort=False).agg(
            purchased_mwh=("purchased_mwh", "sum"), sold_mwh=("sold_mwh", "sum"),
            cross_auction_offsetting_mwh=("cross_auction_offsetting_mwh", "sum"),
            net_committed_mwh=("final_position_mwh", "sum")).reset_index()
    else:
        trade_daily = pd.DataFrame()
    output.mkdir(parents=True, exist_ok=True)
    trade_daily.to_csv(output / "trade_volumes_daily.csv", index=False)
    if not trade_daily.empty:
        trade_monthly = trade_daily.assign(delivery_month=trade_daily.delivery_day.str[:7]).groupby(
            ["case", "strategy", "profile", "delivery_month"], sort=False).agg(
                purchased_mwh=("purchased_mwh", "sum"), sold_mwh=("sold_mwh", "sum"),
                cross_auction_offsetting_mwh=("cross_auction_offsetting_mwh", "sum"),
                net_committed_mwh=("net_committed_mwh", "sum")).reset_index()
    else:
        trade_monthly = pd.DataFrame()
    trade_monthly.to_csv(output / "trade_volumes_monthly.csv", index=False)
    paired_trade_volumes = []
    for case in sorted(trade_daily.case.unique()) if not trade_daily.empty else []:
        selected = trade_daily.loc[(trade_daily.case.eq(case)) &
            trade_daily.profile.eq("DA+IDA1+IDA2+IDA3")].pivot(index="delivery_day", columns="strategy",
                values=["cross_auction_offsetting_mwh", "purchased_mwh", "sold_mwh", "net_committed_mwh"])
        if not {"integrated", "sequential_all_auctions"}.issubset(set(selected.columns.get_level_values(1))):
            continue
        result = pd.DataFrame(index=selected.index)
        for field in ("cross_auction_offsetting_mwh", "purchased_mwh", "sold_mwh", "net_committed_mwh"):
            result[f"integrated_{field}"] = selected[(field, "integrated")]
            result[f"sequential_{field}"] = selected[(field, "sequential_all_auctions")]
            result[f"difference_{field}"] = result[f"integrated_{field}"] - result[f"sequential_{field}"]
        result["delivery_day"] = result.index
        result["case"] = case
        paired_trade_volumes.append(result.reset_index(drop=True))
    paired_volumes = pd.concat(paired_trade_volumes, ignore_index=True) if paired_trade_volumes else pd.DataFrame()
    paired_volumes.to_csv(output / "integrated_minus_sequential_trade_volumes_daily.csv", index=False)
    if not paired_volumes.empty:
        paired_volumes.groupby("case", sort=False).sum(numeric_only=True).reset_index().to_csv(
            output / "integrated_minus_sequential_trade_volumes_monthly.csv", index=False)

    # Reconcile auction revenue and DA-reference decomposition to the saved dispatch ledger.
    if not products.empty:
        for (case, strategy, profile, day), ledger_rows in products.groupby(
                ["case", "strategy", "profile", "delivery_day"], sort=False):
            financial_row = finances.loc[(finances.case.eq(case)) & finances.strategy.eq(strategy)
                & finances.profile.eq(profile) & finances.delivery_day.eq(day)]
            if len(financial_row) != 1:
                continue
            recorded_revenue = float(financial_row.auction_revenue_eur.iloc[0])
            decomposed_revenue = float(ledger_rows.trade_cashflow_eur.sum())
            tolerances = continuous_tolerances(config)
            if not tolerances.close(recorded_revenue, decomposed_revenue, monetary=True):
                raise ValueError(f"Auction trades do not reconcile to realized dispatch for {case}/{strategy}/{profile}/{day}")
            if ledger_rows.da_reference_available.all():
                da_value = float(ledger_rows.loc[ledger_rows.auction.eq("DA"), "final_position_da_value_eur"].sum())
                spread_value = float(ledger_rows.da_referenced_spread_eur.sum())
                if not tolerances.close(recorded_revenue, da_value + spread_value, monetary=True):
                    raise ValueError(f"DA value plus intraday spreads do not reconcile for {case}/{strategy}/{profile}/{day}")
    daily_comparisons = []
    for case in sorted(products.case.unique()):
        paired = _paired_comparison(products, positions, finances,
            "DA+IDA1+IDA2+IDA3", case)
        if paired.empty:
            continue
        paired["case"] = case
        daily_comparisons.append(paired)
    trade_comparison = pd.concat(daily_comparisons, ignore_index=True) if daily_comparisons else pd.DataFrame()
    trade_comparison.to_csv(output / "integrated_minus_sequential_by_auction_product.csv", index=False)
    monthly_fields = ["difference_signed_submitted_mwh", "difference_gross_traded_mwh",
        "difference_trade_cashflow_eur", "difference_da_referenced_spread_eur",
        "difference_final_position_da_value_eur"]
    monthly_trade = trade_comparison.groupby(["case", "auction"], sort=False)[monthly_fields].sum(min_count=1).reset_index() if not trade_comparison.empty else pd.DataFrame()
    monthly_trade.to_csv(output / "integrated_minus_sequential_monthly_by_auction.csv", index=False)

    financial_pairs = []
    for case in sorted(finances.case.unique()):
        current = finances.loc[(finances.case.eq(case)) & finances.profile.eq("DA+IDA1+IDA2+IDA3")]
        financial_columns = ["delivery_day", "auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur",
            "import_fee_eur", "transaction_fee_eur", "imbalance_penalty_eur", "net_profit_eur"]
        intg = current.loc[current.strategy.eq("integrated"), financial_columns].set_index("delivery_day")
        seq = current.loc[current.strategy.eq("sequential_all_auctions"), financial_columns].set_index("delivery_day")
        if intg.empty or seq.empty:
            continue
        pair = intg.add_prefix("integrated_").join(seq.add_prefix("sequential_"), how="outer")
        for field in ("auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur",
                      "import_fee_eur", "transaction_fee_eur", "imbalance_penalty_eur", "net_profit_eur"):
            pair[f"difference_{field}"] = pair[f"integrated_{field}"] - pair[f"sequential_{field}"]
        pair["case"] = case
        pair["delivery_day"] = pair.index
        financial_pairs.append(pair.reset_index(drop=True))
    financial = pd.concat(financial_pairs, ignore_index=True) if financial_pairs else pd.DataFrame()
    financial.to_csv(output / "integrated_minus_sequential_financials_daily.csv", index=False)
    monthly_financial = financial.groupby("case", sort=False).sum(numeric_only=True).reset_index() if not financial.empty else pd.DataFrame()
    monthly_financial.to_csv(output / "integrated_minus_sequential_financials_monthly.csv", index=False)

    for profile in profile_names:
        if profile == "DA+IDA1+IDA2+IDA3":
            _plots(output / profile, products.loc[products.profile.eq(profile)],
                   positions.loc[positions.profile.eq(profile)], financial, profile, config)
    # Save an interpretable concise comparison and explicitly separate attribution from causality.
    lines = [f"# Farm {config['plant']['farm_id']} continuous auction trade attribution", "",
        f"Run: `{root.name}`. Forecast policy: `{config['continuous_optimization']['pv_ensemble'].get('planning_policy', 'deterministic')}`.",
        "", "Trade products come from the immutable accepted-trade ledger and retain each run's recorded delivery resolution; position paths are also retained on the quarter-hour execution grid. Every submitted trade passed the gate check against the configured auction gate less its decision buffer, and IDA3 products passed the delivery-start check. In some January 2026 saved runs, DA prices differ by quarter-hour, so DA products are quarter-hourly in those ledgers rather than being forced into hourly buckets.",
        "", "DA-referenced revenue is available only when an actual DA price curve exists in this run's DA trades. Profiles without DA use the same-run full-auction DA curve if present; otherwise their DA-value and spread decomposition is explicitly unavailable. We do not impute prices.",
        "", "Integrated minus sequential accounting differences are descriptive attribution. They do not establish causal effects because the strategies' bids, dispatch and state trajectories differ.", ""]
    if not financial.empty:
        for case, group in financial.groupby("case", sort=False):
            row = group.sum(numeric_only=True)
            lines += [f"## {case}", "", f"Profit difference: **{row.get('difference_net_profit_eur', np.nan):,.2f} EUR**.",
                f"Auction revenue: {row.get('difference_auction_revenue_eur', np.nan):,.2f} EUR; "
                f"imbalance settlement: {row.get('difference_imbalance_cashflow_eur', np.nan):,.2f} EUR; "
                f"degradation cost difference: {row.get('difference_degradation_eur', np.nan):,.2f} EUR; "
                f"import cost difference: {row.get('difference_import_fee_eur', np.nan):,.2f} EUR; "
                f"transaction fee difference: {row.get('difference_transaction_fee_eur', np.nan):,.2f} EUR.", ""]
            lines.extend(["Full-auction revenue components and cross-auction offsetting volume:", "",
                "| Strategy | DA value (EUR) | IDA1 spread (EUR) | IDA2 spread (EUR) | IDA3 spread (EUR) | Cross-auction offsets (MWh) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |"])
            for strategy in ("integrated", "sequential_all_auctions"):
                revenue_rows = product_monthly.loc[(product_monthly.case.eq(case))
                    & product_monthly.profile.eq("DA+IDA1+IDA2+IDA3")
                    & product_monthly.strategy.eq(strategy)]
                da_value = revenue_rows.loc[revenue_rows.auction.eq("DA"), "final_position_da_value_eur"].sum()
                spreads = {auction: revenue_rows.loc[revenue_rows.auction.eq(auction),
                    "da_referenced_spread_eur"].sum(min_count=1) for auction in AUCTIONS[1:]}
                offset_rows = trade_monthly.loc[(trade_monthly.case.eq(case))
                    & trade_monthly.profile.eq("DA+IDA1+IDA2+IDA3")
                    & trade_monthly.strategy.eq(strategy)] if not trade_monthly.empty else pd.DataFrame()
                offsets = offset_rows.cross_auction_offsetting_mwh.sum() if not offset_rows.empty else np.nan
                lines.append("| " + " | ".join([strategy, f"{da_value:,.2f}",
                    *(f"{spreads[auction]:,.2f}" if pd.notna(spreads[auction]) else "unavailable" for auction in AUCTIONS[1:]),
                    f"{offsets:,.3f}"]) + " |")
            lines.append("")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output

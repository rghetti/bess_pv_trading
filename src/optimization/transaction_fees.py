"""Exact monthly progressive fees on node-owned signed market decisions."""
from __future__ import annotations

from .market_regime import configured_auction_product_groups


def build_transaction_fees(model, inputs, variables, gp):
    """Use LP absolute values for flat bands and exact ABS/PWL at breakpoints.

    Increasing total cost makes the flat-band epigraph exact at an optimum.
    Across tariff breakpoints, Gurobi general constraints enforce both absolute
    volumes and the nonconvex piecewise-linear equality exactly. Native-product
    shares reuse one absolute-value variable per information node.
    """
    settings = inputs.transaction_fees
    count = len(inputs.scenarios.scenario_ids)
    costs = {s: gp.LinExpr(0.0) for s in range(count)}
    diagnostics = {"mode": settings.mode, "absolute_volume_variables": 0,
                   "exact_absolute_constraints": 0, "piecewise_constraints": 0,
                   "monthly_regimes": {}}
    if settings.mode == "none" or (settings.mode == "flat" and settings.flat_eur_per_mwh == 0):
        variables["transaction_fee_formulation"] = diagnostics
        return costs
    state = inputs.market.transaction_fee_state or settings.initial_state
    months, representatives = {}, {}
    for auction, mask in inputs.auction_eligibility.items():
        if not any(mask):
            continue
        month = settings.month(inputs.auction_submission_times[auction])
        months.setdefault(month, []).append(auction)
        representatives[auction] = {t: t for t, enabled in enumerate(mask) if enabled}
        for group in configured_auction_product_groups(inputs.grid, auction,
                inputs.native_interval_minutes_by_auction.get(auction), inputs.market_timezone):
            if all(mask[t] for t in group):
                for t in group:
                    representatives[auction][t] = group[0]
    absolute, monthly_costs = {}, {}
    for month, auctions in months.items():
        starting = settings.starting_volume(state, month)
        maximum = sum(max(inputs.plant.poi_import_mw, inputs.plant.poi_export_mw)
            * inputs.grid.delta_hours[t] for auction in auctions
            for t, enabled in enumerate(inputs.auction_eligibility[auction]) if enabled)
        if maximum == 0:
            continue
        knots = [0.0, maximum]
        if settings.mode == "gme_real":
            knots = [0.0, *[threshold - starting for threshold in settings.tariff(month).thresholds_mwh
                if starting < threshold < starting + maximum], maximum]
        values = [settings.increment(month, starting, volume) for volume in knots]
        slopes = [(right_y-left_y)/(right-left)
            for left, right, left_y, right_y in zip(knots, knots[1:], values, values[1:])]
        linear = max(slopes) - min(slopes) < 1e-12
        diagnostics["monthly_regimes"][month] = {
            "starting_volume_mwh": starting, "maximum_new_volume_mwh": maximum,
            "representation": "linear_band" if linear else "exact_abs_and_nonconvex_pwl"}
        if linear and slopes[0] == 0:
            continue  # A provably free horizon needs no new model variables.
        for s in range(count):
            signature = (month, tuple((auction, inputs.tree.bid_nodes[auction][s]) for auction in auctions))
            if signature in monthly_costs:
                costs[s] += monthly_costs[signature]
                continue  # Later price/PV revelations cannot change identical trades' fee.
            volume = gp.LinExpr()
            for auction in auctions:
                owner = inputs.tree.bid_nodes[auction][s]
                owner_mass = sum(float(inputs.scenarios.probabilities[index])
                    for index, node in enumerate(inputs.tree.bid_nodes[auction]) if node == owner)
                for t, enabled in enumerate(inputs.auction_eligibility[auction]):
                    if not enabled:
                        continue
                    representative = representatives[auction][t]
                    key = (auction, owner, representative)
                    if key not in absolute:
                        bound = max(inputs.plant.poi_import_mw, inputs.plant.poi_export_mw) * min(
                            inputs.grid.delta_hours[index] for index, source in representatives[auction].items()
                            if source == representative)
                        traded = variables["trades"][auction, owner, representative]
                        absolute[key] = model.addVar(ub=bound, name=f"gross_trade[{auction},{owner},{representative}]")
                        if not linear or owner_mass == 0:
                            model.addGenConstrAbs(absolute[key], traded,
                                name=f"exact_gross_trade[{auction},{owner},{representative}]")
                            diagnostics["exact_absolute_constraints"] += 1
                        else:
                            model.addConstr(absolute[key] >= traded)
                            model.addConstr(absolute[key] >= -traded)
                    volume += absolute[key]
            if linear:
                monthly_costs[signature] = slopes[0] * volume
            else:
                gross = model.addVar(ub=maximum, name=f"monthly_gross[{month},s{s}]")
                fee = model.addVar(ub=values[-1], name=f"monthly_transaction_fee[{month},s{s}]")
                model.addConstr(gross == volume)
                model.addGenConstrPWL(gross, fee, knots, values,
                    name=f"progressive_monthly_fee[{month},s{s}]")
                diagnostics["piecewise_constraints"] += 1
                monthly_costs[signature] = fee
            costs[s] += monthly_costs[signature]
    diagnostics["absolute_volume_variables"] = len(absolute)
    variables["transaction_fee_formulation"] = diagnostics
    return costs

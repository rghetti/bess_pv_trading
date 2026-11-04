"""Matched-date economic comparisons with calendar-block uncertainty reporting.

Each bootstrap draw resamples delivery dates jointly for every farm. Separate
calendar segments retain their observed lengths: January and August are never
joined into a fictitious consecutive block or treated as independent farm rows.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "realized_operating_profit_eur", "absolute_imbalance_mwh", "degradation_eur",
    "throughput_mwh", "solver_runtime_total_seconds",
)


def _daily_frame(frame, metric):
    """Canonicalize daily identities while retaining missing numeric observations."""
    required = {"case_id", "farm_id", "delivery_day", "controller", metric}
    if not isinstance(frame, pd.DataFrame) or frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Research reporting requires nonempty daily columns: {sorted(required)}")
    data = frame[list(required)].copy()
    identifiers = ["case_id", "farm_id", "delivery_day", "controller"]
    if data[identifiers].isna().any().any():
        raise ValueError("Research daily identities cannot be missing")
    for name in identifiers:
        data[name] = data[name].astype(str)
    dates = pd.to_datetime(data.delivery_day, format="%Y-%m-%d", errors="raise")
    if not (dates.dt.strftime("%Y-%m-%d") == data.delivery_day).all():
        raise ValueError("Delivery days must be calendar dates in YYYY-MM-DD form")
    if data.duplicated(identifiers).any():
        raise ValueError("Duplicate case, farm, day and controller rows in research reporting")
    data[metric] = pd.to_numeric(data[metric], errors="raise")
    return data


def _bootstrap_settings(block_days, samples, seed, confidence):
    """Reject ambiguous bootstrap requests before any random draws occur."""
    for name, value, minimum in (("block_days", block_days, 1), ("samples", samples, 0), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"Bootstrap {name} must be an integer >= {minimum}")
    if isinstance(confidence, bool) or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Bootstrap confidence must lie strictly between zero and one")


def _calendar_blocks(days, block_days, samples, seed):
    """Draw moving blocks inside contiguous segments and preserve segment lengths."""
    dates = pd.DatetimeIndex(days)
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Bootstrap dates must be unique and sorted")
    boundaries = np.flatnonzero((dates[1:] - dates[:-1]) != pd.Timedelta(days=1)) + 1
    segments = np.split(np.arange(len(dates)), boundaries)
    random = np.random.default_rng(seed)
    pieces, details = [], []
    for segment in segments:
        size = len(segment)
        width = min(block_days, size)
        details.append({"start_day": str(days[int(segment[0])]), "end_day": str(days[int(segment[-1])]),
                        "days": size, "block_days_used": width,
                        "nonoverlapping_blocks": size // width})
        if samples:
            starts = random.integers(0, size - width + 1, size=(samples, int(np.ceil(size / width))))
            local = (starts[:, :, None] + np.arange(width)).reshape(samples, -1)[:, :size]
            pieces.append(segment[local])
    draws = np.concatenate(pieces, axis=1) if pieces else np.empty((0, len(dates)), dtype=int)
    return draws, details


def _statistics(values, draws, confidence, segments):
    """Summarize a date-by-farm difference matrix using the same resampled dates."""
    if values.ndim == 1:
        values = values[:, None]
    means = values.mean(axis=0)
    lower = upper = np.full(values.shape[1], np.nan)
    informative_draws = bool(len(draws)) and any(segment["days"] > segment["block_days_used"] for segment in segments)
    if informative_draws:
        replicates = values[draws].mean(axis=1)
        lower, upper = np.quantile(replicates, [(1 - confidence) / 2, (1 + confidence) / 2], axis=0)
    block_count = sum(segment["nonoverlapping_blocks"] for segment in segments)
    return [{"mean_daily_difference": float(means[index]), "total_difference": float(values[:, index].sum()),
             "ci_lower": None if not informative_draws else float(lower[index]),
             "ci_upper": None if not informative_draws else float(upper[index]),
             "paired_days": len(values), "nonoverlapping_calendar_blocks": block_count,
             "uncertainty_status": "not_requested" if not len(draws) else
                 "insufficient_calendar_blocks" if not informative_draws else
                 "few_calendar_blocks" if block_count < 5 else "conditional_calendar_block_bootstrap"}
            for index in range(values.shape[1])]


def _cohort(series_by_label, farms, days):
    """Expose every missing case/farm/day and retain complete dates across all farms."""
    parsed = pd.to_datetime(days, format="%Y-%m-%d", errors="raise")
    if list(parsed.strftime("%Y-%m-%d")) != list(days):
        raise ValueError("Expected delivery days must be YYYY-MM-DD calendar dates")
    expected = pd.MultiIndex.from_product([farms, days], names=["farm_id", "delivery_day"])
    missing, matrices = {}, {}
    valid = np.ones((len(days), len(farms)), dtype=bool)
    for label, series in series_by_label.items():
        aligned = series.reindex(expected)
        finite = np.isfinite(aligned.to_numpy(float))
        missing[label] = [{"farm_id": farm, "delivery_day": day,
            "reason": "missing_row" if (farm, day) not in series.index else "nonfinite_metric"}
            for (farm, day), present in zip(expected, finite) if not present]
        matrix = aligned.to_numpy(float).reshape(len(farms), len(days)).T
        matrices[label] = matrix
        valid &= np.isfinite(matrix)
    complete = valid.all(axis=1)
    coverage = {"expected_farms": farms, "expected_days": days, "missing": missing,
                "paired_days": [day for day, keep in zip(days, complete) if keep],
                "excluded_days": [day for day, keep in zip(days, complete) if not keep],
                "pairing_rule": "all selected farms and cases present on a common delivery date"}
    return {label: matrix[complete] for label, matrix in matrices.items()}, coverage


def empirical_loss_cvar(losses, alpha=0.95, *, weights=None, minimum_tail_observations=5):
    """Integrate exactly the worst ``1-alpha`` probability mass of signed EUR losses.

    A boundary observation contributes fractional mass when required. Negative
    loss is profit. Tail diagnostics describe this empirical distribution only;
    a small tail is explicitly unsuitable for a precise extreme-risk claim.
    """
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Empirical CVaR requires a nonempty finite loss vector")
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("CVaR alpha must lie in [0, 1)")
    if (isinstance(minimum_tail_observations, bool) or not isinstance(minimum_tail_observations, int)
            or minimum_tail_observations < 1):
        raise ValueError("Minimum tail observations must be a positive integer")
    probabilities = np.full(len(values), 1 / len(values)) if weights is None else np.asarray(weights, dtype=float)
    if (probabilities.shape != values.shape or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError("Empirical CVaR weights must be a normalized nonnegative vector")
    order = np.argsort(-values, kind="stable")
    ordered_weights = probabilities[order]
    remaining = np.maximum(1 - alpha - (np.cumsum(ordered_weights) - ordered_weights), 0)
    tail_mass = np.minimum(ordered_weights, remaining)
    effective = float(tail_mass.sum() ** 2 / np.sum(tail_mass ** 2))
    active = tail_mass > 0
    return {"alpha": float(alpha), "loss_cvar_eur": float(tail_mass @ values[order] / (1 - alpha)),
            "tail_boundary_loss_eur": float(values[order][active][-1]), "tail_probability": float(1 - alpha),
            "observations": len(values), "tail_observations_used": int(active.sum()),
            "effective_tail_observations": effective,
            "fractional_boundary_mass": float(tail_mass[active][-1]),
            "tail_status": "insufficient_tail_observations" if effective < minimum_tail_observations
                else "empirical_tail_only", "loss_definition": "signed monetary loss; positive means loss"}


def paired_case_comparison(frame, baseline_case, metric="realized_operating_profit_eur", *,
                           baseline_controller=None, block_days=7, samples=2000, seed=0,
                           confidence=0.95, expected_farms=None, expected_days=None, cvar_alpha=0.95):
    """Compare every case/controller with a declared baseline on common whole dates.

    With multiple baseline controllers, omitting ``baseline_controller`` matches
    controller names. A single baseline controller is unambiguous and is used
    for all cases. Explicit expected cohorts expose dates missing from both runs.
    """
    data = _daily_frame(frame, metric)
    _bootstrap_settings(block_days, samples, seed, confidence)
    baseline_case = str(baseline_case)
    baseline = data[data.case_id == baseline_case]
    if baseline.empty:
        raise ValueError("Baseline case is absent from daily results")
    available = sorted(baseline.controller.unique())
    if baseline_controller is not None and baseline_controller not in available:
        raise ValueError("Requested baseline controller is absent from daily results")
    comparisons, daily = [], []
    for (case, controller), selected in data.groupby(["case_id", "controller"], sort=True):
        reference_controller = baseline_controller or (available[0] if len(available) == 1 else controller)
        if case == baseline_case and controller == reference_controller:
            continue
        reference = baseline[baseline.controller == reference_controller]
        identity = {"case_id": case, "controller": controller, "baseline_case": baseline_case,
                    "baseline_controller": reference_controller}
        if reference.empty:
            comparisons.append({**identity, "status": "missing_baseline_controller", "aggregate": None,
                                "farms": [], "available_baseline_controllers": available})
            continue
        farms = sorted(set(map(str, expected_farms))) if expected_farms is not None else sorted(
            set(selected.farm_id) | set(reference.farm_id))
        days = sorted(set(map(str, expected_days))) if expected_days is not None else sorted(
            set(selected.delivery_day) | set(reference.delivery_day))
        if not farms or not days:
            raise ValueError("Expected pairing cohorts cannot be empty")
        matrices, coverage = _cohort({"case": selected.set_index(["farm_id", "delivery_day"])[metric],
            "baseline": reference.set_index(["farm_id", "delivery_day"])[metric]}, farms, days)
        if not coverage["paired_days"]:
            comparisons.append({**identity, "status": "no_complete_paired_days", "coverage": coverage,
                                "aggregate": None, "farms": []})
            continue
        differences = matrices["case"] - matrices["baseline"]
        draws, segments = _calendar_blocks(coverage["paired_days"], block_days, samples, seed)
        farm_statistics = _statistics(differences, draws, confidence, segments)
        aggregate = _statistics(differences.sum(axis=1), draws, confidence, segments)[0]
        result = {**identity, "status": "complete" if not coverage["excluded_days"] else "partial_coverage",
                  "coverage": coverage, "calendar_segments": segments, "aggregate": aggregate,
                  "farms": [{"farm_id": farm, **statistics} for farm, statistics in zip(farms, farm_statistics)]}
        if metric == "realized_operating_profit_eur":
            result["downside"] = {label: empirical_loss_cvar(-matrix.sum(axis=1), cvar_alpha)
                                  for label, matrix in matrices.items()}
        comparisons.append(result)
        for day_index, day in enumerate(coverage["paired_days"]):
            for farm_index, farm in enumerate(farms):
                daily.append({**identity, "farm_id": farm, "delivery_day": day, "metric": metric,
                              "case_value": float(matrices["case"][day_index, farm_index]),
                              "baseline_value": float(matrices["baseline"][day_index, farm_index]),
                              "difference": float(differences[day_index, farm_index])})
    return {"metric": metric, "baseline_case": baseline_case, "comparisons": comparisons,
            "interpretation": "Continuous optimization outcomes include consequent changes to later auction submissions",
            "daily_differences": daily, "bootstrap": {"block_days": int(block_days), "samples": int(samples),
                "seed": int(seed), "confidence": float(confidence), "unit": "common calendar date across all farms",
                "segments": "resample within contiguous segments, preserving observed segment lengths",
                "scope": "conditional on selected dates, farms, policies and forecasts; no independent-farm claim"}}


def factorial_effects(frame, cases, *, controller, physical_controls, metric="realized_operating_profit_eur",
                      block_days=7, samples=2000, seed=0, confidence=0.95, expected_farms=None, expected_days=None):
    """Estimate matched 2-by-2 information-update and auction-access differences.

    ``cases[(updates_enabled, auctions_enabled)]`` gives each case ID. Explicit
    identical physical-control metadata is required, so asset-size or controller
    changes cannot silently be attributed to information or auction access.
    """
    required = {(False, False), (True, False), (False, True), (True, True)}
    if set(cases) != required or len(set(cases.values())) != 4:
        raise ValueError("Factorial analysis needs four distinct update/access case IDs")
    case_ids = [str(cases[key]) for key in sorted(required)]
    if (not isinstance(physical_controls, dict) or set(physical_controls) != set(case_ids)
            or any(not isinstance(value, dict) or not value for value in physical_controls.values())
            or len({json.dumps(physical_controls[case], sort_keys=True, allow_nan=False) for case in case_ids}) != 1):
        raise ValueError("Factorial analysis requires identical explicit physical controls for all four cases")
    data = _daily_frame(frame, metric)
    _bootstrap_settings(block_days, samples, seed, confidence)
    selected = data[(data.case_id.isin(case_ids)) & (data.controller == controller)]
    farms = sorted(set(map(str, expected_farms))) if expected_farms is not None else sorted(selected.farm_id.unique())
    days = sorted(set(map(str, expected_days))) if expected_days is not None else sorted(selected.delivery_day.unique())
    if not farms or not days:
        raise ValueError("Factorial controller has no daily results")
    matrices, coverage = _cohort({case: selected[selected.case_id == case].set_index(
        ["farm_id", "delivery_day"])[metric] for case in case_ids}, farms, days)
    result = {"metric": metric, "controller": controller,
              "cases": [{"updates": key[0], "auction_access": key[1], "case_id": str(value)}
                        for key, value in cases.items()], "coverage": coverage, "effects": [],
              "physical_controls": physical_controls[case_ids[0]], "status": "no_complete_paired_days"}
    if not coverage["paired_days"]:
        return result
    cell = {key: matrices[str(value)] for key, value in cases.items()}
    effects = {
        "updates_without_auction_access": cell[True, False] - cell[False, False],
        "updates_with_auction_access": cell[True, True] - cell[False, True],
        "auction_access_without_updates": cell[False, True] - cell[False, False],
        "auction_access_with_updates": cell[True, True] - cell[True, False],
        "interaction_difference_in_differences": cell[True, True] - cell[True, False]
            - cell[False, True] + cell[False, False],
        "mean_update_effect": (cell[True, False] - cell[False, False] + cell[True, True] - cell[False, True]) / 2,
        "mean_auction_access_effect": (cell[False, True] - cell[False, False] + cell[True, True] - cell[True, False]) / 2,
    }
    draws, segments = _calendar_blocks(coverage["paired_days"], block_days, samples, seed)
    for name, values in effects.items():
        statistics = _statistics(values, draws, confidence, segments)
        result["effects"].append({"effect": name,
            "aggregate": _statistics(values.sum(axis=1), draws, confidence, segments)[0],
            "farms": [{"farm_id": farm, **row} for farm, row in zip(farms, statistics)]})
    result.update(status="complete" if not coverage["excluded_days"] else "partial_coverage", calendar_segments=segments)
    return result


def forecast_flexibility_factorial(frame, cases, *, controller, physical_controls,
                                   changed_fields, **settings):
    """Compare calibration effects at two physical settings on matched dates.

    Controls are keyed by case and then by farm/window. Within each physical
    setting the calibrated and raw cases must match exactly. Across settings,
    only the explicitly named battery or POI fields may differ.
    """
    required = {(False, False), (True, False), (False, True), (True, True)}
    if set(cases) != required or len(set(cases.values())) != 4:
        raise ValueError("Forecast/flexibility analysis requires four distinct cells")
    if not changed_fields or any(not isinstance(path, (list, tuple)) or len(path) != 2
                                 or path[0] not in {"battery", "plant"} for path in changed_fields):
        raise ValueError("Declare physical fields that the flexibility contrast changes")
    if not isinstance(physical_controls, dict) or set(physical_controls) != set(cases.values()):
        raise ValueError("Forecast/flexibility analysis requires controls for each cell")
    raw_base = physical_controls[cases[False, False]]
    calibrated_base = physical_controls[cases[True, False]]
    raw_changed = physical_controls[cases[False, True]]
    calibrated_changed = physical_controls[cases[True, True]]
    if not raw_base or raw_base != calibrated_base or raw_changed != calibrated_changed:
        raise ValueError("Raw and calibrated cases must have matching physical controls")
    if set(raw_base) != set(raw_changed):
        raise ValueError("Flexibility cells must cover the same farms and windows")
    normalized_base = deepcopy(raw_base)
    normalized_changed = deepcopy(raw_changed)
    any_change = False
    for identity in normalized_base:
        for section, field in changed_fields:
            try:
                before = normalized_base[identity][section].pop(field)
                after = normalized_changed[identity][section].pop(field)
            except (KeyError, AttributeError) as error:
                raise ValueError("Declared flexibility field is missing from physical controls") from error
            any_change |= before != after
    if not any_change or normalized_base != normalized_changed:
        raise ValueError("Only the declared flexibility fields may differ across physical settings")
    # Reuse the same complete-case matching and calendar-block resampling engine.
    # Explicit matching was checked above; the generic helper sees identical
    # normalized controls while computing the same four-cell arithmetic.
    controls = {case: normalized_base for case in cases.values()}
    result = factorial_effects(frame, cases, controller=controller, physical_controls=controls, **settings)
    names = {"updates_without_auction_access": "calibration_at_base_flexibility",
        "updates_with_auction_access": "calibration_at_modified_flexibility",
        "auction_access_without_updates": "flexibility_with_raw_forecasts",
        "auction_access_with_updates": "flexibility_with_calibrated_forecasts",
        "mean_update_effect": "mean_calibration_effect",
        "mean_auction_access_effect": "mean_flexibility_effect"}
    for effect in result["effects"]:
        effect["effect"] = names.get(effect["effect"], effect["effect"])
    for cell in result["cases"]:
        cell["calibrated"] = cell.pop("updates")
        cell["flexibility_modified"] = cell.pop("auction_access")
    result["changed_fields"] = [list(path) for path in changed_fields]
    result["physical_controls"] = {"base": raw_base, "modified": raw_changed}
    return result


def save_research_report(frame, destination, baseline_case, *, metrics=DEFAULT_METRICS,
                         plots=True, factorial=None, **comparison_settings):
    """Write compact reviewable artifacts from completed daily KPI tables only."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metrics = tuple(metrics)
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("Research report metrics must be nonempty and unique")
    reports = [paired_case_comparison(frame, baseline_case, metric, **comparison_settings) for metric in metrics]
    summaries, differences = [], []
    for report in reports:
        differences.extend(report["daily_differences"])
        for comparison in report["comparisons"]:
            identity = {name: comparison[name] for name in ("case_id", "controller", "baseline_case", "baseline_controller")}
            rows = [{"farm_id": "portfolio", **comparison["aggregate"]}] if comparison["aggregate"] else []
            rows.extend(comparison["farms"])
            for row in rows:
                summaries.append({**identity, "metric": report["metric"], "status": comparison["status"], **row})
    payload = {"schema": "energy-research-comparison-v1", "reports": reports, "factorial": factorial}
    (destination / "research_comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(summaries).to_csv(destination / "research_comparison.csv", index=False)
    pd.DataFrame(differences).to_csv(destination / "research_daily_differences.csv", index=False)
    lines = ["# Energy-market research comparison", "",
        "Differences are case minus baseline. Portfolio rows sum farms on the same delivery date; farms are resampled together.", "",
        "Continuous optimization comparisons include consequent changes to later auction submissions; they measure the complete policy outcome.", "",
        "Uncertainty is conditional on the selected dates. Calendar blocks never cross gaps; short tails and few blocks are flagged in JSON.", "",
        "| Case | Controller | Farm | Metric | Paired days | Mean daily difference | Interval | Coverage |",
        "|---|---|---|---|---:|---:|---|---|"]
    for row in summaries:
        interval = row["uncertainty_status"] if row["ci_lower"] is None else f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        lines.append(f"| {row['case_id']} | {row['controller']} | {row['farm_id']} | {row['metric']} | "
                     f"{row['paired_days']} | {row['mean_daily_difference']:.3f} | {interval} | {row['status']} |")
    lines.extend(["", "Coverage exclusions:", ""])
    for report in reports:
        for comparison in report["comparisons"]:
            missing = comparison.get("coverage", {}).get("missing", {})
            count = sum(len(rows) for rows in missing.values())
            lines.append(f"- {report['metric']}, {comparison['case_id']}/{comparison['controller']}: "
                         f"{comparison['status']}; {count} missing or nonfinite case/farm/day cells.")
    (destination / "research_comparison.md").write_text("\n".join(lines) + "\n")
    if plots and differences:
        _comparison_plots(destination, pd.DataFrame(differences))
    return payload


def _comparison_plots(destination, frame):
    """Plot matched daily portfolio differences without filling calendar gaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = destination / "plots"
    output.mkdir(exist_ok=True)
    for metric, metric_frame in frame.groupby("metric", sort=True):
        figure, axis = plt.subplots(figsize=(10, 4))
        for (case, controller), group in metric_frame.groupby(["case_id", "controller"], sort=True):
            daily = group.groupby("delivery_day", sort=True).difference.sum()
            # Scatter prevents a line from implying forecasts or results in missing months.
            axis.scatter(pd.to_datetime(daily.index), daily.to_numpy(), s=18, label=f"{case}/{controller}")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(xlabel="Common delivery date", ylabel=f"Case minus baseline: {metric}")
        axis.legend(fontsize=7)
        figure.tight_layout()
        safe_metric = "".join(character if character.isalnum() or character == "_" else "_" for character in metric)
        figure.savefig(output / f"paired_daily_{safe_metric}.png", dpi=130)
        plt.close(figure)

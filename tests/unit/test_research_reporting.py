"""Synthetic accounting fixtures verify pairing, temporal blocks and finite tails."""

import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.reporting.research_study import (
    _calendar_blocks, empirical_loss_cvar, factorial_effects, paired_case_comparison, save_research_report,
)


METRIC = "realized_operating_profit_eur"


def daily_results(days=8, farms=("1", "345"), *, gains=None):
    """One common date shock drives all farms, so independent-farm resampling is wrong."""
    dates = pd.date_range("2026-01-01", periods=days).strftime("%Y-%m-%d")
    gains = np.arange(days, dtype=float) if gains is None else np.asarray(gains, dtype=float)
    rows = []
    for case, controller in (("baseline", "track_committed_grid"), ("adaptive", "ensemble_mpc_adaptive")):
        for index, day in enumerate(dates):
            for farm in farms:
                rows.append({"case_id": case, "controller": controller, "delivery_day": day, "farm_id": farm,
                    METRIC: 100 + (gains[index] if case == "adaptive" else 0),
                    "absolute_imbalance_mwh": 2.0 if case == "baseline" else 1.0,
                    "degradation_eur": 3.0, "throughput_mwh": 4.0, "solver_runtime_total_seconds": 5.0})
    return pd.DataFrame(rows)


def test_case_comparison_resamples_common_dates_not_independent_farms():
    frame = daily_results()
    report = paired_case_comparison(frame, "baseline", block_days=2, samples=100, seed=3)
    comparison = report["comparisons"][0]
    assert comparison["status"] == "complete"
    aggregate = comparison["aggregate"]
    farms = comparison["farms"]
    assert aggregate["paired_days"] == 8
    assert aggregate["mean_daily_difference"] == pytest.approx(7)
    assert aggregate["total_difference"] == pytest.approx(56)
    # Perfectly correlated farm effects produce exactly twice either farm's interval.
    assert aggregate["ci_lower"] == pytest.approx(2 * farms[0]["ci_lower"])
    assert aggregate["ci_upper"] == pytest.approx(2 * farms[0]["ci_upper"])
    assert farms[0]["ci_lower"] == farms[1]["ci_lower"]
    assert len(report["daily_differences"]) == 16
    assert comparison["downside"]["case"]["observations"] == 8
    assert comparison["downside"]["case"]["tail_status"] == "insufficient_tail_observations"
    assert "auction submissions" in report["interpretation"]


def test_calendar_bootstrap_never_bridges_gaps_and_preserves_season_counts():
    days = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-08-02", "2026-08-03"]
    draws, segments = _calendar_blocks(days, block_days=2, samples=100, seed=0)
    assert draws.shape == (100, 5)
    assert np.all(draws[:, :3] < 3)
    assert np.all(draws[:, 3:] >= 3)
    assert np.all(np.diff(draws[:, :2], axis=1) == 1)
    assert np.all(np.diff(draws[:, 3:], axis=1) == 1)
    assert segments[0]["days"] == 3
    assert segments[1]["days"] == 2
    repeated, _ = _calendar_blocks(days, block_days=2, samples=100, seed=0)
    np.testing.assert_array_equal(draws, repeated)


def test_missing_one_farm_excludes_whole_common_date_and_reports_missing_cells():
    frame = daily_results()
    remove = (frame.case_id == "adaptive") & (frame.farm_id == "345") & (frame.delivery_day == "2026-01-03")
    report = paired_case_comparison(frame.loc[~remove], "baseline", samples=0)
    comparison = report["comparisons"][0]
    assert comparison["status"] == "partial_coverage"
    assert comparison["aggregate"]["paired_days"] == 7
    assert comparison["coverage"]["excluded_days"] == ["2026-01-03"]
    assert comparison["coverage"]["missing"]["case"] == [
        {"farm_id": "345", "delivery_day": "2026-01-03", "reason": "missing_row"}]
    assert len(report["daily_differences"]) == 14
    assert comparison["aggregate"]["ci_lower"] is None


def test_explicit_cohort_exposes_dates_missing_from_both_cases():
    report = paired_case_comparison(daily_results(days=2), "baseline", samples=0,
        expected_days=["2026-01-01", "2026-01-02", "2026-01-03"])
    coverage = report["comparisons"][0]["coverage"]
    assert coverage["excluded_days"] == ["2026-01-03"]
    assert len(coverage["missing"]["case"]) == len(coverage["missing"]["baseline"]) == 2


def test_nonfinite_metric_is_explicit_missing_evidence_not_zero():
    frame = daily_results(days=3)
    frame.loc[(frame.case_id == "adaptive") & (frame.delivery_day == "2026-01-02"), METRIC] = np.nan
    comparison = paired_case_comparison(frame, "baseline", samples=0)["comparisons"][0]
    assert comparison["aggregate"]["paired_days"] == 2
    assert all(row["reason"] == "nonfinite_metric" for row in comparison["coverage"]["missing"]["case"])
    assert comparison["aggregate"]["mean_daily_difference"] == 2


def test_no_complete_dates_preserves_failed_comparison_in_report():
    frame = daily_results(days=1)
    frame.loc[frame.case_id == "adaptive", METRIC] = np.nan
    report = paired_case_comparison(frame, "baseline")
    assert report["comparisons"][0]["status"] == "no_complete_paired_days"
    assert report["comparisons"][0]["aggregate"] is None
    assert report["daily_differences"] == []


def test_single_available_calendar_block_does_not_claim_zero_width_uncertainty():
    report = paired_case_comparison(daily_results(days=3), "baseline", block_days=7, samples=50)
    aggregate = report["comparisons"][0]["aggregate"]
    assert aggregate["uncertainty_status"] == "insufficient_calendar_blocks"
    assert aggregate["ci_lower"] is None
    assert aggregate["ci_upper"] is None


def test_multi_controller_baseline_requires_match_or_explicit_reference():
    frame = daily_results(days=2)
    extra = frame[frame.case_id == "baseline"].copy()
    extra["controller"] = "ensemble_mpc"
    frame = pd.concat([frame, extra], ignore_index=True)
    report = paired_case_comparison(frame, "baseline", samples=0)
    assert report["comparisons"][0]["status"] == "missing_baseline_controller"
    explicit = paired_case_comparison(frame, "baseline", baseline_controller="track_committed_grid", samples=0)
    assert {row["case_id"] for row in explicit["comparisons"]} == {"adaptive", "baseline"}
    assert all(row["status"] == "complete" for row in explicit["comparisons"])


@pytest.mark.parametrize("mutation", ["duplicate", "missing_identity", "timestamp_day"])
def test_ambiguous_daily_identities_fail(mutation):
    frame = daily_results(days=2)
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "missing_identity":
        frame.loc[0, "farm_id"] = None
    else:
        frame.loc[0, "delivery_day"] = "2026-01-01T12:00:00"
    with pytest.raises(ValueError):
        paired_case_comparison(frame, "baseline")


def test_empirical_cvar_uses_fractional_boundary_probability():
    result = empirical_loss_cvar([0, 10, 20, 30], alpha=0.6)
    # Worst 40% is 25% at 30 plus 15% at 20, rather than the worst two full rows.
    assert result["loss_cvar_eur"] == pytest.approx((0.25 * 30 + 0.15 * 20) / 0.4)
    assert result["tail_observations_used"] == 2
    assert result["fractional_boundary_mass"] == pytest.approx(0.15)
    assert result["tail_boundary_loss_eur"] == 20
    assert empirical_loss_cvar([0, 10, 20, 30], alpha=0)["loss_cvar_eur"] == 15
    assert empirical_loss_cvar([-10, -5], alpha=0.5)["loss_cvar_eur"] == -5
    assert empirical_loss_cvar([10, 20], alpha=np.nextafter(1.0, 0.0))["loss_cvar_eur"] == 20


def test_weighted_cvar_keeps_declared_distribution_and_reports_tail_support():
    result = empirical_loss_cvar([1, 5, 100], alpha=0.8, weights=[0.6, 0.3, 0.1])
    assert result["loss_cvar_eur"] == pytest.approx((0.1 * 100 + 0.1 * 5) / 0.2)
    assert result["effective_tail_observations"] == pytest.approx(2)
    with pytest.raises(ValueError, match="weights"):
        empirical_loss_cvar([1, 2], weights=[0.5, 0.6])
    with pytest.raises(ValueError, match="alpha"):
        empirical_loss_cvar([1, 2], alpha=1)


def factorial_fixture():
    """Updates add 2, access adds 3, and their interaction adds 7 per farm/day."""
    cases = {(False, False): "none", (True, False): "updates", (False, True): "auctions", (True, True): "both"}
    template = daily_results(days=4)
    template = template[template.case_id == "baseline"]
    rows = []
    for (updates, auctions), case in cases.items():
        selected = template.copy()
        selected["case_id"] = case
        selected[METRIC] += 2 * updates + 3 * auctions + 7 * updates * auctions
        rows.append(selected)
    physical = {case: {"battery_capacity_mwh": 1, "poi_mw": 1, "initial_soc_percent": 50} for case in cases.values()}
    return pd.concat(rows, ignore_index=True), cases, physical


def test_factorial_effects_distinguish_main_conditional_and_interaction_values():
    frame, cases, physical = factorial_fixture()
    result = factorial_effects(frame, cases, controller="track_committed_grid", physical_controls=physical, samples=20)
    effects = {row["effect"]: row["aggregate"]["mean_daily_difference"] for row in result["effects"]}
    assert effects["updates_without_auction_access"] == 4
    assert effects["auction_access_without_updates"] == 6
    assert effects["updates_with_auction_access"] == 18
    assert effects["auction_access_with_updates"] == 20
    assert effects["interaction_difference_in_differences"] == 14
    assert effects["mean_update_effect"] == 11
    assert effects["mean_auction_access_effect"] == 13
    assert result["status"] == "complete"


def test_factorial_rejects_asset_changes_and_empty_matching_claims():
    frame, cases, physical = factorial_fixture()
    physical["both"]["battery_capacity_mwh"] = 2
    with pytest.raises(ValueError, match="identical"):
        factorial_effects(frame, cases, controller="track_committed_grid", physical_controls=physical)
    with pytest.raises(ValueError, match="identical"):
        factorial_effects(frame, cases, controller="track_committed_grid", physical_controls={case: {} for case in cases.values()})


def test_factorial_missing_cell_excludes_date_across_all_four_cases_and_farms():
    frame, cases, physical = factorial_fixture()
    remove = (frame.case_id == "both") & (frame.farm_id == "345") & (frame.delivery_day == "2026-01-02")
    result = factorial_effects(frame.loc[~remove], cases, controller="track_committed_grid", physical_controls=physical, samples=0)
    assert result["status"] == "partial_coverage"
    assert result["coverage"]["excluded_days"] == ["2026-01-02"]
    assert result["effects"][0]["aggregate"]["paired_days"] == 3


def test_report_artifacts_expose_per_farm_comparisons_and_continuous_interpretation(tmp_path):
    result = save_research_report(daily_results(days=3), tmp_path, "baseline", samples=0, plots=True)
    assert len(result["reports"]) == 5
    saved = json.loads((tmp_path / "research_comparison.json").read_text())
    assert saved["schema"] == "energy-research-comparison-v1"
    csv = pd.read_csv(tmp_path / "research_comparison.csv")
    assert set(csv.farm_id.astype(str)) == {"portfolio", "1", "345"}
    assert len(csv) == 15
    markdown = (tmp_path / "research_comparison.md").read_text()
    assert "auction submissions" in markdown
    assert "resampled together" in markdown
    assert len(list((tmp_path / "plots").glob("*.png"))) == 5


@pytest.mark.parametrize("settings", [{"block_days": 0}, {"samples": -1}, {"seed": True}, {"confidence": 1}])
def test_invalid_bootstrap_settings_reject_before_comparison(settings):
    with pytest.raises(ValueError):
        paired_case_comparison(daily_results(days=2), "baseline", **settings)

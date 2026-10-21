"""Matched-day sequential comparisons, chart placement and configuration contracts."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import ConfigurationError, load_config, resolve_config
from bess_pv_trading.participation import profile_id
from bess_pv_trading.simulation.participation_study import (
    SEQUENTIAL_PROFILE, participation_comparison_table, summarize_participation_days, summarize_sequential_days,
)


def test_sequential_summary_pairs_days_and_places_row_below_integrated_reference(tmp_path):
    cases = load_config("tests/fixtures/configs/participation_cases.yaml")
    case = cases["ranking"]
    daily = pd.DataFrame([{**case["metrics"], **row} for row in case["daily"]])
    daily, ranking = summarize_participation_days(daily, case["profiles"], case["delivery_days"], case["profit_tolerance_eur"])
    auctions = case["profiles"][-1]
    sequential = daily.loc[daily.profile == profile_id(case["profiles"][0])].copy()
    sequential["infeasible_auction_decisions"] = 0
    paired, summary = summarize_sequential_days(sequential, daily, case["delivery_days"], auctions)
    reference = daily.loc[daily.profile == profile_id(auctions)].set_index("study_date").net_profit_eur
    expected = sequential.study_date.map(reference) - sequential.net_profit_eur
    np.testing.assert_allclose(paired.integrated_minus_sequential_profit_eur, expected)
    assert summary["total_integrated_minus_sequential_profit_eur"] == pytest.approx(expected.sum())
    chart = participation_comparison_table(ranking, sequential=summary)
    anchor = list(chart.profile).index(profile_id(auctions))
    assert chart.profile.iloc[anchor + 1] == SEQUENTIAL_PROFILE
    assert chart.mean_net_profit_eur.iloc[anchor + 1] == pytest.approx(sequential.net_profit_eur.mean())
    from bess_pv_trading.simulation.participation_study import _plot_participation
    perfect = {"label": "Perfect foresight (all auctions)", "selected_auctions": auctions,
        "mean_objective_eur": cases["perfect_foresight"]["expected_mean_objective_eur"],
        "std_daily_objective_eur": cases["perfect_foresight"]["expected_std_objective_eur"],
        "mean_final_energy_mwh": case["metrics"]["final_energy_mwh"]}
    chart = participation_comparison_table(ranking, perfect, summary)
    assert chart.profile.iloc[0] == perfect["label"]
    _plot_participation(ranking, tmp_path / "comparison.png", perfect, summary)
    assert (tmp_path / "comparison.png").is_file()
    with pytest.raises(ValueError, match="identical requested delivery day"):
        summarize_sequential_days(sequential.iloc[:-1], daily, case["delivery_days"], auctions)
    invalid = sequential.copy()
    invalid["net_profit_eur"] = np.nan
    with pytest.raises(ValueError, match="fully priced"):
        summarize_sequential_days(invalid, daily, case["delivery_days"], auctions)


def test_config_rejects_unknown_modes_and_nonboolean_sequential_flags():
    config = load_config("configs/standard.yaml")
    assert config["gme_oracle_experiment"]["auction_planning_mode"] == "integrated"
    assert config["participation_study"]["sequential_all_auctions_baseline"] is False
    cases = load_config("tests/fixtures/configs/participation_cases.yaml")["sequential"]
    for mode in cases["invalid_modes"]:
        invalid = deepcopy(config)
        invalid["gme_oracle_experiment"]["auction_planning_mode"] = mode
        with pytest.raises(ConfigurationError, match="auction_planning_mode"):
            resolve_config(invalid)
    for value in cases["invalid_baseline_flags"]:
        invalid = deepcopy(config)
        invalid["participation_study"]["sequential_all_auctions_baseline"] = value
        with pytest.raises(ConfigurationError, match="YAML booleans"):
            resolve_config(invalid)


def test_batch_runs_separate_sequential_cases_and_resumes_verified_outputs(tmp_path, monkeypatch):
    from bess_pv_trading.simulation import participation_study as module
    from bess_pv_trading.simulation.synthetic import write_json
    case = load_config("tests/fixtures/configs/participation_cases.yaml")["ranking"]
    config = load_config("configs/full_experiments/gme_participation_2025-01-01_03.yaml")
    auctions = list(config["gme_oracle_experiment"]["auction_datasets"])
    config["participation_study"].update(output_root=str(tmp_path), run_id="matched", plots=False,
        perfect_foresight_reference=False, sequential_all_auctions_baseline=True,
        delivery_days=case["delivery_days"], profiles=[auctions])
    seen = []

    def run(child):
        mode = child["gme_oracle_experiment"]["auction_planning_mode"]
        day = child["gme_oracle_experiment"]["delivery_day"]
        seen.append((day, mode))
        output = tmp_path / "child" / child["experiment_outputs"]["run_id"] / day
        (output / "csv").mkdir(parents=True)
        metrics = case["metrics"]
        profit_profile = case["profiles"][-1 if mode == "integrated" else 0]
        profit = next(row["net_profit_eur"] for row in case["daily"]
            if row["study_date"] == day and row["profile"] == profile_id(profit_profile))
        write_json(output / "json/kpis.json", {"total_gain_eur": profit,
            "auction_revenue_eur": profit, "degradation_cost_eur": metrics["degradation_eur"],
            **{key: metrics[key] for key in ("transaction_fee_eur", "curtailment_mwh", "absolute_imbalance_mwh", "final_energy_mwh", "final_soc_percent")},
            "auction_decision_solves": len(auctions), "active_auctions": auctions, "skipped_auctions": []})
        write_json(output / "json/manifest.json", {"status": "complete"})
        values = case["realized_pv_mw"]
        times = pd.date_range(day, periods=len(values), freq="15min", tz="UTC").astype(str)
        pd.DataFrame({"interval_id": times, "pv_realized_execution_ac_mw": values,
            "pv_initial_forecast_ac_mw": values}).to_csv(output / "csv/pv_forecast_and_realized.csv", index=False)
        pd.DataFrame({"interval_id": times, **{auction: values for auction in auctions}}).to_csv(
            output / "csv/auction_prices_expanded.csv", index=False)
        return {"output_dir": str(output)}

    monkeypatch.setattr(module, "run_gme_oracle_experiment", run)
    result = module.run_participation_study(config)
    assert result["completed_experiments"] == len(case["delivery_days"]) * 2
    assert len(seen) == result["completed_experiments"]
    baseline = pd.read_csv(tmp_path / "matched/csv/sequential_daily_results.csv")
    assert set(baseline.study_date) == set(case["delivery_days"])
    chart = pd.read_csv(tmp_path / "matched/csv/participation_comparison.csv")
    assert chart.profile.tolist() == [profile_id(auctions), SEQUENTIAL_PROFILE]
    config["participation_study"]["resume"] = True
    monkeypatch.setattr(module, "run_gme_oracle_experiment", lambda child: pytest.fail("Completed baseline must resume without solving"))
    assert module.run_participation_study(config)["completed_experiments"] == result["completed_experiments"]

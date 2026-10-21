"""Participation is a subset choice with matched-date, full-profit evaluation."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.participation import auction_subset, participation_profiles, profile_id
from bess_pv_trading.simulation.participation_study import summarize_participation_days


def cases():
    return load_config(Path(__file__).resolve().parents[1] / "fixtures/configs/participation_cases.yaml")


def test_profile_enumeration_supports_all_subsets_required_da_and_explicit_choices():
    case = cases()
    profiles = participation_profiles({"candidate_auctions": case["candidates"]})
    assert len(profiles) == case["expected_nonempty_profiles"]
    required = participation_profiles({"candidate_auctions": case["candidates"], "required_auctions": case["required_da"]})
    assert len(required) == case["expected_da_required_profiles"]
    assert all(set(case["required_da"]).issubset(profile) for profile in required)
    explicit = participation_profiles({"profiles": case["selected_profiles"]})
    assert explicit == tuple(tuple(row) for row in case["selected_profiles"])
    for invalid in case["invalid_selections"]:
        with pytest.raises(ValueError):
            auction_subset(invalid, "participation")
    for invalid in case["invalid_profile_settings"]:
        with pytest.raises(ValueError):
            participation_profiles(invalid)


def test_ranking_uses_equal_days_and_prices_retains_inventory_and_does_not_select_daily_winners():
    case = cases()["ranking"]
    daily = pd.DataFrame([{**case["metrics"], **row} for row in case["daily"]])
    daily, ranking = summarize_participation_days(daily, case["profiles"], case["delivery_days"], case["profit_tolerance_eur"])
    assert ranking.profile.tolist() == case["expected_order"]
    np.testing.assert_allclose(ranking.mean_net_profit_eur, case["expected_means_eur"])
    np.testing.assert_allclose(ranking.mean_profit_minus_reference_eur, case["expected_paired_differences_eur"])
    assert ranking.daily_wins_including_ties.tolist() == case["expected_daily_wins"]
    assert daily.is_daily_best_within_tolerance.sum() == len(case["delivery_days"])
    with pytest.raises(ValueError, match="identical requested delivery day"):
        summarize_participation_days(daily.iloc[:-1], case["profiles"], case["delivery_days"], case["profit_tolerance_eur"])
    daily["net_profit_eur"] = np.nan
    with pytest.raises(ValueError, match="fully priced"):
        summarize_participation_days(daily, case["profiles"], case["delivery_days"], case["profit_tolerance_eur"])


def test_batch_keeps_common_forecast_initialization_and_full_day_inputs(tmp_path, monkeypatch):
    from bess_pv_trading.simulation import participation_study
    from bess_pv_trading.simulation.synthetic import write_json
    case = cases()["ranking"]
    config = load_config("configs/full_experiments/gme_participation_2025-01-01_03.yaml")
    config["participation_study"].update(output_root=str(tmp_path), run_id="batch", plots=False,
        perfect_foresight_reference=False, delivery_days=case["delivery_days"], profiles=case["profiles"])
    seen = []
    def run(child):
        profile = profile_id(child["gme_oracle_experiment"]["participating_auctions"])
        day = child["gme_oracle_experiment"]["delivery_day"]
        seen.append((day, child["forecast"]["initialization_cutoff"], child["state"]["initial_soc_percent"]))
        row = next(row for row in case["daily"] if row["profile"] == profile and row["study_date"] == day)
        output = Path(child["experiment_outputs"]["root"]) / profile / day
        output.mkdir(parents=True)
        metrics = case["metrics"]
        write_json(output / "json/kpis.json", {"total_gain_eur": row["net_profit_eur"],
            "auction_revenue_eur": row["net_profit_eur"], "degradation_cost_eur": metrics["degradation_eur"],
            **{key: metrics[key] for key in ("transaction_fee_eur", "curtailment_mwh", "absolute_imbalance_mwh", "final_energy_mwh", "final_soc_percent")},
            "auction_decision_solves": len(profile.split("+")), "active_auctions": profile.split("+"), "skipped_auctions": []})
        write_json(output / "json/manifest.json", {"status": "complete"})
        (output / "csv").mkdir()
        ids = list(pd.date_range(day, periods=len(case["realized_pv_mw"]),
            freq=f"{child['gme_oracle_experiment']['decision_resolution_minutes']}min", tz="UTC").astype(str))
        pd.DataFrame({"interval_id": ids, "pv_realized_execution_ac_mw": case["realized_pv_mw"],
            "pv_initial_forecast_ac_mw": case["realized_pv_mw"]}).to_csv(output / "csv/pv_forecast_and_realized.csv", index=False)
        pd.DataFrame({"interval_id": ids, **{auction: case["realized_pv_mw"] for auction in profile.split("+")}}).to_csv(
            output / "csv/auction_prices_expanded.csv", index=False)
        return {"output_dir": str(output)}
    monkeypatch.setattr(participation_study, "run_gme_oracle_experiment", run)
    result = participation_study.run_participation_study(config)
    assert result["best_mean_profile"] == case["expected_order"][0]
    assert result["completed_experiments"] == len(case["profiles"]) * len(case["delivery_days"])
    assert all(len({cutoff for recorded_day, cutoff, _ in seen if recorded_day == day}) == 1 for day in case["delivery_days"])
    assert len({soc for _, _, soc in seen}) == 1
    config["participation_study"]["resume"] = True
    monkeypatch.setattr(participation_study, "run_gme_oracle_experiment",
        lambda child: pytest.fail("A verified completed checkpoint must not rerun optimizations"))
    resumed = participation_study.run_participation_study(config)
    assert resumed["completed_experiments"] == result["completed_experiments"]
    different = deepcopy(config)
    different["state"]["initial_soc_percent"] = None
    with pytest.raises(ValueError, match="different configuration"):
        participation_study.run_participation_study(different)
    config["participation_study"]["run_id"] = "failed_batch"
    calls = []
    def fail_after_one(child):
        if calls:
            raise ValueError("missing study input")
        calls.append(child)
        return run(child)
    monkeypatch.setattr(participation_study, "run_gme_oracle_experiment", fail_after_one)
    with pytest.raises(ValueError, match="missing study input"):
        participation_study.run_participation_study(config)
    manifest = participation_study.json_load(tmp_path / "failed_batch/json/manifest.json")
    assert manifest["status"] == "failed" and not manifest["ranking_published"]
    assert not (tmp_path / "failed_batch/csv/profile_ranking.csv").exists()


@pytest.mark.parametrize("selection", cases()["selected_profiles"])
def test_market_reader_only_requests_selected_auction_prices(monkeypatch, selection):
    from bess_pv_trading.markets import AuctionCalendar
    from bess_pv_trading.simulation import gme_oracle
    config = load_config("configs/full_experiments/gme_oracle_2025-01-01.yaml")
    config["gme_oracle_experiment"]["participating_auctions"] = selection
    datasets = config["gme_oracle_experiment"]["auction_datasets"]
    allowed = {datasets[auction]["dataset"] for auction in selection}
    calendar = AuctionCalendar.from_config(config)
    day = config["gme_oracle_experiment"]["delivery_day"]
    grid = calendar.delivery_grid(day).to_domain()
    pattern = load_config("tests/fixtures/configs/gme_workbook_cases.yaml")["prices_eur_mwh"]
    def read(database, *, dataset, **kwargs):
        assert dataset in allowed
        auction = next(name for name, item in datasets.items() if item["dataset"] == dataset)
        event = next(item for item in calendar.schedule(day) if item.auction == auction)
        minutes = datasets[auction]["native_interval_minutes"]
        start = grid.starts[event.eligible_intervals[0]]
        times = pd.date_range(start, grid.ends[-1], freq=f"{minutes}min", inclusive="left")
        return pd.DataFrame({"product_start": times, "product_end": times + pd.Timedelta(minutes=minutes),
            "price_eur_mwh": np.resize(pattern, len(times)), "native_interval_minutes": minutes}), []
    monkeypatch.setattr(gme_oracle, "load_api_cache_records", read)
    prices, _, skipped = gme_oracle._load_auction_prices(config, calendar, grid, calendar.schedule(day))
    assert set(prices) == set(selection)
    assert {row["auction"] for row in skipped if row["reason"] == "not_participating"} == set(cases()["candidates"]) - set(selection)


def test_perfect_foresight_reuses_all_auction_objectives_with_cancellations_and_checks_realized_pv(tmp_path, monkeypatch):
    from bess_pv_trading.simulation import participation_study
    from bess_pv_trading.simulation.synthetic import write_json
    fixture = cases()
    case = fixture["perfect_foresight"]
    config = load_config("configs/full_experiments/gme_participation_2025-01-01_03.yaml")
    days = fixture["ranking"]["delivery_days"]
    config["participation_study"]["delivery_days"] = days
    initial = config["battery"]["capacity_mwh"] * config["state"]["initial_soc_percent"] / 100
    pv = fixture["ranking"]["realized_pv_mw"]
    experiments = []
    for index, day in enumerate(days):
        output = tmp_path / "runs" / day
        (output / "csv").mkdir(parents=True)
        identifiers = pd.date_range(day, periods=len(pv), freq="15min", tz="UTC").astype(str).tolist()
        pd.DataFrame({"interval_id": identifiers, "pv_realized_execution_ac_mw": pv}).to_csv(
            output / "csv/pv_forecast_and_realized.csv", index=False)
        pd.DataFrame({"interval_id": identifiers, "pv_ac_mw": pv, "soc_end_mwh": initial}).to_csv(
            output / "csv/perfect_pv_counterfactual_schedule.csv", index=False)
        write_json(output / "json/kpis.json", {"active_auctions": case["active_auctions_by_day"][index]})
        write_json(output / "json/perfect_pv_counterfactual.json", {
            "objective_eur": case["objectives_eur"][index], "validation": {"valid": True},
            "active_auctions": case["active_auctions_by_day"][index],
            "status": case["model_status"], "mip_gap": case["model_mip_gap"]})
        child = deepcopy(config)
        child["gme_oracle_experiment"]["delivery_day"] = day
        write_json(output / "json/study_config.json", child)
        experiments.append({"profile": profile_id(fixture["candidates"]), "study_date": day, "output_dir": str(output)})
    monkeypatch.setattr(participation_study, "_solve_all_auction_reference",
        lambda *args: pytest.fail("Existing all-auction reference should avoid additional solves"))
    (tmp_path / "csv").mkdir()
    (tmp_path / "json").mkdir()
    daily, summary = participation_study.collect_perfect_foresight_reference(config, experiments, tmp_path)
    assert daily.active_auctions.tolist() == case["active_auctions_by_day"]
    assert summary["mean_objective_eur"] == pytest.approx(case["expected_mean_objective_eur"])
    assert summary["std_daily_objective_eur"] == pytest.approx(case["expected_std_objective_eur"])
    assert summary["additional_optimizer_solves"] == 0
    assert summary["mean_final_soc_percent"] == config["state"]["initial_soc_percent"]
    changed = pd.read_csv(Path(experiments[0]["output_dir"]) / "csv/perfect_pv_counterfactual_schedule.csv")
    changed["pv_ac_mw"] = list(reversed(pv))
    changed.to_csv(Path(experiments[0]["output_dir"]) / "csv/perfect_pv_counterfactual_schedule.csv", index=False)
    with pytest.raises(ValueError, match="identical effective realized PV"):
        participation_study.collect_perfect_foresight_reference(config, experiments, tmp_path)
    changed["pv_ac_mw"] = pv
    changed.to_csv(Path(experiments[0]["output_dir"]) / "csv/perfect_pv_counterfactual_schedule.csv", index=False)
    calls = []
    def supply_missing_reference(child, source, destination):
        calls.append(child["gme_oracle_experiment"]["delivery_day"])
        return source
    monkeypatch.setattr(participation_study, "_solve_all_auction_reference", supply_missing_reference)
    restricted = [{**row, "profile": profile_id(fixture["selected_profiles"][0])} for row in experiments]
    _, summary = participation_study.collect_perfect_foresight_reference(config, restricted, tmp_path)
    assert calls == days
    assert summary["additional_optimizer_solves"] == len(days)
    assert summary["mean_objective_eur"] == pytest.approx(case["expected_mean_objective_eur"])


def test_missing_full_profile_reference_solves_all_available_auctions_on_same_physical_inputs(tmp_path, monkeypatch):
    from bess_pv_trading.markets import AuctionCalendar
    from bess_pv_trading.simulation import gme_oracle, participation_study
    fixture = cases()
    case = fixture["perfect_foresight"]
    config = load_config("configs/full_experiments/gme_oracle_2025-01-01.yaml")
    config["gme_oracle_experiment"]["participating_auctions"] = fixture["selected_profiles"][0]
    config["experiment_outputs"]["save_model_files"] = False
    calendar = AuctionCalendar.from_config(config)
    grid = calendar.delivery_grid(config["gme_oracle_experiment"]["delivery_day"]).to_domain()
    pv = np.resize(fixture["ranking"]["realized_pv_mw"], len(grid.interval_ids))
    source = tmp_path / "source"
    (source / "csv").mkdir(parents=True)
    pd.DataFrame({"interval_id": grid.interval_ids, "pv_realized_execution_ac_mw": pv}).to_csv(
        source / "csv/pv_forecast_and_realized.csv", index=False)
    available = case["active_auctions_by_day"][-1]
    skipped = [{"auction": name, "reason": "cancelled"} for name in fixture["candidates"] if name not in available]
    def prices(reference, *args):
        assert reference["gme_oracle_experiment"]["participating_auctions"] == fixture["candidates"]
        return {name: pv for name in available}, {}, skipped
    monkeypatch.setattr(gme_oracle, "_load_auction_prices", prices)
    def solve(reference, actual_grid, plant, schedule, prices, effective_pv, initial, destination):
        assert set(prices) == {event.auction for event in schedule} == set(available)
        np.testing.assert_array_equal(effective_pv, pv)
        assert plant.battery.capacity_mwh == config["battery"]["capacity_mwh"]
        assert initial == config["battery"]["capacity_mwh"] * config["state"]["initial_soc_percent"] / 100
        assert reference["state"]["terminal_target_mwh"] == config["battery"]["capacity_mwh"] * config["state"]["terminal_soc_percent"] / 100
        return pd.DataFrame({"interval_id": actual_grid.interval_ids, "pv_ac_mw": pv, "soc_end_mwh": initial}), {
            "objective_eur": case["objectives_eur"][-1], "validation": {"valid": True},
            "status": case["model_status"], "mip_gap": case["model_mip_gap"]}
    monkeypatch.setattr(gme_oracle, "_solve_perfect_information_schedule", solve)
    output = participation_study._solve_all_auction_reference(config, source, tmp_path / "reference")
    saved = participation_study.json_load(output / "json/perfect_pv_counterfactual.json")
    assert saved["active_auctions"] == available
    assert saved["skipped_auctions"] == skipped
    assert config["gme_oracle_experiment"]["participating_auctions"] == fixture["selected_profiles"][0]


def test_fully_cancelled_selected_profile_keeps_the_day_but_missing_data_still_fail(monkeypatch):
    from bess_pv_trading.markets import AuctionCalendar
    from bess_pv_trading.simulation import gme_oracle
    fixture = cases()
    config = load_config("configs/full_experiments/gme_oracle_2025-01-01.yaml")
    day = fixture["cancelled_auction_day"]
    selection = fixture["selected_profiles"][1]
    config["market"].update(valid_from=day, valid_to=day)
    config["gme_oracle_experiment"].update(delivery_day=day, participating_auctions=selection)
    calendar = AuctionCalendar.from_config(config)
    grid = calendar.delivery_grid(day).to_domain()
    monkeypatch.setattr(gme_oracle, "load_api_cache_records", lambda *args, **kwargs:
        (pd.DataFrame(), [{"cancelled_delivery_days": [day]}]))
    prices, _, skipped = gme_oracle._load_auction_prices(config, calendar, grid, calendar.schedule(day))
    assert not prices
    assert {row["auction"] for row in skipped if row["reason"] == "documented_cancelled_auction"} == set(selection)
    monkeypatch.setattr(gme_oracle, "load_api_cache_records", lambda *args, **kwargs: (pd.DataFrame(), []))
    with pytest.raises(ValueError, match="missing data do not establish auction cancellation"):
        gme_oracle._load_auction_prices(config, calendar, grid, calendar.schedule(day))


def test_no_active_auction_day_executes_zero_position_without_optimizer_or_invented_expected_objective(tmp_path, monkeypatch):
    from bess_pv_trading.simulation import synthetic
    from bess_pv_trading.simulation.settlement import expected_realized_kpis
    config = load_config("configs/full_experiments/synthetic_small.yaml")
    config["outputs"]["plots"] = False
    config["execution"] = {"reoptimize_physical_controls": False, "dispatch_policy": "track_committed_grid"}
    fixture = synthetic.synthetic_inputs(config)
    fixture.update(schedule=(), actual_prices={})
    fixture["forecast_available_at"] = pd.Timestamp(fixture["grid"].starts[0]) - pd.Timedelta(hours=1)
    monkeypatch.setattr(synthetic, "solve_model", lambda *args, **kwargs: pytest.fail("A cancelled profile must not insert an auction/control solve"))
    summary = synthetic._replay_strategy(config, fixture, "oracle", tmp_path / "cancelled")
    dispatch = pd.read_csv(tmp_path / "cancelled/dispatch.csv")
    ledger = pd.read_csv(tmp_path / "cancelled/ledger.csv")
    plans = pd.read_csv(tmp_path / "cancelled/expected_dispatch_plans.csv")
    np.testing.assert_allclose(dispatch.grid_mw, 0, atol=1e-9)
    np.testing.assert_allclose(dispatch.net_position_mwh, 0, atol=1e-9)
    assert ledger.empty and plans.empty
    assert summary["auction_decision_solves"] == summary["physical_control_solves"] == 0
    assert summary["initial_expected_profit_eur"] is None and summary["first_solve"] is None
    assert summary["expectation_baseline"] == "no_auction_decision"
    battery = fixture["plant"].battery
    assert dispatch.energy_end_mwh.between(battery.energy_min_mwh-1e-9, battery.maximum_energy+1e-9).all()
    kpis = expected_realized_kpis(initial_expected_profit_eur=None, initial_expected_auction_revenue_eur=0,
        realized_profit_after_imbalance_eur=summary["profit_before_imbalance_settlement_eur"], auction_revenue_eur=0)
    assert kpis["realized_minus_expected_profit_eur"] is None

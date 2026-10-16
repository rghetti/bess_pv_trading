"""Synthetic canonical forecast inputs exercise causality and MILP contracts."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.gaussian_prices import calibrate_gaussian_prices
from bess_pv_trading.forecasting.intraday_price_features import ConditionalSpreadPriceForecast
from bess_pv_trading.forecasting.prices import RidgePriceModel
from bess_pv_trading.markets import AuctionCalendar
from bess_pv_trading.optimization.forecast_inputs import build_forecast_inputs, load_forecast_inputs
from bess_pv_trading.optimization.prepared import PreparedInputError


def prepared_case(auctions=None):
    """Two hypothetical weather members with an initially indistinguishable PV history."""
    day = "2025-10-02"
    calendar = AuctionCalendar()
    grid = calendar.delivery_grid(day)
    cutoff = calendar.schedule(day)[0].submission_at
    source = cutoff - pd.Timedelta(hours=8)
    available = source + pd.Timedelta(hours=4)
    config = {"optimization": {"forecast_inputs": {"delivery_day": day, "decision_cutoff": cutoff.isoformat(),
        "price_distribution": "point", "auctions": auctions or ["DA", "IDA1", "IDA2", "IDA3"],
        "allow_retrospective": True, "pv_model_fit_status": "unfitted_explicit_assumption",
        "settlement_point_price_eur_per_mwh": 30., "settlement_point_price_available_at": source.isoformat(),
        "settlement_point_price_status": "study_assumption", "branches": 2, "seed": 13}},
        "plant": {"farm_id": 1, "poi_import_mw": 1., "poi_export_mw": 1.},
        "battery": {"capacity_mwh": 1., "max_charge_mw": .5, "max_discharge_mw": .5},
        "state": {"timestamp": grid.starts[0].isoformat(), "initial_energy_mwh": .5,
                  "terminal_target_mwh": .5, "origin": "known_delivery_start_assumption"},
        "trading": {"trade_limit_mwh": .25, "position_limit_mwh": .25,
                    "imbalance_penalty_eur_per_mwh": 100., "grid_import_fee_eur_per_mwh": 0.},
        "degradation": {"model": "throughput", "cost_eur_per_mwh_throughput": 5.,
                        "assumptions": "explicit synthetic fixture coefficient"}}
    frames = []
    for member, probability, level in (("weather-a", .3, .2), ("weather-b", .7, .6)):
        values = np.full(len(grid), level)
        values[:4] = 0
        frames.append(pd.DataFrame({"farm_id": 1, "member_id": member, "member_origin": "synthetic_weather_forecast",
            "member_probability": probability, "forecast_cycle_time": source,
            "forecast_available_at": available, "interval_start_utc": grid.starts,
            "interval_end_utc": grid.ends, "pv_ac_mw": values, "input_eligibility": True,
            "review_status": "eligible", "review_excluded": False}))
    specs = {}
    for index, event in enumerate(calendar.schedule(day)):
        if event.auction not in config["optimization"]["forecast_inputs"]["auctions"]:
            continue
        rows = list(event.eligible_intervals)
        mean = pd.DataFrame({"product_start": grid.starts[rows], "product_end": grid.ends[rows],
            "mean_price_eur_mwh": -5. + 10 * index, "source_time": source, "available_at": available,
            "fit_cutoff": source, "origin": "supplied_synthetic_linear_forecast", "zone": "NORD",
            "historical_feature_eligible": False})
        specs[event.auction] = {"native_interval_minutes": 15, "mean_curve": mean}
    return config, pd.concat(frames, ignore_index=True), specs


def write_prepared_case(directory, *, gaussian=False):
    """Export explicitly synthetic canonical files for an offline CLI smoke test.

    Point mode includes all four auctions and two PV members. Gaussian mode is
    a small DA-only four-leaf fixture; it never claims actual market performance.
    The returned configuration can be written with yaml.safe_dump by a caller.
    """
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config, pv, specs = prepared_case(["DA"] if gaussian else None)
    settings = config["optimization"]["forecast_inputs"]
    pv.to_parquet(directory / "synthetic_pv_members.parquet", index=False)
    settings["pv_forecast_path"] = str((directory / "synthetic_pv_members.parquet").resolve())
    settings["auction_inputs"] = {}
    for auction, value in specs.items():
        path = directory / f"synthetic_{auction}_means.parquet"
        value["mean_curve"].to_parquet(path, index=False)
        settings["auction_inputs"][auction] = {"native_interval_minutes": 15, "mean_curve_path": str(path.resolve())}
        if gaussian:
            calibration = gaussian_fixture(auction, pd.Timestamp(settings["decision_cutoff"]))
            path = directory / f"synthetic_{auction}_gaussian.json"
            calibration.save(path)
            settings["auction_inputs"][auction]["calibration_path"] = str(path.resolve())
    if gaussian:
        settings["price_distribution"] = "gaussian"
    config["outputs"] = {"root": str((directory / "optimization").resolve()), "plots": False}
    config["solver"] = {"threads": 1, "time_limit_seconds": 30., "mip_gap": 1e-4, "seed": 13, "output_flag": False}
    return config


def gaussian_fixture(auction, cutoff):
    """Two complete synthetic calibration days, never actual historical data."""
    frames = []
    for index, day in enumerate(("2025-09-20", "2025-09-21", "2025-09-22")):
        start = pd.Timestamp(day, tz="Europe/Rome")
        starts = pd.date_range(start, start + pd.Timedelta(days=1), freq="15min", inclusive="left").tz_convert("UTC")
        if auction == "IDA3":
            starts = starts[starts.tz_convert("Europe/Rome").hour >= 12]
        frames.append(pd.DataFrame({"auction": auction, "delivery_day": day, "product_start": starts,
            "product_end": starts + pd.Timedelta(minutes=15), "actual": 80 + index + np.sin(np.arange(len(starts))),
            "prediction": 80., "fit_cutoff": start - pd.Timedelta(days=3),
            "forecast_cutoff": start - pd.Timedelta(days=1),
            "training_max_delivery_day": (start - pd.Timedelta(days=4)).date().isoformat(),
            "prediction_provenance": "chronological_out_of_sample", "native_interval_minutes": 15}))
    return calibrate_gaussian_prices(pd.concat(frames), calibration_cutoff=cutoff - pd.Timedelta(days=1),
        auction=auction, minimum_complete_days=2)


def test_joint_members_keep_weights_and_battery_shared_until_completed_pv_observation():
    config, pv, specs = prepared_case()
    result = build_forecast_inputs(config, pv, specs)
    assert result.scenarios.probabilities.tolist() == [.3, .7]
    assert result.scenarios.pv_ac_mw.shape == (2, 96)
    assert np.all(result.scenarios.auction_prices_eur_per_mwh["DA"] == -5)
    assert result.tree.control_nodes[0][4] == result.tree.control_nodes[1][4]
    assert result.tree.control_nodes[0][5] != result.tree.control_nodes[1][5]
    assert result.tree.curtailment_nodes[0][4] != result.tree.curtailment_nodes[1][4]
    assert len(set(result.tree.bid_nodes["DA"])) == 1
    assert sum(result.auction_eligibility["IDA3"]) == 48
    assert np.all(result.scenarios.auction_prices_eur_per_mwh["IDA3"][:, :48] == 0)
    root = next(node for node in result.tree.nodes if node.parent_id is None)
    assert not root.observations
    assert result.scenarios.provenance["price_paths_per_pv_member"] == 1
    assert result.scenarios.provenance["operational_eligible"] is False


def test_delayed_pv_telemetry_cannot_reveal_member_to_earlier_controls_or_curtailment():
    config, pv, specs = prepared_case(["DA"])
    config["optimization"]["forecast_inputs"]["pv_observation_delay_minutes"] = 30
    result = build_forecast_inputs(config, pv, specs)
    assert result.tree.control_nodes[0][6] == result.tree.control_nodes[1][6]
    assert result.tree.control_nodes[0][7] != result.tree.control_nodes[1][7]
    assert result.tree.curtailment_nodes[0][4] == result.tree.curtailment_nodes[1][4]


def test_gaussian_paths_cross_pv_weights_without_revealing_price_draws_before_results():
    config, pv, specs = prepared_case(["DA"])
    settings = config["optimization"]["forecast_inputs"]
    settings["price_distribution"] = "gaussian"
    specs["DA"]["calibration"] = gaussian_fixture("DA", pd.Timestamp(settings["decision_cutoff"]))
    result = build_forecast_inputs(config, pv, specs)
    assert result.scenarios.probabilities.tolist() == [.15, .15, .35, .35]
    assert len(set(result.tree.bid_nodes["DA"])) == 1
    prices = result.scenarios.auction_prices_eur_per_mwh["DA"]
    assert np.array_equal(prices[:2], prices[2:])
    assert not np.array_equal(prices[0], prices[1])
    assert result.scenarios.provenance["price_paths_per_pv_member"] == 2
    assert np.array_equal(result.scenarios.pv_ac_mw[0], result.scenarios.pv_ac_mw[1])


@pytest.mark.parametrize("field", ["source_time", "available_at", "fit_cutoff"])
def test_future_price_provenance_is_rejected(field):
    config, pv, specs = prepared_case(["DA"])
    specs["DA"]["mean_curve"].loc[0, field] = pd.Timestamp("2025-10-02T08:00Z")
    with pytest.raises(PreparedInputError, match="Future/unavailable|unavailable"):
        build_forecast_inputs(config, pv, specs)


@pytest.mark.parametrize("change,match", [({"input_eligibility": False}, "eligibility"),
    ({"review_excluded": True}, "review exclusions"), ({"pv_ac_mw": np.nan}, "finite"),
    ({"member_probability": .5}, "constant probability"),
    ({"forecast_available_at": "2025-10-02T08:00Z"}, "Future/unavailable")])
def test_bad_pv_or_future_weather_cannot_enter_optimizer(change, match):
    config, pv, specs = prepared_case(["DA"])
    for name, value in change.items():
        pv.loc[0, name] = value
    with pytest.raises(PreparedInputError, match=match):
        build_forecast_inputs(config, pv, specs)


def test_missing_products_and_fabricated_morning_ida3_are_rejected():
    config, pv, specs = prepared_case()
    missing = deepcopy(specs)
    missing["IDA1"]["mean_curve"] = missing["IDA1"]["mean_curve"].iloc[1:]
    with pytest.raises(PreparedInputError, match="cover every eligible"):
        build_forecast_inputs(config, pv, missing)
    invented = specs["DA"]["mean_curve"].iloc[:1].copy()
    specs["IDA3"]["mean_curve"] = pd.concat([invented, specs["IDA3"]["mean_curve"]])
    with pytest.raises(PreparedInputError, match="outside its eligible"):
        build_forecast_inputs(config, pv, specs)


def known_results_for(config, specs, auctions):
    """Known prices retain recorded publication clocks and native eligibility."""
    calendar = AuctionCalendar.from_config(config)
    result = []
    for event in calendar.schedule(config["optimization"]["forecast_inputs"]["delivery_day"]):
        if event.auction in auctions:
            rows = specs[event.auction]["mean_curve"].copy().rename(columns={"mean_price_eur_mwh": "price_eur_mwh"})
            rows["auction"], rows["published_at"] = event.auction, event.results_at
            result.append(rows)
    return pd.concat(result, ignore_index=True)


def test_residual_reoptimization_fixes_known_results_and_carries_accepted_positions():
    config, pv, specs = prepared_case()
    calendar = AuctionCalendar()
    day = config["optimization"]["forecast_inputs"]["delivery_day"]
    known = known_results_for(config, specs, ["DA", "IDA1", "IDA2"])
    for auction in ("DA", "IDA1", "IDA2"):
        del specs[auction]
    cutoff = pd.Timestamp("2025-10-02T08:00+02:00")
    config["optimization"]["forecast_inputs"].update({"decision_cutoff": cutoff.isoformat(), "residual_start": cutoff.isoformat()})
    config["state"].update({"timestamp": cutoff.isoformat(), "origin": "carried_executor_state", "initial_energy_mwh": .4,
                            "cumulative_throughput_mwh": .1, "cumulative_degradation_eur": .5})
    interval = calendar.delivery_grid(day).product_ids[32]
    da_event = calendar.schedule(day)[0]
    trades = pd.DataFrame([{"auction": "DA", "submission_time": da_event.submission_at,
        "publication_time": da_event.results_at, "interval_id": interval, "quantity_mwh": .1,
        "price_eur_per_mwh": -5., "trade_id": "known-da-1"}])
    result = build_forecast_inputs(config, pv, specs, known_results=known, accepted_trades=trades)
    assert len(result.grid.starts) == 64 and result.grid.interval_ids[0] == interval
    assert set(result.tree.bid_nodes) == {"IDA3"}
    assert result.market.positions(result.grid)[0] == .1
    assert result.state.cumulative_throughput_mwh == .1
    assert result.state.cumulative_degradation_eur == .5
    assert np.all(result.scenarios.auction_prices_eur_per_mwh["DA"] == -5)
    assert sum(result.auction_eligibility["IDA3"]) == 48


def test_future_known_actual_results_and_missing_published_curves_are_rejected():
    config, pv, specs = prepared_case()
    known = known_results_for(config, specs, ["DA"])
    del specs["DA"]
    with pytest.raises(PreparedInputError, match="Future/unavailable"):
        build_forecast_inputs(config, pv, specs, known_results=known)
    config["optimization"]["forecast_inputs"]["decision_cutoff"] = "2025-10-01T14:00+02:00"
    with pytest.raises(PreparedInputError, match="Previously published DA"):
        build_forecast_inputs(config, pv, specs)


def test_intraday_spread_rebuilds_latent_ancestor_inside_each_gaussian_branch():
    config, pv, specs = prepared_case(["DA", "IDA1"])
    settings = config["optimization"]["forecast_inputs"]
    settings["price_distribution"] = "gaussian"
    cutoff = pd.Timestamp(settings["decision_cutoff"])
    for auction in specs:
        specs[auction]["calibration"] = gaussian_fixture(auction, cutoff)
    frame = specs["IDA1"].pop("mean_curve")
    frame = frame.drop(columns=["mean_price_eur_mwh", "fit_cutoff", "available_at"])
    frame["static_feature_available_at"] = frame.source_time + pd.Timedelta(hours=4)
    frame["forecast_cutoff"] = "2025-10-01T14:55+02:00"
    frame["previous_auction_price"] = -999.
    model = RidgePriceModel(("previous_auction_price",), np.zeros(1), np.ones(1), np.array([.1]),
                           3., 1., frame.source_time.iloc[0].isoformat(), 100, "IDA1_spread")
    specs["IDA1"].update({"features": frame, "spread_model": ConditionalSpreadPriceForecast("IDA1", model, "2025-09-29")})
    result = build_forecast_inputs(config, pv, specs)
    da = result.scenarios.auction_prices_eur_per_mwh["DA"]
    ida = result.scenarios.auction_prices_eur_per_mwh["IDA1"]
    assert len(np.unique(da[:, 0])) == 2 and len(np.unique(ida[:, 0])) == 4
    assert result.scenarios.provenance["forecast_input_audit"]["IDA1"]["future_ancestor_policy"] == "rebuilt from branch history"
    bad = deepcopy(specs)
    bad["IDA1"]["features"]["static_feature_available_at"] = cutoff + pd.Timedelta(hours=1)
    with pytest.raises(PreparedInputError, match="Future/unavailable"):
        build_forecast_inputs(config, pv, bad)


def test_file_loader_keeps_source_hashes_and_uses_no_provider_catalog(tmp_path, monkeypatch):
    config, pv, specs = prepared_case(["DA"])
    settings = config["optimization"]["forecast_inputs"]
    pv.to_parquet(tmp_path / "pv.parquet", index=False)
    specs["DA"]["mean_curve"].to_csv(tmp_path / "mean.csv", index=False)
    settings.update({"pv_forecast_path": str(tmp_path / "pv.parquet"),
                     "auction_inputs": {"DA": {"native_interval_minutes": 15, "mean_curve_path": str(tmp_path / "mean.csv")}}})
    from bess_pv_trading.data import api_catalog
    monkeypatch.setattr(api_catalog, "load_api_cache_records", lambda *args, **kwargs: pytest.fail("No provider cache reads at prepared boundary"))
    result = load_forecast_inputs(config)
    provenance = result.scenarios.provenance["prepared_file_sha256"]
    assert set(provenance) == {"pv_forecast", "DA_mean_curve"}
    assert all(len(value["sha256"]) == 64 for value in provenance.values())


def test_unverified_sources_require_opt_in_and_tree_size_is_bounded():
    config, pv, specs = prepared_case(["DA"])
    config["optimization"]["forecast_inputs"]["allow_retrospective"] = False
    with pytest.raises(PreparedInputError, match="Unverified historical price features"):
        build_forecast_inputs(config, pv, specs)
    config["optimization"]["forecast_inputs"].update({"allow_retrospective": True, "maximum_joint_scenarios": 1})
    with pytest.raises(PreparedInputError, match="Joint tree could contain"):
        build_forecast_inputs(config, pv, specs)


def test_post_submission_reoptimization_requires_explicit_pending_positions():
    config, pv, specs = prepared_case(["DA"])
    settings = config["optimization"]["forecast_inputs"]
    settings["decision_cutoff"] = "2025-10-01T12:30+02:00"
    with pytest.raises(PreparedInputError, match="pending_position_mwh"):
        build_forecast_inputs(config, pv, specs)
    settings["pending_position_mwh"] = [0.] * 96
    result = build_forecast_inputs(config, pv, specs)
    assert not result.tree.bid_nodes and np.array_equal(result.pending_position_mwh, np.zeros(96))


def test_wrong_auction_scope_and_empty_member_ids_are_rejected():
    config, pv, specs = prepared_case(["DA"])
    specs["DA"]["mean_curve"]["auction"] = "IDA1"
    with pytest.raises(PreparedInputError, match="auction column"):
        build_forecast_inputs(config, pv, specs)
    specs["DA"]["mean_curve"]["auction"] = "DA"
    pv.loc[pv.member_id.eq("weather-a"), "member_id"] = " "
    with pytest.raises(PreparedInputError, match="member ID"):
        build_forecast_inputs(config, pv, specs)


def test_fixture_file_loader_produces_complete_four_auction_plan(tmp_path):
    config = write_prepared_case(tmp_path)
    result = load_forecast_inputs(config)
    assert set(result.tree.bid_nodes) == {"DA", "IDA1", "IDA2", "IDA3"}
    assert len(result.scenarios.scenario_ids) == 2
    assert all("synthetic" in origin["origin"] for origin in result.scenarios.provenance["pv_vintages"])


@pytest.mark.solver
def test_prepared_joint_forecasts_solve_actual_milp_and_independent_accounting(tmp_path):
    """A licensed solve validates the new boundary against the physical optimizer."""
    gurobi = pytest.importorskip("gurobipy")
    try:
        with gurobi.Model() as check:
            check.Params.OutputFlag = 0
    except gurobi.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")
    from bess_pv_trading.domain import SolverSettings
    from bess_pv_trading.optimization import build_model, solve_model, validate_solution
    config = write_prepared_case(tmp_path)
    inputs = load_forecast_inputs(config)
    built = build_model(inputs)
    result = solve_model(built, SolverSettings(**config["solver"]))
    assert result.status == "optimal"
    validation = validate_solution(inputs, result)
    assert validation.valid, validation.violations
    assert built.model.Params.MIPGap == pytest.approx(1e-4)
    assert set(auction for auction, _, _ in result.trades) == {"DA", "IDA1", "IDA2", "IDA3"}
    assert result.objective_eur == pytest.approx(float(np.dot(inputs.scenarios.probabilities, result.components["operating_profit"])))


@pytest.mark.solver
def test_cli_optimize_reads_canonical_files_and_publishes_result(tmp_path, capsys):
    """Exercise real CLI routing, file loader, licensed solve and output root."""
    import json
    import yaml
    gurobi = pytest.importorskip("gurobipy")
    try:
        with gurobi.Model() as check:
            check.Params.OutputFlag = 0
    except gurobi.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")
    from bess_pv_trading.cli import main
    config = write_prepared_case(tmp_path / "canonical")
    config["outputs"]["run_id"] = "synthetic_cli_fixture"
    config_path = tmp_path / "synthetic_forecast_inputs.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["optimize", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    result = json.loads(output[output.index("{"):])
    assert result["status"] == "optimal" and result["valid"]
    from pathlib import Path
    destination = Path(result["output_dir"])
    report = json.loads((destination / "result.json").read_text())
    assert (destination / "solution.npz").is_file()
    assert report["scenario_provenance"]["pv_members"] == 2
    assert "prepared_file_sha256" in report["scenario_provenance"]
    assert set(report["scenario_provenance"]["auction_eligible_product_indices"]) == {"DA", "IDA1", "IDA2", "IDA3"}

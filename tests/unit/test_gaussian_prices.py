"""Chronological Gaussian calibration, native DST grids and MILP tree contracts."""
from dataclasses import replace
from math import pi, sqrt

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.domain import DeliveryGrid
from bess_pv_trading.forecasting.gaussian_prices import (
    GaussianPriceCalibration, calibrate_gaussian_prices, evaluate_gaussian_prices,
    gaussian_price_tree,
)
from bess_pv_trading.markets import AuctionCalendar


def prediction_rows(days=None, *, minutes=60):
    """Create declared OOS native daily predictions with coherent residual vectors."""
    days = pd.date_range("2025-01-01", periods=12) if days is None else pd.to_datetime(days)
    random = np.random.default_rng(7)
    rows = []
    for day in days:
        local_start = pd.Timestamp(day.date()).tz_localize("Europe/Rome")
        local_end = pd.Timestamp(day.date() + pd.Timedelta(days=1)).tz_localize("Europe/Rome")
        starts = pd.date_range(local_start.tz_convert("UTC"), local_end.tz_convert("UTC"),
                               freq=f"{minutes}min", inclusive="left")
        factor = random.normal()
        cutoff = local_start - pd.Timedelta(hours=12)
        count = len(starts)
        prediction = 100 + np.sin(np.arange(count) / count * 2 * pi)
        error = 5 + factor * np.linspace(1, 3, count) + random.normal(0, .1, count)
        rows.append(pd.DataFrame({"delivery_day": day.date().isoformat(), "product_start": starts,
            "product_end": starts + pd.Timedelta(minutes=minutes), "actual": prediction + error,
            "prediction": prediction, "forecast_cutoff": cutoff,
            "fit_cutoff": cutoff - pd.Timedelta(hours=1),
            "training_max_delivery_day": (day.date() - pd.Timedelta(days=1)).isoformat(),
            "prediction_provenance": "chronological_out_of_sample", "native_interval_minutes": minutes}))
    return pd.concat(rows, ignore_index=True)


def fit_fixture(rows=None, **kwargs):
    """Fit only the early calibration slice, leaving future fixture days untouched."""
    rows = prediction_rows() if rows is None else rows
    return calibrate_gaussian_prices(rows, calibration_cutoff=rows.product_end.max(),
                                     minimum_complete_days=2, **kwargs)


def test_gaussian_bias_is_separate_from_zero_mean_innovations_and_seeded():
    rows = prediction_rows()
    calibration = fit_fixture(rows, covariance_shrinkage=.25)
    products = prediction_rows(["2025-02-01"])
    model = next(iter(calibration.grids.values()))
    expected_bias = (rows.actual - rows.prediction).to_numpy().reshape(12, 24).mean(axis=0)
    assert np.allclose(model.bias_eur_mwh, expected_bias)
    samples = calibration.sample_innovations(products, 20000, seed=42)
    assert np.allclose(samples.mean(axis=0), 0, atol=.05)
    assert np.corrcoef(samples[:, 0], samples[:, -1])[0, 1] > .6
    assert np.array_equal(calibration.sample_innovations(products, 5, seed=8),
                          calibration.sample_innovations(products, 5, seed=8))
    assert calibration.to_dict()["innovation_mean_eur_mwh"] == 0
    assert not calibration.operational_eligible


def test_diagonal_gaussian_is_a_configurable_dependence_baseline():
    calibration = fit_fixture(covariance_shrinkage=1)
    covariance = next(iter(calibration.grids.values())).covariance_eur2_mwh2
    assert np.array_equal(covariance, np.diag(np.diag(covariance)))
    correlated = next(iter(fit_fixture(covariance_shrinkage=0).grids.values()))
    assert np.linalg.eigvalsh(correlated.covariance_eur2_mwh2)[0] > 0


def test_paths_preserve_negative_prices_and_match_bias_plus_zero_mean_noise():
    calibration = fit_fixture()
    products = prediction_rows(["2025-02-01"])
    mean = np.full(24, -100.)
    paths = calibration.sample_future(mean, products, 3, seed=5, decision_cutoff="2025-01-31T10:00Z",
                                      require_operational=False)
    bias, _ = calibration.parameters_for(products)
    assert np.array_equal(paths, mean + bias + calibration.sample_innovations(products, 3, seed=5))
    assert (paths < 0).all()
    with pytest.raises(ValueError, match="operational"):
        calibration.sample_future(mean, products, 2, decision_cutoff="2025-01-31T10:00Z")


def test_artifact_round_trip_and_variance_units(tmp_path):
    calibration = fit_fixture()
    path = tmp_path / "gaussian.json"
    calibration.save(path)
    loaded = GaussianPriceCalibration.load(path)
    assert calibration.to_dict() == loaded.to_dict()
    assert "never raw price variance" in loaded.to_dict()["variance_source"]
    products = prediction_rows(["2025-02-01"])
    assert np.array_equal(calibration.sample_innovations(products, 4, seed=9),
                          loaded.sample_innovations(products, 4, seed=9))


def test_timestamp_resolution_does_not_change_complete_native_grid_detection():
    rows = prediction_rows()
    nanoseconds = rows.copy()
    microseconds = rows.copy()
    for column in ("product_start", "product_end", "fit_cutoff", "forecast_cutoff"):
        nanoseconds[column] = rows[column].dt.as_unit("ns")
        microseconds[column] = rows[column].dt.as_unit("us")
    from_ns = fit_fixture(nanoseconds)
    from_us = fit_fixture(microseconds)
    assert from_ns.to_dict() == from_us.to_dict()
    products = prediction_rows(["2025-02-01"])
    assert np.array_equal(from_ns.sample_innovations(products, 2, seed=6),
                          from_us.sample_innovations(products, 2, seed=6))


@pytest.mark.parametrize("column,value,match", [
    ("prediction_provenance", "in_sample", "out-of-sample"),
    ("training_max_delivery_day", "2025-02-01", "In-sample"),
    ("fit_cutoff", "2026-01-01T00:00Z", "fitting"),
    ("forecast_cutoff", "2026-01-01T00:00Z", "fitting"),
    ("native_interval_minutes", 15, "native interval"),
    ("delivery_day", "2025-02-01", "delivery_day"),
])
def test_bad_oos_evidence_or_native_metadata_is_rejected(column, value, match):
    rows = prediction_rows()
    rows[column] = value
    with pytest.raises(ValueError, match=match):
        fit_fixture(rows)


def test_naive_cutoffs_future_labels_and_duplicate_products_are_rejected():
    rows = prediction_rows()
    with pytest.raises(ValueError, match="timezone-aware"):
        calibrate_gaussian_prices(rows, calibration_cutoff="2025-01-20")
    with pytest.raises(ValueError, match="keep test rows separate"):
        calibrate_gaussian_prices(rows, calibration_cutoff="2025-01-05T00:00Z")
    with pytest.raises(ValueError, match="one out-of-sample"):
        fit_fixture(pd.concat([rows, rows.iloc[:1]]))
    naive = rows.copy()
    naive["fit_cutoff"] = naive.fit_cutoff.dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        fit_fixture(naive)


def test_missing_products_are_excluded_without_interpolating_prices():
    rows = prediction_rows()
    rows = rows.drop(index=[1])
    calibration = fit_fixture(rows)
    assert next(iter(calibration.grids.values())).calibration_days == 11
    assert calibration.excluded_days["2025-01-01"] == "incomplete or noncontiguous native civil day"
    with pytest.raises(ValueError, match="Insufficient complete"):
        fit_fixture(prediction_rows(["2025-01-01", "2025-01-02"]).drop(index=[1]))


def test_dst_and_native_duration_are_explicitly_distinct_grids():
    ordinary = prediction_rows(["2025-10-01", "2025-10-02"], minutes=15)
    fall = prediction_rows(["2024-10-27", "2025-10-26"], minutes=15)
    spring = prediction_rows(["2024-03-31", "2025-03-30"], minutes=15)
    calibration = fit_fixture(pd.concat([ordinary, fall, spring], ignore_index=True))
    assert set(calibration.grids) == {"15min_96products_ordinary", "15min_100products_repeated", "15min_92products_ordinary"}
    fall_model = calibration.grids["15min_100products_repeated"]
    assert "02:00/fold0" in fall_model.slot_labels and "02:00/fold1" in fall_model.slot_labels
    products = prediction_rows(["2026-10-25"], minutes=15)
    assert calibration.sample_innovations(products, 2, seed=0).shape == (2, 100)
    with pytest.raises(ValueError, match="No calibrated Gaussian grid"):
        calibration.sample_innovations(prediction_rows(["2026-10-25"], minutes=60), 2)
    with pytest.raises(ValueError, match="No calibrated Gaussian grid"):
        fit_fixture(ordinary).sample_innovations(products, 2)


def test_partial_reoptimization_selects_exact_covariance_submatrix():
    calibration = fit_fixture()
    products = prediction_rows(["2025-02-01"])
    bias, covariance = calibration.parameters_for(products)
    selected_bias, selected_covariance = calibration.parameters_for(products.iloc[8:])
    assert np.array_equal(selected_bias, bias[8:])
    assert np.array_equal(selected_covariance, covariance[8:, 8:])
    with pytest.raises(ValueError, match="native product boundaries"):
        wrong = products.copy()
        wrong["product_start"] += pd.Timedelta(minutes=1)
        wrong["product_end"] += pd.Timedelta(minutes=1)
        calibration.sample_innovations(wrong, 2)


def test_held_out_labels_never_refit_calibration_and_scoring_has_known_crps():
    calibration = fit_fixture()
    original = calibration.to_dict()
    model = next(iter(calibration.grids.values()))
    model = replace(model, bias_eur_mwh=np.zeros(24), covariance_eur2_mwh2=4 * np.eye(24))
    controlled = replace(calibration, grids={model.key: model})
    test = prediction_rows(["2025-02-01", "2025-02-02"])
    test["actual"] = test.prediction
    evaluation = evaluate_gaussian_prices(controlled, test)
    assert evaluation["metrics"]["mean_crps_eur_mwh"] == pytest.approx(2 * (sqrt(2) - 1) / sqrt(pi))
    assert evaluation["metrics"]["coverage"]["95"] == 1
    assert evaluation["metrics"]["mae_eur_mwh"] == 0
    altered = test.copy()
    altered["actual"] += 1000
    assert evaluate_gaussian_prices(controlled, altered)["metrics"]["coverage"]["95"] == 0
    assert calibration.to_dict() == original
    with pytest.raises(ValueError, match="follow the calibration cutoff"):
        evaluate_gaussian_prices(calibration, prediction_rows(["2025-01-11"]))


def test_publication_vintage_eligibility_requires_actual_evidence():
    rows = prediction_rows()
    with pytest.raises(ValueError, match="verified publication-vintage"):
        fit_fixture(rows, historical_vintages_verified=True)
    with pytest.raises(ValueError, match="publication timestamps"):
        fit_fixture(rows, historical_vintages_verified=True, availability_policy="verified_historical_publication_vintages")
    rows["feature_available_at"] = rows.forecast_cutoff
    rows["target_available_at"] = rows.product_end
    rows["historical_feature_eligible"] = False
    with pytest.raises(ValueError, match="Unverified"):
        fit_fixture(rows, historical_vintages_verified=True, availability_policy="verified_historical_publication_vintages")
    rows["historical_feature_eligible"] = True
    assert fit_fixture(rows, historical_vintages_verified=True,
                       availability_policy="verified_historical_publication_vintages").operational_eligible


def test_actual_gaussian_paths_build_existing_milp_scenario_contract():
    calibration = fit_fixture(prediction_rows(pd.date_range("2025-10-01", periods=12), minutes=15))
    products = prediction_rows(["2025-11-01"], minutes=15)
    grid = DeliveryGrid(tuple(f"t{i}" for i in range(96)), tuple(products.product_start), tuple(products.product_end))
    events = [AuctionCalendar().schedule("2025-11-01")[0]]
    kwargs = {"mean_prices": {"DA": np.full(96, -20.)}, "calibrations": {"DA": calibration},
              "feature_available_at": {"DA": "2025-10-31T08:00Z"},
              "model_fit_cutoffs": {"DA": "2025-10-30T18:00Z"},
              "decision_cutoff": events[0].submission_at, "branches": 3, "seed": 4}
    with pytest.raises(ValueError, match="allow_retrospective"):
        gaussian_price_tree(grid, np.ones(96), events, **kwargs)
    scenarios, tree = gaussian_price_tree(grid, np.ones(96), events, allow_retrospective=True, **kwargs)
    assert scenarios.auction_prices_eur_per_mwh["DA"].shape == (3, 96)
    assert np.isclose(scenarios.probabilities.sum(), 1)
    assert np.all(scenarios.auction_prices_eur_per_mwh["DA"] < 0)
    assert len(set(tree.bid_nodes["DA"])) == 1
    assert not scenarios.provenance["operational_eligible"]
    assert "zero-mean" in scenarios.provenance["residual_method"]
    tree.paths()


def test_invalid_gaussian_counts_and_parameters_are_rejected():
    rows = prediction_rows()
    for shrinkage in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="shrinkage"):
            fit_fixture(rows, covariance_shrinkage=shrinkage)
    for minimum in (0, 1, 2.5, True):
        with pytest.raises(ValueError, match="at least two"):
            calibrate_gaussian_prices(rows, calibration_cutoff=rows.product_end.max(), minimum_complete_days=minimum)
    calibration = fit_fixture()
    for count in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            calibration.sample_innovations(prediction_rows(["2025-02-01"]), count)

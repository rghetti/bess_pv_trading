"""Exercise search chronology, paired coverage and relative-error definitions."""

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.energy_prices import PriceForecastSettings
from bess_pv_trading.forecasting.price_search import (complete_shared_days, point_metrics, prepare_search_matrix,
                                                      project_search_matrix, daily_price_curve_features,
                                                      restrict_to_common_training_rows)


def test_search_uses_only_labels_released_before_cutoff_and_shared_complete_days():
    starts = pd.date_range("2024-12-01T00:00Z", "2025-01-04T00:00Z", freq="h", inclusive="left")
    history = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(hours=1),
        "available_at": starts + pd.Timedelta(hours=1), "price_eur_per_mwh": np.arange(len(starts), dtype=float)})
    settings = PriceForecastSettings(lag_hours=(24,), interval_minutes=60, timezone="UTC",
                                     decision_local_time="00:00", decision_day_offset=-1)
    cutoff = pd.Timestamp("2024-12-31T00:00Z")
    result = prepare_search_matrix(history, settings, None, cutoff=cutoff,
                                   validation_start="2025-01-01", validation_end="2025-01-04")
    assert (history.loc[result["training"], "product_end"] <= cutoff).all()
    assert complete_shared_days([result], start="2025-01-01", end="2025-01-04", settings=settings) == [
        "2025-01-01", "2025-01-02", "2025-01-03"]
    altered = dict(result)
    altered["validation"] = result["validation"].copy()
    altered["validation"][np.flatnonzero(result["days"] == "2025-01-02")[0]] = False
    assert complete_shared_days([result, altered], start="2025-01-01", end="2025-01-04", settings=settings) == [
        "2025-01-01", "2025-01-03"]


def test_incomplete_days_distinguishes_missing_products_from_missing_features():
    """Audit missing observed labels and inputs without imputing either one."""
    from bess_pv_trading.forecasting.price_search import explain_incomplete_days

    starts = pd.date_range("2025-01-01", "2025-01-03", freq="h", tz="UTC", inclusive="left").delete(2)
    matrix = np.ones((len(starts), 2))
    matrix[starts.get_loc(pd.Timestamp("2025-01-02T03:00:00Z")), 1] = np.nan
    settings = PriceForecastSettings(interval_minutes=60, timezone="UTC")
    details = explain_incomplete_days({"combined": {"starts": starts, "matrix": matrix}},
        {"combined": ["lag", "load"]}, start="2025-01-01", end="2025-01-03", settings=settings)
    assert details[0]["feature_sets"]["combined"]["missing_target_products"] == 1
    assert details[0]["feature_sets"]["combined"]["missing_by_input"] == {}
    assert details[1]["feature_sets"]["combined"]["missing_target_products"] == 0
    assert details[1]["feature_sets"]["combined"]["missing_by_input"] == {"load": 1}


def test_point_metrics_preserve_zero_negative_prices_and_naive_scale():
    metrics = point_metrics([0, -10], [1, -7], [0, -5])
    assert metrics["wape_percent"] == 40
    assert metrics["relative_mae_to_naive"] == pytest.approx(0.8)
    assert point_metrics([0], [1], [0])["wape_percent"] is None


def test_daily_price_curves_average_native_quarters_and_respect_release():
    """Previous-day curve features never expose a product published after the decision."""
    starts = pd.date_range("2026-01-01T00:00Z", periods=96, freq="15min")
    availability = starts + pd.Timedelta(hours=1)
    history = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(minutes=15),
                            "available_at": availability, "price_eur_per_mwh": np.arange(96, dtype=float)})
    targets = pd.DatetimeIndex([pd.Timestamp("2026-01-02T12:00Z")] * 2)
    cutoffs = pd.to_datetime(["2026-01-01T01:45Z", "2026-01-02T00:45Z"])
    settings = PriceForecastSettings(interval_minutes=15, timezone="UTC")
    result = daily_price_curve_features(history, targets, cutoffs, settings, [1])
    assert result.shape == (2, 24)
    assert result[0, 0] == 1.5
    assert np.isnan(result[0, 1])
    np.testing.assert_allclose(result[1], np.arange(24) * 4 + 1.5)


def test_daily_price_curves_leave_dst_missing_hour_explicit():
    """A skipped local hour is not manufactured in a previous-day price curve."""
    starts = pd.date_range("2026-03-29", "2026-03-30", freq="h", inclusive="left", tz="Europe/Rome").tz_convert("UTC")
    history = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(hours=1),
                            "available_at": starts + pd.Timedelta(hours=1), "price_eur_per_mwh": 20.0})
    settings = PriceForecastSettings(interval_minutes=60, timezone="Europe/Rome")
    result = daily_price_curve_features(history, pd.DatetimeIndex([pd.Timestamp("2026-03-30T12:00Z")]),
                                        pd.DatetimeIndex([pd.Timestamp("2026-03-30T00:00Z")]), settings, [1])
    assert np.isnan(result[0, 2])
    assert np.isfinite(result[0]).sum() == 23


def test_feature_projection_does_not_transfer_other_groups_missingness():
    """A missing optional feature cannot remove otherwise valid baseline rows."""
    dataset = {"matrix": np.array([[1.0, 2.0, np.nan], [1.0, np.nan, 3.0]]),
               "training_clock": np.array([True, True]), "validation_clock": np.array([False, True])}
    baseline = project_search_matrix(dataset, [0])
    first = project_search_matrix(dataset, [0, 1])
    second = project_search_matrix(dataset, [0, 2])
    assert baseline["training"].tolist() == [True, True]
    assert first["training"].tolist() == [True, False]
    assert second["training"].tolist() == [False, True]
    assert baseline["validation"].tolist() == [False, True]


def test_common_training_rows_match_feature_groups_without_altering_validation():
    """A feature ablation can use identical training support and retain test masks."""
    datasets = {"base": {"training": np.array([True, True, False]), "validation": np.array([False, True, True])},
                "border": {"training": np.array([False, True, False]), "validation": np.array([True, True, True])}}
    result = restrict_to_common_training_rows(datasets)
    assert result["base"]["training"].tolist() == result["border"]["training"].tolist() == [False, True, False]
    assert result["base"]["validation"].tolist() == [False, True, True]
    assert datasets["base"]["training"].tolist() == [True, True, False]


def test_projected_union_matches_independently_prepared_features():
    """Shared feature preparation produces the same values and masks as separate joins."""
    starts = pd.date_range("2024-12-01T00:00Z", "2025-01-03T00:00Z", freq="h", inclusive="left")
    history = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(hours=1),
        "available_at": starts + pd.Timedelta(hours=1), "price_eur_per_mwh": np.arange(len(starts), dtype=float)})
    exogenous = pd.DataFrame({"product_start": starts[::2], "available_at": starts[::2] - pd.Timedelta(days=2),
                              "feature": "external", "value": 5.0})
    base = dict(lag_hours=(24,), interval_minutes=60, timezone="UTC",
                decision_local_time="00:00", decision_day_offset=-1)
    arguments = dict(cutoff="2024-12-31T00:00Z", validation_start="2025-01-01", validation_end="2025-01-03")
    union = prepare_search_matrix(history, PriceForecastSettings(**base, exogenous_features=("external",)), exogenous, **arguments)
    independent = prepare_search_matrix(history, PriceForecastSettings(**base), exogenous, **arguments)
    projected = project_search_matrix(union, list(range(8)))
    np.testing.assert_allclose(projected["matrix"], independent["matrix"], equal_nan=True)
    np.testing.assert_array_equal(projected["training"], independent["training"])
    np.testing.assert_array_equal(projected["validation"], independent["validation"])


def test_search_rejects_cutoff_after_first_decision():
    starts = pd.date_range("2024-12-01", "2025-01-02", freq="h", tz="UTC")
    history = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(hours=1),
        "available_at": starts + pd.Timedelta(hours=1), "price_eur_per_mwh": 1.0})
    settings = PriceForecastSettings(lag_hours=(24,), interval_minutes=60, timezone="UTC",
                                     decision_local_time="00:00", decision_day_offset=-1)
    with pytest.raises(ValueError, match="first validation decision"):
        prepare_search_matrix(history, settings, None, cutoff="2024-12-31T12:00Z",
                              validation_start="2025-01-01", validation_end="2025-01-02")


def test_worker_saves_predictions_and_scores(tmp_path):
    """The isolated worker trains and scores only the provided split arrays."""
    from scripts.search_energy_price_models import _fit_candidate

    x = np.arange(60, dtype=float).reshape(30, 2)
    dataset = tmp_path / "data.npz"
    output = tmp_path / "predictions.csv"
    np.savez(dataset, train_x=x[:20], train_y=x[:20, 0], validation_x=x[20:],
             validation_y=x[20:, 0], reference=x[20:, 0] + 1,
             starts=np.array([f"2025-01-01T{hour:02}:00:00Z" for hour in range(10)]))
    result = _fit_candidate({"dataset": str(dataset), "output": str(output), "threads": 1,
        "seed": 4, "group": "hourly", "feature_set": "price_only",
        "candidate": {"id": "ridge", "family": "ridge", "parameters": {"alpha": 1}}})
    assert result["training_rows"] == 20
    assert result["observations"] == 10
    assert len(pd.read_csv(output)) == 10


def test_spread_worker_adds_cleared_price_back_before_scoring(tmp_path):
    """A constant spread over a varying anchor reconstructs the actual price."""
    from scripts.search_energy_price_models import _fit_candidate

    x = np.arange(30, dtype=float).reshape(-1, 1)
    dataset = tmp_path / "data.npz"
    output = tmp_path / "predictions.csv"
    np.savez(dataset, train_x=x[:20], train_y=x[:20, 0] + 3, validation_x=x[20:],
             validation_y=x[20:, 0] + 3, reference=x[20:, 0],
             starts=np.array([f"2025-01-01T{hour:02}:00:00Z" for hour in range(10)]))
    result = _fit_candidate({"dataset": str(dataset), "output": str(output), "threads": 1, "seed": 4,
        "group": "ida1", "feature_set": "spread", "residual_feature_index": 0,
        "candidate": {"id": "ridge", "family": "ridge", "parameters": {"alpha": 1}}})
    assert result["mae_eur_per_mwh"] < 1e-10


def test_rolling_blocks_use_only_complete_published_prior_labels():
    """A day gap and publication clock both restrict every refit."""
    from scripts.search_energy_price_models import _rolling_blocks

    starts = pd.date_range("2026-05-01T00:00Z", periods=6, freq="D")
    decisions = starts - pd.Timedelta(hours=1)
    available = starts + pd.Timedelta(hours=1)
    available = available.where(starts != pd.Timestamp("2026-05-03T00:00Z"), pd.Timestamp("2026-05-10T00:00Z"))
    ends = starts + pd.Timedelta(hours=1)
    blocks = _rolling_blocks(starts.astype(str), decisions.asi8, available.asi8, ends.asi8, "UTC", 2, 1)
    assert len(blocks) == 3
    assert blocks[1][2].tolist() == [0]
    assert blocks[2][2].tolist() == [0, 1]
    assert np.concatenate([block[3] for block in blocks]).tolist() == list(range(6))


def test_rolling_worker_records_refit_counts_and_predictions(tmp_path):
    """A weekly-style refit produces one prediction per product and saves its evidence."""
    import json
    from scripts.search_energy_price_models import _fit_candidate

    starts = pd.date_range("2026-05-01T00:00Z", periods=6, freq="D")
    data = tmp_path / "rolling.npz"
    output = tmp_path / "forecast.csv"
    train_x = np.arange(20, dtype=float).reshape(-1, 1)
    test_x = np.arange(20, 26, dtype=float).reshape(-1, 1)
    np.savez(data, train_x=train_x, train_y=train_x[:, 0], validation_x=test_x, validation_y=test_x[:, 0],
        reference=test_x[:, 0] + 1, starts=np.asarray(starts.astype(str), dtype=str),
        decisions_ns=(starts - pd.Timedelta(hours=1)).as_unit("ns").asi8,
        available_ns=(starts + pd.Timedelta(hours=1)).as_unit("ns").asi8,
        product_end_ns=(starts + pd.Timedelta(hours=1)).as_unit("ns").asi8)
    result = _fit_candidate({"dataset": str(data), "output": str(output), "threads": 1, "seed": 1,
        "group": "hourly", "feature_set": "price_only", "timezone": "UTC",
        "rolling_refit": {"frequency_days": 2, "gap_days": 1},
        "candidate": {"id": "ridge", "family": "ridge", "parameters": {"alpha": 1}}})
    refits = json.loads(output.with_suffix(".refits.json").read_text())
    assert result["rolling_refit_count"] == len(refits) == 3
    assert [row["added_prior_intervals"] for row in refits] == [0, 1, 3]
    assert len(pd.read_csv(output)) == 6


def test_local_slot_models_fit_separate_relations_without_future_labels():
    """Each delivery clock slot has its own model and missing slots fail explicitly."""
    from scripts.search_energy_price_models import _candidate_predict

    days = pd.date_range("2025-01-01T00:00Z", periods=20, freq="D")
    starts = pd.DatetimeIndex([day + pd.Timedelta(hours=hour) for day in days for hour in (0, 1)])
    features = np.arange(40, dtype=float).reshape(-1, 1)
    targets = features[:, 0] + np.where(starts.hour == 0, 20.0, -20.0)
    spec = {"family": "slot_ridge", "parameters": {"alpha": 0.0, "min_rows_per_slot": 1}}
    prediction = _candidate_predict(spec, features[:30], targets[:30], features[30:],
        train_starts=starts[:30], test_starts=starts[30:], timezone="UTC", seed=1, threads=1)
    np.testing.assert_allclose(prediction, targets[30:], atol=1e-9)
    spec["parameters"]["min_rows_per_slot"] = 16
    with pytest.raises(ValueError, match="Insufficient training rows"):
        _candidate_predict(spec, features[:30], targets[:30], features[30:],
            train_starts=starts[:30], test_starts=starts[30:], timezone="UTC", seed=1, threads=1)


def test_slot_preflight_rejects_missing_or_sparse_delivery_slots():
    """Total training volume must not hide a missing delivery slot."""
    from scripts.search_energy_price_models import _validate_slot_coverage

    starts = pd.to_datetime(["2025-01-01T00:00Z", "2025-01-02T00:00Z", "2025-01-01T01:00Z"])
    assert _validate_slot_coverage(starts, starts[:1], "UTC", 2) == 2
    with pytest.raises(ValueError, match="Insufficient per-slot"):
        _validate_slot_coverage(starts, starts[2:], "UTC", 2)


def test_slot_worker_reads_training_timestamps_and_writes_predictions(tmp_path):
    """The isolated worker carries local-slot timestamps through the artifact path."""
    from scripts.search_energy_price_models import _fit_candidate

    days = pd.date_range("2025-01-01T00:00Z", periods=20, freq="D")
    starts = pd.DatetimeIndex([day + pd.Timedelta(hours=hour) for day in days for hour in (0, 1)])
    features = np.arange(40, dtype=float).reshape(-1, 1)
    targets = features[:, 0] + np.where(starts.hour == 0, 20., -20.)
    data = tmp_path / "slots.npz"
    output = tmp_path / "prediction.csv"
    np.savez(data, train_x=features[:30], train_y=targets[:30], validation_x=features[30:],
        validation_y=targets[30:], reference=targets[30:] + 1,
        train_starts=np.asarray(starts[:30].astype(str), dtype=str), starts=np.asarray(starts[30:].astype(str), dtype=str))
    result = _fit_candidate({"dataset": str(data), "output": str(output), "threads": 1, "seed": 1,
        "group": "hourly", "feature_set": "curve", "timezone": "UTC",
        "candidate": {"id": "slot", "family": "slot_ridge", "parameters": {"alpha": 0., "min_rows_per_slot": 5}}})
    assert result["mae_eur_per_mwh"] < 1e-9


def test_intraday_search_declares_separate_native_market_models():
    """Each auction and settlement is fitted as its own quarter-hour task."""
    from pathlib import Path
    from bess_pv_trading.config import load_config

    root = Path(__file__).resolve().parents[2]
    study = load_config(root / "configs/research/price_model_search_intraday.yaml")["price_model_search"]
    assert {group["target"] for group in study["product_groups"].values()} == {"IDA1", "IDA2", "IDA3", "settlement"}
    assert all(group["native_minutes"] == 15 for group in study["product_groups"].values())


def test_holdout_selection_uses_validation_and_rejects_early_cutoff(tmp_path):
    """Selection cannot consult held-out scores or occur before validation ends."""
    import json
    from scripts.search_energy_price_models import _validation_selection

    candidates = [{"id": "a", "family": "ridge", "parameters": {"alpha": 1}},
                  {"id": "b", "family": "ridge", "parameters": {"alpha": 2}}]
    source = {"lag_hours": [24], "feature_sets": ["price_only"], "seed": 1, "timezone": "UTC",
              "source_study": "source", "candidates": candidates,
              "product_groups": {"hourly": {"native_minutes": 60, "validation_end": "2025-07-01"}}}
    manifest = {"status": "complete", "failures": [], "config": source,
        "source_lineage": {"hourly": {"availability": {"kind": "assumed_delivery_day_publication_clock",
                                                            "timezone": "UTC", "day_offset": -1}}},
        "results": [{"group": "hourly", "feature_set": "price_only", "candidate": candidate,
                     "mae_eur_per_mwh": error} for candidate, error in [("a", 2), ("b", 1)]]}
    (tmp_path / "validation.json").write_text(json.dumps(manifest))
    study = {**source, "product_groups": {"hourly": {"native_minutes": 60, "cutoff": "2025-07-01T01:00Z"}},
             "selection": {"source_manifest": "validation.json", "top_n": 1, "baseline_candidates": ["a"]}}
    selected, evidence = _validation_selection(study, tmp_path)
    assert selected["hourly"] == {("price_only", "a"), ("price_only", "b")}
    assert evidence["criterion"] == "pooled_validation_mae"
    study["product_groups"]["hourly"]["cutoff"] = "2025-06-30T23:00Z"
    with pytest.raises(ValueError, match="precedes"):
        _validation_selection(study, tmp_path)


def test_holdout_selection_keeps_matching_model_features(tmp_path):
    """Preselected candidates retain the same hyperparameters in each ablation arm."""
    import json
    from scripts.search_energy_price_models import _validation_selection

    source = {"lag_hours": [24], "feature_sets": ["base", "added"], "seed": 1, "timezone": "UTC",
              "source_study": "source", "candidates": [{"id": "best"}, {"id": "reference"}],
              "product_groups": {"quarter": {"native_minutes": 15, "validation_end": "2025-07-01"}}}
    records = [{"group": "quarter", "feature_set": feature, "candidate": candidate,
                "mae_eur_per_mwh": 1 if (feature, candidate) == ("added", "best") else 2}
               for feature in source["feature_sets"] for candidate in ["best", "reference"]]
    manifest = {"status": "complete", "failures": [], "config": source, "results": records,
                "source_lineage": {"quarter": {"availability": {
                    "kind": "assumed_delivery_day_publication_clock", "timezone": "UTC", "day_offset": -1}}}}
    (tmp_path / "validation.json").write_text(json.dumps(manifest))
    study = {**source, "product_groups": {"quarter": {"native_minutes": 15, "cutoff": "2025-07-01T01:00Z"}},
             "selection": {"source_manifest": "validation.json", "top_n": 1,
                           "matched_feature_pairs": [["base", "added"]]}}
    selected, evidence = _validation_selection(study, tmp_path)
    assert selected["quarter"] == {("base", "best"), ("added", "best")}
    assert evidence["matched_feature_pairs"] == [["base", "added"]]


def test_search_report_renders_metrics_and_plot(tmp_path):
    """Report files contain relative metrics and a comparison plot."""
    from scripts.search_energy_price_models import _write_report

    manifest = {"status": "complete", "selection": None, "failures": [], "results": [{
        "group": "hourly", "feature_set": "price_only", "candidate": "ridge", "training_rows": 100,
        "mae_eur_per_mwh": 5.0, "wape_percent": 5.0, "relative_mae_to_naive": 0.8}]}
    _write_report(tmp_path, manifest)
    assert "Relative MAE" in (tmp_path / "summary.md").read_text()
    assert (tmp_path / "hourly__mae.png").is_file()


def test_only_earlier_auction_markets_can_be_used_as_cleared_features():
    """The same or a future auction cannot leak into its target price."""
    from scripts.search_energy_price_models import _eligible_auction_features

    assert _eligible_auction_features("IDA2", ["DA", "IDA1"]) == ["cleared_da", "cleared_ida1"]
    with pytest.raises(ValueError, match="earlier-auction"):
        _eligible_auction_features("IDA2", ["DA", "IDA2"])
    with pytest.raises(ValueError, match="earlier-auction"):
        _eligible_auction_features("IDA1", ["IDA3"])
    with pytest.raises(ValueError, match="earlier-auction"):
        _eligible_auction_features("IDA2", ["IDA1", "DA"])


def test_cleared_auction_feature_respects_publication_clock():
    """An earlier auction becomes usable only after its assumed publication."""
    from bess_pv_trading.forecasting.energy_prices import _exogenous_values, prepare_exogenous_features

    start = pd.Timestamp("2026-02-02T00:00Z")
    exogenous = prepare_exogenous_features(pd.DataFrame({"product_start": [start],
        "available_at": [pd.Timestamp("2026-02-01T12:00Z")], "feature": ["cleared_da"], "value": [75.0]}))
    starts = pd.DatetimeIndex([start, start])
    cutoffs = pd.to_datetime(["2026-02-01T11:59Z", "2026-02-01T12:00Z"])
    values = _exogenous_values(exogenous, ["cleared_da"], starts, cutoffs).ravel()
    assert np.isnan(values[0])
    assert values[1] == 75.0


def test_combined_prepared_feature_indexes_preserve_asof_values():
    """Reusing the base index and adding an auction index gives the same join."""
    from bess_pv_trading.forecasting.energy_prices import PreparedExogenous, _exogenous_values, prepare_exogenous_features

    starts = pd.to_datetime(["2026-02-02T00:00Z", "2026-02-02T00:15Z"])
    available = pd.Timestamp("2026-02-01T12:00Z")
    base = pd.DataFrame({"product_start": starts, "available_at": available, "feature": "load", "value": [1, 2]})
    auction = pd.DataFrame({"product_start": starts, "available_at": available, "feature": "da", "value": [3, 4]})
    first = prepare_exogenous_features(base)
    second = prepare_exogenous_features(auction)
    merged = PreparedExogenous({**first.groups, **second.groups})
    full = prepare_exogenous_features(pd.concat([base, auction], ignore_index=True))
    cutoff = pd.DatetimeIndex([available, available])
    np.testing.assert_array_equal(_exogenous_values(merged, ["load", "da"], starts, cutoff),
                                  _exogenous_values(full, ["load", "da"], starts, cutoff))


def test_auction_neighbors_keep_original_publication_and_correct_offset():
    """A prior-hour value joins the target hour but keeps its original release."""
    from scripts.search_energy_price_models import _neighbor_price_feature

    source = pd.DataFrame({"product_start": pd.to_datetime(["2026-02-02T11:00Z"]),
                           "available_at": pd.to_datetime(["2026-02-01T12:00Z"]), "value": [75.0]})
    result = _neighbor_price_feature(source, "prior_hour", -60)
    assert result.product_start.iloc[0] == pd.Timestamp("2026-02-02T12:00Z")
    assert result.available_at.iloc[0] == source.available_at.iloc[0]
    assert source.product_start.iloc[0] == pd.Timestamp("2026-02-02T11:00Z")


def test_forward_auction_neighbor_stays_within_delivery_day():
    """Use the same cleared product at midnight without borrowing tomorrow's auction."""
    from scripts.search_energy_price_models import _neighbor_price_feature

    source = pd.DataFrame({"product_start": pd.to_datetime(["2026-02-02T22:30Z", "2026-02-02T22:45Z",
                                                               "2026-02-02T23:00Z"]),
                           "available_at": pd.to_datetime(["2026-02-01T12:00Z", "2026-02-01T12:00Z",
                                                             "2026-02-02T12:00Z"]),
                           "value": [10.0, 20.0, 30.0]})
    result = _neighbor_price_feature(source, "forward", 15, boundary="same_product", timezone="Europe/Rome")
    result = result.set_index("product_start").sort_index()
    assert result.loc[pd.Timestamp("2026-02-02T22:30Z"), "value"] == 20.0
    assert result.loc[pd.Timestamp("2026-02-02T22:45Z"), "value"] == 20.0
    assert result.loc[pd.Timestamp("2026-02-02T22:45Z"), "available_at"] == pd.Timestamp("2026-02-01T12:00Z")
    assert not result.index.has_duplicates


def test_lagged_auction_price_keeps_original_release_clock():
    """Yesterday's IDA feature must still pass the target decision's as-of check."""
    from scripts.search_energy_price_models import _lagged_auction_feature
    from bess_pv_trading.forecasting.energy_prices import _exogenous_values

    source = pd.DataFrame({"product_start": pd.to_datetime(["2026-02-01T10:00Z"]),
                           "available_at": pd.to_datetime(["2026-02-01T09:00Z"]), "value": [75.0]})
    feature = _lagged_auction_feature(source, "IDA1", 24)
    starts = pd.DatetimeIndex(["2026-02-02T10:00Z"])
    early = pd.DatetimeIndex(["2026-02-01T08:59Z"])
    late = pd.DatetimeIndex(["2026-02-01T09:00Z"])
    assert np.isnan(_exogenous_values(feature, ["past_ida1_24h"], starts, early)[0, 0])
    assert _exogenous_values(feature, ["past_ida1_24h"], starts, late)[0, 0] == 75.0
    with pytest.raises(ValueError, match="at least 24 hours"):
        _lagged_auction_feature(source, "IDA1", 0)


def test_cleared_search_includes_same_and_neighboring_historical_products():
    """Retain the same-product lag as the reference and add nearby hours."""
    from pathlib import Path
    from bess_pv_trading.config import load_config

    root = Path(__file__).resolve().parents[2]
    study = load_config(root / "configs/research/price_model_search_cleared_auctions.yaml")["price_model_search"]
    assert study["lag_hours"][0] == 24
    assert {23, 24, 25, 47, 48, 49, 167, 168, 169}.issubset(study["lag_hours"])
    assert all(group["market_neighbor_offsets_minutes"] == [-60, -15]
               for group in study["product_groups"].values())
    assert study["product_groups"]["ida3_quarter_hourly"]["lag_hours"] == [24, 48, 72, 168, 336]


def test_expanded_day_ahead_search_respects_historical_feature_coverage():
    """Do not force Swiss-only 2025 features into the hourly 2024 training split."""
    from pathlib import Path
    from bess_pv_trading.config import load_config

    root = Path(__file__).resolve().parents[2]
    study = load_config(root / "configs/research/price_model_search_da_expanded.yaml")["price_model_search"]
    assert study["product_groups"]["hourly"].get("feature_sets", study["feature_sets"]) == ["price_only", "fundamentals"]
    assert "combined" in study["product_groups"]["quarter_hourly"]["feature_sets"]
    assert study["lag_hours"][0] == 24


def test_historical_ida_export_search_starts_after_market_transition():
    """The hourly-export study is separate from the earlier auction regime."""
    from pathlib import Path
    from bess_pv_trading.config import load_config

    root = Path(__file__).resolve().parents[2]
    study = load_config(root / "configs/research/price_model_search_ida_hourly_2024.yaml")["price_model_search"]
    assert study["common_training_rows"] is True
    assert all(group["native_minutes"] == 60 and group["history_start"].startswith("2024-06-14")
               for group in study["product_groups"].values())
    assert all(name.endswith("_hourly_export_2024") for name in study["product_groups"])
    assert all(group["market_neighbor_offsets_minutes"] == [-60] for group in study["product_groups"].values())


def test_ida_2025_validation_retains_quarter_hour_targets_and_prior_training():
    """The additional historical regime cannot silently mix hourly IDA targets."""
    from pathlib import Path
    from bess_pv_trading.config import load_config

    root = Path(__file__).resolve().parents[2]
    study = load_config(root / "configs/research/price_model_search_cleared_2025.yaml")["price_model_search"]
    assert all(group["native_minutes"] == 15 and group["validation_start"] == "2025-06-01"
               for group in study["product_groups"].values())


def test_supplemental_holdout_rechecks_selection_period_and_rejects_changed_rows(monkeypatch):
    """An extended holdout cannot silently reuse revised selection features."""
    from scripts import search_energy_price_models as search

    original = {"database": "/catalog.duckdb", "features": [{"name": "french", "selected_row_sha256": "old"}]}
    validation = {"config": {"product_groups": {"ida": {
        "history_start": "2025-01-01T00:00:00Z", "history_end": "2026-05-01T00:00:00Z"}}},
        "supplemental_features": original}
    study = {"database": "/catalog.duckdb", "timezone": "Europe/Rome", "threads_per_worker": 1,
             "supplemental_features": [{"name": "french"}], "product_groups": {"ida": {
                 "history_start": "2025-01-01T00:00:00Z", "history_end": "2026-09-01T00:00:00Z"}}}
    calls = []

    def reload_features(database, selections, **kwargs):
        calls.append(kwargs)
        return None, original

    monkeypatch.setattr(search, "load_catalog_features", reload_features)
    verification = search._verify_supplemental_selection_sources(study, validation)
    assert verification["selection_end"] == "2026-05-01T00:00:00+00:00"
    assert calls[0]["end"].isoformat() == verification["selection_end"]

    monkeypatch.setattr(search, "load_catalog_features", lambda *args, **kwargs: (None, {
        "database": "/catalog.duckdb", "features": [{"name": "french", "selected_row_sha256": "revised"}]}))
    with pytest.raises(ValueError, match="changed since validation"):
        search._verify_supplemental_selection_sources(study, validation)


def test_rolling_ida_uses_later_dates_and_earlier_market_selection():
    """IDA refits are scheduled after model selection, with market-specific lags retained."""
    from pathlib import Path
    from bess_pv_trading.config import load_config

    root = Path(__file__).resolve().parents[2]
    study = load_config(root / "configs/research/price_model_rolling_ida.yaml")["price_model_search"]
    assert study["rolling_refit"] == {"frequency_days": 7, "gap_days": 1}
    assert all(group["validation_start"] == "2026-05-02" for group in study["product_groups"].values())
    assert study["product_groups"]["ida3_quarter_hourly"]["lag_hours"] == [24, 48, 72, 168, 336]


def test_swiss_reservoir_ablation_uses_only_declared_state_features_and_matched_training():
    """Reservoir levels are additional past state inputs, not target reconstruction."""
    from pathlib import Path
    from bess_pv_trading.config import load_config

    root = Path(__file__).resolve().parents[2]
    study = load_config(root / "configs/research/price_model_search_swiss_reservoir.yaml")["price_model_search"]
    assert study["common_training_rows"] is True
    assert {item["selectors"]["variable"] for item in study["supplemental_features"]} == {
        "total_ch_percent", "ticino_gwh"}
    assert all(item["kind"] == "released_state" and item["max_age_hours"] == 240
               for item in study["supplemental_features"])
    assert set(study["add_supplemental_features_to"]) == {"cleared_reservoir", "spread_reservoir"}


@pytest.mark.parametrize("filename", ["price_model_search_da_french_balance.yaml",
                                      "price_model_search_ida_french_balance.yaml"])
def test_french_balance_ablations_use_prior_day_and_matched_training(filename):
    """French national inputs are lagged and trained on the same rows as their baseline."""
    from pathlib import Path
    from bess_pv_trading.config import load_config

    root = Path(__file__).resolve().parents[2]
    study = load_config(root / "configs/research" / filename)["price_model_search"]
    assert study["common_training_rows"] is True
    assert {item["json_component"]["key"] for item in study["supplemental_features"]} == {
        "consommation", "eolien", "solaire", "ech_comm_suisse"}
    assert all(item["target_shift_hours"] == 24 for item in study["supplemental_features"])


def test_terna_load_study_is_lagged_matched_and_restricted_to_covered_2025():
    """Zonal historical load is not treated as same-day or 2026 coverage."""
    from pathlib import Path
    from bess_pv_trading.config import load_config

    root = Path(__file__).resolve().parents[2]
    study = load_config(root / "configs/research/price_model_search_ida_2025_terna_load.yaml")["price_model_search"]
    item = study["supplemental_features"][0]
    assert study["common_training_rows"] is True
    assert (item["dataset"], item["kind"], item["target_shift_hours"]) == ("total_load", "reference_interval", 24)
    assert all(group["validation_end"] == "2025-09-01" for group in study["product_groups"].values())

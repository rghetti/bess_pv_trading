"""Keep optional price-study report evidence numerical and explicitly qualified."""
from __future__ import annotations

import json
import hashlib

import pytest

from bess_pv_trading.reporting.implementation import (
    _components, _gaussian_studies_tex, _gaussian_study_evidence,
    _linear_price_evidence, _linear_price_studies_tex,
    _price_forward_check_evidence, _price_forward_checks_tex, _forecast_input_boundary_tex,
    _render)


def _study():
    """Supply a selected model whose final baseline happens to perform better."""
    return {"status": "completed", "report_label": "prices", "first_delivery_day": "2025-01-01",
        "last_delivery_day": "2025-09-30", "rows": 6000,
        "prior_evaluation_overlap": "Earlier exploratory holdouts overlap; not a fresh prospective test.",
        "regimes": [{"status": "evaluated", "native_interval_minutes": 60,
            "selected_candidate": {"kind": "ridge", "feature_set": "lags", "alpha": 100.,
                "training_window_days": 180, "target": "delta_day"},
            "selection_metrics": {"mae_eur_mwh": 9.1}, "test_metrics": {"mae_eur_mwh": 8.8},
            "split": {"validation_start_day": "2025-06-01", "validation_end_day": "2025-06-28",
                "calibration_start_day": "2025-06-29", "calibration_end_day": "2025-08-09",
                "test_start_day": "2025-08-11", "test_end_day": "2025-09-21"}}],
        "test_comparisons": [{"native_interval_minutes": 60, "model": "baseline_previous_day", "mae_eur_mwh": 11.0},
            {"native_interval_minutes": 60, "model": "baseline_day_week_average", "mae_eur_mwh": 8.6}],
        "uncertainty": {"regimes": [{"status": "evaluated", "native_interval_minutes": 60,
            "calibration_grids": {"60min_24products_ordinary": {"days": 42, "mean_error_variance_eur2_mwh2": 120.}},
            "test_metrics": {"coverage": {"95": .89}, "mean_crps_eur_mwh": 6.5,
                "scored_days": 40, "scored_rows": 960},
            "test_excluded_days": {"2025-08-12": "incomplete"}}]},
        "uncertainty_artifacts": {"60": {"grids": {"60min_24products_ordinary": {"bias_eur_mwh": [1., 3.]}}}}}


def test_linear_report_keeps_selection_baseline_and_gaussian_scored_subset_distinct():
    text = _linear_price_studies_tex([_study()])
    assert "9.100" in text and "8.800" in text
    assert "day week average: 8.600" in text
    assert "120.00" in text and "2.000" in text and "6.500" in text
    assert "89.0" in text and "not yet reliable" in text
    assert "40 days and 960 products" in text
    assert "final test never" not in text  # Descriptive baseline does not relabel selection.
    assert "does not change the validation-selected forecast" in text
    assert "fresh prospective" in text
    assert "not unconditional price variance" in text
    assert "ineligible for operational historical deployment" in text


def test_optional_report_loader_reads_nearby_uncertainty_and_artifact(tmp_path):
    folder = tmp_path / "outputs" / "prices" / "run"
    uncertainty = folder / "uncertainty"
    uncertainty.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"status": "completed", "rows": 100}))
    artifact = uncertainty / "gaussian.json"
    artifact.write_text(json.dumps({"grids": {"grid": {"bias_eur_mwh": [2.]}}}))
    (uncertainty / "manifest.json").write_text(json.dumps({"status": "complete", "regimes": [
        {"native_interval_minutes": 60, "artifact": str(artifact.relative_to(tmp_path))}]}))
    evidence = _linear_price_evidence(tmp_path, ["outputs/prices/run/manifest.json", "missing.json"])
    assert evidence[0]["report_label"] == "prices"
    assert evidence[0]["uncertainty_artifacts"]["60"]["grids"]["grid"]["bias_eur_mwh"] == [2.]
    assert evidence[1]["status"] == "manifest-missing"
    with pytest.raises(ValueError, match="list"):
        _linear_price_evidence(tmp_path, "wrong")


def test_no_optional_studies_add_no_claim_and_missing_manifest_is_explicit():
    assert _linear_price_studies_tex([]) == ""
    text = _linear_price_studies_tex([{"report_label": "example", "status": "manifest-missing"}])
    assert "manifest-missing" in text and "no result is claimed" in text
    assert "Error variance is" not in text


def test_component_rows_distinguish_executed_study_from_deployment():
    tests = {"status": "executed", "failed": 0, "errors": 0}
    components = _components(tests, [], {}, {}, linear_price_studies=[_study()])
    by_name = {row["component"]: row for row in components}
    assert by_name["Rolling linear price selection and forecast artifacts"]["status"] == "implemented-and-tested"
    assert by_name["Zero-mean Gaussian paths and MILP adapter"]["status"] == "implemented-and-tested"
    assert by_name["Historical learned price/PV scenarios"]["status"] == "blocked"


def _gaussian_study():
    """Keep the linear point accuracy distinct from worse corrected-mean MAE."""
    return {"status": "completed", "report_label": "hourly_structure",
        "source_manifest_copy_verified": True,
        "prior_evaluation_overlap": "Earlier exploratory holdouts overlap; not a fresh prospective test.",
        "source_storage_evidence": {"backends": ["duckdb"], "databases": ["data/api/market.duckdb"],
                                    "snapshot_count": 608, "checksum_verified_snapshots": 608},
        "regimes": [{"status": "evaluated", "native_interval_minutes": 15,
            "selected_policy": {"window_days": 42, "covariance_shrinkage": .3,
                                "selection_max_product_end": "2026-07-19T22:00:00+00:00"},
            "sample_count": 100, "sample_day": "2026-07-21",
            "static_metrics": {"uncorrected_point_mae_eur_mwh": 16.435604,
                "mae_eur_mwh": 17.139980, "mean_crps_eur_mwh": 12.259423,
                "coverage": {"95": .912698}, "consecutive_product_ramps": {"coverage_95": .987218}},
            "adaptive_metrics": {"uncorrected_point_mae_eur_mwh": 16.435604,
                "mae_eur_mwh": 17.195439, "mean_crps_eur_mwh": 12.221246,
                "coverage": {"95": .922619}, "consecutive_product_ramps": {"coverage_95": .983208},
                "scored_days": 42, "scored_rows": 4032}}]}


def test_adaptive_gaussian_report_separates_point_mean_coverage_and_online_update_policy():
    text = _gaussian_studies_tex([_gaussian_study()])
    assert "16.436" in text and "17.195" in text and "17.140" in text
    assert "12.259" in text and "12.221" in text
    assert "91.27" in text and "92.26" in text
    assert "92.3 percent against a nominal 95 percent" in text
    assert "increases MAE" in text and "do not establish an improvement" in text
    assert "earlier mature test errors" in text.lower()
    assert "Final-test outcomes never select policy hyperparameters" in text
    assert "Operational eligibility remains false" in text
    assert "DuckDB" in text and "608 checksum-verified snapshots" in text
    assert "SHA-256" in text and "fresh prospective" in text
    assert "17.195 & 92.26 & 12.221 & 98.32" in text
    assert r"\end{longtable}}\n" not in text


def test_report_loader_recovers_duckdb_provenance_and_verifies_frozen_source_manifest(tmp_path):
    source = tmp_path / "outputs" / "hourly_structure" / "mean_run"
    source.mkdir(parents=True)
    source_data = {"status": "completed"}
    source_bytes = json.dumps(source_data).encode()
    (source / "manifest.json").write_bytes(source_bytes)
    (source / "settings.json").write_text(json.dumps({"cache_directory": "data/api/market.duckdb"}))
    (source / "source_caches.json").write_text(json.dumps({"prices": [{
        "checksums_verified": True,
        "storage": {"backend": "duckdb", "database": "data/api/market.duckdb", "snapshot_id": "snapshot"}}]}))
    uncertainty = tmp_path / "outputs" / "uncertainty" / "gaussian_run"
    uncertainty.mkdir(parents=True)
    (uncertainty / "source_manifest.json").write_bytes(source_bytes)
    manifest = uncertainty / "manifest.json"
    manifest.write_text(json.dumps({"status": "completed", "source_study_manifest": str(source / "manifest.json"),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest()}))
    evidence = _gaussian_study_evidence(tmp_path, [str(manifest.relative_to(tmp_path))])[0]
    assert evidence["source_manifest_copy_verified"] is True
    assert evidence["report_label"] == "hourly_structure"
    storage = evidence["source_storage_evidence"]
    assert storage["backends"] == ["duckdb"] and storage["snapshot_count"] == 1
    assert storage["checksum_verified_snapshots"] == 1
    assert storage["databases"] == ["data/api/market.duckdb"]
    assert len(storage["source_provenance_sha256"]) == 64
    (uncertainty / "source_manifest.json").write_text("{}")
    changed = _gaussian_study_evidence(tmp_path, [str(manifest.relative_to(tmp_path))])[0]
    assert changed["source_manifest_copy_verified"] is False
    with pytest.raises(ValueError, match="list"):
        _gaussian_study_evidence(tmp_path, "not-a-list")
    assert _gaussian_studies_tex([]) == ""


def test_linear_report_renders_by_hour_structure_and_saved_duckdb_ingestion():
    study = _study()
    study["regimes"][0]["selected_candidate"]["model_structure"] = "by_hour"
    study["storage_evidence"] = {"backends": ["duckdb"], "databases": ["data/api/market.duckdb"],
                                  "snapshot_count": 608, "checksum_verified_snapshots": 608}
    text = _linear_price_studies_tex([study])
    assert "ridge by hour" in text
    assert "608 of 608 saved source snapshots" in text
    assert "DuckDB snapshots" in text


def test_gme_and_adaptive_components_do_not_claim_user_acquisition_or_operational_eligibility():
    tests = {"status": "executed", "failed": 0, "errors": 0, "test_ids": ["tests.unit.test_gme_api.test_parser"]}
    rows = _components(tests, [], {}, {}, gaussian_studies=[_gaussian_study()])
    components = {row["component"]: row for row in rows}
    assert components["GME API request and zonal-price parser"]["status"] == "implemented-and-tested"
    assert components["GME authenticated historical acquisition"]["status"] == "blocked"
    assert "no actual GME IDA prices" in components["GME authenticated historical acquisition"]["evidence"]
    assert components["Calibration-selected online Gaussian uncertainty"]["status"] == "implemented-and-tested"


@pytest.mark.parametrize("method,label", [("ols", "OLS"), ("huber", "Huber"),
    ("weighted_ridge", "recency-weighted ridge"), ("lasso", "lasso"), ("elastic_net", "elastic net")])
def test_linear_report_preserves_selected_estimator_family_and_penalty_conventions(method, label):
    study = _study()
    candidate = study["regimes"][0]["selected_candidate"]
    candidate.update(kind="linear", model_structure="by_hour", estimator={
        "method": method, "alpha": .1, "half_life_days": 14, "epsilon": 1.35, "l1_ratio": .5})
    text = _linear_price_studies_tex([study])
    assert label + " by hour" in text
    if method != "weighted_ridge":
        assert "ridge by hour" not in text
    else:
        assert "half-life 14d" in text
    assert "not equivalent penalties" in text
    assert "validation MAE only" in text and "silent ridge substitution" in text
    assert "OLS uses rank-safe" in text


def test_frozen_feature_report_distinguishes_reuse_from_new_database_ingestion():
    study = _study()
    study["storage_evidence"] = {"backends": ["duckdb"], "databases": ["data/api/market.duckdb"],
                                  "snapshot_count": 608, "checksum_verified_snapshots": 608}
    study["frozen_feature_source"] = {"manifest": "outputs/old/manifest.json", "manifest_sha256": "a" * 64,
                                       "features_sha256": "b" * 64, "policy": "Frozen prior feature bytes only"}
    text = _linear_price_studies_tex([study])
    assert "reused frozen features previously derived from DuckDB" in text
    assert "without reopening the current database" in text
    assert "ingested DuckDB snapshots" not in text


def _forward_check():
    """Supply a complete historical extension with separately reported means."""
    return {"status": "completed", "new_test_start": "2026-09-01", "new_test_end_exclusive": "2026-10-17",
        "maximum_previously_seen_delivery_day": "2026-08-31", "fresh_days": 46, "fresh_rows": 4416,
        "database": "data/api/market.duckdb", "new_snapshot_count": 46,
        "source_historical_features_unchanged": True, "copied_protocol_verified": True,
        "source_ancillary_hashes_pinned_in_protocol": True, "test_labels_used_for_selection": False,
        "overlap_verification": {"rows": 8064, "maximum_prediction_difference_eur_mwh": 0.,
                                 "comparison_tolerance_eur_mwh": 1e-9, "cutoffs_and_training_days_identical": True},
        "point_metrics": {"selected": {"mae_eur_mwh": 14.1, "rmse_eur_mwh": 19.2},
            "previous_day": {"mae_eur_mwh": 15., "rmse_eur_mwh": 20., "selected_mae_skill": .06},
            "previous_week": {"mae_eur_mwh": 16., "rmse_eur_mwh": 21., "selected_mae_skill": .11875},
            "day_week_average": {"mae_eur_mwh": 14.8, "rmse_eur_mwh": 19.5, "selected_mae_skill": .047297}},
        "static_metrics": {"mae_eur_mwh": 15.1, "coverage": {"95": .91}, "mean_crps_eur_mwh": 10.2,
                           "consecutive_product_ramps": {"coverage_95": .99}},
        "adaptive_metrics": {"mae_eur_mwh": 14.5, "coverage": {"95": .94}, "mean_crps_eur_mwh": 10.,
                             "consecutive_product_ramps": {"coverage_95": .96}}}


def test_forward_report_shows_historical_extension_replay_baselines_and_distinct_uncertainty():
    text = _price_forward_checks_tex([_forward_check()])
    assert "Previously uninspected historical extension" in text
    assert "not future market outcomes observed prospectively" in text
    assert "2026-09-01 through 2026-10-16" in text and "exclusive end 2026-10-17" in text
    assert "46 complete days and 4416" in text
    assert "8064 prior products" in text and "0.000000000 EUR/MWh" in text
    assert "Copied protocol SHA-256 verified: True" in text
    assert "Frozen selected model & 14.100 & 19.200" in text
    assert "Previous day & 15.000 & 20.000 & 6.00" in text
    assert "Static & 15.100 & 91.00 & 10.200 & 99.00" in text
    assert "Online frozen policy & 14.500 & 94.00 & 10.000 & 96.00" in text
    assert "below nominal coverage" in text
    assert "operational eligibility remains false" in text


def test_forward_loader_hashes_saved_protocol_and_waiting_checks_claim_no_partial_scores(tmp_path):
    directory = tmp_path / "outputs" / "check"
    directory.mkdir(parents=True)
    protocol = json.dumps({"schema_version": "fixture", "selected_mean_specification": {"kind": "ridge"}}).encode()
    (directory / "protocol.json").write_bytes(protocol)
    check = _forward_check()
    check["protocol_sha256"] = hashlib.sha256(protocol).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(check))
    loaded = _price_forward_check_evidence(tmp_path, ["outputs/check/manifest.json"])[0]
    assert loaded["copied_protocol_verified"] is True
    assert loaded["point_metrics"] == check["point_metrics"]
    waiting = {"status": "waiting_for_data", "missing_days": ["2026-10-16"], "requested_days": 46}
    text = _price_forward_checks_tex([waiting])
    assert "1 missing or incomplete days out of 46" in text
    assert "No partial-period accuracy" in text
    assert "Frozen selected model &" not in text
    assert _price_forward_checks_tex([]) == ""
    with pytest.raises(ValueError, match="list"):
        _price_forward_check_evidence(tmp_path, "wrong")


def test_stochastic_input_boundary_status_requires_its_own_fixture_evidence():
    tests = {"status": "executed", "failed": 0, "errors": 0,
             "test_ids": ["tests.unit.test_prices.test_ridge"]}
    component = next(item for item in _components(tests, [], {}, {})
                     if item["component"] == "Artifact-to-stochastic optimization input boundary")
    assert component["status"] == "implemented-not-executed"
    text = _forecast_input_boundary_tex(tests)
    assert "Matching fixture execution is not yet recorded" in text
    assert "independent price-times-PV product measure" in text
    assert "common price draws across PV members" in text
    assert "Every submitted quantity executes in full" in text
    assert "committed quantity awaiting price publication" in text
    assert "publication books quantity times actual clearing price once" in text
    assert "optimization objective measures incremental" in text
    assert "Residual reoptimization" in text and "IDA3" in text
    assert "No actual-data economic optimization backtest" in text
    tested = {**tests, "test_ids": ["tests.unit.test_forecast_inputs.test_joint_members"]}
    component = next(item for item in _components(tested, [], {}, {})
                     if item["component"] == "Artifact-to-stochastic optimization input boundary")
    assert component["status"] == "implemented-and-tested"
    assert "Matching fixture execution is recorded" in _forecast_input_boundary_tex(tested)


def test_report_states_immediate_commitments_and_publication_accounting():
    evidence = {"tests": {"status": "executed", "failed": 0, "errors": 0, "test_ids": []},
                "analysis": {}, "pv": {}, "runs": [], "components": [],
                "snapshot": {"generated_at_utc": "2026-10-16T00:00:00Z", "revision": "a" * 40},
                "environment": {"python": "3.12", "packages": {}},
                "tex_validation": {"status": "not-executed"}}
    text = _render(evidence)
    assert "Submission\nfixes the fully executed quantity immediately" in text
    assert "Price publication books its actual\ncash once" in text
    assert "refreshes conditional future price forecasts" in text
    assert "preserving committed quantities and physical commands" in text
    assert "forecast objective values remain separate from realized ledger cash flow" in text
    assert "diagnostic residual solve\nsubmits no additional quantity" in text
    assert "accepted only at publication" not in text
    assert "Nothing was committed" not in text

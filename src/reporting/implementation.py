"""Generate the implementation report from test logs and actual run manifests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

from ..simulation.synthetic import write_json


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _tex(value: Any) -> str:
    """Escape text content; equations and commands use explicit source strings."""
    substitutions = {"\\":r"\textbackslash{}", "&":r"\&", "%":r"\%", "$":r"\$",
        "#":r"\#", "_":r"\_", "{":r"\{", "}":r"\}", "~":r"\textasciitilde{}", "^":r"\textasciicircum{}"}
    return "".join(substitutions.get(character,character) for character in str(value))


def _test_evidence(path: Path) -> dict:
    """Read JUnit counts without mistaking unexecuted tests for successful checks."""
    if not path.is_file():
        return {"status":"not-run","passed":0,"failed":0,"skipped":0,"errors":0,"path":str(path)}
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    counts = {"failed":sum(case.find("failure") is not None for case in cases),
              "errors":sum(case.find("error") is not None for case in cases),
              "skipped":sum(case.find("skipped") is not None for case in cases)}
    counts["passed"] = len(cases)-sum(counts.values())
    return {"status":"executed","total":len(cases),**counts,
        "seconds":sum(float(case.get("time","0")) for case in cases),"path":str(path),
        "test_ids":[f"{case.get('classname')}.{case.get('name')}" for case in cases]}


def _components(tests: dict, runs: list[dict], pv: dict, analysis: dict,
                price_experiment: dict | None = None, fundamental_experiment: dict | None = None,
                linear_price_studies: list[dict] | None = None,
                gaussian_studies: list[dict] | None = None) -> list[dict]:
    """Assign evidence-based component statuses, including explicit omissions."""
    verified = tests["status"]=="executed" and not tests["failed"] and not tests["errors"]
    tested = "implemented-and-tested" if verified else "implemented-not-executed"
    completed_linear = any(study.get("status") == "completed" for study in (linear_price_studies or []))
    completed_gaussian = any(study.get("status") == "completed" for study in (gaussian_studies or []))
    gme_tested = verified and any("test_gme" in name for name in tests.get("test_ids", []))
    forecast_inputs_tested = verified and any("test_forecast_inputs" in name for name in tests.get("test_ids", []))
    definitions = [
        ("Deterministic and stochastic MILP","optimization/model.py",tested,"Licensed analytical/unit tests; ordinary and DST run manifests"),
        ("Nonanticipativity and explicit information tree","domain.py; scenarios.py",tested,"Bid/control owner tests, unresolved descendants, duplicate-observation grouping"),
        ("Masking, interpolation and evidence as-of views","data/policy.py",tested,"Policy fixtures and selected-farm actual slice manifest"),
        ("Read-only farm adapter","data/adapters.py",tested,"Actual Zarr metadata and labeled Parquet joins"),
        ("pvlib AC forecast and calibration interface","forecasting/pv.py",tested,"Fixed/tracker fixtures; actual forecast has flagged invalid intervals"),
        ("Actual full-day PV eligibility","forecasting/pv.py","partial",f"{pv.get('missing_pv_intervals','unknown')} invalid intervals in the executed real-data example; no imputation"),
        ("Manual GME CSV/mapped XML import and cache","data/market_prices.py",tested,"Synthetic canonical CSV import/cache-hit and gaps; mapped XML tests"),
        ("GME API request and zonal-price parser","data/gme.py; data/api.py",
         "implemented-and-tested" if gme_tested else "implemented-not-executed",
         "Documented MGP/MI-A1/2/3 contracts; offline authentication, JSON ZIP, native-product and cache tests; live coverage pending"),
        ("GME authenticated historical acquisition","data/gme.py","blocked",
         "User account access and authenticated download evidence remain pending; no actual GME IDA prices are claimed"),
        ("Prepared forecast-to-optimization adapter","optimization/prepared.py",tested,"Initial pre-DA point planning; complete/causal CSV and missing-input rejection tests"),
        ("Artifact-to-stochastic optimization input boundary","optimization/forecast_inputs.py",
         "implemented-and-tested" if forecast_inputs_tested else "implemented-not-executed",
         "Matching JUnit fixture execution; joint price/PV members, known results, guaranteed commitments, residual grids and IDA3 eligibility; no actual-market economic backtest" if forecast_inputs_tested else
         "Implemented canonical artifact boundary; matching fixture execution is not yet recorded in supplied JUnit; actual-market economic backtest unavailable"),
        ("Terna API fundamentals and OAuth","data/terna.py; data/api.py",tested,"Verified provider schemas; offline OAuth/cache/parser tests; live downloads left to the user"),
        ("ENTSO-E API prices and forecasts","data/entsoe.py; data/api.py",tested,"A44 DA/native product and forecast parser tests; IDA/imbalance not assumed"),
        ("Monthly downloads and offline cache catalog","data/api.py; data/api_catalog.py",tested,"All requested datasets per local month; unchanged daily cache keys; readiness retains no-data gaps"),
        ("Consolidated DuckDB API storage","data/duckdb_store.py",tested,
         "Transactional raw/normalized snapshots, SQL records and retained provenance; new mean-model run ingested DuckDB snapshots"),
        ("Ridge level/spread fits and OOS errors","forecasting/prices.py",tested,"Four synthetic CLI-trained models; causal lag and rolling-origin tests"),
        ("Retrospective DA price experiments","forecasting/price_experiment.py",tested if price_experiment else "implemented-not-executed",
         "Configured retrieved-revision experiment manifest; native durations and chronological holdout" if price_experiment else "No experiment manifest supplied to this report"),
        ("Gaussian/bootstrap residuals and fitted tree interface","forecasting/prices.py; scenarios.py",tested,"Legacy causal fixture interface; retrieved-revision calibration is reported separately"),
        ("Regional forecast feature comparison","forecasting/fundamental_experiment.py",tested if fundamental_experiment else "implemented-not-executed",
         "January retrieved-revision load/solar/wind comparison; complete intervals and shared holdout" if fundamental_experiment else "No regional comparison manifest supplied"),
        ("Rolling linear price selection and forecast artifacts","forecasting/linear_price_study.py; forecasting/price_features.py",tested if completed_linear else "implemented-not-executed",
         "Actual retrieved-revision studies; validation-only specification choice, weekly refits and independent later calibration/holdout" if completed_linear else "No completed rolling linear study supplied"),
        ("Alternative standardized linear price estimators","forecasting/linear_estimators.py",tested,
         "OLS, recency-weighted ridge, robust Huber IRLS, lasso and elastic-net; training-only scalers, saved-predictor and convergence tests"),
        ("Zero-mean Gaussian paths and MILP adapter","forecasting/gaussian_prices.py; forecasting/price_pipeline.py",tested,
         "Separate OOS bias/error covariance, native/DST grids and scenario-contract tests; historical deployment remains ineligible"),
        ("Calibration-selected online Gaussian uncertainty","forecasting/gaussian_adaptive.py; forecasting/gaussian_study.py",
         tested if completed_gaussian else "implemented-not-executed",
         "Frozen inner-calibration policy; only earlier complete mature errors update daily bias/covariance; static comparison and scored coverage" if completed_gaussian else "No completed adaptive Gaussian study supplied"),
        ("Actual-price perfect-information scenario benchmark","forecasting/price_oracle.py",tested,
         "Deterministic actual clearing prices per PV member; latent PV histories, native labels and licensed analytical upper-bound tests"),
        ("Historical learned price/PV scenarios","forecasting/prices.py","blocked","Verified historical vintages, IDA labels, causal multiauction calibration and settlement unavailable"),
        ("Commitment ledger and synthetic replay","simulation/synthetic.py",tested,"Guaranteed full execution; quantities fixed at submission, actual cash booked once at publication; four benchmarks and 92/96/100 product days"),
        ("Historical market backtest","cli.py backtest","blocked","Requires complete causal auction/settlement inputs, eligible PV and confirmed BESS/POI"),
        ("Constant throughput aging","optimization/model.py",tested,"AC charge-plus-discharge convention and carried cumulative cost"),
        ("Xu cycle-depth segmented aging","optimization/degradation.py",tested,"Eq.5/7 segment costs, appendix priority/state carry; analytical/rainflow/solver/replay tests"),
        ("Independent numerical validation","optimization/validation.py",tested,"Recomputed accounting/bounds; intentionally altered solutions rejected"),
    ]
    return [{"component":name,"module":"src/"+module,"status":status,"evidence":evidence}
            for name,module,status,evidence in definitions]


def _price_experiment_tex(experiment: dict, title: str = "Retrieved-revision DA holdout experiment") -> str:
    """Render optional native-resolution holdout evidence without cache path tables."""
    if not experiment:
        return ("No completed retrieved-revision DA experiment manifest was supplied "
                "to this report; no real-data regression result is claimed here.\n")
    lines=[r"\subsection{"+_tex(title)+"}",
        "The configured experiment reads completed user-downloaded DA caches and "
        "preserves hourly and quarter-hour products as separate native-resolution "
        "studies. Training and evaluation follow delivery-day order, with "
        "training-only feature scaling. Prices retain the retrieved source revision; "
        "their historical publication and revision vintages are unverified. These "
        "results therefore measure retrospective price prediction, not executable "
        "historical trading or a causally calibrated scenario distribution.",
        f"Source coverage: {_tex(experiment.get('first_delivery_day','unknown'))} "
        f"through {_tex(experiment.get('last_delivery_day','unknown'))}; "
        f"{_tex(experiment.get('complete_days','unknown'))} complete days and "
        f"{_tex(experiment.get('incomplete_days','unknown'))} incomplete days. "
        "Native durations are never filled with synthetic quarter-hour observations."]
    if experiment.get("feature_coverage"):
        lines.append("This short regional-feature study compares price/calendar lags, added load, "
            "and added load/solar/onshore-wind forecasts. All models use identical complete "
            "training and holdout products. Regional MW forecasts are duration-averaged "
            "within native price products; incomplete coverage remains missing. Forecast "
            "revision and publication availability remain unverified. This is a pipeline "
            "prototype, not evidence of broad forecasting skill or executable trading.")
    rows=[]
    for regime in experiment.get("metrics",[]):
        minutes=regime.get("native_interval_minutes","unknown")
        if regime.get("status")!="evaluated":
            lines.append(f"{_tex(minutes)}-minute study status: "
                         f"{_tex(regime.get('status','not evaluated'))}.")
            continue
        lines.append(f"{_tex(minutes)}-minute study: "
            f"{_tex(regime.get('training_rows','unknown'))} training rows from "
            f"{_tex(regime.get('training_days','unknown'))} days; holdout "
            f"{_tex(regime.get('holdout_start_day','unknown'))} through "
            f"{_tex(regime.get('holdout_end_day','unknown'))}, "
            f"{_tex(regime.get('holdout_days_observed','unknown'))} observed days.")
        for model,score in regime.get("models",{}).items():
            mae,rmse=score.get("mae_eur_mwh"),score.get("rmse_eur_mwh")
            mae_text=f"{mae:.3f}" if isinstance(mae,(int,float)) else "unavailable"
            rmse_text=f"{rmse:.3f}" if isinstance(rmse,(int,float)) else "unavailable"
            rows.append(f"{_tex(minutes)} & {_tex(model.replace('_',' '))} & "
                f"{_tex(score.get('rows','unknown'))} & {mae_text} & {rmse_text} \\\\")
    if experiment.get("feature_coverage"):
        for regime in experiment.get("metrics",[]):
            scores=regime.get("models",{})
            if scores:
                best=min(scores,key=lambda name:scores[name]["mae_eur_mwh"])
                lines.append(f"Lowest MAE in this holdout: {_tex(best.replace('_',' '))}. "
                    "The holdout is too short to choose a generally superior forecasting method.")
    if rows:
        lines.append("\n".join([r"{\small\begin{longtable}{rlrrr}",
            r"\toprule Native min & Predictor & Holdout rows & MAE EUR/MWh & RMSE EUR/MWh \\ \midrule",
            r"\endhead",*rows,r"\bottomrule\end{longtable}}"]))
        lines.append(
            "Previous-day and previous-week prices provide seasonal baselines. "
            "These point-error summaries do not fit multiauction residual vectors, "
            "validate Gaussian noise or supply missing forecast publication vintages.")
    return "\n\n".join(lines)+"\n"


def _study_storage_evidence(directory: Path) -> dict:
    """Summarize saved ingestion provenance without reopening market databases."""
    settings = _read(directory / "settings.json")
    source_path = directory / "source_caches.json"
    sources = _read(source_path)
    snapshots = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("storage"), dict):
                snapshots.append(value)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(sources)
    return {"configured_cache": settings.get("cache_directory"),
        "source_provenance_path": str(source_path),
        "source_provenance_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.is_file() else None,
        "backends": sorted({str(item["storage"].get("backend")) for item in snapshots if item["storage"].get("backend")}),
        "databases": sorted({str(item["storage"].get("database")) for item in snapshots if item["storage"].get("database")}),
        "snapshot_count": len(snapshots),
        "checksum_verified_snapshots": sum(item.get("checksums_verified") is True for item in snapshots)}


def _linear_price_evidence(root: Path, paths: list[str]) -> list[dict]:
    """Read explicitly configured study and nearby uncertainty manifests offline."""
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError("report.linear_price_study_manifests must be a list of manifest paths")
    studies = []
    for name in paths:
        path = root / name
        study = _read(path)
        entry = {**study, "manifest_path": name,
                 "report_label": path.parent.parent.name,
                 "status": study.get("status", "manifest-missing")}
        uncertainty = _read(path.parent / "uncertainty" / "manifest.json")
        entry["uncertainty"] = uncertainty
        artifacts = {}
        for regime in uncertainty.get("regimes", []):
            artifact = regime.get("artifact")
            if artifact:
                artifacts[str(regime["native_interval_minutes"])] = _read(root / artifact)
        entry["uncertainty_artifacts"] = artifacts
        entry["storage_evidence"] = _study_storage_evidence(path.parent)
        studies.append(entry)
    return studies


def _gaussian_study_evidence(root: Path, paths: list[str]) -> list[dict]:
    """Read independent uncertainty runs and their frozen source-study metadata."""
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError("report.gaussian_study_manifests must be a list of manifest paths")
    studies = []
    for name in paths:
        path = root / name
        study = _read(path)
        source_path = root / study["source_study_manifest"] if study.get("source_study_manifest") else None
        frozen_path = path.parent / "source_manifest.json"
        source = _read(frozen_path)
        frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest() if frozen_path.is_file() else None
        entry = {**study, "manifest_path": name, "status": study.get("status", "manifest-missing"),
            "report_label": source_path.parent.parent.name if source_path else path.parent.parent.name,
            "source_study": source,
            "source_manifest_copy_sha256": frozen_hash,
            "source_manifest_copy_verified": bool(frozen_hash and frozen_hash == study.get("source_manifest_sha256")),
            "source_storage_evidence": _study_storage_evidence(source_path.parent) if source_path else {}}
        studies.append(entry)
    return studies


def _price_forward_check_evidence(root: Path, paths: list[str]) -> list[dict]:
    """Read saved forward-check evidence without reopening its market database."""
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError("report.price_forward_check_manifests must be a list of manifest paths")
    checks = []
    for name in paths:
        path = root / name
        result = _read(path)
        protocol_path = path.parent / "protocol.json"
        protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest() if protocol_path.is_file() else None
        checks.append({**result, "manifest_path": name,
            "status": result.get("status", "manifest-missing"),
            "saved_protocol": _read(protocol_path),
            "copied_protocol_sha256": protocol_hash,
            "copied_protocol_verified": bool(protocol_hash and protocol_hash == result.get("protocol_sha256"))})
    return checks


def _selected_price_model_label(candidate: dict) -> str:
    """Name the persisted estimator family rather than its compatible container."""
    if candidate.get("kind") == "baseline":
        return candidate.get("baseline", "baseline").replace("_", " ")
    if not candidate:
        return "unreported"
    structure = str(candidate.get("model_structure", "pooled")).replace("_", " ")
    estimator = candidate.get("estimator", {}) if candidate.get("kind") == "linear" else {"method": "ridge", "alpha": candidate.get("alpha")}
    method = str(estimator.get("method", "unreported"))
    family = {"ols": "OLS", "weighted_ridge": "recency-weighted ridge", "huber": "Huber",
              "elastic_net": "elastic net"}.get(method, method)
    parameters = [] if method == "ols" else [f"alpha {estimator.get('alpha', candidate.get('alpha'))}"]
    if method == "weighted_ridge":
        parameters.append(f"half-life {estimator.get('half_life_days')}d")
    if method == "huber":
        parameters.append(f"epsilon {estimator.get('epsilon')}")
    if method == "elastic_net":
        parameters.append(f"L1 ratio {estimator.get('l1_ratio')}")
    return ", ".join([f"{family} {structure}", str(candidate.get("feature_set", "")),
        *parameters, f"{candidate.get('training_window_days')}d", candidate.get("target", "").replace("_", " ")])


def _linear_price_studies_tex(studies: list[dict]) -> str:
    """Render executed linear means and separately held-out Gaussian reliability."""
    if not studies:
        return ""
    number = lambda value, places=3: f"{value:.{places}f}" if isinstance(value, (int, float)) else "unavailable"
    lines = [r"\subsection{Rolling linear models and Gaussian trajectories}",
        "Each study selects its feature set, fitting method and penalty settings, level or previous-day "
        "difference target, and training window using an earlier validation block. "
        "Previous-day, previous-week and equal day/week forecasts also compete in "
        "selection. The selected specification is then fixed; coefficients refit "
        "weekly using only earlier delivery days and training-only scaling. "
        "A later calibration block estimates bias and error covariance; a delivery "
        "maturity gap separates it from the final chronological holdout. "
        "All candidates score identical complete-feature validation products. "
        "Hourly and quarter-hour regimes remain separate. Structure candidates include "
        "pooled regression and separate coefficients by local delivery hour, with "
        "quarter-hour observations retained inside their hour groups."]
    lines.append("Candidate features include local-clock price lags, complete prior-day "
        "price summaries and profile blocks, rolling histories, calendar cycles and "
        "weekday/hour indicators. Regional studies add load, solar and wind forecasts "
        "or a separate residual-load specification; no realized target-day demand "
        "or generation is substituted for a forecast.")
    lines.append("Additional fitting configurations compare ordinary least squares, "
        "recency-weighted ridge, fixed-scale Huber IRLS, lasso and elastic-net. "
        "Their family and hyperparameters compete on earlier validation MAE only; "
        "final-test results do not choose the estimator. OLS uses rank-safe least "
        "squares; recency weights enter both training scaling and the fitting loss. "
        "Iterative nonconvergence is reported and excludes that candidate, with no "
        "silent ridge substitution. The saved artifacts retain the actual fitting family.")
    lines.append(r"Penalty magnitudes use distinct conventions: ridge minimizes "
        r"$\sum_i w_i e_i^2+\alpha\lVert\beta\rVert_2^2$ (unit weights for ordinary ridge), "
        r"where recency weights sum to the training row count. Lasso and elastic-net use "
        r"$\sum_i e_i^2/(2n)+\alpha r\lVert\beta\rVert_1"
        r"+\alpha(1-r)\lVert\beta\rVert_2^2/2$, with $r=1$ for lasso. "
        "Identical numeric alpha values are therefore not equivalent penalties "
        "between these families. Huber uses a training-residual MAD scale and an "
        "unpenalized intercept; all scalers use the training slice only.")
    point_rows, noise_rows = [], []
    for study in studies:
        label = study.get("report_label", "study")
        if study.get("status") != "completed":
            lines.append(f"Configured {_tex(label)} study status: {_tex(study.get('status'))}; no result is claimed.")
            continue
        lines.append(f"{_tex(label.capitalize())}: {_tex(study.get('first_delivery_day'))} through "
            f"{_tex(study.get('last_delivery_day'))}; {_tex(study.get('rows'))} native products. "
            + _tex(study.get("prior_evaluation_overlap", "This is a retrospective retrieved-revision study.")))
        storage = study.get("storage_evidence", {})
        if "duckdb" in storage.get("backends", []):
            storage_action = ("reused frozen features previously derived from DuckDB snapshots in "
                              if study.get("frozen_feature_source") else "ingested DuckDB snapshots from ")
            lines.append("This mean-model run " + storage_action
                + ", ".join(r"\path{" + str(name) + "}" for name in storage.get("databases", []))
                + f"; {_tex(storage.get('checksum_verified_snapshots'))} of "
                f"{_tex(storage.get('snapshot_count'))} saved source snapshots have verified checksums. "
                "The report reads the run's saved provenance and frozen predictions.")
        if study.get("frozen_feature_source"):
            frozen = study["frozen_feature_source"]
            lines.append("Frozen-feature source: " + r"\path{" + str(frozen.get("manifest", "unreported")) + "}. "
                "The source manifest and feature SHA-256s are retained; this "
                "development comparison reuses its saved inputs without reopening "
                "the current database or adding newly downloaded extension labels.")
        for regime in study.get("regimes", []):
            if regime.get("status") != "evaluated":
                lines.append(f"{_tex(regime.get('native_interval_minutes'))}-minute status: {_tex(regime.get('status'))}.")
                continue
            minutes = regime["native_interval_minutes"]
            candidate = regime["selected_candidate"]
            selected = _selected_price_model_label(candidate)
            baselines = [item for item in study.get("test_comparisons", [])
                         if item.get("native_interval_minutes") == minutes and item.get("model", "").startswith("baseline_")]
            best = min(baselines, key=lambda item: item["mae_eur_mwh"]) if baselines else {}
            best_text = (best.get("model", "unavailable").replace("baseline_", "").replace("_", " ")
                + ": " + number(best.get("mae_eur_mwh")))
            point_rows.append(f"{_tex(label)} & {_tex(minutes)} & {_tex(selected)} & "
                f"{number(regime.get('selection_metrics', {}).get('mae_eur_mwh'))} & "
                f"{number(regime.get('test_metrics', {}).get('mae_eur_mwh'))} & {_tex(best_text)} \\\\")
            split = regime.get("split", {})
            lines.append(f"{_tex(label)}, {_tex(minutes)} minutes: validation "
                f"{_tex(split.get('validation_start_day'))}--{_tex(split.get('validation_end_day'))}; "
                f"calibration {_tex(split.get('calibration_start_day'))}--{_tex(split.get('calibration_end_day'))}; "
                f"final holdout {_tex(split.get('test_start_day'))}--{_tex(split.get('test_end_day'))}.")
        for regime in study.get("uncertainty", {}).get("regimes", []):
            if regime.get("status") != "evaluated":
                continue
            minutes = regime["native_interval_minutes"]
            metrics = regime.get("test_metrics", {})
            artifact = study.get("uncertainty_artifacts", {}).get(str(minutes), {})
            for grid_name, grid in regime.get("calibration_grids", {}).items():
                bias = artifact.get("grids", {}).get(grid_name, {}).get("bias_eur_mwh", [])
                mean_bias = sum(bias) / len(bias) if bias else None
                coverage = metrics.get("coverage", {}).get("95")
                noise_rows.append(f"{_tex(label)} & {_tex(minutes)} & {_tex(grid.get('days'))} & "
                    f"{number(grid.get('mean_error_variance_eur2_mwh2'), 2)} & {number(mean_bias)} & "
                    f"{number(100 * coverage if isinstance(coverage, (int, float)) else None, 1)} & "
                    f"{number(metrics.get('mean_crps_eur_mwh'))} \\\\")
            excluded = regime.get("test_excluded_days", {})
            lines.append(f"Gaussian scoring for {_tex(label)}, {_tex(minutes)} minutes covers "
                f"{_tex(metrics.get('scored_days', 'unknown'))} days and {_tex(metrics.get('scored_rows', 'unknown'))} products; "
                f"{len(excluded)} days are excluded for incomplete or unsupported native/DST grids. "
                "Its MAE/CRPS subset can therefore differ from the point-model comparison.")
            coverage = metrics.get("coverage", {}).get("95")
            if isinstance(coverage, (int, float)) and coverage < .93:
                lines.append(f"The nominal 95 percent intervals cover only {coverage * 100:.1f} percent "
                    "of scored products in this holdout. This undercoverage is evidence "
                    "that the current Gaussian uncertainty is not yet reliable in this regime.")
    if point_rows:
        lines.append("\n".join([r"{\scriptsize\begin{longtable}{p{.15\textwidth}rp{.26\textwidth}rrp{.18\textwidth}}",
            r"\toprule Study & Min & Validation-selected predictor & Val. MAE & Test MAE & Best test baseline \\ \midrule",
            r"\endhead", *point_rows, r"\bottomrule\end{longtable}}",
            "All MAEs are EUR/MWh. The best test baseline is descriptive and does not change the validation-selected forecast."]))
    if noise_rows:
        lines.append("\n".join([r"{\scriptsize\begin{longtable}{p{.15\textwidth}rrrrrr}",
            r"\toprule Study & Min & Cal. days & Error variance & Mean bias & 95\% cover & CRPS \\ \midrule",
            r"\endhead", *noise_rows, r"\bottomrule\end{longtable}}",
            r"Error variance is mean product variance in $(\mathrm{EUR/MWh})^2$; bias and CRPS are EUR/MWh, coverage is percent."]))
    lines.append(r"Sampled prices use $P_s=\widehat\mu+b+\varepsilon_s$, with "
        r"$\varepsilon_s\sim\mathcal N(0,\Sigma)$. Bias $b$ is estimated separately; "
        "the innovation mean is zero, while a finite Monte Carlo sample need not average "
        "to zero. Covariance comes from chronological out-of-sample forecast errors, "
        "not unconditional price variance, and shrinkage preserves temporal dependence. "
        "The Gaussian scenario adapter supplies native quarter-hour DA scenarios to "
        "the MILP information tree; hourly observations are not expanded into "
        "quarter-hour labels or dispatch prices. Current study artifacts remain explicitly "
        "ineligible for operational historical deployment. Missing IDA labels, "
        "verified publication vintages, PV eligibility and settlement semantics "
        "continue to prevent a validated historical trading backtest.")
    lines.append("The actual-price benchmark builder in \\path{forecasting/price_oracle.py} "
        "returns the same scenario-bundle and information-tree contracts with realized "
        "clearing-price curves. Each supplied PV member receives the identical actual "
        "price curve, with no Gaussian draws, while member probabilities are retained. "
        "PV member identities remain latent until completed-interval PV observations "
        "distinguish their histories; configurable telemetry delays can postpone that "
        "revelation. The oracle relaxes price information and is explicitly non-executable. "
        "Missing IDA auction prices are rejected, and native hourly observations are "
        "not interpolated into quarter-hour labels. For multiple PV members, a fair "
        "comparison uses the same physical constraints, member weights, observations "
        "and end-of-interval curtailment information for both forecasting policy and "
        "oracle. Unit and licensed analytical solve tests verify this construction; "
        "they do not establish historical trading profit.")
    return "\n\n".join(lines) + "\n"


def _gaussian_studies_tex(studies: list[dict]) -> str:
    """Distinguish linear point error, corrected Gaussian means and uncertainty."""
    if not studies:
        return ""
    number = lambda value, places=3: f"{value:.{places}f}" if isinstance(value, (int, float)) else "unavailable"
    percent = lambda value: number(100 * value, 2) if isinstance(value, (int, float)) else "unavailable"
    lines = [r"\subsection{Frozen-policy online Gaussian comparison}",
        "The following independent uncertainty runs reuse frozen out-of-sample point "
        "predictions. Inner chronological calibration days select the rolling error "
        "window and covariance shrinkage using the proper score recorded with each policy. "
        "Those policy settings remain fixed through the later test. Each online "
        "forecast may update error bias and covariance only from complete earlier "
        "delivery days that have matured before its issuance cutoff; a D-1 noon "
        "forecast therefore excludes all errors from delivery D-1. The static "
        "comparison keeps its original calibration bias and covariance fixed. "
        "Final-test outcomes never select policy hyperparameters. Earlier mature "
        "test errors do enter the explicitly online updates.",
        r"The linear point forecast is $\widehat\mu$; the Gaussian mean is "
        r"$\widehat\mu+b$ and innovations remain $\varepsilon\sim\mathcal N(0,\Sigma)$. "
        "Point MAE below measures the unchanged linear model. Mean MAE measures "
        "the separate bias-corrected Gaussian mean, so the two need not agree. "
        "MAE and CRPS use EUR/MWh; interval and consecutive-product ramp coverage "
        "are percentages with a nominal target of 95."]
    for study in studies:
        label = study.get("report_label", "study")
        if study.get("status") != "completed":
            lines.append(f"Configured {_tex(label)} Gaussian study status: {_tex(study.get('status'))}; no result is claimed.")
            continue
        lines.append("Source mean study: " + _tex(label.replace("_", " ")) + ". "
                     + _tex(study.get("prior_evaluation_overlap", "This is retrospective evaluation.")))
        if study.get("source_manifest_copy_verified"):
            lines.append("The saved source-manifest copy matches the recorded SHA-256. "
                "Frozen calibration and test prediction files accompany the uncertainty run.")
        else:
            lines.append("The report has not verified a matching frozen source-manifest hash for this entry.")
        storage = study.get("source_storage_evidence", {})
        if "duckdb" in storage.get("backends", []):
            lines.append("Source ingestion provenance identifies DuckDB "
                + ", ".join(r"\path{" + str(name) + "}" for name in storage.get("databases", []))
                + f" and {_tex(storage.get('checksum_verified_snapshots'))} checksum-verified "
                "snapshots. Gaussian fitting reads the point study's frozen prediction files.")
        rows = []
        for regime in study.get("regimes", []):
            if regime.get("status") != "evaluated":
                lines.append(f"{_tex(regime.get('native_interval_minutes'))}-minute uncertainty status: "
                             f"{_tex(regime.get('status'))}.")
                continue
            minutes = regime["native_interval_minutes"]
            for mode, key in (("Static", "static_metrics"), ("Online", "adaptive_metrics")):
                metrics = regime.get(key, {})
                rows.append(f"{_tex(minutes)} & {mode} & "
                    f"{number(metrics.get('uncorrected_point_mae_eur_mwh'))} & "
                    f"{number(metrics.get('mae_eur_mwh'))} & "
                    f"{percent(metrics.get('coverage', {}).get('95'))} & "
                    f"{number(metrics.get('mean_crps_eur_mwh'))} & "
                    f"{percent(metrics.get('consecutive_product_ramps', {}).get('coverage_95'))} \\\\")
            policy = regime.get("selected_policy", {})
            adaptive = regime.get("adaptive_metrics", {})
            selection_metric = {"mean_joint_nll_per_product": "joint Gaussian negative log likelihood per product",
                                "mean_crps_eur_mwh": "marginal CRPS"}.get(
                                    policy.get("selection_metric"), str(policy.get("selection_metric", "unreported")))
            lines.append(f"{_tex(minutes)} minutes: selected {_tex(policy.get('window_days'))}-day "
                f"window and shrinkage {_tex(policy.get('covariance_shrinkage'))} using {_tex(selection_metric)}; "
                f"policy-selection information ends {_tex(policy.get('selection_max_product_end'))}. "
                f"Online scoring covers {_tex(adaptive.get('scored_days'))} days and "
                f"{_tex(adaptive.get('scored_rows'))} native products. "
                f"{_tex(regime.get('sample_count'))} seeded paths are saved for "
                f"{_tex(regime.get('sample_day'))}.")
            point_mae, mean_mae = adaptive.get("uncorrected_point_mae_eur_mwh"), adaptive.get("mae_eur_mwh")
            if isinstance(point_mae, (int, float)) and isinstance(mean_mae, (int, float)) and mean_mae > point_mae:
                lines.append(f"For {_tex(minutes)} minutes, rolling bias correction increases MAE "
                    f"from {point_mae:.3f} to {mean_mae:.3f}. Its uncertainty scores do not "
                    "establish an improvement in the linear point forecast's accuracy.")
            coverage = adaptive.get("coverage", {}).get("95")
            if isinstance(coverage, (int, float)) and coverage < .95:
                lines.append(f"The {_tex(minutes)}-minute online intervals cover {coverage * 100:.1f} "
                    "percent against a nominal 95 percent. This remaining undercoverage "
                    "does not establish reliable tail probabilities or operational readiness.")
        if rows:
            lines.append("\n".join([r"{\scriptsize\begin{longtable}{rlrrrrr}",
                r"\toprule Min & Error policy & Point MAE & Mean MAE & 95\% cover & CRPS & Ramp 95\% \\ \midrule",
                r"\endhead", *rows, r"\bottomrule\end{longtable}}"]))
    lines.append("These are retrospective latest-retrieved revisions with unverified "
        "historical publication clocks. Operational eligibility remains false. The "
        "comparison does not provide missing IDA observations, causal forecast "
        "vintages, verified settlement inputs or historical trading-profit evidence.")
    return "\n\n".join(lines) + "\n"


def _price_forward_checks_tex(checks: list[dict]) -> str:
    """Render previously uninspected historical periods under frozen settings."""
    if not checks:
        return ""
    number = lambda value, places=3: f"{value:.{places}f}" if isinstance(value, (int, float)) else "unavailable"
    percent = lambda value: number(value * 100, 2) if isinstance(value, (int, float)) else "unavailable"
    lines = [r"\subsection{Previously uninspected historical extension}",
        "A saved protocol fixes the selected mean specification, weekly coefficient "
        "refit schedule and Gaussian policy before examining the additional historical "
        "period. These dates are a previously uninspected extension of the research "
        "dataset, not future market outcomes observed prospectively. The check reads "
        "checksum-verified DuckDB snapshots, preserves the earlier feature history, "
        "and replays the original refit anchor before scoring new dates. It performs "
        "no model-family, feature, penalty or covariance-policy selection using the "
        "extension outcomes. Earlier eligible labels may refit coefficients; only "
        "complete earlier mature errors may update online uncertainty."]
    for check in checks:
        status = check.get("status")
        if status != "completed":
            if status == "waiting_for_data":
                lines.append(f"The configured extension check is waiting for data: "
                    f"{len(check.get('missing_days', []))} missing or incomplete days out of "
                    f"{_tex(check.get('requested_days', 'unknown'))}. "
                    "No partial-period accuracy or reliability result is claimed.")
            else:
                lines.append(f"Configured extension-check status: {_tex(status)}; no result is claimed.")
            continue
        first, stop = check.get("new_test_start"), check.get("new_test_end_exclusive")
        try:
            last = (date.fromisoformat(str(stop)) - timedelta(days=1)).isoformat()
        except ValueError:
            last = "unreported"
        lines.append(f"Scored extension: {_tex(first)} through {_tex(last)} "
            f"(exclusive end {_tex(stop)}), following previously inspected "
            f"{_tex(check.get('maximum_previously_seen_delivery_day'))}; "
            f"{_tex(check.get('fresh_days'))} complete days and {_tex(check.get('fresh_rows'))} "
            "native quarter-hour products. Database: " + r"\path{" + str(check.get("database", "unreported")) + "}; "
            f"{_tex(check.get('new_snapshot_count'))} new source snapshots.")
        candidate = check.get("selected_mean_specification", {})
        policy = check.get("selected_gaussian_policy", {})
        lines.append("Frozen point specification: " + _tex(_selected_price_model_label(candidate))
            + f". Gaussian window {_tex(policy.get('window_days', 'unreported'))} days, "
            f"shrinkage {_tex(policy.get('covariance_shrinkage', 'unreported'))}. "
            "Saved protocol creation (UTC): "
            + _tex(check.get("saved_protocol", {}).get("created_at_utc", "unreported")) + ".")
        audit = check.get("overlap_verification", {})
        lines.append("Saved-history immutability flag: "
            + _tex(check.get("source_historical_features_unchanged", "unreported"))
            + f". The overlap replay compares {_tex(audit.get('rows'))} prior products; "
            f"maximum forecast difference {number(audit.get('maximum_prediction_difference_eur_mwh'), 9)} EUR/MWh, "
            f"with tolerance {number(audit.get('comparison_tolerance_eur_mwh'), 9)}. "
            "Identical cutoff and training-day audit: " + _tex(audit.get("cutoffs_and_training_days_identical", "unreported"))
            + ". Copied protocol SHA-256 verified: " + _tex(check.get("copied_protocol_verified", False))
            + "; ancillary source hashes pinned in the protocol: "
            + _tex(check.get("source_ancillary_hashes_pinned_in_protocol", "unreported")) + ".")
        if check.get("test_labels_used_for_selection") is not False:
            lines.append("The saved manifest does not establish that extension labels were excluded "
                         "from selection; this entry cannot certify the frozen-selection claim.")
        point_rows = []
        for name, label in (("selected", "Frozen selected model"), ("previous_day", "Previous day"),
                            ("previous_week", "Previous week"), ("day_week_average", "Equal day/week average")):
            metrics = check.get("point_metrics", {}).get(name)
            if metrics is None:
                continue
            skill = "---" if name == "selected" else percent(metrics.get("selected_mae_skill"))
            point_rows.append(f"{_tex(label)} & {number(metrics.get('mae_eur_mwh'))} & "
                f"{number(metrics.get('rmse_eur_mwh'))} & {skill} \\\\")
        if point_rows:
            lines.append("\n".join([r"{\small\begin{longtable}{lrrr}",
                r"\toprule Point predictor & MAE & RMSE & Selected MAE skill (\%) \\ \midrule",
                r"\endhead", *point_rows, r"\bottomrule\end{longtable}}",
                "Point MAE/RMSE are EUR/MWh. Skill is one minus selected-model MAE "
                "divided by the named baseline's MAE; positive values favor the selected model."]))
        probability_rows = []
        for name, label in (("static_metrics", "Static"), ("adaptive_metrics", "Online frozen policy")):
            metrics = check.get(name, {})
            if not metrics:
                continue
            probability_rows.append(f"{label} & {number(metrics.get('mae_eur_mwh'))} & "
                f"{percent(metrics.get('coverage', {}).get('95'))} & {number(metrics.get('mean_crps_eur_mwh'))} & "
                f"{percent(metrics.get('consecutive_product_ramps', {}).get('coverage_95'))} & "
                f"{number(metrics.get('mean_interval_width_eur_mwh', {}).get('95'))} \\\\")
        if probability_rows:
            lines.append("\n".join([r"{\small\begin{longtable}{lrrrrr}",
                r"\toprule Gaussian policy & Mean MAE & 95\% cover & CRPS & Ramp 95\% & 95\% width \\ \midrule",
                r"\endhead", *probability_rows, r"\bottomrule\end{longtable}}",
                "Mean MAE includes the separately estimated Gaussian bias correction; "
                "it is distinct from linear point MAE. The static distribution retains "
                "its original calibration parameters; the online distribution updates "
                "earlier mature errors under the already-fixed policy. Interval width "
                "is EUR/MWh: closer nominal coverage must be assessed together with "
                "width and CRPS, rather than interpreted alone as better forecasts."]))
        coverage = check.get("adaptive_metrics", {}).get("coverage", {}).get("95")
        if isinstance(coverage, (int, float)) and coverage < .95:
            lines.append(f"Online 95 percent intervals cover {coverage * 100:.1f} percent "
                "in this extension, below nominal coverage. The new period does not "
                "establish reliable tail calibration.")
    lines.append("Historical publication and revision vintages remain unverified, "
        "and operational eligibility remains false. This extension tests statistical "
        "prediction under a frozen protocol; it does not establish a causal historical "
        "trading backtest or actual strategy profit.")
    return "\n\n".join(lines) + "\n"


def _forecast_input_boundary_tex(tests: dict) -> str:
    """State the new executable input contract separately from market evidence."""
    verified = (tests.get("status") == "executed" and not tests.get("failed") and not tests.get("errors")
                and any("test_forecast_inputs" in name for name in tests.get("test_ids", [])))
    execution = ("Matching fixture execution is recorded in the supplied passing JUnit suite. " if verified else
        "Matching fixture execution is not yet recorded in the supplied JUnit evidence. ")
    return (r"\subsection{Artifact-to-stochastic optimization input boundary}" + "\n"
        r"The executable \path{optimization/forecast_inputs.py} interface provides "
        r"\texttt{build\_forecast\_inputs} for in-memory canonical tables and "
        r"\texttt{load\_forecast\_inputs} for saved forecast/model artifacts. "
        r"The \texttt{optimize} command routes an explicit "
        r"\texttt{optimization.forecast\_inputs} configuration to the same "
        r"\texttt{OptimizationInput} contract and MILP builder. "
        "This boundary accepts explicitly weighted PV members, supplied price means "
        "or fitted DA/conditional IDA spread artifacts, with Gaussian calibration when "
        "uncertainty is requested. It forms an independent price-times-PV product "
        "measure using common price draws across PV members; no calibrated dependence "
        "between weather/PV and market prices is claimed. Member identities remain "
        "latent until completed PV observations distinguish histories.\n\n"
        "Every submitted quantity executes in full under the price-taking assumption "
        "and becomes a fixed commitment immediately. A committed quantity awaiting "
        "price publication remains fixed in every residual scenario. Recorded known "
        "clearing curves remain fixed; publication books quantity times actual clearing "
        "price once in the immutable priced ledger. The legacy accepted-trade field "
        "names denote already executed, priced quantities. Prior committed cash is "
        "accounted for externally; the optimization objective measures incremental "
        "new trading and dispatch profit. Residual reoptimization uses an exact remaining product "
        "grid and an explicit boundary physical state. IDA3 data and calibration "
        "cover afternoon products only; coefficients outside its false trading mask "
        "are inert placeholders, not observed clearing prices. Source, fit and "
        "availability cutoffs are checked, and retrospective artifacts require an "
        "explicit study opt-in. " + execution
        + r"The schema and assumptions are documented in \path{forecast_optimization_inputs.md}. "
        "These contracts do not supply absent GME labels, verified historical "
        "publication vintages, settlement economics or operational farm forecasts. "
        "No actual-data economic optimization backtest is established by the fixtures.\n")


def _render(evidence: dict) -> str:
    """Write an English technical article that matches the current executable scope."""
    tests = evidence["tests"]
    analysis,pv = evidence["analysis"],evidence["pv"]
    energy = next(iter(analysis.get("daily_ac_energy",[])),{})
    evaluation = next(iter(pv.get("ex_post_point_evaluation",{}).get("members",[])),{}).get("policy_treated_AC_target",{})
    runs = evidence["runs"]
    rows = "\n".join(f"{_tex(item['component'])} "+r"\newline{\scriptsize\path{"+item["module"]+"}} & "
                     +f"{_tex(item['status'])} & {_tex(item['evidence'])} \\\\"
                     for item in evidence["components"])
    results = "\n".join(f"{_tex(run['delivery_day'])} & {run['intervals']} & "
        +_tex(item['strategy']+(' (Xu)' if run.get('config',{}).get('degradation',{}).get('model')=='xu_segmented' else ''))+" & "
        f"{item['realized_operating_profit_eur']:.4f} & {item['runtime_seconds']:.3f} & {item['max_validation_error']:.2e} \\\\"
        for run in runs for item in run["strategies"])
    versions = ", ".join(f"{_tex(name)} {_tex(value)}" for name,value in evidence["environment"]["packages"].items())
    first = next((run for run in runs if len(run["strategies"])==4),runs[0] if runs else {})
    stochastic = next((item for item in first.get("strategies",[]) if item["strategy"]=="stochastic_integrated"),{})
    solve = stochastic.get("first_solve",{})
    figure = (first.get("output_dir","")+"/stochastic_integrated/dispatch.png")
    artifact_rows = "\n".join(r"\item \path{"+run["manifest_path"]+"}" for run in runs)
    price_experiment_tex=_price_experiment_tex(evidence.get("price_experiment",{}))
    fundamental_experiment_tex=(_price_experiment_tex(evidence["fundamental_experiment"],"Regional forecast feature comparison")
        if evidence.get("fundamental_experiment") else "")
    linear_price_studies_tex = _linear_price_studies_tex(evidence.get("linear_price_studies", []))
    gaussian_studies_tex = _gaussian_studies_tex(evidence.get("gaussian_studies", []))
    price_forward_checks_tex = _price_forward_checks_tex(evidence.get("price_forward_checks", []))
    forecast_input_boundary_tex = _forecast_input_boundary_tex(evidence["tests"])
    return r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage{amsmath,booktabs,longtable,array,graphicx}
\usepackage[T1]{fontenc}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue]{hyperref}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{5pt}
\setlength{\parindent}{0pt}
\title{PV+BESS Auction Trading: Executed Implementation Status}
\author{Repository implementation evidence}
\date{"""+_tex(evidence["snapshot"]["generated_at_utc"])+r"""}
\begin{document}
\maketitle
\section{Snapshot and scope}
This report describes delivered code and observed executions, with machine-readable
evidence in \path{implementation_status.json}. The repository is
\path{/home/rghetti/bess_pv_trading}; revision \texttt{"""+_tex(evidence["snapshot"]["revision"][:12])+r"""}
identifies the base revision at report generation; the accompanying JSON records
the working-tree state used for this snapshot. The original audit modules and raw
stores remain intact. A compatibility
bridge exposes their existing imports alongside the new \path{src/} application.

The solved system is one AC PV farm with an AC-coupled lithium-ion battery and one
POI, using guaranteed fixed-quantity DA/IDA1/IDA2/IDA3 energy adjustments. Every
submitted quantity executes at its auction clearing price; that price becomes
known at publication. Each model
contains all represented scenarios with node-owned decisions. No portfolio POI,
continuous trading, ancillary services or price impact is modeled. Each run is an
independent single-delivery-day study. Current farm weather has one deterministic
member; multiple prices constitute \emph{stochastic-price / deterministic-PV}.

Python version: \texttt{"""+_tex(evidence["environment"]["python"].splitlines()[0])+r"""}.
Imported library versions used: """+versions+r""". Scikit-learn was unavailable; Ridge uses
NumPy linear algebra. No packages, environments or licenses were changed. A tiny
Gurobi model failed with sandbox HostID masking and succeeded outside the sandbox;
licensed solver tests and experiments used the latter execution context.

\section{Component status and evidence}
Module paths and exact test identities are recorded in the JSON evidence. The
status below distinguishes tested software from unavailable historical inputs.
{\small
\begin{longtable}{p{0.29\textwidth}p{0.20\textwidth}p{0.43\textwidth}}
\toprule Component & Status & Executed evidence or blocker \\ \midrule
\endhead
"""+rows+r"""
\bottomrule
\end{longtable}}

\section{Dataset contract and derived observations}
The verified files are
\path{data/farms/pv_farms_timeseries_20250101_20260831.zarr} and the matching
\path{pv_farms_timeseries_20250101_20260831.parquet}. Metadata inspection observed
four farms, 58,368 historical timestamps per farm, 2,440 forecast cycles, and 49
hourly leads. Observations cover 1 January 2025 through 31 August 2026 in UTC;
forecast boundary coverage differs. Leads are timedeltas from zero to 48 hours.
Installed DC capacities are 953.28, 1524.60, 4160 and 6638 kW for IDs 1, 345, 901
and 910 respectively. Each is a separate plant. The measured \texttt{power}
variable is already AC kW, regardless of obsolete DC column labels retained inside
the unchanged audit. These meanings follow the user's corrected contract and
local metadata; public documentation cannot verify private source provenance.

Power has dimensions \texttt{(time,farm\_id)}; satellite GHI and diffuse fraction
use \texttt{(farm\_id,time)}. Forecasts use
\texttt{(farm\_id,cycle\_time,lead\_time)}. Named transposes and farm-label joins
prevent positional assumptions. Overlapping vintages are not ensemble members.
Hourly shortwave radiation is forecast GHI, while the separate quarter-hour GHI
is historical satellite data. The direct-radiation-as-BHI meaning is a working
assumption. \texttt{fraction\_diffuse} remains disabled as a model input.

The versioned policy first detects native nonzero anchor-matched power runs of
at least four samples, using assumed tolerances 0.01 kW absolute and 0.0001
relative, and zeroes the entire qualifying run. Zero, missing value or timestamp
gap breaks a run. It then zeroes positive production from 21:30 inclusive through
05:00 exclusive in Europe/Rome, floors only allowlisted negative forecast
magnitudes, and linearly interpolates internally bracketed missing historical
power/GHI. The default has no maximum interior gap. It neither extrapolates nor
fills forecast gaps, and it reapplies the production night window. Temperatures,
signed wind and prices retain valid negative values. Original values, reason
flags, run/gap lengths, evidence times and energy changes are exported.

The executed farm-1 UTC slice on 1 October 2025 contains """+_tex(analysis.get("rows","not executed"))+r"""
long-format rows. Adjusted rows: """+_tex(analysis.get("adjusted_rows","unknown"))+r""";
remaining missing: """+_tex(analysis.get("remaining_missing","unknown"))+r""".
Raw/treated finite AC energies were """+_tex(energy.get("raw_finite_energy_mwh","unknown"))+r""" and
"""+_tex(energy.get("treated_finite_energy_mwh","unknown"))+r""" MWh, with
"""+_tex(energy.get("energy_change_mwh","unknown"))+r""" MWh change.
This slice interpolated GHI while AC production energy was unchanged. Detailed
counts and daily raw/treated energies are retained in the analysis manifest.
Policy treatment is an analysis choice, not independent verification of faults or
unconstrained PV availability. Real clipping plateaus may be removed. No upstream
review file was configured; a missing review file is not a completed review.

\section{Time semantics and causal information}
Each historical label starts a 15-minute interval; its final average is available
at interval end plus configured telemetry latency. Retroactive plateau evidence
for the first three samples arrives only at the fourth sample's availability;
later flags arrive no earlier than their sample. A two-sided interpolation also
waits for its right endpoint and applicable mask evidence. As-of observation
views reconstruct what was available at a cutoff; finalized treated observations
are used only for labeled ex-post accounting.

Forecast availability defaults to cycle plus four hours. At the executed cutoff
"""+_tex(pv.get("provenance",{}).get("decision_cutoff_utc","not executed"))+r""",
cycle """+_tex(pv.get("forecast_vintage_utc","unknown"))+r""" was selected, available at
"""+_tex(pv.get("available_at_utc","unknown"))+r""". The 06:00 cycle was not yet
available. End-labeled radiation at 12:00 describes 11:00--12:00; uniform
redistribution conserves its irradiation over the four contained quarters.
Instantaneous weather uses within-vintage interpolation at interval midpoints.
No structural forecast gap is replaced with measurements or another future cycle.

The calendar uses Europe/Rome local gates: DA D-1 12:00, IDA1 D-1 15:00, IDA2
D-1 22:00, IDA3 D 10:00, with a five-minute computation buffer. IDA3 products
start at noon. IDA results use documented publication deadlines; DA 13:30 is a
conservative study assumption because the rule states an earliest final time,
not a guaranteed deadline. Recorded timestamps override schedules and cancellation
can remove an auction. Rule revisions 13 and 14 are dated separately
\cite{gme_dtf03_rev13,gme_dtf03_rev14}. Native quarter-hour MGP delivery begins 1 October 2025
\cite{gme_quarter_hour}. Earlier hourly DA products remain native in retrospective
price analysis; mapping them to quarter-hour dispatch is not implemented. Local days
have 92, 96 or 100 products, all represented internally with aware UTC clocks.

\section{Weather-to-AC implementation}
The installed pvlib implementation uses solar position, total plane-of-array
irradiance, \texttt{irradiance.dni}, \texttt{tracking.singleaxis}, Faiman
temperature and \texttt{pvsystem.pvwatts\_dc} \cite{pvlib_dni,pvlib_tracking,pvlib_pvwatts}.
Wind is the magnitude of signed 10-m components, explicitly a module-wind proxy.
The limited-metadata DC model uses nameplate watts and a configured temperature
coefficient, returning diagnostic MW. The example applies a separate assumed,
unfitted PV-output factor 0.96. An AC calibration helper enforces chronological
availability; no real calibration fit was claimed. Nameplate is not an automatic
hard output cap.

Milan coordinates are acknowledged proxy geometry, not fault evidence. Fixed
surfaces retain their own tilt/azimuth; single-axis farms use separately assumed
axis tilt/azimuth, rotation, GCR and backtracking. Farm 901 explicitly omits rear
gain under a front-side-only approximation. Section 8.2 of the user's prompt
confirms west=-90 degrees; the configurable east-positive conversion and its
alternative are tested. No undocumented module technology is inferred.

The BHI working interpretation computes DHI=GHI-BHI and converts to DNI with
representative quarter-hour zenith, suppressing low-sun direct irradiance and
flagging an explicit DNI cap. This is an interval-mean geometry approximation.
Contradictory BHI exceeding GHI invalidates intervals under the example policy.
The actual forecast produced """+_tex(pv.get("intervals","unknown"))+r""" products with
"""+_tex(pv.get("missing_pv_intervals","unknown"))+r""" invalid intervals. Its finite-only
energy is not a complete-day energy estimate. Ex-post evaluation uses exactly
matched measured/policy-treated AC intervals, with coverage and exclusions.
On """+_tex(evaluation.get("finite_eligible_pairs","unknown"))+r""" eligible pairs, AC MAE and RMSE were
"""+_tex(evaluation.get("mae_mw","unknown"))+r""" and
"""+_tex(evaluation.get("rmse_mw","unknown"))+r""" MW respectively. These are an unfitted
one-vintage comparison, not held-out calibrated skill. The separately configured
radiation sensitivity completed 96 products while retaining the contradiction flags;
its detailed numerical metrics are included in JSON evidence.

Measured historical AC power is divided by 1000 to MW without another conversion.
The separately configured battery converter uses 0.97 per direction, giving
0.9409 converter-only round-trip efficiency. Direct PV exports bypass it.
The final positive AC forecast receives the same requested production night mask.

\section{Prices, residuals and information trees}
Canonical CSV and explicitly mapped XML import preserve negative prices, reject
duplicates/nonfinite labels and report gaps. Cache identity contains provider,
dataset, zone, product, dates, vintage/revision and schema. Retrieval time does
not establish historical publication time. Verified ENTSO-E and Terna API adapters
are implemented and tested with injected offline transports. Authenticated
downloads are user-run; this report reads completed local cache artifacts and
does not make authenticated requests or infer account permissions. ENTSO-E
supports native DA prices and load/generation/solar/wind
forecasts \cite{entsoe_api_reference,entsoe_extraction_guide}. Terna supports load observations/forecasts, MGP load forecasts, MSD wind
forecasts and national renewable actuals, preserving MW versus GWh
\cite{terna_access_token,terna_total_load,terna_load_forecast,terna_wind_forecast,terna_renewables}.
IDA prices are not silently substituted. Terna fee-route ambiguity and the
historical settlement adapter remain unresolved.

The GME API adapter implements authenticated PublicMarketResults requests for
MGP and MI-A1/2/3 zonal prices, bounded JSON ZIP decoding and explicit native
products. Documented-format offline fixtures exercise authentication, parser,
cache and failure contracts. GME account access and user-authenticated historical
downloads are still pending in the supplied evidence. No actual GME IDA price
coverage or historical publication vintages are claimed.

The downloader plans daily Italy-local windows, reads caches before credentials,
retries bounded transient failures, and reuses identical source requests. The
default \texttt{month\_first} order completes all configured datasets for one local
month before advancing; \texttt{dataset\_first} remains available. Scheduling is
outside cache identity, so completed price and forecast days survive a restart.
A process already running retains its original plan until it is restarted.
The offline \texttt{api-cache-status} command reports per-month and per-dataset
coverage while acquisition continues. A ready month requires every configured
daily payload; no-data receipts remain gaps. Readiness establishes payload
presence, not historical information availability or experimental suitability.
The cache catalog excludes unfinished writes and archived revisions, and model
ingestion verifies raw and normalized checksums. Hidden
terminal credentials are resolved once per process; token refresh uses the provider
expiry. Current API configurations store requests in \path{data/api/market.duckdb}.
The transactional store retains exact raw and normalized Parquet bytes, SQL
records, request identities, retrieval clocks and snapshot provenance. Provider,
dataset, zone and local day remain reusable metadata keys; existing folder caches
can be migrated without another provider download. Experiment tables and plots
remain grouped by run. Cache reads verify raw and normalized checksums;
explicit refresh retains earlier snapshots. No-data receipts remain gaps and
have a configurable lifetime. Historical publication and unverified Terna interval
boundaries remain missing. Request schemas, credentials, subscriptions and actual
coverage are recorded from the user's resulting caches when available.

Ridge uses training-only scaling, calendar cycles, local-clock one-day/seven-day
lags and eligible prior auction prices. IDAs predict spreads from the preceding
auction; lag joins preserve DST fold rather than shifting 96 rows. This causal
interface omits regional fundamentals and local-to-regional proxies; the separate
retrospective prototype below evaluates regional forecasts. Expanding chronological
fits produce out-of-sample errors. Residual interfaces support shrunk Gaussian
covariance, a labeled independent Gaussian, and whole-vector bootstrap; differently
sized DST trajectories require separate calibration. The executed fixture has
288 synthetic records over 18 days and four selected products, not complete
historical days. Four synthetic Ridge models each fitted 44 rows and retained
28 OOS residual rows; gaps remain explicitly reported.

"""+price_experiment_tex+fundamental_experiment_tex+linear_price_studies_tex+gaussian_studies_tex+price_forward_checks_tex+r"""

The finite tree branches at results publication, keeps later descendants unresolved,
and groups identical sibling observed vectors to avoid leaking latent labels.
DA innovations belong to its level equation; later innovations belong to
conditional spread equations. No second cross-auction residual is added. Known
results remain fixed. Current weather is frozen until a new publication; no
probabilistic forecast-revision model is supplied. The synthetic replay publishes
one perfect deterministic AC forecast; it is not a weather-skill experiment.
The earlier prepared-input adapter also connects complete exported AC forecasts and supplied
point-price forecasts to the same builder. It checks forecast source, fit and
availability cutoffs, one farm/member, review exclusions, equipment and an explicit
settlement study assumption. Its contract was tested on fixtures; no real price feed
was supplied to this adapter. This route supports initial pre-DA planning, not historical replay.
Imbalance prices use an explicit synthetic point estimate. Solver-independent
scenario contracts accept joint PV arrays. The new artifact boundary below accepts
explicit weighted PV members; actual meteorological ensemble acquisition and
calibration remain unexecuted.

"""+forecast_input_boundary_tex+r"""

\section{Formulation matching the code}
For leaf $s$ with mass $p_s$, delivery interval $t$ with duration $h_t$ hours,
control owner $n(t,s)$ and auction owner $m(a,s)$, charge $c$, discharge $d$,
curtailment $u$ and signed grid power $x$ use MW. Internal energy $e$ and financial
trades $v$ use MWh. The battery's charge mode $z$ is binary and shared by node:
\begin{align}
x_{ts}&=g_{ts}-u_{ts}+d_{t,n(t,s)}-c_{t,n(t,s)},\\
e_{t+1,s}&=r_t e_{ts}+\eta_c c_{t,n(t,s)}h_t-d_{t,n(t,s)}h_t/\eta_d,\\
0\le c_{tn}&\le\bar c z_{tn},\qquad 0\le d_{tn}\le\bar d(1-z_{tn}),\\
\underline e\le e_{ts}&\le\bar e,\quad -\bar x^{in}\le x_{ts}\le\bar x^{out},\quad 0\le u_{ts}\le g_{ts}.
\end{align}
Efficiencies are converter times cell efficiencies, or a validated combined
convention; losses are never applied twice. Retention is $(1-\ell)^{h_t}$.
Initial state timestamps equal the residual grid start; terminal energy is a
fixed equality in every leaf. Power/energy reachability is prechecked; remaining
infeasibility is reported with optional LP/IIS diagnostics.

Financial positions and imbalances obey
\begin{equation}
q_{ts}=q^{priced}_t+q^{unpriced}_t+\sum_a v_{a,m(a,s),t},\qquad
b_{ts}=x_{ts}h_t-q_{ts}.
\end{equation}
Only eligible open auctions receive variables. Both per-trade and cumulative
commercial positions are bounded. Submitted quantities execute in full and remain
fixed while awaiting price publication. There is no acceptance uncertainty.
Their unknown cash flow is a scenario constant excluded from residual optimization
and identified in diagnostics. On publication, quantity times the actual clearing
price is booked once in the priced ledger. The internal pending and accepted field
names distinguish unpriced and priced commitments, respectively. Opposite financial trades incur no battery
efficiency loss.

The exact signed settlement uses $b=b^+-b^-$ and a bounded binary direction split,
so simultaneous positive and negative components cannot create fictitious profit:
\begin{align}
R_s^{imb}&=\sum_t(\pi^+_{ts}b^+_{ts}-\pi^-_{ts}b^-_{ts}),\\
C_s^{imb}&=\rho\sum_t(b^+_{ts}+b^-_{ts}),\\
C_s^{age}&=k\sum_t(c_{t,n(t,s)}+d_{t,n(t,s)})h_t,\\
\max\quad&\sum_s p_s\left[\sum_{a,t}\pi^a_{ts}v_{a,m(a,s),t}
+R_s^{imb}-C_s^{imb}-C_s^{age}-C_s^{other}\right].
\end{align}
The fixed absolute-imbalance penalty is the requested study deterrent; it does
not establish historical law or guarantee compliance. Synthetic imports have no
extra grid fee; excluded taxes/brokerage remain explicit. Throughput is AC charge
plus discharge, with cumulative accounting carried across events. The optional
\texttt{xu\_segmented} model implements Xu's Eq. (5), (7), (8)--(15) and appendix
Eq. (37)--(40) \cite{xu2017aging}. It uses a convex depth stress function and equal
internal-energy segments:
\begin{align}
\Phi(\delta)&=\alpha\delta^\beta,\quad \bar e_j=E/J,\\
k_j&=\frac{RJ}{\eta_d E}\left[\Phi(j/J)-\Phi((j-1)/J)\right],\\
C_s^{age,Xu}&=\sum_{t,j}h_t k_j d^j_{t,n(t,s)},\\
e^j_{t+1,s}&=e^j_{ts}+h_t\left(\eta_c c^j_{t,n(t,s)}-d^j_{t,n(t,s)}/\eta_d\right).
\end{align}
Segment powers sum to the physical battery command and segment energies sum to
internal energy. Explicit priority binaries select the appendix's shallow-first
allocation for both charging and discharging; these are cycle-memory segments,
not fixed absolute-SoC buckets. Charge-only half cycles have zero aging charge;
discharge half cycles incur the full same-depth cost. No half factor is applied.
Initial occupancy is assigned once, and the segment vector is carried through
committed commands, execution and residual optimization. Independent numerical
replay and rainflow comparisons check the segment policy and cost.

The demonstrated Xu configuration uses eight segments, a separately assumed
EUR300,000 pack replacement cost, and the paper's illustrative
$\alpha=0.000524,\beta=2.03$. These are not calibrated farm-battery parameters.
Xu mode currently requires zero self-discharge; the throughput baseline remains
available. The priority representation adds $2(J-1)$ binaries per physical control
node and interval. Details and exact run evidence are in \path{xu_degradation.md}.

One Gurobi deterministic equivalent holds shared decisions; its related-model
multiple-scenario facility is not used \cite{gurobi_scenarios}. Solve results distinguish
optimality, time limits with/without incumbents, infeasibility and unboundedness
\cite{gurobi_status}. Extraction requires an incumbent. A separate numerical validator
recomputes balances, modes, positions, settlement, node sharing and weighted profit.
The configured relative optimality tolerance is $10^{-4}$ (0.01\%); no absolute
MIP gap is configured, following the user's general rule.

\section{Execution and observed results}
Events order forecast publications, auction results and completed observations
before coincident submissions/physical controls, then settlement. A computation
buffer shifts submission cutoffs. Only the current auction quantity is submitted;
future modeled adjustments are discarded until their actionable event. Submission
fixes the fully executed quantity immediately. Price publication books its actual
cash once, refreshes conditional future price forecasts and triggers residual
reoptimization while preserving committed quantities and physical commands.
These forecast objective values remain separate from realized ledger cash flow.
The synthetic results handler rebuilds conditional descendant prices from all
published actual curves at each publication. Its diagnostic residual solve
submits no additional quantity and replaces no command already fixed for the
current physical interval. The solve trace records the publication trigger,
known-price checks, booked quantity and actual cash increment. The four-auction
fixture checks these updates, single cash booking and carried state across an
intrainterval publication. Saved replay totals below retain the run-specific
configuration and timestamps of their listed manifests.
Completed interval averages never enter the
command for their own interval. State and cumulative aging carry forward.
At a cutoff within an already committed interval, its known battery command
projects the next boundary without reading its final PV average. The safety
response is idealized at interval-average resolution and corrections are recorded;
the synthetic support has perfect PV and required none. This is not high-frequency
hardware validation or continuous overlapping multi-day simulation.

The final JUnit evidence records """+str(tests.get("passed",0))+r""" passed,
"""+str(tests.get("failed",0))+r""" failed, """+str(tests.get("errors",0))+r""" errors and
"""+str(tests.get("skipped",0))+r""" skipped tests. Historical market backtests,
causally verified historical scenario calibration and rear-side bifacial validation
were not executed. Authenticated acquisition remains user-run; any supplied DA
experiment evidence describes a retrieved-revision retrospective price study.
Xu aging was executed on analytical cycles, nested-cycle/rainflow
comparisons, shared-scenario controls and a full 96-product auction replay.
The ordinary synthetic day compares four different policy definitions on the same
realization and constraints. Equal integrated-policy profits on this case do not
establish value from stochastic optimization or justify modifying the economics.
The oracle uses future truth and is non-executable. Realized operating profit is
distinct from initial expected profit and excludes any salvage credit.

{\small\begin{longtable}{llp{0.25\textwidth}rrr}
\toprule Day & Products & Policy & Profit EUR & Replay s & Max error \\ \midrule
\endhead
"""+results+r"""
\bottomrule\end{longtable}}
The initial stochastic ordinary-day model had """+_tex(solve.get("variables","not run"))+r"""
variables, """+_tex(solve.get("binary_variables","not run"))+r""" binaries and
"""+_tex(solve.get("constraints","not run"))+r""" constraints, with
"""+_tex(solve.get("leaves","not run"))+r""" leaves. Measured build/solve seconds were
"""+_tex(round(solve.get("build_seconds",0),4))+r""" and
"""+_tex(round(solve.get("solve_seconds",0),4))+r"""; status
"""+_tex(solve.get("status","not run"))+r""", bound
"""+_tex(solve.get("objective_bound_eur","not run"))+r""" EUR, gap
"""+_tex(solve.get("mip_gap","not run"))+r""". All values are synthetic, not financial
performance estimates. Model size/runtime for the one-leaf configuration is saved
separately as a reproducible scaling reference, if executed.

\IfFileExists{../"""+figure+r"""}{\begin{figure}[htbp]\centering
\includegraphics[width=0.94\textwidth]{../"""+figure+r"""}
\caption{Executed synthetic stochastic policy, AC powers, energy, positions and prices.}
\end{figure}}{The optional synthetic plot was not available at report generation.}

\section{Reproducibility, artifacts and limitations}
Activate the existing \texttt{bess} environment and run from the repository;
no installation step is needed. The application is executable with
\texttt{PYTHONPATH=src python -m bess\_pv\_trading}. Concrete commands:
\begin{verbatim}
PYTHONPATH=src python -m bess_pv_trading validate-config \
  --config configs/farms_nord.yaml
PYTHONPATH=src python -m bess_pv_trading inspect-data \
  --config configs/farms_nord.yaml
PYTHONPATH=src python -m bess_pv_trading prepare-analysis-data \
  --config configs/farms_nord.yaml
PYTHONPATH=src python -m bess_pv_trading forecast-pv \
  --config configs/farms_nord.yaml
PYTHONPATH=src python -m bess_pv_trading run-synthetic \
  --config configs/full_experiments/synthetic_small.yaml
PYTHONPATH=src python -m bess_pv_trading import-market-data \
  --config configs/price_experiments/synthetic_prices.yaml
PYTHONPATH=src python -m bess_pv_trading train-price-models \
  --config configs/price_experiments/synthetic_prices.yaml
PYTHONPATH=src python -m bess_pv_trading api-cache-status \
  --config configs/api_experiments/api_history.yaml
PYTHONPATH=src python -m bess_pv_trading experiment-api-prices \
  --config configs/price_experiments/price_experiment.yaml
PYTHONPATH=src python -m bess_pv_trading study-linear-prices \
  --config configs/price_experiments/price_linear_study.yaml
PYTHONPATH=src python -m bess_pv_trading study-linear-prices \
  --config configs/price_experiments/price_linear_methods.yaml
PYTHONPATH=src python -m bess_pv_trading calibrate-price-scenarios \
  --config configs/price_experiments/price_gaussian_adaptive.yaml
PYTHONPATH=src python -m bess_pv_trading optimize \
  --config configs/full_experiments/stochastic_prepared.example.yaml
PYTHONPATH=src python -m pytest -q --junitxml=docs/generated/tests.xml
PYTHONPATH=src python -m bess_pv_trading report-implementation \
  --config configs/report.yaml
\end{verbatim}
The long commands above may be line-wrapped by the PDF viewer; README contains
copyable commands. Report generation reads the actual manifests and JUnit file;
it does not rerun tests or synthesize missing evidence. Run directories cannot
overwrite existing experiment results. The real-data example remains exploratory
and writes explicit ineligibility rather than inventing missing physical inputs.
The prepared stochastic optimization configuration is a schema template: supply
its documented forecast artifacts and equipment/state assumptions before solving.

Selected numerical evidence paths, relative to the repository:
\begin{itemize}
"""+artifact_rows+r"""
\item \path{outputs/analysis/farm1_20251001/manifest.json}
\item \path{outputs/pv/farm1_20251001/manifest.json}
\item \path{outputs/market_price_fixture/models/training_manifest.json}
\item \path{docs/generated/tests.xml}
\end{itemize}
The historical eligible intersection for executable strategy validation is presently
empty because complete causal auction and verified settlement inputs remain
unavailable. Retrieved DA prices support retrospective price analysis separately.
PV eligibility and upstream review
would further restrict it. BESS/POI hardware, actual coordinates, tracker geometry,
rear response and PV-output conversion/calibration remain assumed. No independent
historical strategy validation or meteorological ensemble calibration is claimed.
Any Gaussian CRPS and coverage above concern retrieved-revision price studies
and the explicitly reported scored subset. Four farms must continue to be evaluated separately. The source
documentation and unresolved meanings are detailed in \path{dataset_contract.md},
\path{assumptions.md}, \path{formulation.md} and \path{sources.md}.

TeX validation status: """+_tex(evidence["tex_validation"]["status"])+r""".
PDF visual inspection status is separately recorded in the JSON evidence.
\bibliographystyle{plain}
\bibliography{implementation_status}
\end{document}
"""


def report_implementation(config: dict[str, Any]) -> dict[str, Any]:
    """Collect final evidence, regenerate source/JSON, and compile if available."""
    settings = config.get("report",{})
    root = Path(settings.get("project_root",".")).resolve()
    destination = root/"docs"
    destination.mkdir(exist_ok=True)
    tests = _test_evidence(root/settings.get("test_junit","docs/generated/tests.xml"))
    manifests = sorted((root/"outputs/trading").glob("*/farm_*/*/manifest.json"))
    selected: dict[tuple,dict] = {}
    for path in manifests:
        run = _read(path)
        run["manifest_path"] = str(path.relative_to(root))
        if run.get("delivery_day") in settings.get("delivery_days",["2025-10-02","2025-10-26","2026-03-29"]):
            mode=run.get("config",{}).get("degradation",{}).get("model","throughput")
            strategies=tuple(sorted(item["strategy"] for item in run.get("strategies",[])))
            selected[run["delivery_day"],mode,strategies] = run
    runs = [selected[key] for key in sorted(selected)]
    revision = subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True,check=False).stdout.strip()
    dirty = subprocess.run(["git","status","--porcelain"],cwd=root,text=True,capture_output=True,check=False).stdout.splitlines()
    packages, distribution_versions = {}, {}
    for name in ("numpy","pandas","scipy","gurobipy","pvlib","xarray","zarr","pyarrow","PyYAML","pytest","matplotlib"):
        try:
            distribution_versions[name]=version(name)
            packages[name]=getattr(import_module("yaml" if name=="PyYAML" else name),"__version__",distribution_versions[name])
        except (PackageNotFoundError, ImportError):
            packages[name]="missing"
    pv = _read(root/settings.get("pv_manifest","outputs/pv/farm1_20251001/manifest.json"))
    analysis = _read(root/settings.get("analysis_manifest","outputs/analysis/farm1_20251001/manifest.json"))
    price_path=settings.get("price_experiment_manifest")
    price_experiment=_read(root/price_path) if price_path else {}
    fundamental_path=settings.get("fundamental_experiment_manifest")
    fundamental_experiment=_read(root/fundamental_path) if fundamental_path else {}
    linear_price_studies = _linear_price_evidence(root, settings.get("linear_price_study_manifests", []))
    gaussian_studies = _gaussian_study_evidence(root, settings.get("gaussian_study_manifests", []))
    price_forward_checks = _price_forward_check_evidence(root, settings.get("price_forward_check_manifests", []))
    evidence = {"snapshot":{"generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "repository":str(root),"revision":revision,"dirty":bool(dirty),"changed_paths":dirty},
        "environment":{"python":sys.version,"packages":packages,"distribution_versions":distribution_versions,
            "distribution_metadata_mismatches":{name:{"imported":value,"metadata":distribution_versions[name]}
                for name,value in packages.items() if name in distribution_versions and value!=distribution_versions[name]},
            "modified":False},
        "tests":tests,"runs":runs,"analysis":analysis,"pv":pv,
        "dataset_inspection":_read(root/"docs/generated/dataset_inspection.json"),
        "data_pv_validation":_read(root/"outputs/data_pv_validation/evidence.json"),
        "api_plan":_read(root/"docs/generated/api_plan.json"),
        "api_download_status":"Authenticated acquisition is user-run; report reads completed local artifacts without API credentials",
        "api_cache_status":_read(root/settings.get("api_cache_status_manifest","docs/generated/api_cache_status.json")),
        "price_experiment":price_experiment,
        "price_experiment_manifest":price_path,
        "fundamental_experiment":fundamental_experiment,
        "fundamental_experiment_manifest":fundamental_path,
        "linear_price_studies": linear_price_studies,
        "gaussian_studies": gaussian_studies,
        "price_forward_checks": price_forward_checks,
        "optimization_scaling":[{"path":str(path.relative_to(root)),
            "diagnostics":_read(path).get("diagnostics",{}),
            "status":_read(path).get("status"),"objective_eur":_read(path).get("objective_eur")}
            for path in sorted((root/"outputs/trading").glob("optimize_*/result.json"))],
        "synthetic_price_training":_read(root/"outputs/market_price_fixture/models/training_manifest.json"),
        "historical_eligible_intersection":{"delivery_days":0,"reason":"Complete causal auction and verified settlement inputs unavailable"},
        "tex_validation":{"status":"not-executed","visual_inspection":False},
        "known_execution_failures":["Sandboxed Gurobi host identification failed; licensed external execution passed",
            "Preliminary CLI timestamp-resolution/efficiency-validation errors were fixed before final tests",
            "Initial extension suite had one null-valued template-validation failure; corrected before final passing suite",
            "Real-data forecast retains invalid BHI>GHI intervals; no historical market backtest executed"]}
    evidence["components"] = _components(tests,runs,pv,analysis,price_experiment,fundamental_experiment,linear_price_studies,gaussian_studies)
    tex = destination/"implementation_status.tex"
    tex.write_text(_render(evidence))
    compiler = shutil.which("latexmk")
    if settings.get("compile",True) and compiler:
        command = [compiler,"-pdf","-interaction=nonstopmode","-halt-on-error",
                   "-outdir=generated/latex","implementation_status.tex"]
        result = subprocess.run(command,cwd=destination,capture_output=True,text=True,check=False)
        (destination/"generated/latex_build_output.txt").write_text(result.stdout+result.stderr)
        evidence["tex_validation"].update(command="cd docs && "+" ".join(command),
            status="compiled-successfully" if result.returncode==0 else "compilation-failed",
            returncode=result.returncode)
        if result.returncode==0:
            tex.write_text(_render(evidence))
            final = subprocess.run(command,cwd=destination,capture_output=True,text=True,check=False)
            if final.returncode:
                evidence["tex_validation"]["status"]="compilation-failed"
                (destination/"generated/latex_build_output.txt").write_text(final.stdout+final.stderr)
            else:
                (destination/"generated/latex_build_output.txt").write_text(final.stdout+final.stderr)
                shutil.copy2(destination/"generated/latex/implementation_status.pdf",destination/"implementation_status.pdf")
        log = destination/"generated/latex/implementation_status.log"
        if log.is_file():
            evidence["tex_validation"]["log_findings"]=[line for line in log.read_text(errors="replace").splitlines()
                if any(word in line for word in ("Overfull", "undefined", "LaTeX Error"))]
    else:
        evidence["tex_validation"]["status"]="not-executed-compiler-absent" if not compiler else "not-executed-disabled"
        evidence["tex_validation"]["intended_command"]="cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error implementation_status.tex"
    write_json(destination/"implementation_status.json",evidence)
    return {"tex":str(tex),"json":str(destination/"implementation_status.json"),
        "pdf":str(destination/"implementation_status.pdf") if evidence["tex_validation"]["status"]=="compiled-successfully" else None,
        "tests":{key:value for key,value in tests.items() if key!="test_ids"},
        "tex_validation":evidence["tex_validation"]}

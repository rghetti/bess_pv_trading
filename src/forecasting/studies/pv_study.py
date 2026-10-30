"""Reproducible monthly physical PV studies, retraining and artifact publication."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import multiprocessing
import sys
import time

import numpy as np
import pandas as pd
import pvlib
import scipy

from ...data import config_path
from ..inference.pv_artifacts import digest, fixed_contract
from ..training.pv_fitting import available_methods, fixed_fit, predict_fit, select_and_refit
from ..data.pv_study_inputs import frame_fingerprint, prepare_corpus
from ..plotting.pv_study_reporting import error_metrics, report_farm


def _write(path: Path, value):
    """Replace complete JSON documents atomically; interrupted writes are reusable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+"\n")
    temporary.replace(path)


def validate_study(config: dict, *, retrain=False):
    """Reject impossible bounds and chronology before opening or fitting any data."""
    section = config["pv_calibration"]
    # Reject impossible bounds and chronology before opening or fitting any data for each `name` in `('start_day', 'end_day_exclusive')`.
    for name in ("start_day", "end_day_exclusive"):
        # Enforce the input contract: pv_calibration.`name` is required.
        if not section.get(name):
            raise ValueError(f"pv_calibration.{name} is required")
    # Enforce the input contract: PV calibration end must follow start.
    if pd.Timestamp(section["end_day_exclusive"]) <= pd.Timestamp(section["start_day"]):
        raise ValueError("PV calibration end must follow start")
    # Enforce the input contract: PV calibration farm_ids must be nonempty and unique.
    if not section["farm_ids"] or len(set(section["farm_ids"])) != len(section["farm_ids"]):
        raise ValueError("PV calibration farm_ids must be nonempty and unique")
    allowed={"fixed"}|{f"{family}_{loss}" for family in ("scale","thermal","tracker","bifacial") for loss in ("mse","huber")}
    # Enforce the input contract: PV calibration requires fixed and supported calibration methods.
    if not set(section["methods"]) <= allowed or "fixed" not in section["methods"]:
        raise ValueError("PV calibration requires fixed and supported calibration methods")
    # Enforce the input contract: PV calibration training windows are too short.
    if not section["training_windows_days"] or any(x is not None and int(x) <= int(section["validation_days"])+int(section["minimum_training_days"]) for x in section["training_windows_days"]):
        raise ValueError("PV calibration training windows are too short")
    # Enforce the input contract: Invalid calibration transposition models.
    if not section["transposition_models"] or not set(section["transposition_models"]) <= {"isotropic","haydavies","perez-driesse"}:
        raise ValueError("Invalid calibration transposition models")
    # Process each `key` in this collection.
    for key in ("validation_days","minimum_training_days","multistarts","max_function_evaluations","workers","bootstrap_block_days","bootstrap_repetitions","representative_days_per_season"):
        # Enforce the input contract: pv_calibration.`key` must be a positive integer.
        if isinstance(section[key],bool) or int(section[key]) != section[key] or int(section[key]) < 1:
            raise ValueError(f"pv_calibration.{key} must be a positive integer")
    # Reject impossible bounds and chronology before opening or fitting any data for each `(name, bounds)` in `section['parameter_bounds'].items()`.
    for name,bounds in section["parameter_bounds"].items():
        # Enforce the input contract: Invalid PV calibration bounds for `name`.
        if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[1]<=bounds[0] or bounds[0]<0:
            raise ValueError(f"Invalid PV calibration bounds for {name}")
    # Enforce the input contract: PV calibration regularization must be finite and nonnegative.
    if not section["regularization_values"] or any(not np.isfinite(x) or x<0 for x in section["regularization_values"]):
        raise ValueError("PV calibration regularization must be finite and nonnegative")
    # Enforce the input contract: PV calibration loss/optimizer tolerances must be positive.
    if float(section["huber_delta"]) <= 0 or float(section["optimizer_ftol"]) <= 0 or float(section["optimizer_gtol"]) <= 0:
        raise ValueError("PV calibration loss/optimizer tolerances must be positive")
    # Enforce the input contract: retrain-pv-calibration requires timezone-aware pv_calibration.as_of_utc.
    if retrain and (not section.get("as_of_utc") or pd.Timestamp(section["as_of_utc"]).tzinfo is None):
        raise ValueError("retrain-pv-calibration requires timezone-aware pv_calibration.as_of_utc")


def _progress(message):
    print(message, file=sys.stderr, flush=True)


def _fit_month(corpus, issue, farm, settings, section, destination, source, config_hash, label):
    """Checkpoint every completed method, including explicit fallback provenance."""
    fits, entries, history = {}, [], []
    # Checkpoint every completed method, including explicit fallback provenance for each `method` in `available_methods(section, farm)`.
    for method in available_methods(section, farm):
        path=destination/"json"/"fits"/label/f"{method}.json"
        # When `path.exists()`, set `saved` using `json.loads(path.read_text())`.
        if path.exists():
            saved=json.loads(path.read_text())
            # Enforce the input contract: Completed PV fit fingerprint does not match current study.
            if saved["source_fingerprint"] != source or saved["configuration_fingerprint"] != config_hash:
                raise ValueError("Completed PV fit fingerprint does not match current study")
            fit=saved["fit"]
        else:
            _progress(f"farm {farm.farm_id}: {label} fitting {method}")
            fit,diagnostics=select_and_refit(corpus,issue,farm,settings,section,method)
            _write(path,{"source_fingerprint":source,"configuration_fingerprint":config_hash,
                "issued_at_utc":issue.isoformat(),"fit":fit,"diagnostics":diagnostics})
        fits[method]=fit
        history.append({"issue_utc":issue.isoformat(),"method":method,
            **fit["parameters"],"transposition_model":fit["transposition_model"],
            "validation_nrmse":fit["validation_nrmse"],"window_days":fit.get("window_days"),
            "converged":fit["converged"],"bound_hits":",".join(fit["bound_hits"]),
            "weakly_identified":fit["weakly_identified"],"runtime_seconds":fit["runtime_seconds"]})
    winner=min(fits,key=lambda method:fits[method]["validation_nrmse"])
    fit=fits[winner]
    artifact={"schema_version":1,"farm_metadata":asdict(farm),"issued_at_utc":issue.isoformat(),
        "latest_target_available_at_utc":fit.get("latest_target_available_at_utc",issue.isoformat()),
        "fixed_contract":fixed_contract(settings,section["night_policy"]),"fit":fit,
        "source_fingerprint":source,"configuration_fingerprint":config_hash,
        "training_data_fingerprint":frame_fingerprint(corpus.loc[corpus.target_available_at_utc<=issue]),
        "versions":{"pvlib":pvlib.__version__,"scipy":scipy.__version__,"numpy":np.__version__},
        "selection":"earlier_validation_only_including_fixed_baseline"}
    path=destination/"json"/"artifacts"/f"{label}_{winner}.json"
    # When `path.exists()`, set `existing` using `json.loads(path.read_text())`.
    if path.exists():
        existing=json.loads(path.read_text())
        # Enforce the input contract: Refusing to overwrite a different immutable PV parameter artifact.
        if existing!=json.loads(json.dumps(artifact,default=str)):
            raise ValueError("Refusing to overwrite a different immutable PV parameter artifact")
    else:
        _write(path,artifact)
    entries.append({"farm_id":farm.farm_id,"issued_at_utc":issue.isoformat(),
        "artifact_path":str(path.relative_to(destination.parent)),
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    return fits,winner,entries,history


def _sensitivity(corpus, monthly_fits, farm, settings, section, destination):
    """Change one unmeasured rear-side assumption at a time without fitting to test data."""
    # Change one unmeasured rear-side assumption at a time without fitting to test data when `farm.tracking_type_id != 3`.
    if farm.tracking_type_id!=3:
        return
    rows=[]
    # Change one unmeasured rear-side assumption at a time without fitting to test data for each `(month, fits)` in `monthly_fits.items()`.
    for month,fits in monthly_fits.items():
        rear=[fit for name,fit in fits.items() if name.startswith("bifacial_")]
        # Skip the current item when `not rear`.
        if not rear:
            continue
        fitted=min(rear,key=lambda fit:fit["validation_nrmse"])
        sample=corpus.loc[corpus.delivery_day.str.startswith(month)]
        # Change one unmeasured rear-side assumption at a time without fitting to test data for each `(field, values)` in `section['bifacial_sensitivity'].items()`.
        for field,values in section["bifacial_sensitivity"].items():
            # Change one unmeasured rear-side assumption at a time without fitting to test data for each `value` in `values`.
            for value in values:
                changed=deepcopy(fitted)
                # When `field == 'albedo'`, set `changed['albedo']` using `value`.
                if field=="albedo":
                    changed["albedo"]=value
                else:
                    changed["bifacial_settings"][field]=value
                prediction=predict_fit(sample,farm,settings,section["night_policy"],changed)
                # Change one unmeasured rear-side assumption at a time without fitting to test data for each `(auction, group)` in `sample.groupby('auction')`.
                for auction,group in sample.groupby("auction"):
                    positions=sample.index.get_indexer(group.index)
                    valid=(group.eligible & group.daylight).to_numpy()
                    rows.append({"month":month,"auction":auction,"parameter":field,"value":value,
                        "source_method":fitted["method"],**error_metrics(group.measured_ac_mw.to_numpy()[valid],
                            prediction[positions][valid],farm.ac_capacity_mw)})
    pd.DataFrame(rows).to_csv(destination/"csv"/"bifacial_assumption_sensitivity.csv",index=False)


def _farm_worker(config, farm_id, root, config_hash, retrain=False):
    """Execute an independently resumable farm trajectory without shared model state."""
    return _run_farm(config,farm_id,root,config_hash,retrain)


def _run_farm(config,farm_id,root,config_hash,retrain):
    destination=Path(root)/f"farm_{farm_id}"
    # Perform run farm for each `folder` in `('json', 'csv', 'parquet', 'plots')`.
    for folder in ("json","csv","parquet","plots"):
        (destination/folder).mkdir(parents=True,exist_ok=True)
    corpus,farm,settings,provenance=prepare_corpus(config,farm_id,destination,_progress)
    section={**config["pv_calibration"],"night_policy":config["policy"]["night"]}
    source=provenance["source_fingerprint"]
    methods=available_methods(section,farm)
    # When `retrain`, set `issue` using `pd.Timestamp(section['as_of_utc']).tz_convert('UTC')`.
    if retrain:
        issue=pd.Timestamp(section["as_of_utc"]).tz_convert("UTC")
        fits,winner,entries,history=_fit_month(corpus,issue,farm,settings,section,destination,
            source,config_hash,"retrain")
        return {"farm_id":farm_id,"winner":winner,"registry_entries":entries,"source_fingerprint":source}
    evaluation=(pd.Timestamp(section["evaluation_start_day"]) if section.get("evaluation_start_day")
                else pd.Timestamp(section["start_day"])+pd.offsets.MonthBegin(1))
    end=pd.Timestamp(section["end_day_exclusive"])
    data=corpus.copy()
    data["phase"]=np.where(data.delivery_day>=evaluation.date().isoformat(),"held_out","warmup")
    # Perform run farm for each `method` in `methods + ['selected']`.
    for method in methods+["selected"]:
        data[f"prediction_{method}_mw"]=data.baseline_mw
    history,entries,monthly_fits=[],[],{}
    months=pd.date_range(evaluation,end,freq="MS",inclusive="left")
    # Perform run farm for each `month` in `months`.
    for month in months:
        issue=(month-pd.Timedelta(days=1)).tz_localize(section["delivery_timezone"]).tz_convert("UTC")
        label=month.strftime("%Y-%m")
        fits,winner,new_entries,new_history=_fit_month(corpus,issue,farm,settings,section,destination,
            source,config_hash,label)
        monthly_fits[label]=fits
        entries.extend(new_entries);history.extend(new_history)
        mask=data.delivery_day.str.startswith(label)
        sample=data.loc[mask]
        # Perform run farm for each `(method, fit)` in `fits.items()`.
        for method,fit in fits.items():
            data.loc[mask,f"prediction_{method}_mw"]=predict_fit(sample,farm,settings,section["night_policy"],fit)
        data.loc[mask,"prediction_selected_mw"]=data.loc[mask,f"prediction_{winner}_mw"]
        data.loc[mask,"selected_method"]=winner
        _progress(f"farm {farm_id}: {label} complete; prior-validation selection={winner}")
    data.to_parquet(destination/"parquet"/"predictions.parquet",index=False)
    summary=report_farm(data,methods+["selected"],farm,section,destination,history)
    _sensitivity(corpus,monthly_fits,farm,settings,section,destination)
    # When `section['retrospective_fit']`, set `issue` using `end.tz_localize(section['delivery_timezone']).tz_convert('UTC')`.
    if section["retrospective_fit"]:
        issue=end.tz_localize(section["delivery_timezone"]).tz_convert("UTC")
        retrospective_section={**section,"training_windows_days":[None]}
        fits,winner,new_entries,_=_fit_month(corpus,issue,farm,settings,retrospective_section,destination,
            source,config_hash,"retrospective_full_year")
        entries.extend(new_entries)
        retrospective=corpus.copy()
        retrospective["phase"]="retrospective_in_sample"
        # Perform run farm for each `(method, fit)` in `fits.items()`.
        for method,fit in fits.items():
            retrospective[f"prediction_{method}_mw"]=predict_fit(corpus,farm,settings,section["night_policy"],fit)
        retrospective["prediction_selected_mw"]=retrospective[f"prediction_{winner}_mw"]
        retrospective.to_parquet(destination/"parquet"/"retrospective_predictions.parquet",index=False)
        summary["retrospective_in_sample"]=report_farm(retrospective,methods+["selected"],farm,section,destination,[],retrospective=True)
    _write(destination/"json"/"summary.json",summary)
    return {"farm_id":farm_id,"summary":summary,"registry_entries":entries,"source_fingerprint":source}


def run_pv_calibration_study(config: dict, *, retrain=False, progress=None) -> dict:
    """Build inputs once per farm, compare methods, and publish versioned local artifacts."""
    config=deepcopy(config)
    # null/empty farm overrides explicitly request the shared reference model.
    # Canonicalize this no-op so older frozen baseline studies remain resumable
    # when calibrated farm metadata is later added to the standard settings.
    for farm in config.get("farms",{}).values():
        # Build inputs once per farm, compare methods, and publish versioned local artifacts when `farm.get('pv') is None or farm.get('pv') == {}`.
        if farm.get("pv") is None or farm.get("pv")=={}:
            farm.pop("pv",None)
    section=config["pv_calibration"]
    # Update `section['end_day_exclusive']` when this condition is met.
    if retrain and section.get("as_of_utc"):
        section["end_day_exclusive"]=pd.Timestamp(section["as_of_utc"]).tz_convert(section["delivery_timezone"]).date().isoformat()
    validate_study(config,retrain=retrain)
    started=time.perf_counter()
    run_id=section.get("run_id") or datetime.now(timezone.utc).strftime("physical_pv_%Y%m%dT%H%M%S%fZ")
    # Update `run_id` when this condition is met.
    if retrain:
        run_id+= "_retrain_"+pd.Timestamp(section["as_of_utc"]).strftime("%Y%m%dT%H%M%S%z")
    root=config_path(config,section["output_root"])/run_id
    config_hash=digest(config)
    configuration=root/"json"/"study_config.json"
    # Build inputs once per farm, compare methods, and publish versioned local artifacts when `root.exists()`.
    if root.exists():
        # Enforce the input contract: PV study output exists; enable resume or choose a fresh run ID.
        if not section["resume"] or not configuration.exists():
            raise ValueError("PV study output exists; enable resume or choose a fresh run ID")
        # Enforce the input contract: PV study configuration fingerprint changed; choose a fresh run ID.
        if digest(json.loads(configuration.read_text()))!=config_hash:
            raise ValueError("PV study configuration fingerprint changed; choose a fresh run ID")
    # Build inputs once per farm, compare methods, and publish versioned local artifacts for each `folder` in `('json', 'csv', 'parquet', 'plots')`.
    for folder in ("json","csv","parquet","plots"):
        (root/folder).mkdir(parents=True,exist_ok=True)
    _write(configuration,config)
    completed=[]
    workers=min(int(section["workers"]),len(section["farm_ids"]))
    # Build inputs once per farm, compare methods, and publish versioned local artifacts when `progress`.
    if progress:
        progress(f"PV calibration: {len(section['farm_ids'])} farms, {workers} workers, local data only; output={root}")
    # Build inputs once per farm, compare methods, and publish versioned local artifacts when `workers == 1`.
    if workers==1:
        # Build inputs once per farm, compare methods, and publish versioned local artifacts for each `farm` in `section['farm_ids']`.
        for farm in section["farm_ids"]:
            completed.append(_farm_worker(config,int(farm),str(root),config_hash,retrain))
    else:
        # Fresh interpreters honor these limits before importing NumPy/SciPy;
        # no optional package or environment installation is required.
        thread_variables=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")
        old_threads={name:os.environ.get(name) for name in thread_variables}
        # Build inputs once per farm, compare methods, and publish versioned local artifacts for each `name` in `thread_variables`.
        for name in thread_variables:
            os.environ[name]="1"
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            futures={pool.submit(_farm_worker,config,int(farm),str(root),config_hash,retrain):farm for farm in section["farm_ids"]}
            # Build inputs once per farm, compare methods, and publish versioned local artifacts for each `future` in `as_completed(futures)`.
            for future in as_completed(futures):
                completed.append(future.result())
                # Build inputs once per farm, compare methods, and publish versioned local artifacts when `progress`.
                if progress:
                    progress(f"PV calibration: farm {futures[future]} completed ({len(completed)}/{len(futures)})")
        # Build inputs once per farm, compare methods, and publish versioned local artifacts for each `(name, value)` in `old_threads.items()`.
        for name,value in old_threads.items():
            # Build inputs once per farm, compare methods, and publish versioned local artifacts when `value is None`.
            if value is None:
                os.environ.pop(name,None)
            else:
                os.environ[name]=value
    entries=[entry for result in completed for entry in result["registry_entries"]]
    _write(root/"json"/"parameter_registry.json",{"schema_version":1,"farm_ids":section["farm_ids"],
        "entries":[{**entry,"artifact_path":"../"+entry["artifact_path"]} for entry in entries]})
    comparisons=[]
    # Build inputs once per farm, compare methods, and publish versioned local artifacts when `not retrain`.
    if not retrain:
        # Build inputs once per farm, compare methods, and publish versioned local artifacts for each `result` in `completed`.
        for result in completed:
            # Build inputs once per farm, compare methods, and publish versioned local artifacts for each `row` in `result['summary']['paired_daylight_comparison']`.
            for row in result["summary"]["paired_daylight_comparison"]:
                comparisons.append({"farm_id":result["farm_id"],**row})
        pd.DataFrame(comparisons).to_csv(root/"csv"/"farm_method_comparison.csv",index=False)
        _study_plot(pd.DataFrame(comparisons),root)
    manifest={"status":"complete","run_id":run_id,"output_dir":str(root),
        "kind":"retrain" if retrain else "chronological_physical_pv_calibration",
        "farm_ids":section["farm_ids"],"configuration_fingerprint":config_hash,
        "inputs":{str(x["farm_id"]):x["source_fingerprint"] for x in completed},
        "parameter_registry":str(root/"json"/"parameter_registry.json"),
        "runtime_seconds":time.perf_counter()-started,"comparison":comparisons,
        "versions":{"pvlib":pvlib.__version__,"scipy":scipy.__version__}}
    # When `not retrain`, set `manifest['status']` using `'validating'`.
    if not retrain:
        manifest["status"]="validating"
    _write(root/"json"/"manifest.json",manifest)
    # Build inputs once per farm, compare methods, and publish versioned local artifacts when `not retrain`.
    if not retrain:
        from ..evaluation.pv_study_audit import audit_saved_study
        manifest["validation_audit"]=audit_saved_study(root)
        manifest["status"]="complete"
        _write(root/"json"/"manifest.json",manifest)
    return {key:value for key,value in manifest.items() if key!="comparison"}


def _study_plot(comparisons,root):
    """Compare farms in normalized units so the largest farm cannot dominate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout="constrained")
    # Compare farms in normalized units so the largest farm cannot dominate for each `(axis, (farm, group))` in `zip(axes.flat, comparisons.groupby('farm_id'))`.
    for axis,(farm,group) in zip(axes.flat,comparisons.groupby("farm_id")):
        ordered=group.sort_values("daylight_balanced_rmse_mw")
        axis.barh(ordered.method,ordered.rmse_improvement_mw,color=["tab:orange" if x=="selected" else "tab:blue" for x in ordered.method])
        axis.errorbar(ordered.rmse_improvement_mw,np.arange(len(ordered)),
            xerr=np.vstack([np.maximum(ordered.rmse_improvement_mw-ordered.improvement_ci_2_5_mw,0),
                            np.maximum(ordered.improvement_ci_97_5_mw-ordered.rmse_improvement_mw,0)]),fmt="none",color="grey")
        axis.set(title=f"Farm {farm}",xlabel="Daylight RMSE reduction versus fixed (MW)")
        axis.axvline(0,color="grey",linewidth=.5)
    fig.suptitle("2025 held-out physical PV calibration — paired seven-day block intervals")
    fig.savefig(root/"plots"/"farm_method_comparison.png",dpi=150);plt.close(fig)

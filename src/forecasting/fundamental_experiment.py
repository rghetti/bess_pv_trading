"""Retrospective comparisons of price lags with regional forecast features."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..data.api_catalog import load_api_cache_records
from .price_experiment import (FEATURES, _configured_bound, _daily_quality,
    _fit_retrospective_ridge, _metrics, _write_frame, build_retrospective_features,
    prepare_native_prices)


LIMITATION = ("Retrospective retrieved-revision experiment. Price and regional forecast "
    "publication/revision availability is unverified. These results do not establish "
    "historical executable forecasts, trading profits or calibrated scenario noise.")


def align_interval_forecast(products: pd.DataFrame, records: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Duration-weight MW forecasts inside complete native price products.

    Missing quarters leave the whole hourly feature missing. Coarser forecasts
    cannot be split into finer products without an explicit additional policy.
    """
    required={"product_start","product_end","retrieved_at","value_mw","unit"}
    if records.empty or not required.issubset(records.columns):
        raise ValueError("Regional forecasts require cached MW records with explicit product intervals")
    source=records.copy()
    for name in ("product_start","product_end","retrieved_at"):
        source[name]=pd.to_datetime(source[name],utc=True,errors="raise").dt.as_unit("ns")
        if source[name].isna().any():
            raise ValueError("Regional forecast intervals and retrieval times must be known")
    if not source.unit.eq("MW").all() or not np.isfinite(source.value_mw.to_numpy(float)).all():
        raise ValueError("Regional forecast features must contain finite MW, not energy or mixed units")
    keys=["product_start","product_end"]
    conflicts=source.groupby([*keys,"retrieved_at"]).value_mw.nunique()
    if conflicts.gt(1).any():
        raise ValueError("Conflicting regional forecast values for the same product and retrieval")
    source=source.sort_values([*keys,"retrieved_at"]).drop_duplicates(keys,keep="last").sort_values("product_start")
    starts=source.product_start.array.asi8
    ends=source.product_end.array.asi8
    target_starts=pd.to_datetime(products.product_start,utc=True).dt.as_unit("ns").array.asi8
    target_ends=pd.to_datetime(products.product_end,utc=True).dt.as_unit("ns").array.asi8
    if (ends<=starts).any() or (starts[1:]<ends[:-1]).any():
        raise ValueError("Regional forecast source intervals overlap or have invalid lengths")
    if not len(products) or (target_ends<=target_starts).any() or (target_starts[1:]<target_ends[:-1]).any():
        raise ValueError("Native price products must be ordered and nonoverlapping")
    keep=(ends>target_starts[0]) & (starts<target_ends[-1])
    starts,ends=starts[keep],ends[keep]
    values=source.value_mw.to_numpy(float)[keep]
    positions=np.searchsorted(target_starts,starts,side="right")-1
    if (positions<0).any() or (ends>target_ends[positions]).any():
        raise ValueError("Forecast intervals cross native price boundaries; coarser forecasts are not split")
    durations=(ends-starts)/1e9
    coverage=np.bincount(positions,weights=durations,minlength=len(products))
    weighted=np.bincount(positions,weights=values*durations,minlength=len(products))
    expected=(target_ends-target_starts)/1e9
    complete=np.isclose(coverage,expected,rtol=0,atol=1e-6)
    aligned=np.full(len(products),np.nan)
    aligned[complete]=weighted[complete]/expected[complete]
    return aligned,{"source_records":int(keep.sum()),"complete_products":int(complete.sum()),
        "missing_products":int((~complete).sum()),"source_interval_minutes":sorted(set((durations/60).tolist())),
        "historically_eligible_source_records":int(source.get("historical_feature_eligible",pd.Series(False,index=source.index)).fillna(False).sum()),
        "aggregation":"Duration-weighted MW mean; complete nonoverlapping coverage required; no filling or interpolation",
        "revision_selection":"Latest retrieval for each source interval; not a historical publication assertion"}


def compare_fundamental_models(features: pd.DataFrame, settings: Mapping[str,Any]) -> tuple[pd.DataFrame,list[dict],dict]:
    """Fit every model on the same earlier rows and score the same held-out rows."""
    extras=settings.get("fundamental_features",{})
    specifications=settings.get("comparison_models",{})
    if not extras or not specifications:
        raise ValueError("Configure fundamental_features and comparison_models")
    model_features={}
    for name,additional in specifications.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}",name) or not isinstance(additional,list):
            raise ValueError("Comparison model names must be simple identifiers with lists of additional features")
        if len(set(additional))!=len(additional) or set(additional)-set(extras):
            raise ValueError("Comparison models must reference unique configured fundamental features")
        model_features[name]=[*FEATURES,*additional]
    required=[*FEATURES,*extras]
    holdout_days=int(settings.get("holdout_days",7))
    min_days=int(settings.get("min_training_days",14))
    alpha=float(settings.get("alpha",1.0))
    if holdout_days<1 or min_days<1 or not np.isfinite(alpha) or alpha<0:
        raise ValueError("Positive holdout/training days and finite nonnegative alpha are required")
    predictions,evaluations,models=[],[],{}
    for minutes,portion in features.groupby("native_interval_minutes",sort=True):
        last=date.fromisoformat(portion.delivery_day.max())
        boundary_day=(last-timedelta(days=holdout_days-1)).isoformat()
        finite=portion.loc[np.isfinite(portion[required].to_numpy(float)).all(axis=1)]
        train=finite.loc[finite.delivery_day<boundary_day]
        holdout=finite.loc[finite.delivery_day>=boundary_day]
        info={"native_interval_minutes":int(minutes),"training_rows":len(train),
            "training_days":int(train.delivery_day.nunique()),"training_start_day":None if train.empty else train.delivery_day.min(),
            "training_end_day":None if train.empty else train.delivery_day.max(),
            "holdout_start_day":boundary_day,"holdout_end_day":last.isoformat(),"holdout_rows":len(holdout),
            "holdout_days_observed":int(holdout.delivery_day.nunique()),"incomplete_feature_rows":len(portion)-len(finite)}
        if info["training_days"]<min_days or holdout.empty:
            evaluations.append({**info,"status":"insufficient_data"})
            continue
        boundary=pd.Timestamp(boundary_day,tz=settings.get("timezone","Europe/Rome")).tz_convert("UTC").isoformat()
        selected=holdout[["product_start","product_end","delivery_day","native_interval_minutes","price_eur_mwh",*required]].copy()
        scores={}
        model_group={}
        for name,names in model_features.items():
            model=_fit_retrospective_ridge(train,alpha,boundary,names)
            selected[f"{name}_prediction"]=model.predict(holdout[names])
            selected[f"{name}_residual"]=selected.price_eur_mwh-selected[f"{name}_prediction"]
            scores[name]=_metrics(selected.price_eur_mwh.to_numpy(),selected[f"{name}_prediction"].to_numpy())
            model_group[name]={**model.to_dict(),"training_start_day":info["training_start_day"],
                "training_end_day":info["training_end_day"],"cutoff_semantics":"Exclusive training delivery-day boundary; historical availability unverified"}
        for name,column in (("previous_day","lag_1d"),("previous_week","lag_7d")):
            selected[f"{name}_prediction"]=selected[column]
            scores[name]=_metrics(selected.price_eur_mwh.to_numpy(),selected[column].to_numpy())
        selected["training_delivery_boundary"]=boundary
        predictions.append(selected)
        models[f"{int(minutes)}min"]=model_group
        evaluations.append({**info,"status":"evaluated","models":scores})
    return (pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame()),evaluations,models


def _plot_comparison(predictions: pd.DataFrame, evaluations: list[dict], directory: Path, zone: str) -> list[str]:
    """Compare held-out errors and actual/forecast traces without adding noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files=[]
    for evaluation in evaluations:
        if evaluation["status"]!="evaluated":
            continue
        minutes=evaluation["native_interval_minutes"]
        frame=predictions.loc[predictions.native_interval_minutes==minutes]
        fig,axes=plt.subplots(3,1,figsize=(12,9),layout="constrained")
        names=list(evaluation["models"])
        axes[0].bar(names,[evaluation["models"][name]["mae_eur_mwh"] for name in names])
        axes[0].tick_params(axis="x",rotation=15)
        axes[0].set(title=f"{zone}: {minutes}-minute retrospective holdout",ylabel="MAE (EUR/MWh)")
        axes[1].plot(frame.product_start,frame.price_eur_mwh,color="black",linewidth=1.3,label="Retrieved price")
        for name in names:
            if name in {"previous_day","previous_week"}:
                continue
            axes[1].plot(frame.product_start,frame[f"{name}_prediction"],linewidth=.8,label=name)
        axes[1].set(ylabel="EUR/MWh",xlabel="UTC product start")
        axes[1].legend(ncol=2)
        for name in names:
            errors=(frame.price_eur_mwh-frame[f"{name}_prediction"]).abs().groupby(frame.delivery_day).mean()
            axes[2].plot(pd.to_datetime(errors.index),errors.values,marker=".",label=name)
        axes[2].set(ylabel="Daily MAE (EUR/MWh)",xlabel="Local delivery day")
        axes[2].legend(ncol=2)
        file=f"comparison_{minutes}min.png"
        fig.savefig(directory/file,dpi=130)
        plt.close(fig)
        files.append(file)
    return files


def run_fundamental_experiment(config: Mapping[str,Any]) -> dict:
    """Read verified caches, freeze joined inputs and save a retrospective comparison."""
    settings=config.get("price_experiment",{})
    timezone_name=settings.get("timezone","Europe/Rome")
    zone=settings.get("zone","NORD")
    bounds={"start":_configured_bound(settings.get("start"),timezone_name),
            "end":_configured_bound(settings.get("end"),timezone_name)}
    cache=settings.get("cache_directory","data/api")
    prices,price_sources=load_api_cache_records(cache,provider=settings.get("provider","entsoe"),
        dataset=settings.get("dataset","day_ahead_prices"),zone=zone,**bounds)
    native=prepare_native_prices(prices,settings)
    features=build_retrospective_features(native,timezone_name)
    sources={"prices":price_sources}
    coverage={}
    for name,specification in settings.get("fundamental_features",{}).items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}",name) or name in features:
            raise ValueError("Fundamental feature names must be unique simple identifiers")
        records,provenance=load_api_cache_records(cache,provider=specification["provider"],
            dataset=specification["dataset"],zone=zone,**bounds)
        features[name],coverage[name]=align_interval_forecast(native,records)
        sources[name]=provenance
    predictions,evaluations,models=compare_fundamental_models(features,settings)
    if predictions.empty:
        raise ValueError("Insufficient overlapping forecast history for the configured comparison; no models fitted")
    quality=_daily_quality(native,timezone_name,settings)
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory=Path(settings.get("output_directory","outputs/price_fundamentals"))/run_id
    directory.mkdir(parents=True,exist_ok=False)
    _write_frame(features,directory,"prices_with_fundamentals")
    _write_frame(predictions,directory,"holdout_predictions")
    _write_frame(quality,directory,"daily_quality")
    for filename,value in (("models.json",models),("metrics.json",evaluations),("source_caches.json",sources),("feature_coverage.json",coverage)):
        (directory/filename).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+"\n")
    plots=_plot_comparison(predictions,evaluations,directory,zone)
    monthly=[]
    for month,frame in features.groupby(features.delivery_day.str[:7],sort=True):
        year,month_number=month.split("-")
        destination=Path(settings.get("analysis_directory","data/analysis/price_fundamentals"))/zone/f"year={year}"/f"month={month_number}"
        _write_frame(frame,destination,"prices_with_fundamentals")
        (destination/"manifest.json").write_text(json.dumps({"month":month,"rows":len(frame),
            "immutable_run":str(directory.resolve()),"source_caches":str((directory/"source_caches.json").resolve()),
            "limitation":LIMITATION},indent=2)+"\n")
        monthly.append(str(destination.resolve()))
    result={"schema_version":"retrospective-fundamentals-v1","status":"completed","run_directory":str(directory.resolve()),
        "study_type":"Fixed chronological holdout; regional forecast feature comparison","limitation":LIMITATION,
        "historical_publication_verified":False,"features":[*FEATURES,*settings["fundamental_features"]],
        "settings":dict(settings),"metrics":evaluations,"feature_coverage":coverage,
        "first_delivery_day":features.delivery_day.min(),"last_delivery_day":features.delivery_day.max(),
        "native_products":len(native),"complete_days":int(quality.complete.sum()),"incomplete_days":int((~quality.complete).sum()),
        "source_cache_count":sum(len(items) for items in sources.values()),"evaluation_comparison":"Identical complete-feature training and holdout rows for all models",
        "aggregation":"Duration-weighted regional MW forecasts within complete native price products; no price interpolation",
        "plots":plots,"derived_monthly_directories":monthly}
    (directory/"manifest.json").write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+"\n")
    return result

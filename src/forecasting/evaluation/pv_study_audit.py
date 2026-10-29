"""Validate saved study chronology, frozen targets and complete method checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...markets import AuctionCalendar
from ...data import FarmMetadata
from ..inference.pv_artifacts import read_json
from ..training.pv_fitting import available_methods
from ..data.pv_study_inputs import frame_fingerprint


def audit_saved_study(root: str | Path) -> dict:
    """Audit persisted inputs/results without rerunning calibration or optimization."""
    root=Path(root)
    config=read_json(root/"json"/"study_config.json")
    section=config["pv_calibration"]
    registry=read_json(root/"json"/"parameter_registry.json")
    evaluation=pd.Timestamp(section.get("evaluation_start_day") or
        (pd.Timestamp(section["start_day"])+pd.offsets.MonthBegin(1)).date().isoformat())
    end=pd.Timestamp(section["end_day_exclusive"])
    months=pd.date_range(evaluation,end,freq="MS",inclusive="left")
    calendar_settings={**config["market"],"valid_from":section["start_day"],
        "valid_to":(end-pd.Timedelta(days=1)).date().isoformat()}
    calendar=AuctionCalendar.from_config(calendar_settings)
    farm_audits=[]
    for farm_id in section["farm_ids"]:
        directory=root/f"farm_{farm_id}"
        corpus=pd.read_parquet(directory/"parquet"/"corpus.parquet")
        data=pd.read_parquet(directory/"parquet"/"predictions.parquet")
        provenance=read_json(directory/"json"/"input_provenance.json")
        farm=FarmMetadata(**provenance["farm_metadata"])
        methods=available_methods(section,farm)
        assert frame_fingerprint(corpus)==provenance["aligned_fingerprint"],"Input snapshot changed"
        pd.testing.assert_frame_equal(corpus,data[corpus.columns])
        assert (data.forecast_available_at_utc<=data.decision_cutoff_utc).all(),"Future weather used"
        assert (data.forecast_cycle_utc<=data.decision_cutoff_utc).all(),"Future cycle used"
        if not config["source"].get("explicit_availability_variable"):
            assert (data.forecast_cycle_utc+pd.Timedelta(hours=config["source"]["availability_lag_hours"])
                    == data.forecast_available_at_utc).all(),"UTC cycle lag changed"
        for (day,auction),group in data.groupby(["delivery_day","auction"]):
            event=next(x for x in calendar.schedule(day) if x.auction==auction)
            expected=calendar.delivery_grid(day).starts[list(event.eligible_intervals)]
            assert pd.DatetimeIndex(group.interval_start_utc).as_unit("ns").equals(expected.as_unit("ns")),"Delivery scope changed"
            assert (group.decision_cutoff_utc==event.submission_at).all(),"Submission cutoff changed"
        for method in methods+["selected"]:
            assert np.isfinite(data.loc[data.eligible,f"prediction_{method}_mw"]).all(),"Candidate dropped evaluation samples"
        np.testing.assert_allclose(data.prediction_fixed_mw,data.baseline_mw,atol=1e-12,equal_nan=True)
        required=(len(months)+int(section["retrospective_fit"]))*len(methods)
        files=list((directory/"json"/"fits").glob("*/*.json"))
        assert len(files)==required,f"Incomplete method checkpoints for farm {farm_id}"
        for path in files:
            checkpoint=read_json(path)
            issue=pd.Timestamp(checkpoint["issued_at_utc"])
            fit=checkpoint["fit"]
            if "latest_target_available_at_utc" in fit:
                assert pd.Timestamp(fit["latest_target_available_at_utc"])<=issue,"Future target in refit"
            for candidate in checkpoint["diagnostics"].get("candidates",[]):
                validation_start=pd.Timestamp(checkpoint["diagnostics"]["validation_start_day"],tz=section["delivery_timezone"])
                assert pd.Timestamp(candidate["fit"]["latest_target_available_at_utc"])<=validation_start,"Validation target entered candidate fit"
        for entry in registry["entries"]:
            if entry["farm_id"]==farm_id:
                artifact=read_json(root/"json"/entry["artifact_path"])
                assert pd.Timestamp(artifact["latest_target_available_at_utc"])<=pd.Timestamp(entry["issued_at_utc"])
        farm_audits.append({"farm_id":farm_id,"forecast_rows":len(data),
            "delivery_days":int(data.delivery_day.nunique()),
            "held_out_days":int(data.loc[data.phase=="held_out","delivery_day"].nunique()),
            "completed_method_checkpoints":len(files),"expected_method_checkpoints":required,
            "missing_unique_targets":int(data.loc[~np.isfinite(data.measured_ac_mw),"interval_start_utc"].nunique()),
            "frozen_targets_unchanged":True,"all_methods_share_evaluation_mask":True,
            "forecast_and_target_cutoffs_valid":True,"native_scopes_and_dst_valid":True})
    result={"status":"passed","farms":farm_audits,
        "completed_method_checkpoints":sum(x["completed_method_checkpoints"] for x in farm_audits)}
    (root/"json"/"validation_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    comparisons=pd.read_csv(root/"csv"/"farm_method_comparison.csv")
    rankings=[]
    macro=[]
    for farm_id,group in comparisons.groupby("farm_id"):
        rating=read_json(root/f"farm_{farm_id}"/"json"/"summary.json")["ac_rating_mw"]
        fixed=group.loc[group.method=="fixed"].iloc[0]
        selected=group.loc[group.method=="selected"].iloc[0]
        best=group.loc[group.method!="selected"].sort_values("daylight_balanced_rmse_mw").iloc[0]
        rankings.append({"farm_id":int(farm_id),"ac_rating_mw":rating,
            "fixed_daylight_rmse_mw":fixed.daylight_balanced_rmse_mw,
            "selected_daylight_rmse_mw":selected.daylight_balanced_rmse_mw,
            "selected_improvement_percent":100*(1-selected.daylight_balanced_rmse_mw/fixed.daylight_balanced_rmse_mw),
            "best_family_in_hindsight":best.method,"best_family_daylight_rmse_mw":best.daylight_balanced_rmse_mw})
        for row in group.itertuples(index=False):
            macro.append({"farm_id":farm_id,"method":row.method,
                "daylight_nrmse_percent_rating":100*row.daylight_balanced_rmse_mw/rating,
                "rmse_improvement_percent":100*row.rmse_improvement_mw/row.baseline_rmse_mw})
    pd.DataFrame(rankings).to_csv(root/"csv"/"selected_and_hindsight_comparison.csv",index=False)
    pd.DataFrame(macro).groupby("method").agg(farms=("farm_id","nunique"),
        mean_daylight_nrmse_percent_rating=("daylight_nrmse_percent_rating","mean"),
        mean_rmse_improvement_percent=("rmse_improvement_percent","mean")).reset_index().to_csv(
            root/"csv"/"macro_method_comparison.csv",index=False)
    return result

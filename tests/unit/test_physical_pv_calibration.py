"""Physical parameter recovery, historical isolation and operational artifact parity."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data import FarmMetadata, ForecastBundle
from bess_pv_trading.forecasting.pv import forecast_pv
from bess_pv_trading.forecasting.pv_artifacts import fixed_contract, resolve_parameters
from bess_pv_trading.forecasting.pv_fitting import (balanced_weights, fit_one, fixed_fit,
    predict_fit, select_and_refit)
from bess_pv_trading.forecasting.pv_physical import parameter_defaults
from bess_pv_trading.forecasting.pv_study import validate_study
from bess_pv_trading.forecasting.pv_study_inputs import farm_settings
from bess_pv_trading.markets import AuctionCalendar


@pytest.fixture
def physical_case():
    """Vary irradiance, temperature, wind and solar angle independently enough to fit."""
    config=load_config("configs/pv_experiments/calibration_2025.yaml")
    settings=deepcopy(config.get("_pv_shared_settings",config["pv"]))
    farm=FarmMetadata(910,"NORD",config["farms"]["910"]["ac_capacity_kw"],1,30,0,22)
    days=pd.date_range("2025-01-01",periods=24,freq="13D",tz="UTC")
    times=pd.DatetimeIndex(np.concatenate([pd.date_range(day,periods=96,freq="15min").to_numpy() for day in days]))
    rng=np.random.default_rng(config["pv_calibration"]["seed"])
    hour=times.hour+times.minute/60
    radiation=np.maximum(0,np.sin((hour-5)/14*np.pi))*rng.uniform(200,1100,len(times))
    values=pd.DataFrame({"forecast_ghi":radiation,"forecast_bhi":radiation*rng.uniform(.1,.9,len(times)),
        "temperature_c":rng.uniform(-5,35,len(times)),"wind_u_10m":rng.uniform(0,10,len(times)),
        "wind_v_10m":rng.uniform(-2,2,len(times))},index=times)
    bundle=ForecastBundle(farm.farm_id,days[0]-pd.Timedelta(days=2),days[0]-pd.Timedelta(days=1),values,{})
    frame=forecast_pv(bundle,farm,settings,config["policy"]["night"]).values
    frame=frame.loc[frame.pv_input_valid & (frame.solar_zenith_deg<85)].copy()
    frame["interval_start_utc"]=frame.index
    frame["delivery_day"]=frame.index.strftime("%Y-%m-%d")
    frame["auction"]="DA"
    frame["target_available_at_utc"]=frame.index+pd.Timedelta(minutes=15)
    frame["daylight"]=True
    frame["eligible"]=True
    frame["measured_ac_mw"]=frame.pv_ac_mw
    frame["baseline_mw"]=frame.pv_ac_mw
    section={**config["pv_calibration"],"night_policy":config["policy"]["night"],
        "multistarts":1,"max_function_evaluations":2000}
    return config,farm,settings,section,frame.reset_index(drop=True)


@pytest.mark.parametrize("method,tracking",[("scale_mse",0),("thermal_mse",0),("tracker_mse",1),("bifacial_mse",3)])
def test_recovers_synthetic_physical_predictions(physical_case,method,tracking):
    _,farm,settings,section,frame=physical_case
    farm=FarmMetadata(**{**asdict(farm),"tracking_type_id":tracking})
    truth=fixed_fit(settings)
    truth["method"]=method
    truth["parameters"]["coefficient"]*=.8
    if not method.startswith("scale"):
        truth["parameters"]["u0"]*=1.1
        truth["parameters"]["u1"]*=.9
    if method.startswith("tracker"):
        truth["parameters"]["gcr"]*=1.1
        truth["parameters"]["max_angle"]*=.9
    if tracking==3:
        truth["bifacial_settings"]["enabled"]=True
        truth["bifacial_enabled"]=True
    frame["measured_ac_mw"]=predict_fit(frame,farm,settings,section["night_policy"],truth)
    fit=fit_one(frame,farm,settings,section,method,"isotropic",0,section["seed"])
    result=predict_fit(frame,farm,settings,section["night_policy"],fit)
    assert np.sqrt(np.mean((result-frame.measured_ac_mw)**2))/farm.ac_capacity_mw < 2e-4
    assert fit["parameters"]["coefficient"]==pytest.approx(truth["parameters"]["coefficient"],rel=.01)
    for parameter in fit["fitted_parameters"]:
        assert fit["parameters"][parameter]==pytest.approx(truth["parameters"][parameter],rel=1e-3)
    assert "reference_conversion_efficiency" not in fit["fitted_parameters"]


def test_clipping_and_huber_outages(physical_case):
    _,farm,settings,section,frame=physical_case
    truth=fixed_fit(settings)
    truth["parameters"]["coefficient"]=section["parameter_bounds"]["coefficient"][1]*.9
    frame["measured_ac_mw"]=predict_fit(frame,farm,settings,section["night_policy"],truth)
    fitted=fit_one(frame,farm,settings,section,"scale_mse","isotropic",0,section["seed"])
    assert fitted["parameters"]["coefficient"]==pytest.approx(truth["parameters"]["coefficient"],rel=1e-4)
    truth["parameters"]=parameter_defaults(settings)
    frame["measured_ac_mw"]=predict_fit(frame,farm,settings,section["night_policy"],truth)
    frame.loc[frame.index[::7],"measured_ac_mw"]=0
    mse=fit_one(frame,farm,settings,section,"scale_mse","isotropic",0,section["seed"])
    huber=fit_one(frame,farm,settings,section,"scale_huber","isotropic",0,section["seed"])
    true=truth["parameters"]["coefficient"]
    assert abs(huber["parameters"]["coefficient"]-true)<abs(mse["parameters"]["coefficient"]-true)


def test_balancing_does_not_overweight_longer_auction_horizons():
    data=pd.DataFrame({"delivery_day":["d"]*6,"auction":["DA"]*4+["IDA3"]*2})
    weights=balanced_weights(data)
    assert weights[:4].sum()==pytest.approx(weights[4:].sum())


def test_future_targets_do_not_change_prior_fit(physical_case):
    _,farm,settings,section,frame=physical_case
    section.update(training_windows_days=[None],validation_days=20,minimum_training_days=2,minimum_pairs=10)
    issue=pd.Timestamp("2025-09-01",tz="UTC")
    before,_=select_and_refit(frame,issue,farm,settings,section,"scale_mse")
    changed=frame.copy()
    changed.loc[changed.target_available_at_utc>issue,"measured_ac_mw"]*=10
    after,_=select_and_refit(changed,issue,farm,settings,section,"scale_mse")
    assert before["parameters"]==after["parameters"]


def test_artifact_cutoffs_farm_checks_and_operational_parity(physical_case,tmp_path):
    config,farm,settings,section,frame=physical_case
    issue=pd.Timestamp("2025-02-01",tz="UTC")
    fit=fixed_fit(settings)
    fit["parameters"]["coefficient"]*=.75
    fit["parameters"]["u0"]*=1.1
    fit["transposition_model"]="haydavies"
    artifact={"farm_metadata":asdict(farm),"issued_at_utc":issue.isoformat(),
        "latest_target_available_at_utc":(issue-pd.Timedelta(hours=1)).isoformat(),
        "fixed_contract":fixed_contract(settings,section["night_policy"]),"fit":fit}
    file=tmp_path/"fit.json";file.write_text(json.dumps(artifact))
    registry=tmp_path/"registry.json"
    registry.write_text(json.dumps({"schema_version":1,"farm_ids":[farm.farm_id],"entries":[{
        "farm_id":farm.farm_id,"issued_at_utc":issue.isoformat(),"artifact_path":file.name,
        "sha256":hashlib.sha256(file.read_bytes()).hexdigest()}]}))
    configured={**settings,"parameter_registry":str(registry)}
    resolved,status=resolve_parameters(configured,farm,section["night_policy"],issue-pd.Timedelta(seconds=1))
    assert status["status"]=="fixed_baseline_before_first_fit"
    chosen=frame.loc[frame.interval_start_utc>issue].iloc[:20]
    bundle=ForecastBundle(farm.farm_id,issue-pd.Timedelta(days=2),issue-pd.Timedelta(days=1),
        chosen.set_index("interval_start_utc"),{},provenance={"decision_cutoff_utc":issue.isoformat()})
    production=forecast_pv(bundle,farm,configured,section["night_policy"])
    expected=predict_fit(chosen,farm,settings,section["night_policy"],fit)
    np.testing.assert_allclose(production.values.pv_ac_mw,expected,rtol=1e-12)
    other=FarmMetadata(**{**asdict(farm),"farm_id":345})
    with pytest.raises(ValueError,match="requested farm"):
        resolve_parameters(configured,other,section["night_policy"],issue)


@pytest.mark.parametrize("day,count",[("2025-03-30",92),("2025-10-26",100)])
def test_study_auction_scopes_and_utc_availability_across_dst(day,count):
    config=load_config("configs/pv_experiments/calibration_2025.yaml")
    calendar=AuctionCalendar.from_config(config)
    grid=calendar.delivery_grid(day)
    assert len(grid.starts)==count
    ida3=next(x for x in calendar.schedule(day) if x.auction=="IDA3")
    starts=grid.starts[list(ida3.eligible_intervals)].tz_convert(config["market"]["timezone"])
    assert len(starts)==48 and starts[0].hour==12
    assert ida3.gate_at.tz_convert(config["market"]["timezone"]).hour==int(config["market"]["gates"]["IDA3"].split(":")[0])


def test_study_validation_rejects_invalid_bounds_and_naive_retraining():
    config=load_config("configs/pv_experiments/calibration_2025.yaml")
    with pytest.raises(ValueError,match="as_of"):
        validate_study(config,retrain=True)
    config["pv_calibration"]["parameter_bounds"]["gcr"]=[1,0]
    with pytest.raises(ValueError,match="bounds"):
        validate_study(config)


def test_per_farm_overrides_do_not_contaminate_other_farms():
    config=load_config("configs/pv_experiments/calibration_2025.yaml",overrides={
        "farms":{"910":{"pv":{"temperature":{"u0":40}}},"1":{"pv":None}}})
    original=load_config("configs/standard.yaml")
    assert config["pv"]["temperature"]["u0"]==40
    other=FarmMetadata(1,"NORD",original["farms"]["1"]["ac_capacity_kw"],0,15,0,25)
    shared=original.get("_pv_shared_settings",original["pv"])
    assert farm_settings(config,other)["temperature"]["u0"]==shared["temperature"]["u0"]


def test_float32_nameplate_roundoff_does_not_block_artifacts(physical_case,tmp_path):
    _,farm,settings,section,_=physical_case
    issue=pd.Timestamp("2025-02-01",tz="UTC")
    artifact={"farm_metadata":asdict(farm),"issued_at_utc":issue.isoformat(),
        "latest_target_available_at_utc":issue.isoformat(),
        "fixed_contract":fixed_contract(settings,section["night_policy"]),"fit":fixed_fit(settings)}
    artifact["fixed_contract"]["ac_limit_mw"]*=1+1e-8
    file=tmp_path/"fit.json";file.write_text(json.dumps(artifact))
    registry=tmp_path/"registry.json"
    registry.write_text(json.dumps({"schema_version":1,"farm_ids":[farm.farm_id],"entries":[{
        "farm_id":farm.farm_id,"issued_at_utc":issue.isoformat(),"artifact_path":file.name,
        "sha256":hashlib.sha256(file.read_bytes()).hexdigest()}]}))
    configured={**settings,"parameter_registry":str(registry)}
    _,status=resolve_parameters(configured,farm,section["night_policy"],issue)
    assert status["status"]=="fitted"
    configured=deepcopy(configured)
    configured["dc"]["reference_conversion_efficiency"]*=.9
    with pytest.raises(ValueError,match="assumptions mismatch"):
        resolve_parameters(configured,farm,section["night_policy"],issue)


@pytest.mark.parametrize("farm_id",load_config("configs/standard.yaml")["pv_calibration"]["farm_ids"])
def test_global_farm_parameters_load_in_experiment_and_apply_before_fitting_year(physical_case,farm_id):
    from pathlib import Path
    config=load_config("configs/full_experiments/continuous_optimization_2025-01.yaml",
                       overrides={"farm_id":farm_id})
    artifact=json.loads(Path(f"configs/pv_parameters/2025/farm_{farm_id}.json").read_text())
    fit=artifact["fit"]
    assert config["pv"]["output"]["calibrated_coefficient"]==fit["parameters"]["coefficient"]
    assert config["pv"]["temperature"]["u0"]==fit["parameters"]["u0"]
    assert config["pv"]["transposition_model"]==fit["transposition_model"]
    _,_,_,_,frame=physical_case
    values=frame.iloc[:20].set_index("interval_start_utc")
    cutoff=pd.Timestamp("2024-12-31T00:00Z")
    weather=ForecastBundle(farm_id,cutoff-pd.Timedelta(days=1),cutoff,values,{},
                           provenance={"decision_cutoff_utc":cutoff.isoformat()})
    forecast=forecast_pv(weather,FarmMetadata(**artifact["farm_metadata"]),config["pv"],config["policy"]["night"])
    assert forecast.provenance["pv_output_model"]["calibration_source"]=="configured_static_metadata"
    assert forecast.provenance["pv_output_model"]["calibrated_dc_to_ac_scale"]==fit["parameters"]["coefficient"]
    assert forecast.provenance["parameter_artifact"]["status"]=="disabled"


def test_ensemble_physical_features_ignore_measured_targets(physical_case):
    """Tracking and rear-side inputs must not expose either production or satellite targets."""
    from bess_pv_trading.forecasting.pv_ensemble_study import physical_features
    config, farm, settings, _, frame = physical_case
    farm = FarmMetadata(**{**asdict(farm), 'tracking_type_id': 3})
    before = physical_features(frame, farm, settings, config['policy']['night'])
    changed = frame.copy()
    changed['measured_ac_mw'] = np.nan
    changed['satellite_ghi_w_m2'] = np.nan
    after = physical_features(changed, farm, settings, config['policy']['night'])
    pd.testing.assert_frame_equal(before, after)
    assert (before.rear_poa_back_w_m2 > 0).any()
    assert before.front_surface_tilt_deg.nunique() > 1

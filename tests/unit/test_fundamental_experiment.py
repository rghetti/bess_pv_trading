"""Regional forecast alignment, chronological fitting and reproducible outputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.fundamental_experiment import (
    align_interval_forecast, compare_fundamental_models, run_fundamental_experiment,
)
from bess_pv_trading.forecasting.price_experiment import build_retrospective_features


def sources(starts,minutes,values):
    """Build invented interval forecasts whose publication remains unverified."""
    return pd.DataFrame({"product_start":starts,"product_end":starts+pd.Timedelta(minutes=minutes),
        "value_mw":values,"unit":"MW","retrieved_at":pd.Timestamp("2026-10-16T00:00:00Z"),
        "historical_feature_eligible":False})


def settings(**overrides):
    """Keep the same declared split and explicit feature comparisons as the prototype."""
    return {"timezone":"Europe/Rome","holdout_days":7,"min_training_days":14,"alpha":1.0,
        "fundamental_features":{"load_mw":{"provider":"entsoe","dataset":"load_forecast"},
                                "solar_mw":{"provider":"entsoe","dataset":"solar_forecast"}},
        "comparison_models":{"ridge_lags":[],"ridge_fundamentals":["load_mw","solar_mw"]},
        "native_product_regimes":[{"start":"2025-01-01","end":None,"minutes":60}],**overrides}


def feature_fixture():
    """Only current regional load predicts the injected independent target signal."""
    clock=pd.date_range("2025-01-01","2025-02-01",freq="h",inclusive="left",tz="Europe/Rome").tz_convert("UTC")
    generator=np.random.default_rng(17)
    load=generator.uniform(20,100,len(clock))
    solar=np.maximum(0,np.sin(2*np.pi*(clock.hour.to_numpy()-6)/24))
    frame=pd.DataFrame({"product_start":clock,"product_end":clock+pd.Timedelta(hours=1),
        "price_eur_mwh":40+2*load+10*solar,"native_interval_minutes":60,
        "retrieved_at":pd.Timestamp("2026-10-16T00:00:00Z")})
    features=build_retrospective_features(frame,"Europe/Rome")
    features["load_mw"],features["solar_mw"]=load,solar
    return features


def test_forecasts_are_duration_weighted_only_for_complete_products():
    starts=pd.date_range("2025-01-01T00:00:00Z",periods=2,freq="h")
    targets=pd.DataFrame({"product_start":starts,"product_end":starts+pd.Timedelta(hours=1)})
    quarters=pd.date_range(starts[0],periods=8,freq="15min")
    frame=sources(quarters,15,np.arange(8,dtype=float))
    aligned,coverage=align_interval_forecast(targets,frame)
    assert np.allclose(aligned,[1.5,5.5]) and coverage["complete_products"]==2
    assert coverage["source_interval_minutes"]==[15.0]
    missing,_=align_interval_forecast(targets,frame.drop(index=1))
    assert np.isnan(missing[0]) and missing[1]==5.5
    unequal=frame.iloc[:2].copy()
    unequal.loc[unequal.index[-1],"product_end"]=starts[0]+pd.Timedelta(hours=1)
    unequal["value_mw"]=[4,8]
    weighted,_=align_interval_forecast(targets.iloc[:1],unequal)
    assert weighted[0]==pytest.approx(7)


def test_alignment_normalizes_timestamp_storage_precision():
    starts=pd.date_range("2025-01-01T00:00:00Z",periods=2,freq="h").as_unit("us")
    targets=pd.DataFrame({"product_start":starts,"product_end":starts+pd.Timedelta(hours=1)})
    quarters=pd.date_range(starts[0],periods=8,freq="15min").as_unit("ns")
    aligned,coverage=align_interval_forecast(targets,sources(quarters,15,[1]*4+[2]*4))
    assert aligned.tolist()==[1,2] and coverage["source_interval_minutes"]==[15.0]


def test_forecast_alignment_keeps_repeated_dst_hours_separate():
    starts=pd.date_range("2025-10-26T00:00:00Z",periods=2,freq="h")
    targets=pd.DataFrame({"product_start":starts,"product_end":starts+pd.Timedelta(hours=1)})
    frame=sources(pd.date_range(starts[0],periods=8,freq="15min"),15,[1]*4+[2]*4)
    aligned,_=align_interval_forecast(targets,frame)
    assert np.array_equal(aligned,[1,2])


def test_forecasts_with_overlap_wrong_units_or_coarser_intervals_are_rejected():
    starts=pd.date_range("2025-01-01T00:00:00Z",periods=4,freq="15min")
    targets=pd.DataFrame({"product_start":starts,"product_end":starts+pd.Timedelta(minutes=15)})
    frame=sources(starts,15,[1,2,3,4])
    wrong_units=frame.assign(unit="GWh")
    with pytest.raises(ValueError,match="finite MW"):
        align_interval_forecast(targets,wrong_units)
    with pytest.raises(ValueError,match="coarser forecasts"):
        align_interval_forecast(targets,sources(starts[:1],60,[1]))
    overlap=frame.copy()
    overlap.loc[0,"product_end"]+=pd.Timedelta(minutes=1)
    with pytest.raises(ValueError,match="overlap"):
        align_interval_forecast(targets,overlap)


def test_fundamental_fit_uses_training_only_and_captures_independent_forecast_signal():
    features=feature_fixture()
    predictions,metrics,models=compare_fundamental_models(features,settings())
    assert metrics[0]["training_rows"]==408 and metrics[0]["holdout_rows"]==168
    scores=metrics[0]["models"]
    assert scores["ridge_fundamentals"]["mae_eur_mwh"]<scores["ridge_lags"]["mae_eur_mwh"]*.1
    altered=features.copy()
    altered.loc[altered.index[-1],"price_eur_mwh"]+=10000
    changed,_,changed_models=compare_fundamental_models(altered,settings())
    assert changed_models==models
    assert np.allclose(predictions.ridge_fundamentals_prediction,changed.ridge_fundamentals_prediction)
    incomplete=features.copy()
    incomplete.loc[incomplete.index[-1],"solar_mw"]=np.nan
    _,scores_missing,_=compare_fundamental_models(incomplete,settings())
    assert {score["rows"] for score in scores_missing[0]["models"].values()}=={167}


def test_realistic_run_freezes_joined_data_provenance_and_comparison_outputs(tmp_path,monkeypatch):
    from bess_pv_trading.forecasting import fundamental_experiment as experiment
    fixture=feature_fixture()
    price_columns=["product_start","product_end","price_eur_mwh","retrieved_at"]
    def load(_cache,*,dataset,**_kwargs):
        if dataset=="day_ahead_prices":
            frame=fixture[price_columns].copy()
        else:
            name="load_mw" if dataset=="load_forecast" else "solar_mw"
            frame=sources(pd.DatetimeIndex(fixture.product_start),60,fixture[name].to_numpy())
        return frame,[{"manifest":f"synthetic/{dataset}/manifest.json","checksums_verified":True}]
    monkeypatch.setattr(experiment,"load_api_cache_records",load)
    config={"price_experiment":settings(start="2025-01-01",end="2025-02-01",
        output_directory=str(tmp_path/"outputs"),analysis_directory=str(tmp_path/"analysis"))}
    result=run_fundamental_experiment(config)
    run=Path(result["run_directory"])
    assert result["source_cache_count"]==3 and result["complete_days"]==31
    assert result["historical_publication_verified"] is False
    frozen=pd.read_parquet(run/"prices_with_fundamentals.parquet")
    assert len(frozen)==744 and frozen.load_mw.notna().all()
    assert (run/"comparison_60min.png").is_file()
    assert json.loads((run/"models.json").read_text())["60min"]["ridge_fundamentals"]["training_end_day"]=="2025-01-24"
    assert len(result["derived_monthly_directories"])==1

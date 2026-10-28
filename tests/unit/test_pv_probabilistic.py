"""Deterministic contract checks for the member-wise PV study."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bess_pv_trading.config import load_config
from bess_pv_trading.data import align_forecast
from bess_pv_trading.forecasting.pv import local_delivery_starts
from bess_pv_trading.forecasting.pv_outages import apply_episode_exclusions
from bess_pv_trading.forecasting.pv_probabilistic_data import _bundle, _cycle, align_cycle
from bess_pv_trading.forecasting.pv_probabilistic_metrics import raw_crps, quantile_mean, predict_calibrator
from bess_pv_trading.forecasting.pv_probabilistic_search import interval_weights, trial_plan


def test_ensemble_midnight_alignment_matches_existing_contract():
    """Missing local midnight weather stays missing while radiation uses its end label."""
    cycle = pd.Timestamp('2025-12-31T00:00:00Z')
    leads = pd.to_timedelta([22, 23, 24, 25, 26], unit='h').to_numpy()
    shape = (1, 1, 5, 2)
    radiation = np.array([np.nan, np.nan, 100, 200, 300])
    temperature = np.array([np.nan, np.nan, 8, 12, 16])
    ds = xr.Dataset({'radiation': (('farm_id','cycle_time','lead_time','member'), np.tile(radiation[:,None], (1,2)).reshape(shape)),
                     'temperature': (('farm_id','cycle_time','lead_time','member'), np.tile(temperature[:,None], (1,2)).reshape(shape))},
        coords={'farm_id':[1], 'cycle_time':[cycle.tz_localize(None).to_datetime64()], 'lead_time':leads, 'member':[1,2]})
    config={'forecast_mapping':{'forecast_ghi':'radiation','temperature_c':'temperature'},
        'forecast_semantics':{'forecast_ghi':'hourly_mean','default':'instantaneous'},
        'source':{'availability_lag_hours':6},'policy':{'nonnegative_forecasts':['forecast_ghi']}}
    starts = local_delivery_starts('2026-01-01')[:8]
    fast = align_cycle(ds, 1, cycle, starts, config, True).values
    reference = align_forecast(_bundle(ds, 1, cycle, config, True), starts,
                               config['forecast_semantics'], config['policy']['nonnegative_forecasts']).values
    for name in ['forecast_ghi','temperature_c','forecast_ghi_raw_aligned','temperature_c_raw_aligned']:
        np.testing.assert_allclose(fast[name], reference[name], equal_nan=True)
    member = fast.loc[fast.member.eq('1')]
    np.testing.assert_allclose(member.forecast_ghi, [100]*4+[200]*4)
    assert member.temperature_c.iloc[:4].isna().all()
    np.testing.assert_allclose(member.temperature_c.iloc[4:], [8.5,9.5,10.5,11.5])
    assert _cycle('2026-01-01') == cycle


def test_member_weights_and_trial_budget():
    """Each observed interval contributes one unit irrespective of members."""
    times = pd.to_datetime(['2026-01-01T00:00Z']*3 + ['2026-01-01T00:15Z']*3)
    w = interval_weights(times, [True,True,False,True,True,True])
    np.testing.assert_allclose(w, [0.5,0.5,0,1/3,1/3,1/3])
    config=load_config('configs/pv_experiments/probabilistic_january_2026.yaml')
    trials=trial_plan(config['pv_probabilistic_study'])
    assert len(trials)==256
    assert len({(x['farm_id'], x['name']) for x in trials})==256


def test_crps_quantile_mean_and_bounded_rearrangement():
    np.testing.assert_allclose(raw_crps(np.array([[0.,2.]]), np.array([1.])), [0.5])
    np.testing.assert_allclose(quantile_mean(np.array([[0.,2.]]), [.25,.75]), [1.])
    class Scaler:
        def transform(self,x): return x
    class Estimator:
        def __init__(self,y): self.y=y
        def predict(self,x): return np.full(len(x), self.y)
    cal={'scaler':Scaler(),'models':[Estimator(.8),Estimator(-.2),Estimator(1.2)],
         'rating_mw':1.,'quantiles':[.1,.5,.9]}
    values, adjustment = predict_calibrator(cal, np.ones((1,50))*.5,
                                             pd.to_datetime(['2026-01-01T12:00Z']))
    np.testing.assert_allclose(values, [[0,.8,1]])
    assert adjustment['crossing_rows']==1 and adjustment['bounded_values']==2


def test_review_exclusion_boundaries_are_inclusive(tmp_path):
    path=tmp_path/'exclusions.csv'
    pd.DataFrame([{'farm_id':1,'start_utc':'2026-01-01T10:00Z','end_utc':'2026-01-01T10:15Z',
                   'decision':'exclude','reason':'reviewed_exclusion'}]).to_csv(path,index=False)
    times=pd.date_range('2026-01-01T09:45Z',periods=4,freq='15min')
    source=pd.DataFrame({'interval_start_utc':times,'eligible':True})
    out=apply_episode_exclusions(source, {'_project_root':str(tmp_path),'pv_data_quality':{'exclusions_csv':Path('exclusions.csv')}},farm_id=1)
    assert out.episode_excluded.tolist()==[False,True,True,False]
    assert out.eligible.tolist()==[True,False,False,True]


def test_all_predictor_families_fit_small_member_fixture():
    """Exercise each approved backend with a fixed, non-random five-day fixture."""
    from threadpoolctl import threadpool_limits
    from bess_pv_trading.forecasting.pv_probabilistic_search import add_time_features, fit_predictor, predict
    from bess_pv_trading.forecasting.pv_probabilistic_search import CORE, EXTRA
    rows=[]
    for day in pd.date_range('2026-01-01',periods=5):
        for member in ['1','2']:
            for quarter in range(8):
                row={'delivery_day':str(day.date()),'member':member,
                     'interval_start_utc':day.tz_localize('UTC')+pd.Timedelta(minutes=15*quarter),
                     'measured_ac_mw':.1+quarter*.04,'eligible':True}
                row.update({name: .2+quarter*.1+int(member)*.01 for name in CORE+EXTRA})
                rows.append(row)
    frame=add_time_features(pd.DataFrame(rows))
    config=load_config('configs/pv_experiments/probabilistic_january_2026.yaml')
    settings=config['pv_probabilistic_study'].copy()
    settings.update(max_epochs=2,early_stopping_patience=1,batch_size=8,tree_estimators=4,boosting_iterations=4)
    with threadpool_limits(limits=1):
        for family in settings['trial_parameters']:
            spec={'farm_id':1,'family':family,'features':'core','parameters':settings['trial_parameters'][family][0],
                  'name':family,'seed':42}
            model=fit_predictor(frame,spec,'2026-01-04',1.,settings)
            values=predict(model,frame,config)
            assert np.isfinite(values).all() and np.all((values>=0)&(values<=1))
            assert model['training']['training_days']==4
            assert model['training']['independent_target_intervals']==32


def test_quantiles_do_not_use_target_exclusion_to_hide_forecasts():
    """Retrospective review masks affect scoring, not weather-based prediction."""
    from bess_pv_trading.forecasting.pv_probabilistic_reporting import apply_calibrator
    class Scaler:
        def transform(self,x): return x
    class Estimator:
        def __init__(self,y): self.y=y
        def predict(self,x): return np.full(len(x), self.y)
    ids=[str(i) for i in range(1,51)]
    frame=pd.DataFrame([{**{i:.5 for i in ids},'interval_start_utc':pd.Timestamp('2026-01-01T12:00Z'),
                         'eligible':False,'episode_excluded':True}])
    cal={'scaler':Scaler(),'models':[Estimator(.2),Estimator(.4),Estimator(.6)],
         'rating_mw':1.,'quantiles':[.1,.5,.9]}
    out, flags=apply_calibrator(frame,ids,cal,[.1,.5,.9],{'enabled':False,'timezone':'Europe/Rome','start':'21:30','end':'05:00'})
    np.testing.assert_allclose(out[['calibrated_q10','calibrated_q50','calibrated_q90']].to_numpy(), [[.2,.4,.6]])
    assert flags['total_rows']==1


def test_validation_only_output_has_no_all_january_label():
    """Validation-only candidate predictions are not mislabeled as full January."""
    from bess_pv_trading.forecasting.pv_probabilistic_reporting import scored_windows
    ids=[str(i) for i in range(1,51)]
    frame=pd.DataFrame([{**{i:.5 for i in ids},'interval_start_utc':pd.Timestamp('2026-01-22T12:00Z'),
                         'delivery_day':'2026-01-22','split':'validation','measured_ac_mw':.5,
                         'deterministic_prediction':.5,'eligible':True,'episode_excluded':False}])
    scores=scored_windows(frame,ids,[.05,.1,.25,.5,.75,.9,.95],1.)
    assert 'all_january' not in scores.window.values
    assert 'all_available_validation' in scores.window.values


def test_fit_ignores_targets_after_fit_end():
    """Changing validation targets cannot change a January training fit."""
    from bess_pv_trading.forecasting.pv_probabilistic_search import add_time_features, fit_predictor, predict, CORE
    config=load_config('configs/pv_experiments/probabilistic_january_2026.yaml')
    rows=[]
    for day in pd.date_range('2026-01-01',periods=5):
        for member in ['1','2']:
            for hour in range(8,12):
                row={'delivery_day':str(day.date()),'member':member,'interval_start_utc':day.tz_localize('UTC')+pd.Timedelta(hours=hour),
                     'measured_ac_mw':hour/20,'eligible':True}
                row.update({name:float(hour)+int(member)/10 for name in CORE})
                rows.append(row)
    original=add_time_features(pd.DataFrame(rows)); changed=original.copy()
    changed.loc[changed.delivery_day.eq('2026-01-05'),'measured_ac_mw']=999.
    trial={'family':'ridge','features':'core','parameters':{'alpha':1.},'seed':1,'name':'ridge'}
    a=fit_predictor(original,trial,'2026-01-04',1.,config['pv_probabilistic_study'])
    b=fit_predictor(changed,trial,'2026-01-04',1.,config['pv_probabilistic_study'])
    np.testing.assert_allclose(predict(a,original,config),predict(b,original,config))


def test_hourly_aggregation_requires_four_paired_quarters():
    """An excluded quarter cannot silently become a complete hourly trajectory."""
    from bess_pv_trading.forecasting.pv_probabilistic_reporting import hourly_frame
    ids=[str(i) for i in range(1,51)]
    rows=[]
    for i, stamp in enumerate(pd.date_range('2026-01-22T10:00Z',periods=8,freq='15min')):
        rows.append({**{name:1. for name in ids},'interval_start_utc':stamp,'delivery_day':'2026-01-22',
                     'split':'validation','measured_ac_mw':1.,'deterministic_prediction':1.,
                     'eligible':i!=5,'episode_excluded':i==5})
    hourly=hourly_frame(pd.DataFrame(rows),ids)
    assert len(hourly)==1 and hourly.quarters.iloc[0]==4
    assert hourly.hour_start_utc.iloc[0]==pd.Timestamp('2026-01-22T10:00Z')


def test_two_archive_chronological_80_20_split_and_summer_cycle():
    """The August archive starts one day after its first usable next-day forecast."""
    from bess_pv_trading.forecasting.pv_probabilistic_data import study_days, delivery_split
    config=load_config('configs/pv_experiments/probabilistic_january_august_2026.yaml')
    section=config['pv_probabilistic_study']; days=study_days(section)
    train=[d for d in days if delivery_split(d,section)=='train']
    validation=[d for d in days if delivery_split(d,section)=='validation']
    assert (len(train),len(validation))==(49,12)
    assert train[-1]==pd.Timestamp('2026-08-19')
    assert validation[0]==pd.Timestamp('2026-08-20')
    assert pd.Timestamp('2026-08-01') not in days
    assert _cycle('2026-08-02')==pd.Timestamp('2026-08-01T00:00Z')
    assert local_delivery_starts('2026-08-02')[0]==pd.Timestamp('2026-08-01T22:00Z')
    for fold in section['calibration_folds']:
        assert fold['fit_end'] < fold['predict_start'] <= fold['predict_end'] <= section['train_end_day']


def test_scores_include_nameplate_normalized_errors():
    """MAE, RMSE, bias and CRPS percentages use installed AC capacity."""
    from bess_pv_trading.forecasting.pv_probabilistic_metrics import score_frame
    ids=[str(i) for i in range(1,51)]
    row={**{i:2. for i in ids},'measured_ac_mw':1.,'deterministic_prediction':2.,'eligible':True}
    result=score_frame(pd.DataFrame([row]),ids,[.05,.1,.25,.5,.75,.9,.95],4.,'validation')
    point=result.set_index('method').loc['member_mean']
    assert point.mae_mw==1 and point.nmae_percent_capacity==25
    assert point.nrmse_percent_capacity==25 and point.bias_percent_capacity==25
    probability=result.set_index('method').loc['raw_ensemble_distribution']
    assert probability.crps_mw==1 and probability.crps_percent_capacity==25


def test_calibration_rejects_future_targets_and_overlapping_folds():
    """Out-of-fold calibration may never consume validation or overlapping days."""
    from copy import deepcopy
    from bess_pv_trading.forecasting.pv_probabilistic_data import validate_study
    config=load_config('configs/pv_experiments/probabilistic_january_august_2026.yaml')
    section=deepcopy(config['pv_probabilistic_study'])
    assert len(validate_study(section))==61
    section['calibration_folds'][0]['predict_start']=section['calibration_folds'][0]['fit_end']
    with pytest.raises(ValueError,match='past training days'):
        validate_study(section)
    section=deepcopy(config['pv_probabilistic_study'])
    section['calibration_folds'][1]['predict_start']='2026-08-10'
    section['calibration_folds'][1]['predict_end']='2026-08-11'
    with pytest.raises(ValueError,match='non-overlapping'):
        validate_study(section)


def test_same_day_retrospective_uses_prior_cycle_only_before_utc_midnight():
    """The midnight fill respects winter/summer local offsets and retains both vintages."""
    from bess_pv_trading.forecasting.pv_probabilistic_data import day_vintage_segments
    for day, count in [('2026-01-02', 4), ('2026-08-02', 8)]:
        grid = local_delivery_starts(day)
        segments = day_vintage_segments(day, grid, 'same_day_retrospective')
        assert len(segments) == 2
        assert segments[0][0] == _cycle(day) and len(segments[0][1]) == count
        assert segments[1][0] == pd.Timestamp(day, tz='UTC') and len(segments[1][1]) == 96-count
        assert segments[0][1][-1] < segments[1][0] == segments[1][1][0]


def test_matched_vintage_comparison_checks_targets_and_exclusions():
    """Compare vintage strategies only where both forecast the same curated target."""
    from bess_pv_trading.forecasting.pv_probabilistic_compare import matched_frames
    ids=[str(i) for i in range(1,51)]
    rows=[]
    for i,t in enumerate(pd.date_range('2026-08-20T08:00Z',periods=2,freq='15min')):
        rows.append({**{name:.3 for name in ids},'interval_start_utc':t,'split':'validation',
                     'measured_ac_mw':.4,'deterministic_prediction':.3,'eligible':True,
                     'episode_excluded':False})
    a=pd.DataFrame(rows);b=a.copy();b.loc[1,'deterministic_prediction']=np.nan
    aa,bb=matched_frames(a,b,ids)
    assert len(aa)==len(bb)==1
    b=a.copy();b.loc[0,'measured_ac_mw']=.5
    with pytest.raises(ValueError,match='identical measured'):
        matched_frames(a,b,ids)

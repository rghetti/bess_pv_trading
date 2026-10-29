"""Check the weighted score, delayed feedback, and auction vintage selection."""
import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.online.pv_online_crps import MLPoly, weighted_crps, weighted_quantile
from bess_pv_trading.forecasting.online.pv_online_crps_prepare import choose_cycle, last_target_available
from bess_pv_trading.forecasting.plotting.pv_online_crps_report import score_one
from bess_pv_trading.markets import AuctionCalendar


def test_weighted_score_equals_direct_pairwise_formula_and_uniform():
    x=np.array([0., 1., 3.]); w=np.array([.1,.3,.6]); y=1.5
    expected=sum(w[i]*abs(x[i]-y) for i in range(3))-.5*sum(w[i]*w[j]*abs(x[i]-x[j]) for i in range(3) for j in range(3))
    assert weighted_crps(x,w,y)==pytest.approx(expected)
    assert weighted_crps([0.,2.],[.5,.5],1.)==pytest.approx(.5)
    assert weighted_crps([2.],[1.],1.)==pytest.approx(1.)


def test_mlpoly_update_uses_frozen_issued_weights_for_delayed_feedback():
    learner=MLPoly(3); issued=learner.weights.copy(); x=np.array([0., .5, 1.])
    learner.update(x,issued,0.)
    gradient=abs(x-.8)-abs(x[:,None]-x[None,:])@issued
    expected=issued@gradient-gradient
    actual=learner.update(x,issued,.8)
    assert actual==pytest.approx(expected)
    assert np.isclose(learner.weights.sum(),1.) and (learner.weights>=0).all()


def test_all_equal_experts_preserve_uniform_weights():
    learner=MLPoly(4)
    learner.update(np.ones(4),learner.weights.copy(),0.)
    assert learner.weights==pytest.approx(np.ones(4)/4)


def test_weighted_empirical_quantile_and_invalid_weights():
    assert weighted_quantile([3,1,2],[.5,.2,.3],[.1,.5,.9]).tolist()==[1,2,3]
    with pytest.raises(ValueError): weighted_crps([1,2],[.8,.8],1.)


@pytest.mark.parametrize('day', ['2026-01-20','2026-08-20'])
def test_latest_cycle_and_ida3_eligible_suffix(day):
    cal=AuctionCalendar(); events={e.auction:e for e in cal.schedule(day)}
    d=pd.Timestamp(day, tz='UTC')
    cycles=pd.date_range(d-pd.Timedelta(days=2),d+pd.Timedelta(days=1),freq='6h')
    expected={'DA':d-pd.Timedelta(days=1),'IDA1':d-pd.Timedelta(hours=18),
              'IDA2':d-pd.Timedelta(hours=12),'IDA3':d}
    for auction,event in events.items():
        cycle=choose_cycle(cycles,event.submission_at,6)
        assert cycle==expected[auction]
        assert cycle+pd.Timedelta(hours=6)<=event.submission_at
    local=cal.delivery_grid(day).starts.tz_convert('Europe/Rome')
    assert all(local[i].hour>=12 for i in events['IDA3'].eligible_intervals)
    previous=(pd.Timestamp(day)-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    assert last_target_available(previous,0)>events['DA'].submission_at
    assert last_target_available(previous,0)<events['IDA3'].submission_at


def test_dst_calendar_grid_and_calibrated_score():
    cal=AuctionCalendar(valid_to='2026-12-31')
    assert len(cal.delivery_grid('2026-03-29'))==92
    assert len(cal.delivery_grid('2026-10-25'))==100
    row={'values':np.array([0.,1.,2.]),'levels':np.array([.1,.5,.9]),'weights':None,'observed_mw':1.}
    score=score_one(row)
    assert score['median_mw']==pytest.approx(1.)
    assert score['coverage_80']==1
    assert score['crps_mw']>=0


def test_replay_respects_availability_and_freezes_weights(tmp_path, monkeypatch):
    from bess_pv_trading.forecasting.online import pv_online_crps_evaluate as replay
    from bess_pv_trading.forecasting.data.pv_probabilistic_data import file_hash, write_json
    root=tmp_path/'farm_1'; root.mkdir()
    ids=[str(i) for i in range(50)]
    levels=[.05,.1,.25,.5,.75,.9,.95]
    rows=[]
    for date in pd.date_range('2026-01-01',periods=8):
        for hour in (12,13):
            interval=(date+pd.Timedelta(hours=hour)).tz_localize('UTC')
            cutoff=interval.normalize()-pd.Timedelta(hours=14)
            row={'interval_start_utc':interval,'delivery_day':str(date.date()),'auction':'DA',
                 'decision_cutoff_utc':cutoff,'target_available_utc':interval+pd.Timedelta(minutes=15),
                 'deterministic_cycle_utc':cutoff-pd.Timedelta(hours=8),
                 'deterministic_available_utc':cutoff-pd.Timedelta(hours=2),
                 'model_fit_available_utc':cutoff-pd.Timedelta(days=1), 'retrospective_model':False,
                 'eligible':True,'episode_excluded':False,'measured_ac_mw':.4,
                 'deterministic_prediction':.5,'split':'train' if date.day<=6 else 'validation'}
            row.update({member:i/50 for i,member in enumerate(ids)})
            row.update({f'calibrated_q{round(q*100):02d}':q for q in levels})
            rows.append(row)
    data=root/'prepared.parquet'; pd.DataFrame(rows).to_parquet(data,index=False)
    write_json(root/'prepared_manifest.json',{'prepared_sha256':file_hash(data),'member_ids':ids,'rating_mw':1.})
    def fake_quantiles(history,current,levels,rating,alpha,night):
        q=np.tile(levels,(len(current),1)); flags=np.zeros(len(current),dtype=bool)
        assert (history.target_available_utc<=current.decision_cutoff_utc.iloc[0]).all()
        assert (history.decision_cutoff_utc<current.decision_cutoff_utc.iloc[0]).all()
        return q,q.copy(),flags,flags,flags
    monkeypatch.setattr(replay,'deterministic_quantiles',fake_quantiles)
    config={'pv_online_crps_study':{'quantiles':list(np.arange(.05,1,.05)), 'minimum_quantile_days':1,
             'quantile_alpha':.01,'block_hours':3}}
    source={'pv_probabilistic_study':{'quantiles':levels},'policy':{'night':{'enabled':False}}}
    replay.evaluate_farm_auction(config,source,1,'DA',tmp_path)
    forecasts=pd.read_parquet(root/'DA'/'forecasts.parquet')
    updates=pd.read_parquet(root/'DA'/'updates.parquet')
    assert (updates.target_available_utc<=updates.update_processed_at_utc).all()
    cold=forecasts.loc[(forecasts.method=='online_ensemble_cold') & (forecasts.delivery_day=='2026-01-07')]
    assert len(cold)==2
    assert all(np.allclose(w,np.ones(50)/50) for w in cold.weights)
    assert np.allclose(cold.weights.iloc[0],cold.weights.iloc[1])

"""Keep measured source values but exclude sustained daylight zero runs."""

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.pv_outages import (
    apply_sustained_zero_exclusion,sustained_daylight_zero_mask)
from bess_pv_trading.forecasting.pv_quality_review import _partial_recovery_mask
from bess_pv_trading.forecasting.pv_quality_review import run_pv_quality_review


def test_sustained_zero_requires_measured_daylight_and_contiguous_quarters():
    config=load_config('configs/pv_experiments/online_ridge_901_910.yaml')
    rule=config['pv_data_quality']['sustained_zero_outage']
    interval=config['source']['historical_interval_minutes']
    count=rule['minimum_consecutive_intervals']
    first=pd.date_range('2026-05-01T08:00Z',periods=count-1,freq=f'{interval}min')
    second=pd.date_range(first[-1]+pd.Timedelta(minutes=2*interval),
        periods=count,freq=f'{interval}min')
    times=first.append(second)
    power=np.zeros(len(times));ghi=np.full(len(times),rule['minimum_satellite_ghi_w_m2']+1)
    flagged=sustained_daylight_zero_mask(times,power,ghi,minimum_intervals=count,
        minimum_ghi_w_m2=rule['minimum_satellite_ghi_w_m2'],interval_minutes=interval)
    assert not flagged[:count-1].any()
    assert flagged[count-1:].all()
    low=ghi.copy();low[count-1:]=0
    assert not sustained_daylight_zero_mask(times,power,low,minimum_intervals=count,
        minimum_ghi_w_m2=rule['minimum_satellite_ghi_w_m2'],interval_minutes=interval).any()


def test_exclusion_preserves_raw_ac_and_applies_to_every_auction_view():
    config=load_config('configs/pv_experiments/online_ridge_901_910.yaml')
    rule=config['pv_data_quality']['sustained_zero_outage']
    count=rule['minimum_consecutive_intervals']
    times=pd.date_range('2026-05-01T09:00Z',periods=count+1,
        freq=f"{config['source']['historical_interval_minutes']}min")
    physical=pd.DataFrame({'interval_start_utc':times,
        'delivery_day':['2026-05-01']*(count+1),
        'measured_ac_mw':[0.0]*count+[.5],
        'satellite_ghi_w_m2':[rule['minimum_satellite_ghi_w_m2']+1]*(count+1),
        'daylight':[True]*(count+1),
        'eligible':[True]*(count+1)})
    frame=pd.concat([physical,physical],ignore_index=True)
    original=frame.measured_ac_mw.copy()
    curated=apply_sustained_zero_exclusion(frame,config)
    pd.testing.assert_series_equal(curated.measured_ac_mw,original)
    assert curated.outage_excluded.sum()==2*count
    assert curated.eligible.sum()==2
    assert curated.loc[curated.measured_ac_mw.eq(.5),'eligible'].all()


def test_review_csv_adds_exclusion_without_changing_measured_power(tmp_path):
    config=load_config('configs/pv_experiments/curate_sustained_zero.yaml')
    times=pd.date_range('2026-04-26T13:45Z',periods=2,
        freq=f"{config['source']['historical_interval_minutes']}min")
    frame=pd.DataFrame({'interval_start_utc':times,'delivery_day':['2026-04-26']*2,
        'measured_ac_mw':[.3536,.8849],'satellite_ghi_w_m2':[692.,653.],
        'daylight':[True,True],'eligible':[True,True],'review_excluded':[False,False]})
    review=tmp_path/'review.csv'
    pd.DataFrame([{'farm_id':910,'start_utc':times[0].isoformat(),
        'end_utc':times[0].isoformat(),'decision':'exclude'}]).to_csv(review,index=False)
    config['review']['exclusions_csv']=str(review)
    curated=apply_sustained_zero_exclusion(frame,config,farm_id=910)
    assert curated.measured_ac_mw.tolist()==frame.measured_ac_mw.tolist()
    assert curated.eligible.tolist()==[False,True]
    assert not curated.outage_excluded.any()
    assert apply_sustained_zero_exclusion(frame,config,farm_id=901).eligible.all()


def test_anchored_daytime_zero_run_includes_satellite_ghi_gap():
    config=load_config('configs/pv_experiments/curate_sustained_zero.yaml')
    rule=config['pv_data_quality']['sustained_zero_outage']
    count=rule['minimum_consecutive_intervals']
    times=pd.date_range('2026-04-25T08:00Z',periods=count+2,
        freq=f"{config['source']['historical_interval_minutes']}min")
    power=np.zeros(len(times))
    ghi=np.full(len(times),rule['minimum_satellite_ghi_w_m2']+1)
    ghi[2]=np.nan
    marked=sustained_daylight_zero_mask(times,power,ghi,minimum_intervals=count,
        minimum_ghi_w_m2=rule['minimum_satellite_ghi_w_m2'],
        interval_minutes=config['source']['historical_interval_minutes'],
        daylight=np.ones(len(times),dtype=bool),
        day_keys=np.array(['2026-04-25']*len(times)),expand_full_daylight_run=True)
    assert marked.all()


def test_short_partial_recovery_is_review_only_with_microsecond_timestamps():
    config=load_config('configs/pv_experiments/curate_sustained_zero.yaml')
    times=pd.date_range('2026-04-26T13:15Z',periods=7,freq='15min').astype('datetime64[us, UTC]')
    physical=pd.DataFrame({'interval_start_utc':times,
        'delivery_day':['2026-04-26']*7,
        'measured_ac_mw':[0,0,.3536,.8849,.2946,0,0],
        'daylight':[True]*7,
        'outage_excluded':[True,True,False,False,False,True,True]})
    marked=_partial_recovery_mask(physical,rating=config['farms']['910']['ac_capacity_kw']/1000,
        max_fraction=config['pv_data_quality_review']['partial_recovery_max_fraction_of_rating'],
        max_intervals=config['pv_data_quality_review']['partial_recovery_max_intervals'],
        interval_minutes=config['source']['historical_interval_minutes'])
    assert marked.tolist()==[False,False,True,True,True,False,False]
    assert not physical.loc[marked,'outage_excluded'].any()


def test_quality_review_emits_empty_review_queue_for_clean_source(tmp_path):
    config=load_config('configs/pv_experiments/audit_data_quality_2025_2026.yaml')
    source=tmp_path/'source'/'farm_910'/'parquet'
    source.mkdir(parents=True)
    times=pd.date_range('2026-04-26T09:00Z',periods=8,freq='15min')
    frame=pd.DataFrame({'interval_start_utc':times,
        'delivery_day':['2026-04-26']*len(times),
        'measured_ac_mw':[1.0]*len(times),
        'satellite_ghi_w_m2':[200.0]*len(times),
        'forecast_ghi':[200.0]*len(times),
        'daylight':[True]*len(times),
        'plateau_flag':[False]*len(times),
        'review_excluded':[False]*len(times),
        'eligible':[True]*len(times)})
    frame.to_parquet(source/'corpus.parquet',index=False)
    config['pv_data_quality_review'].update(farm_ids=[910],source_root=str(tmp_path/'source'),
        output_root=str(tmp_path/'audit'),run_id='clean',save_plots=False)
    result=run_pv_quality_review(config,progress=lambda _:None)
    queue=pd.read_csv(tmp_path/'audit'/'clean'/'csv'/'review_queue.csv')
    assert queue.empty
    assert {'farm_id','start_utc','end_utc','decision'}.issubset(queue.columns)
    assert result['summaries'][0]['candidate_episodes']==0

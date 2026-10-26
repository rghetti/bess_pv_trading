"""Guard causal availability and physical-interval counting for hourly PV correction."""

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.pv_hourly_bias import (
    causal_bucket_prediction, completed_bucket_statistics)


def test_completed_bucket_history_counts_overlap_once_and_waits_for_midnight():
    config=load_config('configs/pv_experiments/hourly_bias_901_910.yaml')
    options=config['pv_hourly_bias_study'];timezone=config['pv_neural']['delivery_timezone']
    day=options['history_start_day']
    midnight=pd.Timestamp(day).tz_localize(timezone)
    start=midnight+pd.Timedelta(hours=12)
    interval=pd.Timedelta(minutes=config['source']['historical_interval_minutes'])
    quarter=pd.DataFrame({'delivery_day':[day],'interval_start_utc':[start.tz_convert('UTC')],
        'target_available_at_utc':[(start+interval).tz_convert('UTC')],
        'measured_ac_mw':[1.0]})
    doubled=pd.concat([quarter,quarter],ignore_index=True)
    history=completed_bucket_statistics(doubled,np.array([.5,.5]),timezone=timezone,
        history_start_day=day,bucket_hours=options['bucket_hours'])
    assert history.interval_count.sum()==1
    assert history.residual_sum.sum()==pytest.approx(.5)
    assert history.available.iloc[0]==(midnight+pd.DateOffset(days=1)).tz_convert('UTC')
    changed=doubled.copy();changed.loc[1,'measured_ac_mw']=2.0
    with pytest.raises(ValueError,match='disagree'):
        completed_bucket_statistics(changed,np.array([.5,.5]),timezone=timezone,
            history_start_day=day,bucket_hours=options['bucket_hours'])


def test_future_measured_power_cannot_change_earlier_hourly_correction():
    config=load_config('configs/pv_experiments/hourly_bias_901_910.yaml')
    options=config['pv_hourly_bias_study'];timezone=config['pv_neural']['delivery_timezone']
    day=options['history_start_day'];midnight=pd.Timestamp(day).tz_localize(timezone)
    interval=pd.Timedelta(minutes=config['source']['historical_interval_minutes'])
    times=[midnight+pd.Timedelta(hours=12),midnight+pd.DateOffset(days=1)+pd.Timedelta(hours=12)]
    frame=pd.DataFrame({'delivery_day':[point.date().isoformat() for point in times],
        'interval_start_utc':[point.tz_convert('UTC') for point in times],
        'target_available_at_utc':[(point+interval).tz_convert('UTC') for point in times],
        'measured_ac_mw':[1.,2.]})
    history=completed_bucket_statistics(frame,np.array([.5,.5]),timezone=timezone,
        history_start_day=day,bucket_hours=options['bucket_hours'])
    forecast=frame.iloc[[1,1]].copy().reset_index(drop=True)
    forecast['decision_cutoff_utc']=[
        (midnight+pd.DateOffset(days=1)-pd.Timedelta(minutes=1)).tz_convert('UTC'),
        (midnight+pd.DateOffset(days=1)+pd.Timedelta(minutes=1)).tz_convert('UTC')]
    def predict(completed):
        return causal_bucket_prediction(forecast,np.array([.5,.5]),completed,timezone=timezone,
            bucket_hours=options['bucket_hours'],window_days=options['window_days'][0],
            ridge_pseudo_intervals=0,minimum_days=1,rating=3.)
    original=predict(history)
    altered=frame.copy();altered.loc[1,'measured_ac_mw']=10.
    revised=completed_bucket_statistics(altered,np.array([.5,.5]),timezone=timezone,
        history_start_day=day,bucket_hours=options['bucket_hours'])
    np.testing.assert_array_equal(original,predict(revised))
    assert original[0]==pytest.approx(.5)
    assert original[1]==pytest.approx(1.)

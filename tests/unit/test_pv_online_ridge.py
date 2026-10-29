"""Check cutoff causality and physical-quarter de-duplication of online ridge."""

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.online.pv_online_ridge import (
    causal_ridge_prediction,completed_day_moments,correction_features)


def _three_day_views():
    delivery=pd.to_datetime(['2026-01-01T08:00Z','2026-01-02T08:00Z',
        '2026-01-03T08:00Z'])
    frame=pd.DataFrame({'delivery_day':np.repeat(['2026-01-01','2026-01-02',
        '2026-01-03'],2),'interval_start_utc':np.repeat(delivery,2),
        'auction':['DA','IDA1']*3,
        'decision_cutoff_utc':np.repeat(pd.to_datetime([
            '2025-12-31T12:00Z','2026-01-01T12:00Z','2026-01-02T12:00Z']),2),
        'target_available_at_utc':np.repeat(pd.to_datetime([
            '2026-01-01T09:00Z','2026-01-02T09:00Z','2026-01-03T09:00Z']),2),
        'measured_ac_mw':np.repeat([.5,.4,.3],2),
        'forecast_ghi':[500]*6})
    return frame,np.full(len(frame),.2)


def _predict(frame,base):
    features,_=correction_features(frame,base,rating=1,timezone='Europe/Rome',family='offset')
    history=completed_day_moments(frame,base,features,rating=1,
        timezone='Europe/Rome',history_start_day='2026-01-01')
    forecast,coefficients=causal_ridge_prediction(frame,base,features,history,rating=1,
        window_days=7,ridge_pseudo_days=.1,minimum_days=1)
    return forecast,coefficients,history


def test_online_ridge_uses_each_physical_quarter_once_and_only_completed_days():
    frame,base=_three_day_views()
    predicted,coefficients,history=_predict(frame,base)
    assert [item['physical_quarters'] for item in history]==[1,1,1]
    np.testing.assert_allclose(predicted[:4],base[:4])
    assert predicted[4]>base[4]
    assert len(coefficients)==1
    assert coefficients.last_history_day.iloc[0]=='2026-01-01'
    assert pd.Timestamp(coefficients.last_history_available_utc.iloc[0])<=pd.Timestamp(
        coefficients.decision_cutoff_utc.iloc[0])
    future=frame.copy();future.loc[future.delivery_day.eq('2026-01-02'),'measured_ac_mw']=.9
    future.loc[future.delivery_day.eq('2026-01-03'),'measured_ac_mw']=.9
    np.testing.assert_allclose(_predict(future,base)[0],predicted)


def test_online_ridge_rejects_singular_unregularized_fit():
    frame,base=_three_day_views()
    features,_=correction_features(frame,base,rating=1,timezone='Europe/Rome',family='offset')
    with pytest.raises(ValueError,match='Invalid online ridge'):
        causal_ridge_prediction(frame,base,features,[],rating=1,window_days=7,
            ridge_pseudo_days=0,minimum_days=1)

"""Causal gain calibration from completed, observed production days; no aging model."""
from __future__ import annotations

import numpy as np
import pandas as pd


def completed_day_statistics(frame,prediction,timezone):
    """Count physical targets once, averaging overlapping forecast views as a reference.

    A calibration day is unavailable until all of its retained measurements have
    been observed. No target from the day being forecast can enter its pre-delivery
    calibration. Auction identity is neither a feature nor a calibration key.
    """
    eligible=frame.eligible.to_numpy(dtype=bool) if 'eligible' in frame else np.ones(len(frame),dtype=bool)
    data=frame.loc[eligible,
        ['delivery_day','interval_start_utc','target_available_at_utc','measured_ac_mw']].copy()
    data['reference']=np.asarray(prediction)[eligible]
    unique=data.groupby(['delivery_day','interval_start_utc'],sort=True).agg(
        reference=('reference','mean'),target=('measured_ac_mw','first'),
        available=('target_available_at_utc','max')).reset_index()
    valid=np.isfinite(unique.reference)&np.isfinite(unique.target)&(unique.target>=0)
    unique=unique.loc[valid].copy()
    unique['xx']=unique.reference**2;unique['xy']=unique.reference*unique.target
    daily=unique.groupby('delivery_day',sort=True).agg(xx=('xx','sum'),xy=('xy','sum'),
        available=('available','max'),quarters=('target','size')).reset_index()
    boundaries=pd.to_datetime([
        (pd.Timestamp(day).tz_localize(timezone)+pd.DateOffset(days=1)).tz_convert('UTC')
        for day in daily.delivery_day],utc=True)
    observed=pd.to_datetime(daily.available,utc=True)
    daily['available']=[max(a,b) for a,b in zip(observed,boundaries)]
    return daily.sort_values('available').reset_index(drop=True)


def causal_gain_prediction(frame,prediction,history,*,window_days,minimum_days,gain_bounds):
    """Issue one constant gain per cutoff using only already observed calibration days."""
    prediction=np.asarray(prediction);gains=np.ones(len(frame),dtype=float)
    available=pd.DatetimeIndex(history.available).as_unit('ns').asi8
    for cutoff,positions in frame.groupby('decision_cutoff_utc',sort=False).indices.items():
        moment=pd.Timestamp(cutoff)
        upper=available.searchsorted(moment.value,side='right')
        lower=available.searchsorted((moment-pd.Timedelta(days=window_days)).value,side='left')
        known=history.iloc[lower:upper]
        if len(known)>=minimum_days and known.xx.sum()>0:
            gain=np.clip(known.xy.sum()/known.xx.sum(),*gain_bounds)
            gains[positions]=gain
    return prediction*gains,gains

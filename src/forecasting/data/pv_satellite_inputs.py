"""Shared retrospective satellite-GHI replacement and physical-time interpolation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def interpolate_satellite(frame, method):
    """Fill interior gaps once on the physical UTC timeline, shared by auction views."""
    # Enforce the input contract: Satellite GHI gap fill must be none or linear_inside.
    if method not in {'none','linear_inside'}:
        raise ValueError('Satellite GHI gap fill must be none or linear_inside')
    grouped=frame.groupby('interval_start_utc',sort=True).satellite_ghi_w_m2
    # Enforce the input contract: Auction views disagree on observed satellite GHI.
    if grouped.nunique(dropna=False).gt(1).any():
        raise ValueError('Auction views disagree on observed satellite GHI')
    raw=grouped.first()
    raw.index=pd.DatetimeIndex(raw.index)
    valid=np.isfinite(raw)&raw.ge(0)
    working=raw.where(valid)
    # When `method == 'linear_inside'`, set `working` using `working.interpolate(method='time', limit_area='inside')`.
    if method=='linear_inside':
        working=working.interpolate(method='time',limit_area='inside')
    filled=(~np.isfinite(raw))&working.notna()
    # Invalid negative observations remain invalid, and raw measurements are preserved.
    working=working.where(raw.ge(0)|~np.isfinite(raw))
    result=frame.copy()
    result['satellite_ghi_input_w_m2']=working.reindex(frame.interval_start_utc).to_numpy()
    result['satellite_ghi_interpolated']=filled.reindex(frame.interval_start_utc).to_numpy()
    return result


def satellite_inputs(frame, settings):
    """Replace GHI, preserving forecast direct fraction; infer undefined fractions with Erbs.

    Observed GHI alone does not specify direct/diffuse radiation. Other weather
    and all saved model parameters remain unchanged. The supplied GHI includes
    explicitly recorded interior interpolation; unbracketed gaps remain missing.
    """
    result=frame.copy(deep=True)
    ghi=frame.satellite_ghi_input_w_m2.to_numpy(dtype=float)
    valid=np.isfinite(ghi)&(ghi>=0)
    old=frame.forecast_ghi.to_numpy(dtype=float)
    direct=frame.forecast_bhi.to_numpy(dtype=float)
    ratio=np.divide(direct,old,out=np.full(len(frame),np.nan),where=old>0)
    ratio=np.clip(ratio,0,1)
    bhi=ghi*ratio
    fallback=valid&(ghi>0)&~np.isfinite(ratio)
    times=pd.DatetimeIndex(frame.interval_start_utc)+pd.Timedelta(minutes=7.5)
    zenith=frame.solar_zenith_deg.to_numpy(dtype=float)
    # Update `decomposition` when this condition is met.
    if fallback.any():
        decomposition=pvlib.irradiance.erbs(ghi[fallback],zenith[fallback],times[fallback])
        bhi[fallback]=np.clip(ghi[fallback]-np.asarray(decomposition['dhi']),0,ghi[fallback])
    bhi[valid&(ghi==0)]=0
    ghi=np.where(valid,ghi,np.nan)
    dhi=ghi-bhi
    dni=np.asarray(pvlib.irradiance.dni(pd.Series(ghi),pd.Series(dhi),pd.Series(zenith),
        zenith_threshold_for_zero_dni=float(settings['low_sun_zenith_deg'])))
    dni=np.where(zenith>=float(settings['low_sun_zenith_deg']),0,dni)
    limited=np.isfinite(dni)&(dni>float(settings['dni_limit_w_m2']))
    # When `settings['dni_limit_action'] == 'cap_and_flag'`, set `dni` using `np.minimum(dni, float(settings['dni_limit_w_m2']))`.
    if settings['dni_limit_action']=='cap_and_flag':
        dni=np.minimum(dni,float(settings['dni_limit_w_m2']))
    # When `settings['dni_limit_action'] == 'invalidate'`, set `valid` using `~limited`.
    elif settings['dni_limit_action']=='invalidate':
        valid&=~limited
    else:
        raise ValueError('Unsupported DNI limit action')
    # Process each `(name, values)` in this collection.
    for name,values in {'forecast_ghi':ghi,'forecast_bhi':bhi,'ghi_w_m2':ghi,
        'bhi_w_m2':bhi,'dhi_w_m2':dhi,'dni_w_m2':dni,
        'pv_input_valid':valid&np.isfinite(bhi)&np.isfinite(dni),
        'satellite_ghi_valid':np.isfinite(ghi),'direct_fraction_erbs_fallback':fallback}.items():
        result[name]=values
    return result



"""Check that diagnostic grouping preserves targets and the shared error population."""

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.evaluation.pv_error_diagnostics import diagnose_farm_errors


def test_weather_diagnostics_preserve_measured_ac_and_partition_error():
    config=load_config('configs/pv_experiments/error_diagnostics_901_910.yaml')
    options=config['pv_error_diagnostics'].copy()
    options['ghi_quantile_count']=2
    starts=pd.date_range('2026-05-01T08:00Z',periods=4,
        freq=f"{config['source']['historical_interval_minutes']}min")
    original=pd.DataFrame({'interval_start_utc':starts,
        'delivery_day':['2026-05-01']*4,'auction':['DA']*4,
        'eligible':[True]*4,'daylight':[True]*4,'split':['test']*4,
        'measured_ac_mw':[.2]*4,'forecast_ghi':[300,310,320,330],
        'satellite_ghi_w_m2':[300,300,300,np.nan],
        'suspected_outage':[False,True,False,False],
        'plateau_flag':[False]*4})
    observed=original.measured_ac_mw.copy()
    rating=config['farms']['1']['ac_capacity_kw']/1000
    forecast=np.array([.2,.3,.2,.5])
    result,groups,hours,auctions,views,labels=diagnose_farm_errors(original,forecast,
        rating=rating,timezone=config['pv_neural']['delivery_timezone'],section=options)
    pd.testing.assert_series_equal(observed,original.measured_ac_mw)
    assert result['views']==len(original)
    assert result['satellite_ghi_coverage']==pytest.approx(.75)
    partition=groups[groups.group.isin(labels+['Satellite GHI missing'])]
    assert partition.views.sum()==len(original)
    assert partition.squared_error_share.sum()==pytest.approx(1)
    assert groups.loc[groups.group.eq('Flagged: suspected_outage'),'views'].iloc[0]==1
    assert len(hours)==1 and auctions.auction.tolist()==['DA']

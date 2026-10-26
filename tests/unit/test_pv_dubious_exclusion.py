"""Approved target exclusions are immutable across vintages and modeling splits."""
import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.pv_outages import apply_sustained_zero_exclusion, modeling_instances
from bess_pv_trading.forecasting.pv_neural_data import pack_sequences
from bess_pv_trading.forecasting.pv_satellite_inputs import interpolate_satellite


def test_approved_episodes_and_plateaus_mask_all_splits_and_views(tmp_path):
    config=load_config('configs/pv_experiments/satellite_ghi_cleaned_benchmark.yaml')
    times=pd.date_range('2026-10-25T00:30Z',periods=4,freq='15min')
    path=tmp_path/'episodes.csv'
    pd.DataFrame([{'farm_id':1,'start_utc':times[0].isoformat(),
        'end_utc':times[1].isoformat(),'decision':'exclude','reason':'low_power'},
        {'farm_id':1,'start_utc':times[2].isoformat(),'end_utc':times[2].isoformat(),
         'decision':'exclude','reason':'nonzero_plateau'},
        {'farm_id':345,'start_utc':times[3].isoformat(),'end_utc':times[3].isoformat(),
         'decision':'exclude','reason':'high_power'}]).to_csv(path,index=False)
    config['pv_data_quality']['exclusions_csv']=str(path)
    physical=pd.DataFrame({'interval_start_utc':times,'delivery_day':['2026-10-25']*4,
        'measured_ac_mw':[.1,.2,.3,.4],'satellite_ghi_w_m2':[500.]*4,
        'daylight':[True]*4,'eligible':[True]*4,'plateau_flag':[False,False,True,False]})
    frame=pd.concat([physical.assign(split=split,auction=auction)
        for split in ['train','validation','test'] for auction in ['DA','IDA1']],ignore_index=True)
    result=apply_sustained_zero_exclusion(frame,config,farm_id=1)
    np.testing.assert_array_equal(result.measured_ac_mw,frame.measured_ac_mw)
    assert result.groupby(['split','auction']).eligible.sum().eq(1).all()
    assert result.loc[result.interval_start_utc.eq(times[-1]),'eligible'].all()
    assert result.groupby('interval_start_utc').dubious_excluded.nunique().eq(1).all()
    changed=frame.copy();changed.loc[result.dubious_excluded,'measured_ac_mw']=999
    repeated=apply_sustained_zero_exclusion(changed,config,farm_id=1)
    np.testing.assert_array_equal(result.eligible,repeated.eligible)
    cleaned=modeling_instances(frame,config,farm_id=1)
    changed_cleaned=modeling_instances(changed,config,farm_id=1)
    pd.testing.assert_frame_equal(cleaned,changed_cleaned)
    assert len(cleaned)==6
    assert cleaned.interval_start_utc.eq(times[-1]).all()
    assert not cleaned.dubious_excluded.any()


def test_satellite_interpolation_keeps_edges_missing_across_dst():
    times=pd.date_range('2025-10-26T00:30Z',periods=7,freq='15min')
    frame=pd.DataFrame({'interval_start_utc':list(times)*2,
        'satellite_ghi_w_m2':[np.nan,100,np.nan,np.nan,400,500,np.nan]*2})
    result=interpolate_satellite(frame,'linear_inside')
    np.testing.assert_allclose(result.satellite_ghi_input_w_m2.iloc[:7],
        [np.nan,100,200,300,400,500,np.nan],equal_nan=True)
    np.testing.assert_array_equal(result.satellite_ghi_w_m2,frame.satellite_ghi_w_m2)
    assert result.satellite_ghi_interpolated.sum()==4


def test_neural_segments_do_not_join_across_removed_quarters():
    times=pd.date_range('2026-05-01T08:00Z',periods=6,freq='15min').delete([2,3])
    frame=pd.DataFrame({'interval_start_utc':times,'delivery_day':['2026-05-01']*4,
        'auction':['DA']*4,'split':['train']*4,'forecast_cycle_utc':[times[0]]*4,
        'measured_ac_mw':[.1,.2,.5,.6],'eligible':[True]*4,'daylight':[True]*4})
    packed=pack_sequences(frame,np.ones((4,2),dtype=np.float32),1.0)
    assert packed['indices'].tolist()==[[0,1],[2,3]]
    assert packed['view_ids'].tolist()==[0,0]
    assert packed['y'].shape==(2,2)


def test_csv_is_the_only_runtime_classification_source(tmp_path):
    config=load_config('configs/pv_experiments/cleaned_all_farms_hpo.yaml')
    times=pd.date_range('2026-10-25T08:00Z',periods=5,freq='15min')
    frame=pd.DataFrame({'interval_start_utc':times,'delivery_day':['2026-10-25']*5,
        'measured_ac_mw':[0.]*5,'satellite_ghi_w_m2':[900.]*5,
        'daylight':[True]*5,'eligible':[True]*5,'plateau_flag':[True]*5})
    path=tmp_path/'episodes.csv'
    pd.DataFrame(columns=['farm_id','start_utc','end_utc','decision','reason']).to_csv(path,index=False)
    config['pv_data_quality']['exclusions_csv']=str(path)
    assert len(modeling_instances(frame,config,farm_id=1))==len(frame)
    pd.DataFrame([{'farm_id':1,'start_utc':times[1].isoformat(),'end_utc':times[2].isoformat(),
        'decision':'exclude','reason':'sustained_zero_outage'}]).to_csv(path,index=False)
    result=modeling_instances(frame,config,farm_id=1)
    assert result.interval_start_utc.tolist()==[times[0],times[3],times[4]]
    assert not result.episode_excluded.any()

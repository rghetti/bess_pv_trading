"""Prepared modeling data must not depend on deleted historical experiments."""
import numpy as np
import pandas as pd
from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.pv_forecast_search import prepare_search


def test_frame_override_does_not_read_old_prediction_files(tmp_path):
    config=load_config('configs/pv_experiments/timing_features_restored.yaml')
    config['pv_data_quality']['exclusions_csv']=None
    config['pv_forecast_search'].update(source_root=str(tmp_path/'deleted_experiment'),feature_sets=['core'])
    blocks=[]
    for day,split in [('2025-01-02','train'),('2026-01-02','validation'),('2026-05-02','test')]:
        starts=pd.date_range(day+'T08:00Z',periods=4,freq=f"{config['source']['historical_interval_minutes']}min")
        cycle=starts[0]-pd.Timedelta(days=1)
        blocks.append(pd.DataFrame({'interval_start_utc':starts,'delivery_day':day,'auction':'DA','split':split,
            'forecast_cycle_utc':cycle,'forecast_available_at_utc':cycle+pd.Timedelta(hours=config['source']['availability_lag_hours']),
            'decision_cutoff_utc':starts[0]-pd.Timedelta(hours=12),'target_available_at_utc':starts+pd.Timedelta(minutes=15),
            'measured_ac_mw':.5,'satellite_ghi_w_m2':500.,'daylight':True,'eligible':True,'plateau_flag':False,
            'forecast_ghi':500.,'forecast_bhi':300.,'temperature_c':20.,'wind_u_10m':1.,'wind_v_10m':0.,
            'lead_hours':np.arange(4)/4+24,'baseline_mw':.5}))
    frame=pd.concat(blocks,ignore_index=True)
    result,_,_,cache,_,_=prepare_search(config,config['farm_id'],tmp_path/'new_run','identity',
        frame_override=frame,sequence_length=len(blocks[0]))
    np.testing.assert_array_equal(result.measured_ac_mw,frame.measured_ac_mw)
    assert np.load(cache/'x_core.npy').shape==(len(frame),15)
    assert not (tmp_path/'deleted_experiment').exists()

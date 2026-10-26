"""Separate GHI-correctable errors from unresolved residuals on fixed retained instances."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from benchmark_satellite_ghi import write,sha
from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path
from bess_pv_trading.forecasting.pv_fitting import balanced_weights


def situations(table,section):
    """Partition profile evidence without interpreting a residual as a confirmed fault."""
    result=table.copy()
    result['ghi_squared_error_reduction_fraction']=1-np.square(
        result.satellite_input_nrmse_percent/np.maximum(result.nrmse_percent,np.finfo(float).eps))
    helps=result.ghi_squared_error_reduction_fraction.ge(section['material_ghi_sse_reduction'])
    residual=result.satellite_input_nrmse_percent.ge(section['large_residual_nrmse_percent'])
    result['situation']=np.select([helps&~residual,helps&residual,~helps&residual],
        ['ghi_correction_largely_resolves','ghi_helps_large_residual_remains','large_residual_little_ghi_benefit'],
        default='small_or_mixed_error')
    return result


def run(config):
    """Summarize matched input counterfactuals, worst profiles and cross-year behavior."""
    section=config['pv_retrained_error_diagnostics']
    root=config_path(config,section['output_root'])/section['run_id']
    profiles=pd.read_csv(root/'csv/profile_ranking.csv')
    episodes=pd.read_csv(root/'csv/review_candidates.csv')
    persistent=episodes.loc[episodes.reason.str.startswith('persistent')].groupby(
        ['farm_id','delivery_day','auction']).duration_hours.sum().rename('persistent_low_hours').reset_index()
    extra=[];auction_rows=[];matched=[]
    for farm in section['farm_ids']:
        f=pd.read_parquet(root/f'farm_{farm}/parquet/diagnostic_instances.parquet')
        rating=config['farms'][str(farm)]['ac_capacity_kw']/1000
        selected=f.loc[f.score_eligible].copy()
        for name,column in [('previous','without_timing_mw'),('physical_sat','pvlib_satellite_mw')]:
            selected[f'{name}_mse']=np.square((selected[column]-selected.measured_ac_mw)/rating*100)
        for (split,day,auction),part in selected.groupby(['split','delivery_day','auction']):
            extra.append({'farm_id':farm,'split':split,'delivery_day':day,'auction':auction,
                'without_timing_nrmse_percent':float(np.sqrt(part.previous_mse.mean())),
                'pvlib_satellite_nrmse_percent':float(np.sqrt(part.physical_sat_mse.mean()))})
        # Compare vintages on the exact same noon-onward targets, not different delivery suffixes.
        local=pd.DatetimeIndex(f.interval_start_utc).tz_convert(config['pv_neural']['delivery_timezone'])
        noon=f.loc[f.score_eligible&(local.hour>=12)]
        counts=noon.groupby('interval_start_utc').auction.nunique()
        shared=noon.loc[noon.interval_start_utc.isin(counts.index[counts.eq(4)])]
        for (split,auction),part in shared.groupby(['split','auction']):
            weights=balanced_weights(part)
            err=(part.neural_mw-part.measured_ac_mw).to_numpy()/rating
            oracle=(part.neural_satellite_mw-part.measured_ac_mw).to_numpy()/rating
            auction_rows.append({'farm_id':farm,'split':split,'auction':auction,'quarters':len(part),
                'nrmse_percent':float(np.sqrt(weights@(err**2))*100),
                'satellite_input_nrmse_percent':float(np.sqrt(weights@(oracle**2))*100),
                'ghi_mae_w_m2':float(weights@part.ghi_error.abs().to_numpy()),
                'mean_lead_hours':float(weights@part.lead_hours.to_numpy()),
                'population':'Same noon-onward physical quarters in all four views'})
        # Descriptive same-GHI/time comparison, without fitting a new power reference.
        low,high=section['matched_ghi_range_w_m2'];begin,end=section['matched_local_hours']
        mask=f.score_eligible&f.auction.eq('DA')&f.satellite_ghi_input_w_m2.between(low,high)
        mask&=(local.hour>=begin)&(local.hour<end)
        for month,part in f.loc[mask].groupby(f.loc[mask,'delivery_day'].str[:7]):
            matched.append({'farm_id':farm,'month':month,'quarters':len(part),
                'median_ac_mw':float(part.measured_ac_mw.median()),
                'median_ac_fraction_of_rating':float(part.measured_ac_mw.median()/rating),
                'median_satellite_ghi_w_m2':float(part.satellite_ghi_input_w_m2.median()),
                'median_temperature_c':float(part.temperature_c.median()),
                'median_wind_speed_m_s':float(part.wind_speed_m_s.median()),
                'median_forecast_bhi_fraction':float((part.forecast_bhi/part.forecast_ghi).median())})
    profiles=profiles.merge(pd.DataFrame(extra),on=['farm_id','split','delivery_day','auction'])
    profiles=profiles.merge(persistent,on=['farm_id','delivery_day','auction'],how='left')
    profiles['persistent_low_hours']=profiles.persistent_low_hours.fillna(0)
    profiles=situations(profiles,section)
    large=profiles.satellite_input_nrmse_percent.ge(section['large_residual_nrmse_percent'])
    physics_close=profiles.pvlib_satellite_nrmse_percent.lt(section['large_residual_nrmse_percent'])
    stable_evidence=profiles.persistent_low_hours.gt(0)|profiles.stable_weather_jump_quarters.gt(0)
    profiles['residual_evidence']=np.select([large&physics_close,large&~physics_close&stable_evidence,
        large&~physics_close],['neural_mapping_or_input_distribution_issue',
        'production_variation_not_explained_by_available_inputs_candidate',
        'residual_unresolved_weather_components_or_model_mapping'],default='no_large_residual')
    profiles['timing_change_nrmse_pp']=profiles.nrmse_percent-profiles.without_timing_nrmse_percent
    profiles.sort_values(['farm_id','split','nrmse_percent'],ascending=[True,True,False]).to_csv(
        root/'csv/situation_profiles.csv',index=False)
    profiles.sort_values('timing_change_nrmse_pp',ascending=False).to_csv(root/'csv/largest_timing_regressions.csv',index=False)
    summaries=profiles.groupby(['farm_id','split','situation']).agg(profiles=('auction','size'),
        distinct_delivery_days=('delivery_day','nunique'),squared_error_share=('squared_error_share','sum')).reset_index()
    summaries.to_csv(root/'csv/situation_error_shares.csv',index=False)
    daily=[]
    for (farm,split,day),part in profiles.groupby(['farm_id','split','delivery_day']):
        daily.append({'farm_id':farm,'split':split,'delivery_day':day,
            'nrmse_percent':float(np.sqrt(np.mean(part.nrmse_percent**2))),
            'satellite_input_nrmse_percent':float(np.sqrt(np.mean(part.satellite_input_nrmse_percent**2))),
            'squared_error_share':float(part.squared_error_share.sum()),
            'ghi_mae_w_m2':float(part.ghi_mae_w_m2.mean())})
    daily=situations(pd.DataFrame(daily),section)
    daily.sort_values('squared_error_share',ascending=False).to_csv(root/'csv/daily_error_ranking.csv',index=False)
    pd.DataFrame(auction_rows).to_csv(root/'csv/common_noon_auction_metrics.csv',index=False)
    pd.DataFrame(matched).to_csv(root/'csv/matched_ghi_monthly.csv',index=False)
    write(root/'json/situation_summary.json',{'status':'complete',
        'counterfactual':'Same trained neural weights; replace GHI while preserving forecast direct fraction and other weather.',
        'material_ghi_sse_reduction':section['material_ghi_sse_reduction'],
        'large_residual_nrmse_percent':section['large_residual_nrmse_percent'],
        'interpretation':'Review evidence only. Residuals do not prove a plant fault, curtailment or bad meter.',
        'code_sha256':sha(__file__),'quality_csv_sha256':sha(config_path(config,config['pv_data_quality']['exclusions_csv']))})
    print(summaries.loc[summaries.split.eq('test')].to_string(index=False))
    print(root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();run(load_config(args.config))

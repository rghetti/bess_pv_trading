"""Cross-fitted, review-only satellite-GHI versus metered-AC episode scan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path
from bess_pv_trading.forecasting.pv_calendar import italian_national_holiday
from bess_pv_trading.reporting.artifacts import prepare_experiment_output


def _features(frame, timezone):
    """Represent measured irradiance, local clock time and season without proxy geometry."""
    starts=pd.DatetimeIndex(pd.to_datetime(frame.interval_start_utc,utc=True))
    local=(starts+pd.Timedelta(minutes=7.5)).tz_convert(timezone)
    hour=local.hour+local.minute/60
    annual=(local.dayofyear-1+hour/24)/np.where(local.is_leap_year,366,365)
    return np.column_stack([frame.satellite_ghi_w_m2.to_numpy(dtype=float),
        np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*annual),np.cos(2*np.pi*annual)])


def _crossfit(frame, rating, settings):
    """Fit median and upper reference curves on alternating completed days."""
    x=_features(frame,settings['timezone'])
    power=frame.measured_ac_mw.to_numpy(dtype=float)
    valid=np.isfinite(x).all(axis=1)&np.isfinite(power)&(power>=0)
    eligible=valid&~frame.outage_excluded.to_numpy(dtype=bool)
    days=pd.to_datetime(frame.delivery_day).map(pd.Timestamp.toordinal).to_numpy()%2
    prediction={quantile:np.full(len(frame),np.nan) for quantile in
        [0.5,float(settings['model']['upper_quantile'])]}
    model_options={key:settings['model'][key] for key in
        ['max_iter','max_leaf_nodes','min_samples_leaf','learning_rate','l2_regularization']}
    for fold in [0,1]:
        train=eligible&(days!=fold)
        infer=valid&(days==fold)
        if train.sum()<100 or not infer.any():
            raise ValueError('Insufficient data for alternating-day reference fit')
        for quantile in prediction:
            model=HistGradientBoostingRegressor(loss='quantile',quantile=quantile,
                random_state=int(settings['model']['seed']),**model_options)
            model.fit(x[train],power[train]/rating)
            prediction[quantile][infer]=np.clip(model.predict(x[infer]),0,1)*rating
    return prediction[0.5],prediction[float(settings['model']['upper_quantile'])],valid


def _episodes(frame, active, *, farm, kind, minimum, interval_minutes):
    """Summarize uninterrupted physical quarters within one local delivery day."""
    times=pd.DatetimeIndex(frame.interval_start_utc)
    days=frame.delivery_day.to_numpy()
    selected=np.flatnonzero(active)
    if not len(selected):return []
    boundaries=np.flatnonzero((np.diff(selected)!=1)
        | ((times[selected[1:]]-times[selected[:-1]])!=pd.Timedelta(minutes=interval_minutes))
        | (days[selected[1:]]!=days[selected[:-1]]))+1
    groups=np.split(selected,boundaries)
    rows=[]
    for positions in groups:
        if len(positions)<minimum:continue
        group=frame.iloc[positions]
        reference=group.reference_median_mw.to_numpy()
        actual=group.measured_ac_mw.to_numpy()
        gap=np.maximum(reference-actual,0) if kind=='low_power' else np.maximum(actual-reference,0)
        ghi_gap=np.abs(group.forecast_ghi.to_numpy()-group.satellite_ghi_w_m2.to_numpy())
        rows.append({'farm_id':farm,'kind':kind,'status':'review_candidate',
            'delivery_day':days[positions[0]],'start_utc':str(times[positions[0]]),
            'end_utc':str(times[positions[-1]]),'physical_quarters':len(positions),
            'duration_hours':len(positions)*interval_minutes/60,
            'mean_measured_ac_mw':float(actual.mean()),
            'mean_reference_median_mw':float(reference.mean()),
            'mean_reference_upper_mw':float(group.reference_upper_mw.mean()),
            'mean_power_ratio':float(np.mean(actual/reference)),
            'mean_satellite_ghi_w_m2':float(group.satellite_ghi_w_m2.mean()),
            'mean_forecast_ghi_w_m2':float(group.forecast_ghi.mean()),
            'fraction_forecast_ghi_agrees':float(np.mean(ghi_gap<=frame.attrs['forecast_ghi_agreement_w_m2'])),
            'mean_neural_forecast_mw':float(group.neural_forecast_mw.mean()),
            'mean_pvlib_forecast_mw':float(group.pvlib_forecast_mw.mean()),
            'fraction_both_forecasts_above_actual':float(np.mean(
                (group.neural_forecast_mw.to_numpy()>actual)
                & (group.pvlib_forecast_mw.to_numpy()>actual))),
            'indicative_gap_mwh':float(gap.sum()*interval_minutes/60),
            'outage_excluded_quarters':int(group.outage_excluded.sum()),
            'plateau_flagged_quarters':int(group.plateau_flag.sum())})
    return rows


def _plot_day(frame, episode, *, farm, rating, output, timezone):
    """Show metered power, GHI-conditioned references, and weather evidence."""
    day=frame.loc[frame.delivery_day.eq(episode['delivery_day'])]
    local=pd.DatetimeIndex(day.interval_start_utc).tz_convert(timezone)
    start=pd.Timestamp(episode['start_utc']).tz_convert(timezone)
    end=pd.Timestamp(episode['end_utc']).tz_convert(timezone)+pd.Timedelta(minutes=15)
    fig,(top,bottom)=plt.subplots(2,1,figsize=(12,6),sharex=True,layout='constrained')
    top.plot(local,day.measured_ac_mw,color='black',label='Measured AC')
    top.plot(local,day.reference_median_mw,color='tab:blue',label='Typical AC given satellite GHI')
    top.plot(local,day.reference_upper_mw,color='tab:purple',alpha=0.7,label='Upper reference')
    top.plot(local,day.pvlib_forecast_mw,color='tab:orange',alpha=0.65,label='pvlib forecast')
    top.plot(local,day.neural_forecast_mw,color='tab:green',alpha=0.65,label='Neural forecast')
    top.axhline(rating,color='gray',linestyle='--',label='AC rating')
    top.set(ylabel='AC power (MW)',title=f"Farm {farm}, {episode['delivery_day']}: review candidate")
    bottom.plot(local,day.satellite_ghi_w_m2,color='black',label='Satellite GHI')
    bottom.plot(local,day.forecast_ghi,color='tab:orange',label='Forecast GHI (auction mean)')
    bottom.set(ylabel='GHI (W/m²)',xlabel=f'Delivery time, {timezone}')
    for axis in (top,bottom):
        axis.axvspan(start,end,color='tab:red',alpha=0.13)
        axis.grid(alpha=0.2)
        axis.legend(loc='upper right',fontsize=8)
    fig.savefig(output,dpi=135)
    plt.close(fig)


def _matched_months(frame, *, farm, rating, settings):
    """Compare AC power under similar GHI and clock hours across months."""
    starts=pd.DatetimeIndex(frame.interval_start_utc).tz_convert(settings['timezone'])
    hour=starts.hour+starts.minute/60
    ghi_low,ghi_high=settings['matched_ghi_range_w_m2']
    hour_low,hour_high=settings['matched_local_hours']
    eligible=(frame.satellite_ghi_w_m2.between(ghi_low,ghi_high)
        & (hour>=hour_low)&(hour<hour_high)
        & frame.measured_ac_mw.ge(0)&~frame.outage_excluded)
    sample=frame.loc[eligible,['delivery_day','measured_ac_mw','satellite_ghi_w_m2',
        'reference_median_mw']].copy()
    sample['month']=sample.delivery_day.str[:7]
    monthly=sample.groupby('month').agg(intervals=('measured_ac_mw','size'),
        median_measured_ac_mw=('measured_ac_mw','median'),
        median_satellite_ghi_w_m2=('satellite_ghi_w_m2','median'),
        median_reference_mw=('reference_median_mw','median')).reset_index()
    monthly.insert(0,'farm_id',farm)
    monthly['median_ac_fraction_of_rating']=monthly.median_measured_ac_mw/rating
    return monthly


def _plot_matched_months(monthly, path):
    """Show same-weather monthly AC fractions without claiming a cause."""
    farms=monthly.farm_id.unique()
    fig,axes=plt.subplots(len(farms),1,figsize=(11,2.7*len(farms)),sharex=True,
        layout='constrained')
    for farm,axis in zip(farms,np.atleast_1d(axes)):
        subset=monthly.loc[monthly.farm_id.eq(farm)].copy()
        subset['year']=subset.month.str[:4]
        subset['calendar_month']=subset.month.str[5:7].astype(int)
        for year,group in subset.groupby('year'):
            axis.plot(group.calendar_month,group.median_ac_fraction_of_rating,
                marker='o',label=year)
        axis.set(ylabel=f'Farm {farm}\nAC / rating',ylim=(0,1.05))
        axis.grid(alpha=0.2)
        axis.legend(loc='upper left')
    axes[-1].set(xlabel='Calendar month',xticks=np.arange(1,13))
    fig.suptitle('Matched satellite GHI and local midday hours; operating conditions unverified')
    fig.savefig(path,dpi=140)
    plt.close(fig)


def run(config):
    """Scan the frozen physical-quarter snapshots and save auditable candidates."""
    settings=config['pv_ghi_power_disconnect']
    source=config_path(config,settings['source_root'])
    output_root=config_path(config,settings['output_root'])
    output=output_root/settings['run_id']
    prepare_experiment_output(output_root,output,
        overwrite=config['experiment_outputs']['overwrite_existing'])
    (output/'json').mkdir(exist_ok=True)
    (output/'csv').mkdir(exist_ok=True)
    (output/'json/resolved_config.json').write_text(json.dumps(config,indent=2,default=str)+'\n')
    all_episodes=[];summaries=[];sensitivity=[];monthly_tables=[];fingerprints={}
    for farm in settings['farm_ids']:
        source_file=source/f'farm_{farm}/parquet/physical_quarters.parquet'
        frame=pd.read_parquet(source_file).sort_values('interval_start_utc').reset_index(drop=True)
        if frame.interval_start_utc.duplicated().any():
            raise ValueError(f'Farm {farm} physical-quarter source contains duplicate timestamps')
        rating=config['farms'][str(farm)]['ac_capacity_kw']/1000
        prediction_file=(config_path(config,settings['prediction_source_root'])
            /f'farm_{farm}/parquet/predictions.parquet')
        prediction_columns=['interval_start_utc','measured_ac_mw',
            settings['neural_prediction_column'],settings['pvlib_prediction_column']]
        predictions=pd.read_parquet(prediction_file,columns=prediction_columns)
        grouped=predictions.groupby('interval_start_utc',sort=True)
        if (grouped.measured_ac_mw.nunique(dropna=False)>1).any():
            raise ValueError(f'Farm {farm} auction views disagree on measured AC')
        compact=grouped.agg(measured_ac_check=('measured_ac_mw','first'),
            neural_forecast_mw=(settings['neural_prediction_column'],'mean'),
            pvlib_forecast_mw=(settings['pvlib_prediction_column'],'mean')).reset_index()
        frame=frame.merge(compact,on='interval_start_utc',how='left',validate='one_to_one')
        if (not np.allclose(frame.measured_ac_mw,frame.measured_ac_check,equal_nan=True)
                or frame[['neural_forecast_mw','pvlib_forecast_mw']].isna().any().any()):
            raise ValueError(f'Farm {farm} physical quarters do not align with forecast outputs')
        frame=frame.drop(columns='measured_ac_check')
        median,upper,valid=_crossfit(frame,rating,settings)
        frame['reference_median_mw']=median
        frame['reference_upper_mw']=upper
        frame.attrs['forecast_ghi_agreement_w_m2']=settings['forecast_ghi_agreement_w_m2']
        monthly_tables.append(_matched_months(frame,farm=farm,rating=rating,settings=settings))
        base=valid&(frame.satellite_ghi_w_m2.to_numpy()>=settings['satellite_ghi_min_w_m2'])
        base&=median>=rating*settings['reference_min_fraction_of_rating']
        actual=frame.measured_ac_mw.to_numpy(dtype=float)
        low=base&(actual<=median*settings['low_power_ratio_max'])
        low&=(median-actual)>=rating*settings['gap_min_fraction_of_rating']
        high=base&(actual>=median*settings['high_power_ratio_min'])
        high&=(actual-median)>=rating*settings['gap_min_fraction_of_rating']
        minimum=settings['minimum_consecutive_intervals']
        interval=config['source']['historical_interval_minutes']
        episodes=_episodes(frame,low,farm=farm,kind='low_power',minimum=minimum,interval_minutes=interval)
        episodes+=_episodes(frame,high,farm=farm,kind='high_power',minimum=minimum,interval_minutes=interval)
        all_episodes.extend(episodes)
        folder=output/f'farm_{farm}'
        for name in ['csv','parquet','plots']:(folder/name).mkdir(parents=True,exist_ok=True)
        frame.to_parquet(folder/'parquet/physical_reference.parquet',index=False)
        pd.DataFrame(episodes).to_csv(folder/'csv/candidate_episodes.csv',index=False)
        for number,episode in enumerate(sorted(episodes,key=lambda row:row['indicative_gap_mwh'],reverse=True)[:settings['plot_top_per_farm']]):
            _plot_day(frame,episode,farm=farm,rating=rating,
                output=folder/'plots'/f"{number+1:02d}_{episode['kind']}_{episode['delivery_day']}.png",
                timezone=settings['timezone'])
        focus_days=settings['focus_days_by_farm'].get(str(farm),
            settings['focus_days_by_farm'].get(farm,[]))
        for day in focus_days:
            candidates=[episode for episode in episodes if episode['delivery_day']==day]
            if candidates:
                selected=max(candidates,key=lambda row:row['indicative_gap_mwh'])
                _plot_day(frame,selected,farm=farm,rating=rating,
                    output=folder/'plots'/f'focus_{day}.png',timezone=settings['timezone'])
        for ratio in settings['ratio_sensitivity']:
            for duration in settings['duration_sensitivity_intervals']:
                selected=base&(actual<=median*ratio)&((median-actual)>=rating*settings['gap_min_fraction_of_rating'])
                rows=_episodes(frame,selected,farm=farm,kind='low_power',minimum=duration,interval_minutes=interval)
                sensitivity.append({'farm_id':farm,'max_power_ratio':ratio,'minimum_quarters':duration,
                    'episodes':len(rows),'days':len({row['delivery_day'] for row in rows}),
                    'quarters':sum(row['physical_quarters'] for row in rows)})
        fingerprints[str(farm)]={'physical_sha256':hashlib.sha256(source_file.read_bytes()).hexdigest(),
            'predictions_sha256':hashlib.sha256(prediction_file.read_bytes()).hexdigest()}
        summary={'farm_id':farm,'physical_quarters':len(frame),'usable_satellite_and_ac_quarters':int(valid.sum()),
            'low_power_episodes':sum(row['kind']=='low_power' for row in episodes),
            'high_power_episodes':sum(row['kind']=='high_power' for row in episodes),
            'candidate_days':len({row['delivery_day'] for row in episodes}),
            'known_zero_outage_quarters':int(frame.outage_excluded.sum())}
        summaries.append(summary)
        print(summary,flush=True)
    table=pd.DataFrame(all_episodes)
    if not table.empty:
        table=table.sort_values('indicative_gap_mwh',ascending=False)
        local_days=pd.to_datetime(table.delivery_day).dt.tz_localize(settings['timezone'])
        table['national_holiday']=italian_national_holiday(
            pd.DatetimeIndex(local_days.dt.tz_convert('UTC')),settings['timezone']).astype(bool)
        table['weekend']=local_days.dt.weekday.ge(5).to_numpy()
        table['known_zero_overlap']=table.outage_excluded_quarters.gt(0)
        low=table.loc[table.kind.eq('low_power')]
        same_day={row.Index:int(low.loc[low.delivery_day.eq(row.delivery_day)
            & low.farm_id.ne(row.farm_id),'farm_id'].nunique()) for row in table.itertuples()}
        table['other_farms_low_same_day']=[same_day[index] for index in table.index]
        starts=pd.to_datetime(table.start_utc,utc=True)
        ends=pd.to_datetime(table.end_utc,utc=True)
        overlap=[]
        for index,row in table.iterrows():
            other=(table.kind.eq('low_power')&table.farm_id.ne(row.farm_id)
                & table.delivery_day.eq(row.delivery_day)
                & starts.le(ends[index])&ends.ge(starts[index]))
            overlap.append(int(table.loc[other,'farm_id'].nunique()))
        table['other_farms_low_time_overlap']=overlap
    table.to_csv(output/'csv/all_episodes.csv',index=False)
    table.loc[table.national_holiday].to_csv(output/'csv/national_holiday_episodes.csv',index=False)
    calendar=table.groupby(['farm_id','kind','national_holiday','known_zero_overlap']).agg(
        episodes=('kind','size'),days=('delivery_day','nunique'),
        quarters=('physical_quarters','sum')).reset_index()
    calendar.to_csv(output/'csv/calendar_summary.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(output/'csv/threshold_sensitivity.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'csv/farm_summary.csv',index=False)
    monthly=pd.concat(monthly_tables,ignore_index=True)
    monthly.to_csv(output/'csv/matched_condition_monthly.csv',index=False)
    (output/'plots').mkdir(exist_ok=True)
    _plot_matched_months(monthly,output/'plots/matched_condition_monthly.png')
    manifest={'status':'complete','method':'alternating-local-day cross-fitted GHI/time/season quantile references',
        'interpretation':'Retrospective candidate screening only; indicative gaps are not confirmed lost energy.',
        'input_sha256':fingerprints,'farm_summaries':summaries,
        'national_holiday_calendar':'Italy-wide fixed holidays plus Easter and Easter Monday; no local patron days',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'raw_data_modified':False}
    (output/'json/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    print(run(load_config(args.config)))

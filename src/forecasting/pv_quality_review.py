"""Full-history physical-quarter PV quality audit and manual review queue."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data import config_path
from ..reporting.artifacts import prepare_experiment_output
from .pv_outages import apply_sustained_zero_exclusion
from .pv_validation_audit import _plot_day

EPISODE_COLUMNS = ['farm_id','issue','status','start_utc','end_utc',
    'delivery_day','physical_quarters','duration_hours','mean_measured_ac_mw',
    'max_measured_ac_mw','mean_satellite_ghi_w_m2','mean_forecast_ghi_w_m2',
    'automatic_excluded_quarters','manual_excluded_quarters']


def _partial_recovery_mask(physical,*,rating,max_fraction,max_intervals,interval_minutes):
    """Find short nonzero spans bracketed by excluded zero runs on the same day."""
    outage=physical.outage_excluded.to_numpy(dtype=bool)
    power=physical.measured_ac_mw.to_numpy(dtype=float)
    days=physical.delivery_day.to_numpy()
    times=pd.DatetimeIndex(physical.interval_start_utc)
    cadence=pd.Timedelta(minutes=interval_minutes)
    marked=np.zeros(len(physical),dtype=bool)
    position=1
    while position<len(physical)-1:
        if outage[position] or not outage[position-1]:
            position+=1;continue
        start=position
        while position<len(physical) and not outage[position] and days[position]==days[start]:
            position+=1
        end=position
        if (end<len(physical) and end-start<=max_intervals and outage[end]
                and days[start-1]==days[start]==days[end]
                and ((times[start:end+1]-times[start-1:end])==cadence).all()
                and np.isfinite(power[start:end]).all()
                and ((power[start:end]>0)&(power[start:end]<=max_fraction*rating)).all()
                and physical.daylight.iloc[start:end].all()):
            marked[start:end]=True
    return marked


def _episode_rows(physical,selected,*,farm,kind,status,minimum_intervals,interval_minutes):
    """Summarize contiguous native intervals without bridging gaps or local days."""
    active=np.asarray(selected,dtype=bool)
    starts=pd.DatetimeIndex(physical.interval_start_utc)
    days=physical.delivery_day.to_numpy()
    cadence=pd.Timedelta(minutes=interval_minutes)
    episodes=[];first=None
    def finish(end):
        if first is None or end-first<minimum_intervals:return
        group=physical.iloc[first:end]
        episodes.append({'farm_id':farm,'issue':kind,'status':status,
            'start_utc':str(starts[first]),'end_utc':str(starts[end-1]),
            'delivery_day':days[first],'physical_quarters':end-first,
            'duration_hours':(end-first)*interval_minutes/60,
            'mean_measured_ac_mw':float(group.measured_ac_mw.mean()),
            'max_measured_ac_mw':float(group.measured_ac_mw.max()),
            'mean_satellite_ghi_w_m2':float(group.satellite_ghi_w_m2.mean()),
            'mean_forecast_ghi_w_m2':float(group.forecast_ghi.mean()),
            'automatic_excluded_quarters':int(group.outage_excluded.sum()),
            'manual_excluded_quarters':int(group.review_excluded.sum())})
    for position,value in enumerate(active):
        continuous=(position>0 and starts[position]-starts[position-1]==cadence
            and days[position]==days[position-1])
        if value:
            if first is None or not continuous:
                finish(position);first=position
        elif first is not None:
            finish(position);first=None
    finish(len(physical))
    return episodes


def run_pv_quality_review(config,progress=print):
    """Write curation evidence without adding candidate flags to the exclusion mask."""
    section=config['pv_data_quality_review']
    source=config_path(config,section['source_root'])
    output_root=config_path(config,section['output_root']);output=output_root/section['run_id']
    if source.resolve().is_relative_to(output.resolve()):
        raise ValueError('PV quality source cannot reside in its output directory')
    if section['save_plots']:
        for farm in section['farm_ids']:
            prediction_path=source/f'farm_{farm}/parquet'/section['plot_prediction_file']
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
    prepare_experiment_output(output_root,output,
        overwrite=config['experiment_outputs']['overwrite_existing'])
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    all_episodes=[];summaries=[]
    for farm in section['farm_ids']:
        path=source/f'farm_{farm}/parquet/corpus.parquet'
        frame=pd.read_parquet(path)
        curated=apply_sustained_zero_exclusion(frame,config,farm_id=farm)
        rating=config['farms'][str(farm)]['ac_capacity_kw']/1000
        population=curated.groupby('interval_start_utc',sort=True)
        physical=population.agg(delivery_day=('delivery_day','first'),
            measured_ac_mw=('measured_ac_mw','first'),
            satellite_ghi_w_m2=('satellite_ghi_w_m2','first'),
            forecast_ghi=('forecast_ghi','mean'),daylight=('daylight','first'),
            plateau_flag=('plateau_flag','first'),
            outage_excluded=('outage_excluded','first'),
            review_excluded=('review_excluded','first')).reset_index()
        high_ghi=physical.satellite_ghi_w_m2.ge(section['low_output_minimum_ghi_w_m2'])
        low=physical.measured_ac_mw.between(0,section['low_output_max_fraction_of_rating']*rating)
        interval=config['source']['historical_interval_minutes']
        recovery=_partial_recovery_mask(physical,rating=rating,
            max_fraction=section['partial_recovery_max_fraction_of_rating'],
            max_intervals=section['partial_recovery_max_intervals'],interval_minutes=interval)
        issue_masks={
            'sustained_zero_outage':(physical.outage_excluded,'automatic_excluded',1),
            'low_output_high_ghi':(low&high_ghi&physical.daylight&~physical.outage_excluded,
                'review_candidate',section['low_output_minimum_intervals']),
            'partial_recovery_between_outages':(recovery,'review_candidate',1),
            'nonzero_plateau':(physical.plateau_flag&high_ghi&physical.daylight,
                'review_candidate',section['plateau_minimum_intervals']),
            'above_ac_rating':(physical.measured_ac_mw.gt(
                rating*(1+section['rating_excess_fraction'])),'review_candidate',1),
            'satellite_ghi_missing':(physical.daylight&physical.satellite_ghi_w_m2.isna(),
                'input_gap',1),
            'weather_ghi_disagreement':(physical.daylight&
                (physical.forecast_ghi-physical.satellite_ghi_w_m2).abs().ge(
                    section['weather_ghi_disagreement_w_m2']),
                'weather_input',section['weather_disagreement_minimum_intervals'])}
        farm_episodes=[]
        for kind,(mask,status,minimum) in issue_masks.items():
            farm_episodes.extend(_episode_rows(physical,mask,farm=farm,kind=kind,
                status=status,minimum_intervals=minimum,interval_minutes=interval))
        episodes=pd.DataFrame(farm_episodes,columns=EPISODE_COLUMNS)
        folder=output/f'farm_{farm}'
        for name in ['csv','json','parquet','plots']:(folder/name).mkdir(parents=True,exist_ok=True)
        physical.to_parquet(folder/'parquet/physical_quarters.parquet',index=False)
        episodes.to_csv(folder/'csv/candidate_episodes.csv',index=False)
        by_kind=(episodes.groupby(['issue','status']).size().rename('episodes').reset_index()
            if not episodes.empty else pd.DataFrame(columns=['issue','status','episodes']))
        by_kind.to_csv(folder/'csv/episode_counts.csv',index=False)
        plateau=episodes.loc[episodes.issue.eq('nonzero_plateau')].copy()
        if not plateau.empty:
            bucket=section['plateau_level_rounding_kw']
            plateau['rounded_level_kw']=(np.round(
                plateau.mean_measured_ac_mw*1000/bucket)*bucket)
            recurring=plateau.groupby('rounded_level_kw').agg(
                episodes=('issue','size'),days=('delivery_day','nunique'),
                total_quarters=('physical_quarters','sum'),
                first_day=('delivery_day','min'),last_day=('delivery_day','max'),
                mean_satellite_ghi_w_m2=('mean_satellite_ghi_w_m2','mean')).reset_index()
            recurring=recurring.sort_values(['days','total_quarters'],ascending=False)
        else:
            recurring=pd.DataFrame(columns=['rounded_level_kw','episodes','days',
                'total_quarters','first_day','last_day','mean_satellite_ghi_w_m2'])
        recurring.to_csv(folder/'csv/recurring_plateau_levels.csv',index=False)
        summary={'farm_id':farm,'physical_quarters':len(physical),
            'outage_excluded_quarters':int(physical.outage_excluded.sum()),
            'manual_review_excluded_quarters':int(physical.review_excluded.sum()),
            'candidate_episodes':int((episodes.status=='review_candidate').sum()) if not episodes.empty else 0,
            'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'status':'Candidate flags do not change model eligibility unless reviewed or automatically excluded.'}
        (folder/'json/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if section['save_plots']:
            prediction_path=source/f'farm_{farm}/parquet'/section['plot_prediction_file']
            column=section['plot_prediction_column']
            comparison_column=section['plot_pvlib_column']
            keys=['interval_start_utc','delivery_day','auction','measured_ac_mw']
            saved=pd.read_parquet(prediction_path,columns=[*keys,column,comparison_column])
            if not frame[keys].equals(saved[keys]):
                raise ValueError(f'Farm {farm} PV plot predictions do not align with measured AC corpus')
            prediction=saved[column].to_numpy(dtype=float)
            comparison_prediction=saved[comparison_column].to_numpy(dtype=float)
            if not np.isfinite(prediction[frame.daylight.to_numpy(dtype=bool)]).all():
                raise ValueError(f'Farm {farm} PV plot predictions are nonfinite in daylight')
            if not np.isfinite(comparison_prediction[frame.daylight.to_numpy(dtype=bool)]).all():
                raise ValueError(f'Farm {farm} pvlib plot predictions are nonfinite in daylight')
            summary['plot_prediction']={'file':str(prediction_path),
                'column':column,'label':section['plot_prediction_label'],
                'comparison_column':comparison_column,
                'comparison_label':section['plot_pvlib_label'],
                'sha256':hashlib.sha256(prediction_path.read_bytes()).hexdigest()}
            (folder/'json/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            days=list(section['focus_days'])
            if not episodes.empty:
                days+=episodes.loc[episodes.status.eq('review_candidate')].nlargest(
                    section['plot_top_candidate_days'],'physical_quarters').delivery_day.tolist()
            for day in dict.fromkeys(days):
                if not frame.delivery_day.eq(day).any():continue
                _plot_day(frame,prediction,farm=farm,day=day,rating=rating,
                    timezone=config['pv_neural']['delivery_timezone'],
                    forecast_label=section['plot_prediction_label'],
                    comparison_prediction=comparison_prediction,
                    comparison_label=section['plot_pvlib_label'],
                    path=folder/'plots'/f'quality_day_{day}.png')
        progress(f"Farm {farm}: {summary['outage_excluded_quarters']} automatic quarters, {summary['candidate_episodes']} candidate episodes")
        all_episodes.extend(farm_episodes);summaries.append(summary)
    all_table=pd.DataFrame(all_episodes,columns=EPISODE_COLUMNS)
    (output/'csv').mkdir(exist_ok=True)
    all_table.to_csv(output/'csv/all_episodes.csv',index=False)
    queue=all_table.loc[all_table.status.eq('review_candidate')].copy()
    queue.insert(3,'decision','')
    queue['review_note']=''
    queue.to_csv(output/'csv/review_queue.csv',index=False)
    (output/'manifest.json').write_text(json.dumps({'status':'complete','summaries':summaries,
        'review_csv_contract':'Copy review_queue.csv to a stable path; set review.exclusions_csv to it, then fill decision with keep or exclude. Interval end is inclusive.',
        'raw_dataset_modified':False,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    return {'output':str(output),'summaries':summaries}

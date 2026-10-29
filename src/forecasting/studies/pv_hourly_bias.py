"""Causal local-hour correction of frozen PV forecasts from completed days."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...data import config_path
from ...reporting.artifacts import prepare_experiment_output
from ..studies.pv_ensemble_study import score
from ..evaluation.pv_forecast_comparison import _paired_intervals
from ..data.pv_outages import modeling_instances


def completed_bucket_statistics(frame, prediction, *, timezone, history_start_day, bucket_hours):
    """Count each measured physical quarter once and release days after midnight."""
    if 24 % bucket_hours:
        raise ValueError('bucket_hours must divide the 24-hour local clock')
    eligible=frame.eligible.to_numpy(dtype=bool) if 'eligible' in frame else np.ones(len(frame),dtype=bool)
    data=frame.loc[eligible,
        ['delivery_day','interval_start_utc','target_available_at_utc','measured_ac_mw']].copy()
    data['reference']=np.asarray(prediction,dtype=float)[eligible]
    grouped=data.groupby(['delivery_day','interval_start_utc'],sort=True)
    if (grouped.measured_ac_mw.nunique(dropna=False)>1).any():
        raise ValueError('Overlapping auction views disagree on measured AC')
    unique=grouped.agg(reference=('reference','mean'),target=('measured_ac_mw','first'),
        available=('target_available_at_utc','max')).reset_index()
    valid=np.isfinite(unique.reference)&np.isfinite(unique.target)&unique.target.ge(0)
    unique=unique.loc[valid&unique.delivery_day.ge(history_start_day)].copy()
    local=pd.DatetimeIndex(unique.interval_start_utc).tz_convert(timezone)
    unique['bucket']=local.hour//bucket_hours
    unique['residual']=unique.target-unique.reference
    daily=unique.groupby(['delivery_day','bucket'],sort=True).agg(
        residual_sum=('residual','sum'),interval_count=('residual','size'),
        available=('available','max')).reset_index()
    days=daily.groupby('delivery_day').available.max()
    ready={day:max(pd.Timestamp(value),
        (pd.Timestamp(day).tz_localize(timezone)+pd.DateOffset(days=1)).tz_convert('UTC'))
        for day,value in days.items()}
    daily['available']=pd.to_datetime(daily.delivery_day.map(ready),utc=True)
    return daily.sort_values(['available','bucket']).reset_index(drop=True)


def causal_bucket_prediction(frame,prediction,statistics,*,timezone,bucket_hours,
                             window_days,ridge_pseudo_intervals,minimum_days,rating):
    """Apply a regularized additive correction known at each issuance cutoff."""
    if window_days<1 or ridge_pseudo_intervals<0 or minimum_days<1:
        raise ValueError('Invalid causal correction settings')
    result=np.asarray(prediction,dtype=float).copy()
    if statistics.empty:return np.clip(result,0,rating)
    count=24//bucket_hours
    totals=statistics.pivot(index='available',columns='bucket',values='residual_sum')
    intervals=statistics.pivot(index='available',columns='bucket',values='interval_count')
    totals=totals.reindex(columns=range(count),fill_value=0).fillna(0).sort_index()
    intervals=intervals.reindex(columns=range(count),fill_value=0).fillna(0).sort_index()
    cumulative=totals.cumsum().to_numpy();population=intervals.cumsum().to_numpy()
    available=pd.DatetimeIndex(totals.index).as_unit('ns').asi8
    buckets=pd.DatetimeIndex(frame.interval_start_utc).tz_convert(timezone).hour.to_numpy()//bucket_hours
    for cutoff,positions in frame.groupby('decision_cutoff_utc',sort=False).indices.items():
        moment=pd.Timestamp(cutoff).value
        last=available.searchsorted(moment,side='right')-1
        first=available.searchsorted(moment-pd.Timedelta(days=window_days).value,side='left')-1
        if last<0 or last-first<minimum_days:continue
        residual=cumulative[last]-(cumulative[first] if first>=0 else 0)
        observed=population[last]-(population[first] if first>=0 else 0)
        correction=np.divide(residual,observed+ridge_pseudo_intervals,
            out=np.zeros_like(residual,dtype=float),where=observed>0)
        result[positions]+=correction[buckets[positions]]
    return np.clip(result,0,rating)


def run_hourly_bias_study(config,progress=print):
    """Choose regularization on validation and report the reused holdout."""
    section=config['pv_hourly_bias_study']
    source=config_path(config,section['source_root'])
    output_root=config_path(config,section['output_root'])
    output=output_root/section['run_id']
    if not 1<=section['bucket_hours']<=24 or 24%section['bucket_hours']:
        raise ValueError('pv_hourly_bias_study.bucket_hours must divide 24')
    if source.resolve().is_relative_to(output.resolve()):
        raise ValueError('The study input cannot be below its output directory')
    prepare_experiment_output(output_root,output,
        overwrite=config['experiment_outputs']['overwrite_existing'])
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2,default=str)+'\n')
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    summaries=[]
    for farm in section['farm_ids']:
        directory=output/f'farm_{farm}'
        for name in ['csv','json','parquet','plots']:(directory/name).mkdir(parents=True,exist_ok=True)
        corpus=source/f'farm_{farm}/parquet/corpus.parquet'
        prior=source/f'farm_{farm}/parquet/comparison_predictions.parquet'
        frame=modeling_instances(pd.read_parquet(corpus),config,farm_id=farm)
        saved=pd.read_parquet(prior).loc[frame.index]
        keys=['interval_start_utc','delivery_day','auction','measured_ac_mw']
        if not frame[keys].equals(saved[keys]):raise ValueError('Forecast and corpus keys/targets differ')
        prediction=saved[f"prediction_{section['reference_model']}"].to_numpy()
        rating=config['farms'][str(farm)]['ac_capacity_kw']/1000
        statistics=completed_bucket_statistics(frame,prediction,
            timezone=config['pv_neural']['delivery_timezone'],
            history_start_day=section['history_start_day'],bucket_hours=section['bucket_hours'])
        statistics.to_parquet(directory/'parquet/completed_bucket_statistics.parquet',index=False)
        validation=(frame.eligible&frame.daylight&frame.split.eq('validation')).to_numpy()
        rows=[];candidates={}
        for window in section['window_days']:
            for ridge in section['ridge_pseudo_intervals']:
                label=f'window_{window}_ridge_{ridge:g}'
                proposed=causal_bucket_prediction(frame,prediction,statistics,
                    timezone=config['pv_neural']['delivery_timezone'],bucket_hours=section['bucket_hours'],
                    window_days=window,ridge_pseudo_intervals=ridge,
                    minimum_days=section['minimum_days'],rating=rating)
                candidates[label]=proposed
                for split in ['validation','test']:
                    mask=(frame.eligible&frame.daylight&frame.split.eq(split)).to_numpy()
                    rows.append({'model':label,'window_days':window,'ridge_pseudo_intervals':ridge,
                        'split':split,**score(frame,proposed,rating,mask)})
        table=pd.DataFrame(rows);table.to_csv(directory/'csv/metrics.csv',index=False)
        selection=table[table.split.eq('validation')].sort_values(['nrmse_percent','model']).iloc[0]
        chosen=candidates[selection.model]
        result=frame[['interval_start_utc','delivery_day','auction','decision_cutoff_utc',
            'measured_ac_mw','split','eligible','daylight']].copy()
        result['previous_ac_mw']=prediction;result['corrected_ac_mw']=chosen
        result.to_parquet(directory/'parquet/selected_predictions.parquet',index=False)
        old={split:score(frame,prediction,rating,(frame.eligible&frame.daylight&frame.split.eq(split)).to_numpy())
            for split in ['validation','test']}
        new={split:score(frame,chosen,rating,(frame.eligible&frame.daylight&frame.split.eq(split)).to_numpy())
            for split in ['validation','test']}
        summary={'farm_id':farm,'selected_model':selection.model,'previous':old,'corrected':new,
            'selection':'January-April 2026 daylight NRMSE only',
            'report_period':'May-August 2026 reused holdout',
            'paired_interval':_paired_intervals(frame,chosen,prediction,rating,config['pv_online_bias']),
            'input_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [corpus,prior]}}
        (directory/'json/selection.json').write_text(json.dumps(summary,indent=2)+'\n')
        months=frame.delivery_day.str[:7]
        monthly=[]
        for month in sorted(months[months.ge(section['history_start_day'][:7])].unique()):
            mask=(frame.eligible&frame.daylight&months.eq(month)).to_numpy()
            for label,curve in [('previous',prediction),('corrected',chosen)]:
                monthly.append({'month':month,'model':label,**score(frame,curve,rating,mask)})
        pd.DataFrame(monthly).to_csv(directory/'csv/monthly_metrics.csv',index=False)
        if section['save_plots']:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            figure,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
            values=pd.DataFrame(monthly)
            for label,color in [('previous','tab:gray'),('corrected','tab:blue')]:
                rows=values[values.model.eq(label)]
                axes[0].plot(rows.month,rows.nrmse_percent,color=color,label=label,marker='o')
                axes[1].plot(rows.month,rows.bias_percent,color=color,label=label,marker='o')
            axes[0].set_ylabel('Daylight RMSE / AC rating (%)')
            axes[1].set_ylabel('Signed bias / AC rating (%)')
            for axis in axes:axis.grid(alpha=.2);axis.legend()
            axes[1].tick_params(axis='x',rotation=45)
            figure.savefig(directory/'plots/monthly_error_comparison.png',dpi=150)
            plt.close(figure)
        progress(f"Farm {farm}: {selection.model}; validation {new['validation']['nrmse_percent']:.3f}%, report {new['test']['nrmse_percent']:.3f}%")
        summaries.append(summary)
    (output/'manifest.json').write_text(json.dumps({'status':'complete','summaries':summaries,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    return {'output':str(output),'summaries':summaries}

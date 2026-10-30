"""Causal, weather-conditional ridge correction of frozen PV forecasts."""
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


def correction_features(frame, prediction, *, rating, timezone, family):
    """Build dimensionless inputs known from the selected forecast issuance."""
    power=np.asarray(prediction,dtype=float)/rating
    ghi=np.clip(frame.forecast_ghi.to_numpy(dtype=float)/1000,0,1.6)
    local=pd.DatetimeIndex(frame.interval_start_utc).tz_convert(timezone)
    angle=2*np.pi*(local.hour.to_numpy()+local.minute.to_numpy()/60)/24
    sine=np.sin(angle);cosine=np.cos(angle)
    columns={'offset':np.ones(len(frame)),'forecast_ac':power,
        'forecast_ghi':ghi,'forecast_ac_x_ghi':power*ghi,
        'hour_sin':sine,'hour_cos':cosine,
        'forecast_ac_x_hour_sin':power*sine,
        'forecast_ac_x_hour_cos':power*cosine}
    names={
        'offset':['offset'],
        'gain':['forecast_ac'],
        'irradiance':['offset','forecast_ac','forecast_ghi','forecast_ac_x_ghi'],
        'diurnal':['offset','forecast_ac','hour_sin','hour_cos',
            'forecast_ac_x_hour_sin','forecast_ac_x_hour_cos'],
        'weather_diurnal':['offset','forecast_ac','forecast_ghi','forecast_ac_x_ghi',
            'hour_sin','hour_cos','forecast_ac_x_hour_sin','forecast_ac_x_hour_cos'],
    }
    # Enforce the input contract: Unsupported online ridge family: `family`.
    if family not in names:raise ValueError(f'Unsupported online ridge family: {family}')
    matrix=np.column_stack([columns[name] for name in names[family]])
    # Enforce the input contract: Online ridge features contain missing forecast inputs.
    if not np.isfinite(matrix).all():raise ValueError('Online ridge features contain missing forecast inputs')
    return matrix,names[family]


def completed_day_moments(frame,prediction,features,*,rating,timezone,history_start_day):
    """Release one set of physical-quarter moments per completed delivery day."""
    keys=['delivery_day','interval_start_utc']
    eligible=frame.eligible.to_numpy(dtype=bool) if 'eligible' in frame else np.ones(len(frame),dtype=bool)
    data=frame.loc[eligible,keys+['target_available_at_utc','measured_ac_mw']].copy()
    data['reference_ac_mw']=np.asarray(prediction,dtype=float)[eligible]
    # Release one set of physical-quarter moments per completed delivery day for each `position` in `range(features.shape[1])`.
    for position in range(features.shape[1]):data[f'x{position}']=features[eligible,position]
    groups=data.groupby(keys,sort=True)
    # Enforce the input contract: Auction views disagree on physical-quarter measured AC.
    if (groups.measured_ac_mw.nunique(dropna=False)>1).any():
        raise ValueError('Auction views disagree on physical-quarter measured AC')
    aggregations={'measured_ac_mw':'first','reference_ac_mw':'mean',
        'target_available_at_utc':'max'}
    aggregations.update({f'x{position}':'mean' for position in range(features.shape[1])})
    unique=groups.agg(aggregations).reset_index()
    x=unique[[f'x{position}' for position in range(features.shape[1])]].to_numpy(dtype=float)
    y=(unique.measured_ac_mw.to_numpy(dtype=float)-unique.reference_ac_mw.to_numpy(dtype=float))/rating
    valid=(unique.delivery_day.ge(history_start_day).to_numpy()&np.isfinite(y)&
        unique.measured_ac_mw.ge(0).to_numpy()&np.isfinite(x).all(axis=1))
    unique=unique.loc[valid].copy();x=x[valid];y=y[valid]
    records=[]
    # Release one set of physical-quarter moments per completed delivery day for each `(day, positions)` in `unique.groupby('delivery_day', sort=True).indices.items()`.
    for day,positions in unique.groupby('delivery_day',sort=True).indices.items():
        day_x=x[positions];day_y=y[positions];count=len(positions)
        last=pd.Timestamp(unique.target_available_at_utc.iloc[positions].max())
        midnight=(pd.Timestamp(day).tz_localize(timezone)+pd.DateOffset(days=1)).tz_convert('UTC')
        records.append({'delivery_day':day,'available':max(last,midnight),
            'physical_quarters':count,'xx':day_x.T@day_x/count,
            'xy':day_x.T@day_y/count})
    return records


def causal_ridge_prediction(frame,prediction,features,moments,*,rating,
                            window_days,ridge_pseudo_days,minimum_days):
    """Fit zero-centered ridge coefficients from days available at each cutoff."""
    # Enforce the input contract: Invalid online ridge correction settings.
    if window_days<1 or ridge_pseudo_days<=0 or minimum_days<1:
        raise ValueError('Invalid online ridge correction settings')
    result=np.asarray(prediction,dtype=float).copy()
    records=sorted(moments,key=lambda record:record['available'])
    # Return `(np.clip(result, 0, rating), pd.DataFrame())` when `not records`.
    if not records:return np.clip(result,0,rating),pd.DataFrame()
    available=np.array([pd.Timestamp(record['available']).value for record in records],dtype=np.int64)
    cumulative_xx=np.cumsum(np.stack([record['xx'] for record in records]),axis=0)
    cumulative_xy=np.cumsum(np.stack([record['xy'] for record in records]),axis=0)
    parameters=[];identity=np.eye(features.shape[1])
    # Fit zero-centered ridge coefficients from days available at each cutoff for each `(cutoff, positions)` in `frame.groupby('decision_cutoff_utc', sort=False).indices.items()`.
    for cutoff,positions in frame.groupby('decision_cutoff_utc',sort=False).indices.items():
        # Skip the current item when `'split' in frame and frame.split.iloc[positions].eq('train').all()`.
        if 'split' in frame and frame.split.iloc[positions].eq('train').all():
            continue
        moment=pd.Timestamp(cutoff).value
        last=available.searchsorted(moment,side='right')-1
        first=available.searchsorted(moment-pd.Timedelta(days=window_days).value,side='left')-1
        days=max(0,last-first)
        # Skip the current item when `days < minimum_days`.
        if days<minimum_days:continue
        xx=cumulative_xx[last]-(cumulative_xx[first] if first>=0 else 0)
        xy=cumulative_xy[last]-(cumulative_xy[first] if first>=0 else 0)
        beta=np.linalg.solve(xx+ridge_pseudo_days*identity,xy)
        result[positions]+=rating*(features[positions]@beta)
        parameters.append({'decision_cutoff_utc':str(cutoff),'available_history_days':days,
            'last_history_day':records[last]['delivery_day'],
            'last_history_available_utc':str(records[last]['available']),
            'coefficients':beta.tolist()})
    return np.clip(result,0,rating),pd.DataFrame(parameters)


def run_online_ridge_study(config,progress=print):
    """Select conditional corrections on validation and report untouched test scores."""
    section=config['pv_online_ridge_study']
    source=config_path(config,section['source_root'])
    output_root=config_path(config,section['output_root']);output=output_root/section['run_id']
    # Enforce the input contract: Online ridge input cannot be within its output directory.
    if source.resolve().is_relative_to(output.resolve()):
        raise ValueError('Online ridge input cannot be within its output directory')
    # Enforce the input contract: Online ridge grid must contain families, windows and penalties.
    if not section['families'] or not section['window_days'] or not section['ridge_pseudo_days']:
        raise ValueError('Online ridge grid must contain families, windows and penalties')
    prepare_experiment_output(output_root,output,
        overwrite=config['experiment_outputs']['overwrite_existing'])
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    summaries=[]
    # Apply score for `farm` from `section['farm_ids']`.
    for farm in section['farm_ids']:
        folder=output/f'farm_{farm}'
        # Select conditional corrections on validation and report untouched test scores for each `name` in `['csv', 'json', 'parquet', 'plots']`.
        for name in ['csv','json','parquet','plots']:(folder/name).mkdir(parents=True,exist_ok=True)
        corpus=source/f'farm_{farm}/parquet/corpus.parquet'
        previous=source/f'farm_{farm}/parquet/comparison_predictions.parquet'
        frame=modeling_instances(pd.read_parquet(corpus),config,farm_id=farm)
        saved=pd.read_parquet(previous).loc[frame.index]
        keys=['interval_start_utc','delivery_day','auction','measured_ac_mw']
        # Enforce the input contract: Online ridge source population differs.
        if not frame[keys].equals(saved[keys]):raise ValueError('Online ridge source population differs')
        base=saved[f"prediction_{section['reference_model']}"].to_numpy(dtype=float)
        source_selection=json.loads((source/f'farm_{farm}/json/selection.json').read_text())
        secondary_name=source_selection.get(section['secondary_model'],section['secondary_model'])
        alternate=saved[f'prediction_{secondary_name}'].to_numpy(dtype=float)
        rating=config['farms'][str(farm)]['ac_capacity_kw']/1000
        masks={split:(frame.eligible&frame.daylight&frame.split.eq(split)).to_numpy()
            for split in ['validation','test']}
        rows=[];selected_curve=base;selected_coefficients=pd.DataFrame()
        reference_metrics={split:score(frame,base,rating,mask) for split,mask in masks.items()}
        # Select conditional corrections on validation and report untouched test scores for each `(split, metric)` in `reference_metrics.items()`.
        for split,metric in reference_metrics.items():
            rows.append({'model':'uncorrected','family':'uncorrected','window_days':None,
                'ridge_pseudo_days':None,'split':split,**metric})
        best_score=reference_metrics['validation']['nrmse_percent']
        chosen='uncorrected';selected_names=[]
        # Apply score for `family` from `section['families']`.
        for family in section['families']:
            x,names=correction_features(frame,base,rating=rating,
                timezone=config['pv_neural']['delivery_timezone'],family=family)
            moments=completed_day_moments(frame,base,x,rating=rating,
                timezone=config['pv_neural']['delivery_timezone'],
                history_start_day=section['history_start_day'])
            # Apply score for `window` from `section['window_days']`.
            for window in section['window_days']:
                # Apply score for `ridge` from `section['ridge_pseudo_days']`.
                for ridge in section['ridge_pseudo_days']:
                    label=f'{family}_w{window}_r{ridge:g}'
                    curve,coefficients=causal_ridge_prediction(frame,base,x,moments,rating=rating,
                        window_days=window,ridge_pseudo_days=ridge,
                        minimum_days=section['minimum_days'])
                    candidate_score=None
                    # Apply score for `(split, mask)` from `masks.items()`.
                    for split,mask in masks.items():
                        metric=score(frame,curve,rating,mask)
                        rows.append({'model':label,'family':family,'window_days':window,
                            'ridge_pseudo_days':ridge,'split':split,
                            **metric})
                        # When `split == 'validation'`, set `candidate_score` using `metric['nrmse_percent']`.
                        if split=='validation':candidate_score=metric['nrmse_percent']
                    # When `candidate_score < best_score`, set `best_score` using `candidate_score`.
                    if candidate_score<best_score:
                        best_score=candidate_score;chosen=label
                        selected_curve=curve;selected_coefficients=coefficients
                        selected_names=names
        table=pd.DataFrame(rows);table.to_csv(folder/'csv/candidate_metrics.csv',index=False)
        # When `not selected_coefficients.empty`, set `selected_coefficients['feature_names']` using `json.dumps(selected_names)`.
        if not selected_coefficients.empty:
            selected_coefficients['feature_names']=json.dumps(selected_names)
            selected_coefficients.to_json(folder/'json/selected_cutoff_coefficients.json',orient='records',indent=2)
        subset=frame[['interval_start_utc','delivery_day','auction','decision_cutoff_utc',
            'measured_ac_mw','split','eligible','daylight']].copy()
        subset['uncorrected_ac_mw']=base
        subset['corrected_ac_mw']=selected_curve
        subset['secondary_ac_mw']=alternate
        subset.to_parquet(folder/'parquet/selected_predictions.parquet',index=False)
        selected={split:score(frame,selected_curve,rating,mask) for split,mask in masks.items()}
        secondary={split:score(frame,alternate,rating,mask) for split,mask in masks.items()}
        summary={'farm_id':farm,'selected_model':chosen,
            'secondary_model_resolved':secondary_name,
            'feature_names':selected_names,'reference':reference_metrics,
            'selected':selected,'secondary':secondary,
            'selection':'January-April 2026 validation NRMSE only; uncorrected is eligible',
            'report_period':'May-August 2026 reused holdout',
            'paired_interval':_paired_intervals(frame,selected_curve,base,rating,config['pv_online_bias']),
            'input_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [corpus,previous]}}
        (folder/'json/selection.json').write_text(json.dumps(summary,indent=2)+'\n')
        monthly=[]
        # Apply score for `month` from `sorted(frame.delivery_day.str[:7].unique())`.
        for month in sorted(frame.delivery_day.str[:7].unique()):
            # Skip the current item when `month < section['history_start_day'][:7]`.
            if month<section['history_start_day'][:7]:continue
            mask=(frame.eligible&frame.daylight&frame.delivery_day.str.startswith(month)).to_numpy()
            # Apply score for `(label, curve)` from `[('reference', base), ('selected', selected_curve), ('secondary', alternate)]`.
            for label,curve in [('reference',base),('selected',selected_curve),('secondary',alternate)]:
                monthly.append({'month':month,'model':label,**score(frame,curve,rating,mask)})
        pd.DataFrame(monthly).to_csv(folder/'csv/monthly_metrics.csv',index=False)
        # Select conditional corrections on validation and report untouched test scores when `section['save_plots']`.
        if section['save_plots']:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            figure,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
            values=pd.DataFrame(monthly)
            # Process each `(label, color)` in this collection.
            for label,color in [('reference','tab:gray'),('selected','tab:blue'),
                                ('secondary','tab:orange')]:
                lines=values[values.model.eq(label)]
                axes[0].plot(lines.month,lines.nrmse_percent,color=color,marker='o',label=label)
                axes[1].plot(lines.month,lines.bias_percent,color=color,marker='o',label=label)
            axes[0].set_ylabel('Daylight RMSE / AC rating (%)')
            axes[1].set_ylabel('Signed bias / AC rating (%)')
            # Select conditional corrections on validation and report untouched test scores for each `axis` in `axes`.
            for axis in axes:axis.grid(alpha=.2);axis.legend()
            axes[1].tick_params(axis='x',rotation=45)
            figure.savefig(folder/'plots/monthly_comparison.png',dpi=150)
            plt.close(figure)
        progress(f"Farm {farm}: {chosen}; validation {selected['validation']['nrmse_percent']:.3f}%, report {selected['test']['nrmse_percent']:.3f}%")
        summaries.append(summary)
    (output/'manifest.json').write_text(json.dumps({'status':'complete','summaries':summaries,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    return {'output':str(output),'summaries':summaries}

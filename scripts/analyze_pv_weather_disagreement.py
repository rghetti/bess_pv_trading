#!/usr/bin/env python
"""Score saved PV forecasts on audited weather-disagreement episodes.

Example (no training, optimization, API calls or source-data changes):
  python scripts/analyze_pv_weather_disagreement.py --config configs/pv_experiments/audit_data_quality_2025_2026.yaml --farm-id 1 --output outputs/pv_weather_disagreement/farm_1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path
from bess_pv_trading.forecasting.pv_fitting import balanced_weights
from bess_pv_trading.forecasting.pv_outages import apply_sustained_zero_exclusion


def metrics(frame, column, rating):
    """Normalize signed MW errors by farm AC nameplate, never measured power."""
    if frame.empty:
        return {}
    residual=frame[column].to_numpy(float)-frame.measured_ac_mw.to_numpy(float)
    if not np.isfinite(residual).all():
        raise ValueError('Saved predictions must be finite on the common evaluation mask')
    weights=balanced_weights(frame)
    return {'auction_views':len(frame),
        'physical_quarters':frame.interval_start_utc.nunique(),
        'days':frame.delivery_day.nunique(),
        'nmae_percent':float(100*np.dot(weights,np.abs(residual))/rating),
        'nrmse_percent':float(100*np.sqrt(np.dot(weights,residual**2))/rating),
        'bias_percent':float(100*np.dot(weights,residual)/rating),
        'max_absolute_error_percent':float(100*np.abs(residual).max()/rating),
        'mae_mw':float(np.dot(weights,np.abs(residual))),
        'rmse_mw':float(np.sqrt(np.dot(weights,residual**2))),
        'uniform_view_nrmse_percent':float(100*np.sqrt(np.mean(residual**2))/rating)}


def plot_episode(frame, episode, column, rating, timezone, interval_minutes, destination):
    """Show AC, weather GHI and signed capacity-normalized AC error on local time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo

    day=frame.loc[frame.delivery_day.eq(episode.delivery_day)].copy()
    physical=day.groupby('interval_start_utc',sort=True).first().reset_index()
    boundary=(pd.Timestamp(episode.delivery_day).tz_localize(timezone)+pd.DateOffset(days=1)).tz_convert('UTC')
    next_point=frame.loc[frame.interval_start_utc.eq(boundary)]
    if not next_point.empty:
        physical=pd.concat([physical,next_point.iloc[[0]]],ignore_index=True)
    figure,axes=plt.subplots(3,1,figsize=(13,10),sharex=True,layout='constrained')
    axes[0].plot(physical.interval_start_utc,physical.measured_ac_mw,color='black',lw=2,label='Measured AC')
    axes[0].axhline(rating,color='gray',ls='--',label='Installed AC rating')
    axes[1].plot(physical.interval_start_utc,physical.satellite_ghi_w_m2,color='black',lw=2,label='Satellite GHI')
    for auction,curve in day.groupby('auction',sort=False):
        line=axes[0].plot(curve.interval_start_utc,curve[column],label=auction)[0]
        axes[1].plot(curve.interval_start_utc,curve.forecast_ghi,color=line.get_color(),label=auction)
        axes[2].plot(curve.interval_start_utc,
            100*(curve[column]-curve.measured_ac_mw)/rating,color=line.get_color(),label=auction)
    left=pd.Timestamp(episode.start_utc)
    right=pd.Timestamp(episode.end_utc)+pd.Timedelta(minutes=interval_minutes)
    for axis in axes:
        axis.axvspan(left,right,color='orange',alpha=.15,label='Audited disagreement episode')
        axis.grid(alpha=.2)
        axis.legend(ncol=3,fontsize=8)
    axes[0].set_ylabel('AC power (MW)')
    axes[1].set_ylabel('GHI (W/m²)')
    axes[2].set_ylabel('Forecast − actual (% AC rating)')
    axes[2].axhline(0,color='gray',lw=.7)
    axes[2].set_xlabel(f'Delivery time ({timezone})')
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=ZoneInfo(timezone)))
    figure.suptitle(f'Farm {episode.farm_id}, {episode.delivery_day}: saved model on weather disagreement')
    figure.savefig(destination,dpi=150)
    plt.close(figure)


def plot_summary(summary,destination):
    """Compare weather-episode and other-daylight RMSE separately by data split."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    splits=[name for name in ('train','validation','test') if name in set(summary.split)]
    figure,axes=plt.subplots(1,len(splits),figsize=(5*len(splits),4),sharey=True,
        squeeze=False,layout='constrained')
    groups=['weather_disagreement','other_daylight']
    for axis,split in zip(axes[0],splits):
        for offset,(method,color,label) in enumerate([
            ('selected_model','tab:blue','Selected model'),('pvlib','tab:orange','Calibrated pvlib')]):
            part=summary.loc[summary['mask'].eq('raw_valid')&summary.split.eq(split)&
                summary.method.eq(method)].set_index('group')
            bars=axis.bar(np.arange(len(groups))+(offset-.5)*.36,
                part.loc[groups,'nrmse_percent'],width=.36,color=color,label=label)
            axis.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
        axis.set_xticks(range(len(groups)),['Weather disagreement','Other daylight'])
        axis.set_title(split.capitalize())
        axis.grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('RMSE (% installed AC power)')
    axes[0,-1].legend(fontsize=8)
    figure.suptitle('Farm forecasts on all audited episodes; valid raw AC targets, no outage exclusion')
    figure.savefig(destination,dpi=150)
    plt.close(figure)


def main():
    """Read versioned predictions and episode boundaries; write a reproducible report."""
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--farm-id',required=True,type=int)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prediction-column',default='prediction_selected_mw')
    parser.add_argument('--baseline-column',default='prediction_pvlib_mw')
    parser.add_argument('--reference-split',default='test')
    args=parser.parse_args()
    config=load_config(args.config)
    section=config['pv_data_quality_review']
    source=config_path(config,section['source_root'])/f'farm_{args.farm_id}'
    audit=config_path(config,section['output_root'])/section['run_id']/f'farm_{args.farm_id}'
    corpus_path=source/'parquet/corpus.parquet'
    prediction_path=source/'parquet/predictions.parquet'
    episode_path=audit/'csv/candidate_episodes.csv'
    corpus=pd.read_parquet(corpus_path)
    saved=pd.read_parquet(prediction_path)
    keys=['interval_start_utc','delivery_day','auction','measured_ac_mw']
    if not corpus[keys].equals(saved[keys]):
        raise ValueError('Saved predictions and corpus have different keys or measured AC')
    frame=apply_sustained_zero_exclusion(corpus,config,farm_id=args.farm_id)
    methods={'selected_model':args.prediction_column,'pvlib':args.baseline_column}
    for column in methods.values():
        frame[column]=saved[column].to_numpy(float)
    audit_episodes=pd.read_csv(episode_path)
    episodes=audit_episodes.loc[audit_episodes.issue.eq('weather_ghi_disagreement')].reset_index(drop=True)
    flagged=np.zeros(len(frame),dtype=bool)
    flagged_any_audit=np.zeros(len(frame),dtype=bool)
    frame['weather_episode_id']=-1
    weather_episode_id=0
    for row in audit_episodes.itertuples(index=False):
        selected=frame.interval_start_utc.between(pd.Timestamp(row.start_utc),pd.Timestamp(row.end_utc))
        if frame.loc[selected,'interval_start_utc'].nunique()!=row.physical_quarters:
            raise ValueError('Episode boundaries do not match source physical quarters')
        flagged_any_audit|=selected.to_numpy()
        if row.issue=='weather_ghi_disagreement':
            flagged|=selected.to_numpy()
            frame.loc[selected,'weather_episode_id']=weather_episode_id
            weather_episode_id+=1
    frame['weather_episode']=flagged
    frame['any_audit_episode']=flagged_any_audit
    rating=float(config['farms'][str(args.farm_id)]['ac_capacity_kw'])/1000
    masks={'raw_valid':frame.daylight&np.isfinite(frame.measured_ac_mw)&
               frame.measured_ac_mw.ge(0)&np.isfinite(frame.baseline_mw),
           'curated':frame.eligible&frame.daylight}
    rows=[];auction_rows=[]
    for mask_name,eligible in masks.items():
        for split in ['all_dates',*frame.split.unique()]:
            population=frame.loc[eligible&(True if split=='all_dates' else frame.split.eq(split))]
            for group,part in [('all_daylight',population),
                               ('weather_disagreement',population.loc[population.weather_episode]),
                               ('other_daylight',population.loc[~population.weather_episode]),
                               ('outside_all_audit_episodes',population.loc[~population.any_audit_episode])]:
                for method,column in methods.items():
                    if not part.empty:
                        rows.append({'mask':mask_name,'split':split,'group':group,'method':method,
                            'episodes':part.loc[part.weather_episode,'weather_episode_id'].nunique(),
                            **metrics(part,column,rating)})
                        for auction,view in part.groupby('auction'):
                            auction_rows.append({'mask':mask_name,'split':split,'group':group,
                                'method':method,'auction':auction,**metrics(view,column,rating)})
    summary=pd.DataFrame(rows)
    episode_rows=[]
    for mask_name,eligible in masks.items():
        for episode_id,row in enumerate(episodes.itertuples(index=False)):
            event=frame.loc[frame.weather_episode_id.eq(episode_id)]
            part=event.loc[eligible.loc[event.index]]
            for auction,view in [('ALL_BALANCED',part),*list(part.groupby('auction'))]:
                for method,column in methods.items():
                    episode_rows.append({'mask':mask_name,'episode_id':episode_id,
                        'split':event.split.iloc[0],'status':'scored' if not view.empty else 'no_eligible_views',
                        'delivery_day':row.delivery_day,'start_utc':row.start_utc,
                        'end_utc':row.end_utc,'auction':auction,'method':method,
                        **metrics(view,column,rating)})
    output=args.output
    for name in ['csv','json','plots']:(output/name).mkdir(parents=True,exist_ok=True)
    summary.to_csv(output/'csv/group_metrics.csv',index=False)
    pd.DataFrame(auction_rows).to_csv(output/'csv/by_auction_metrics.csv',index=False)
    per_episode=pd.DataFrame(episode_rows)
    per_episode.to_csv(output/'csv/episode_metrics.csv',index=False)
    focus=per_episode.loc[per_episode['mask'].eq('raw_valid')&per_episode.split.eq(args.reference_split)&
        per_episode.auction.eq('ALL_BALANCED')&per_episode.method.eq('selected_model')]
    if not focus.empty:
        selected=focus.nlargest(1,'physical_quarters').iloc[0]
        episode=episodes.iloc[int(selected.episode_id)]
        plot_episode(frame,episode,args.prediction_column,rating,
            config['pv_neural']['delivery_timezone'],config['source']['historical_interval_minutes'],
            output/'plots/longest_heldout_episode.png')
        focus=per_episode.loc[per_episode['mask'].eq('raw_valid')&per_episode.episode_id.eq(selected.episode_id)]
    plot_summary(summary,output/'plots/weather_vs_other_errors.png')
    provenance={'farm_id':args.farm_id,'ac_rating_mw':rating,
        'arguments':{key:str(value) for key,value in vars(args).items()},
        'inputs_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [corpus_path,prediction_path,episode_path]},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'metric_definition':'100 * error_MW / installed_AC_MW; equal day/auction-view weight within each reported population',
        'positive_bias':'overprediction',
        'interpretation':'Frozen pre-curation predictors rescored on both raw-valid and curated daylight populations. No retraining. All-dates rows mix fitted and held-out dates and are descriptive only.',
        'episodes_requested':len(episodes),
        'all_audit_episodes_requested':len(audit_episodes),
        'outside_all_audit_episodes_definition':'Remove the union of every candidate_episodes.csv issue during evaluation only; this is a sensitivity check, not a causal diagnosis or a deployable forecast filter.',
        'episode_definition':'Existing audit: averaged auction-view weather GHI vs satellite; physical episode quarters applied to every PV auction forecast.',
        'energy_convention':'Overlapping auction forecasts are never added as production or energy.',
        'summary':summary.to_dict('records'),'longest_reference_episode':focus.to_dict('records')}
    (output/'json/report.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    print(summary.loc[summary['mask'].eq('raw_valid')&summary.group.eq('weather_disagreement'),
        ['split','method','episodes','physical_quarters','nmae_percent','nrmse_percent','bias_percent']].to_string(index=False))
    print('\nLongest reference-split episode:\n'+focus.to_string(index=False))
    print(f'\nOutput: {output}')


if __name__=='__main__':main()

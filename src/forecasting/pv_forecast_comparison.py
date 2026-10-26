"""Validation-only model averaging and causal online-bias comparisons."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import config_path
from .pv_ensemble_study import score
from .pv_forecast_bundle import export_bundle,export_composite
from .pv_online_bias import completed_day_statistics,causal_gain_prediction
from .pv_outages import apply_episode_exclusions


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def _gain_job(job):
    frame=pd.read_parquet(job['frame'])
    prediction=np.load(job['prediction'])
    history=pd.read_parquet(job['history'])
    result,gain=causal_gain_prediction(frame,prediction,history,window_days=job['window'],
        minimum_days=job['minimum'],gain_bounds=job['bounds'])
    result=np.clip(result,0,job['rating']);np.save(job['output'],result)
    return job['name'],str(job['output'])


def _paired_intervals(frame,left,right,rating,section):
    """Resample complete seven-day blocks rather than treating quarters as independent."""
    mask=(frame.eligible&frame.daylight&frame.split.eq('test')).to_numpy()
    f=frame.loc[mask,['delivery_day','auction']].copy()
    f['left']=((left[mask]-frame.measured_ac_mw.to_numpy()[mask])/rating)**2
    f['right']=((right[mask]-frame.measured_ac_mw.to_numpy()[mask])/rating)**2
    views=f.groupby(['delivery_day','auction'])[['left','right']].mean()
    daily=views.groupby(level=0).mean().to_numpy()
    rng=np.random.default_rng(section['bootstrap_seed']);values=[];n=len(daily)
    block=int(section['bootstrap_block_days'])
    for _ in range(section['bootstrap_repetitions']):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        indices=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        means=daily[indices].mean(0)
        values.append(100*(np.sqrt(means[0])-np.sqrt(means[1])))
    return {'quantity':'candidate minus old selected test NRMSE, percentage points of AC rating',
        'block_days':block,'resamples':section['bootstrap_repetitions'],
        'percentile_95_interval':np.quantile(values,[.025,.975]).tolist(),
        'scope':'exploratory reused holdout; does not include HPO/model-selection uncertainty'}


def run_forecast_comparison(config,progress=print):
    """Compare static predictors fairly and label online observations as added information."""
    section=config['pv_online_bias'];output=config_path(config,section['output_root'])/section['run_id']
    for folder in ['json','csv','plots']:(output/folder).mkdir(parents=True,exist_ok=True)
    _write(output/'json/resolved_config.json',config)
    workers=int(section['workers'])
    if not 1<=workers<=42:raise ValueError('Use at most 42 single-threaded comparison workers')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    studies={};jobs=[]
    for farm in section['farm_ids']:
        directory=output/f'farm_{farm}'
        for folder in ['csv','json','arrays','parquet','plots','bundles']:(directory/folder).mkdir(parents=True,exist_ok=True)
        models={};predictions={};frame=None;rating=None
        for number,run in enumerate(section['source_runs'],1):
            source=config_path(config,run)/f'farm_{farm}'
            selection=json.loads((source/'json/selection.json').read_text())
            incoming=apply_episode_exclusions(
                pd.read_parquet(source/'parquet/corpus.parquet'),config,farm_id=farm)
            if frame is None:frame=incoming
            elif not frame.equals(incoming):raise ValueError('Searches do not share identical frozen inputs and targets')
            provenance=json.loads((source/'json/input_provenance.json').read_text());rating=provenance['rating_mw']
            records=sorted((source/'json/trials').glob('*.json'))
            for path in records:
                record=json.loads(path.read_text());model_name=path.stem;label=f'v{number}/{model_name}'
                value=np.load(source/'arrays'/f'{model_name}.npy')
                digest=hashlib.sha256((source/'arrays'/f'{model_name}.npy').read_bytes()).hexdigest()
                if digest!=record['prediction_sha256']:raise ValueError('Search prediction checksum mismatch')
                predictions[label]=value;models[label]={'source':str(source),'model':model_name,
                    'family':record['specification']['family'],'kind':'static'}
        source_config=json.loads((config_path(config,section['source_runs'][0])/'json/resolved_config.json').read_text())
        prior=pd.read_parquet(config_path(config,source_config['pv_forecast_search']['source_root'])/f'farm_{farm}'/'parquet/predictions.parquet')
        for old in ['pvlib','selected']:
            predictions[f'old_{old}']=prior[f'prediction_{old}_mw'].to_numpy()
            models[f'old_{old}']={'kind':'reference','family':'legacy reference'}
        validation=(frame.eligible&frame.daylight&frame.split.eq('validation')).to_numpy()
        ranking=sorted([name for name in models if models[name]['kind']=='static'],
            key=lambda name:score(frame,predictions[name],rating,validation)['nrmse_percent'])
        for size in section['mean_ensemble_sizes']:
            members=ranking[:size];label=f'mean_top_{size}'
            predictions[label]=np.mean([predictions[m] for m in members],axis=0)
            models[label]={'kind':'static_mean','family':'model averaging','members':members}
        bases=ranking[:section['base_top_k']]+[name for name in models if models[name]['kind']=='static_mean']
        frame_path=directory/'parquet/corpus.parquet';frame.to_parquet(frame_path,index=False)
        for base_number,base in enumerate(bases):
            pred_path=directory/'arrays'/f'base_{base_number}.npy';np.save(pred_path,predictions[base])
            history=completed_day_statistics(frame,predictions[base],config['pv_neural']['delivery_timezone'])
            hist_path=directory/'parquet'/f'history_{base_number}.parquet';history.to_parquet(hist_path,index=False)
            for window in section['window_days']:
                for minimum in section['minimum_days']:
                    name=f'online_b{base_number}_w{window}_n{minimum}'
                    models[name]={'kind':'online_gain','family':'causal online gain','base':base,
                        'window_days':window,'minimum_days':minimum,'gain_bounds':section['gain_bounds'],
                        'history':str(hist_path)}
                    jobs.append({'farm':farm,'name':name,'frame':str(frame_path),
                        'prediction':str(pred_path),'history':str(hist_path),'window':window,'minimum':minimum,
                        'bounds':section['gain_bounds'],'rating':rating,'output':str(directory/'arrays'/f'{name}.npy')})
        studies[farm]=(frame,predictions,models,directory,rating)
    progress(f'Running {len(jobs)} causal correction settings with at most {workers} workers')
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(_gain_job,j):j for j in jobs}
        for future in as_completed(pending):
            name,path=future.result();job=pending[future]
            studies[job['farm']][1][name]=np.load(path)
    summaries=[]
    for farm,(frame,predictions,models,directory,rating) in studies.items():
        progress(f'Reporting farm {farm}')
        rows=[];monthly=[]
        for name in sorted(predictions):
            prediction=predictions[name]
            for split in ['train','validation','test']:
                mask=(frame.eligible&frame.daylight&frame.split.eq(split)).to_numpy()
                rows.append({'model':name,'kind':models[name]['kind'],'family':models[name]['family'],
                    'split':split,**score(frame,prediction,rating,mask)})
            for month in sorted(frame.delivery_day.str[:7].unique()):
                mask=(frame.eligible&frame.daylight&frame.delivery_day.str.startswith(month)).to_numpy()
                if mask.any():monthly.append({'model':name,'month':month,**score(frame,prediction,rating,mask)})
        table=pd.DataFrame(rows);table.to_csv(directory/'csv/metrics.csv',index=False)
        pd.DataFrame(monthly).to_csv(directory/'csv/monthly_metrics.csv',index=False)
        eligible=table[table.split.eq('validation') & table.kind.ne('reference')].sort_values(['nrmse_percent','model'])
        static=eligible[eligible.kind.isin(['static','static_mean'])].iloc[0].model
        selected=eligible.iloc[0].model
        neural=eligible[eligible.family.eq('neural')].iloc[0].model
        statistical=eligible[eligible.kind.eq('static') & eligible.family.ne('neural')].iloc[0].model
        _write(directory/'json/model_definitions.json',models)
        summary={'farm_id':farm,'selected_static':static,'selected_including_online':selected,
            'best_new_neural':neural,'best_new_statistical':statistical,
            'selection':'validation Jan-Apr 2026 daylight NRMSE; no test selection',
            'test_is_reused_holdout':True,
            'online_information':'Uses completed production days observed by each issuance cutoff; unlike frozen models, adapts causally within 2026.',
            'test_paired_interval':_paired_intervals(frame,predictions[selected],predictions['old_selected'],rating,section)}
        _write(directory/'json/selection.json',summary)
        # Export every constituent needed by selected static or adaptive predictors.
        wanted=set()
        def collect(name):
            m=models[name]
            if m['kind']=='online_gain':collect(m['base'])
            elif m['kind']=='static_mean':
                for member in m['members']:collect(member)
            elif m['kind']=='static':wanted.add(name)
        for name in [static,selected,neural,statistical]:collect(name)
        for name in wanted:
            export_bundle(models[name]['source'],models[name]['model'],directory/'bundles'/name.replace('/','_'))
        for label,name in [('selected_static',static),('selected_including_online',selected)]:
            export_composite(models,name,directory/'bundles'/label,rating=rating,timezone=config['pv_neural']['delivery_timezone'])
        saved=frame[['interval_start_utc','delivery_day','auction','decision_cutoff_utc','target_available_at_utc','split','eligible','daylight','measured_ac_mw']].copy()
        for name in set([static,selected,neural,statistical,'old_selected','old_pvlib']):saved[f'prediction_{name}']=predictions[name]
        saved.to_parquet(directory/'parquet/comparison_predictions.parquet',index=False)
        _plot(table,monthly,[statistical,neural,static,selected,'old_selected','old_pvlib'],directory)
        summaries.append(summary)
    _write(output/'json/manifest.json',{'status':'complete','summaries':summaries})
    return {'output':str(output),'summaries':summaries}


def _plot(table,monthly,names,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(dict.fromkeys(names));fig,axes=plt.subplots(1,3,figsize=(18,5),layout='constrained')
    for axis,split in zip(axes,['train','validation','test']):
        rows=table[table.split.eq(split)].set_index('model').loc[names]
        axis.barh(rows.index,rows.nrmse_percent);axis.invert_yaxis();axis.set_title(split)
        axis.set_xlabel('Daylight RMSE / AC rating (%)');axis.grid(axis='x',alpha=.2)
    fig.savefig(directory/'plots/model_comparison.png',dpi=150);plt.close(fig)
    monthly=pd.DataFrame(monthly);fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True,layout='constrained')
    for name in names:
        rows=monthly[monthly.model.eq(name)]
        axes[0].plot(rows.month,rows.nrmse_percent,label=name,marker='o',ms=3)
        axes[1].plot(rows.month,rows.bias_percent,label=name,marker='o',ms=3)
    axes[0].set_ylabel('Daylight NRMSE (% AC)');axes[1].set_ylabel('Signed bias (% AC)')
    axes[1].axhline(0,color='gray',lw=1);axes[1].tick_params(axis='x',rotation=45)
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(directory/'plots/monthly_errors.png',dpi=150);plt.close(fig)

"""Chronological monthly refits of validation-selected weather regressors."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import config_path,night_window_mask
from ..reporting.artifacts import prepare_experiment_output
from .pv_ensemble_study import score
from .pv_fitting import balanced_weights
from .pv_forecast_search import _fit_statistical
from .pv_outages import modeling_instances


def plot_monthly_metrics(table,destination):
    """Show the monthly refit and frozen comparators on their common mask."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for farm in sorted(table.farm_id.unique()):
        data=table[table.farm_id.eq(farm)]
        figure,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
        for model,label,color in [('monthly_refit','Monthly refit','tab:blue'),
                                  ('old_selected','Existing frozen model','tab:gray'),
                                  ('old_pvlib','Calibrated pvlib','tab:orange')]:
            rows=data[data.model.eq(model)]
            axes[0].plot(rows.month,rows.nrmse_percent,marker='o',label=label,color=color)
            axes[1].plot(rows.month,rows.bias_percent,marker='o',label=label,color=color)
        axes[0].set_ylabel('Daylight RMSE / AC rating (%)')
        axes[1].set_ylabel('Signed bias / AC rating (%)')
        for axis in axes:axis.legend();axis.grid(alpha=.2)
        axes[1].tick_params(axis='x',rotation=30)
        figure.savefig(destination/f'farm_{farm}_monthly_comparison.png',dpi=150)
        plt.close(figure)


def _monthly_job(job):
    """Fit once using data available before this month's first delivery day."""
    import sklearn  # Load the conda C++ runtime before optional PyTorch.
    from sklearn.linear_model import Ridge
    from threadpoolctl import threadpool_limits
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[name]='1'
    frame=pd.read_parquet(job['corpus'])
    trial=json.loads(Path(job['trial_path']).read_text())['specification']
    x=np.load(job['features'],mmap_mode='r')[job['source_positions']]
    y=frame.measured_ac_mw.to_numpy(dtype=float)/job['rating']
    first=pd.Timestamp(job['month']+'-01').tz_localize(job['timezone'])
    issue=(first-pd.DateOffset(days=1)).tz_convert('UTC')
    train=(frame.eligible & frame.delivery_day.lt(job['month']+'-01') &
        frame.target_available_at_utc.le(issue)).to_numpy()
    target=(frame.eligible&frame.daylight&frame.delivery_day.str.startswith(job['month'])).to_numpy()
    if not train.any() or not target.any():raise ValueError('Missing eligible training or report intervals')
    if frame.loc[train,'target_available_at_utc'].max()>issue:
        raise ValueError('Future observed production entered a monthly fit')
    weight=balanced_weights(frame.loc[train]);weight*=len(weight)
    with threadpool_limits(limits=1):
        reference=Ridge(alpha=job['ridge_alpha']).fit(x[train],y[train],sample_weight=weight)
        baseline=np.clip(reference.predict(x),0,1)
        baseline[night_window_mask(pd.DatetimeIndex(frame.interval_start_utc),job['night'])]=0
        model,fraction=_fit_statistical(x,y,train,weight,baseline,trial)
    prediction=np.clip(fraction,0,1)*job['rating']
    prediction[night_window_mask(pd.DatetimeIndex(frame.interval_start_utc),job['night'])]=0
    path=Path(job['output']);path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,prediction[target],allow_pickle=False)
    metrics=score(frame,prediction,job['rating'],target)
    return {'farm_id':job['farm_id'],'month':job['month'],'trial':trial['name'],
        'issue_cutoff_utc':str(issue),'training_rows':int(train.sum()),
        'last_training_target_available_utc':str(frame.loc[train,'target_available_at_utc'].max()),
        'report_rows':int(target.sum()),'prediction_path':str(path),'metrics':metrics,
        'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'sklearn_version':sklearn.__version__}


def run_monthly_retraining(config,progress=print):
    """Compare expanding-history monthly fits with the same frozen references."""
    section=config['pv_monthly_retraining']
    workers=int(section['workers'])
    available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    if not 1<=workers<=min(42,max(1,available-section['reserved_cpu_cores'])):
        raise ValueError('Monthly worker count must leave the configured CPU reserve')
    source=config_path(config,section['source_root'])
    comparison=config_path(config,section['comparison_root'])
    output_root=config_path(config,section['output_root'])
    output=output_root/section['run_id']
    if any(path.resolve().is_relative_to(output.resolve()) for path in [source,comparison]):
        raise ValueError('A monthly study input cannot be under its output directory')
    prepare_experiment_output(output_root,output,
        overwrite=config['experiment_outputs']['overwrite_existing'])
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2,default=str)+'\n')
    jobs=[];inputs={}
    for farm in section['farm_ids']:
        farm_dir=source/f'farm_{farm}'
        frame=modeling_instances(
            pd.read_parquet(farm_dir/'parquet/corpus.parquet'),config,farm_id=farm)
        previous=pd.read_parquet(comparison/f'farm_{farm}/parquet/comparison_predictions.parquet').loc[frame.index]
        keys=['interval_start_utc','delivery_day','auction','measured_ac_mw']
        if not frame[keys].equals(previous[keys]):raise ValueError('Monthly input and reference population differ')
        selection=json.loads((comparison/f'farm_{farm}/json/selection.json').read_text())
        name=selection['best_new_statistical'].split('/')[-1]
        record=farm_dir/'json/trials'/f'{name}.json'
        trial=json.loads(record.read_text())['specification']
        features=farm_dir/'cache'/f"x_{trial['features']}.npy"
        rating=config['farms'][str(farm)]['ac_capacity_kw']/1000
        inputs[farm]=(frame,previous,rating)
        curated=output/f'farm_{farm}/curated_corpus.parquet'
        curated.parent.mkdir(parents=True,exist_ok=True)
        frame.to_parquet(curated,index=False)
        for month in section['report_months']:
            jobs.append({'farm_id':farm,'month':month,'trial_path':str(record),
                'corpus':str(curated),'features':str(features),'source_positions':frame.index.to_numpy(),
                'timezone':config['pv_neural']['delivery_timezone'],'rating':rating,
                'ridge_alpha':config['pv_forecast_search']['baseline_ridge_alpha'],
                'night':config['policy']['night'],
                'output':str(output/f'farm_{farm}/arrays/{month}.npy')})
    progress(f'Monthly refitting {len(jobs)} farm/month cases with {workers} single-thread CPU workers')
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(_monthly_job,job) for job in jobs]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            progress(f"Farm {result['farm_id']} {result['month']}: {result['metrics']['nrmse_percent']:.3f}%")
    rows=[];summaries=[]
    for result in sorted(results,key=lambda item:(item['farm_id'],item['month'])):
        farm=result['farm_id'];frame,previous,rating=inputs[farm]
        mask=(frame.eligible&frame.daylight&frame.delivery_day.str.startswith(result['month'])).to_numpy()
        model=np.load(result['prediction_path'])
        if len(model)!=mask.sum():raise ValueError('Monthly prediction count changed')
        candidates={'monthly_refit':result['metrics']}
        for label in ['old_selected','old_pvlib']:
            curve=previous[f'prediction_{label}'].to_numpy()
            candidates[label]=score(frame,curve,rating,mask)
        for label,metrics in candidates.items():
            rows.append({'farm_id':farm,'month':result['month'],'model':label,
                'issue_cutoff_utc':result['issue_cutoff_utc'],'training_rows':result['training_rows'],
                **metrics})
        summaries.append(result)
    table=pd.DataFrame(rows);(output/'csv').mkdir(exist_ok=True)
    table.to_csv(output/'csv/monthly_metrics.csv',index=False)
    if section['save_plots']:plot_monthly_metrics(table,output/'plots')
    (output/'json').mkdir(exist_ok=True)
    (output/'json/fit_provenance.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (output/'json/manifest.json').write_text(json.dumps({'status':'complete','cases':len(results),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    return {'output':str(output),'cases':len(results)}

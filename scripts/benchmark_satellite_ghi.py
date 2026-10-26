"""Reproducible cleaned forecast-GHI versus observed-GHI PV benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path, load_metadata
from bess_pv_trading.forecasting.pv_artifacts import digest
from bess_pv_trading.forecasting.pv_ensemble_study import score
from bess_pv_trading.forecasting.pv_fitting import available_methods, fit_one, predict_fit, validation_score
from bess_pv_trading.forecasting.pv_forecast_search import prepare_search, _worker, interpolate_forecast_gaps
from bess_pv_trading.forecasting.pv_forecast_bundle import export_bundle, predict_bundle
from bess_pv_trading.forecasting.pv_outages import apply_episode_exclusions, modeling_instances
from bess_pv_trading.forecasting.pv_physical import PhysicalPVModel
from bess_pv_trading.forecasting.pv_satellite_inputs import interpolate_satellite, satellite_inputs
from bess_pv_trading.reporting.artifacts import prepare_experiment_output


def sha(path):
    """Fingerprint exact frozen sources and reusable model artifacts."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    """Atomically publish inspectable study metadata."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')
    temporary.replace(path)


def prepare_inputs(config,farm_id,root):
    """Freeze one label mask and two paired input regimes without changing measurements."""
    section=config['pv_satellite_benchmark']
    source=config_path(config,section['source_root'])/f'farm_{farm_id}'
    raw=pd.read_parquet(source/'parquet/corpus.parquet')
    original=raw.measured_ac_mw.copy()
    marked=apply_episode_exclusions(raw,config,farm_id=farm_id)
    frame=modeling_instances(marked,config,farm_id=farm_id)
    frame,weather_report=interpolate_forecast_gaps(frame,config['pv_forecast_search']['missing_weather_interpolation'])
    # Never use a later split's satellite endpoints to fill an earlier split.
    frame=pd.concat([interpolate_satellite(group,section['satellite_ghi_gap_fill'])
        for _,group in frame.groupby('split',sort=False)]).sort_index()
    settings=json.loads((source/'json/input_provenance.json').read_text())['settings']
    settings['parameter_registry']=None
    farm=load_metadata(config_path(config,config['inputs']['metadata_parquet']),[farm_id])[farm_id]
    # Recompute physical radiation after weather interpolation through the same shared conversion.
    forecast=frame.copy();forecast['satellite_ghi_input_w_m2']=frame.forecast_ghi
    forecast=satellite_inputs(forecast,settings)
    forecast['satellite_ghi_input_w_m2']=frame.satellite_ghi_input_w_m2
    satellite=satellite_inputs(forecast,settings)
    for target in [forecast,satellite]:
        height=int(settings['temperature']['wind_height_m'])
        target['wind_speed_m_s']=np.hypot(target[f'wind_u_{height}m'],target[f'wind_v_{height}m'])
    night=config['policy']['night']
    predictions=[PhysicalPVModel(f.set_index('interval_start_utc'),farm,settings,night).evaluate()
        for f in [forecast,satellite]]
    common=frame.eligible.to_numpy()&np.isfinite(predictions[0])&np.isfinite(predictions[1])
    # Apply a complete-day embargo before the validation and test boundaries.
    timezone=config['pv_neural']['delivery_timezone'];embargo=pd.DateOffset(days=section['embargo_days'])
    boundaries={'train':section['validation_start'],'validation':section['test_start']}
    for split,boundary in boundaries.items():
        cutoff=(pd.Timestamp(boundary)-embargo).tz_localize(timezone).tz_convert('UTC')
        common&=~(frame.split.eq(split).to_numpy()&(frame.target_available_at_utc>cutoff).to_numpy())
    for target in [forecast,satellite]:
        target['eligible']=common
        np.testing.assert_array_equal(target.measured_ac_mw,original.loc[target.index])
        assert not target.loc[target.dubious_excluded|target.outage_excluded,'eligible'].any()
    folder=root/f'farm_{farm_id}'
    for sub in ['json','csv','parquet','plots']:(folder/sub).mkdir(parents=True,exist_ok=True)
    audit=marked[['interval_start_utc','delivery_day','auction','split','measured_ac_mw',
        'satellite_ghi_w_m2',
        'outage_excluded','dubious_excluded','plateau_flag','daylight']].copy()
    audit['satellite_ghi_input_w_m2']=frame.satellite_ghi_input_w_m2.reindex(audit.index)
    audit['satellite_ghi_interpolated']=frame.satellite_ghi_interpolated.reindex(audit.index,fill_value=False)
    audit['eligible']=pd.Series(common,index=frame.index).reindex(audit.index,fill_value=False)
    audit.to_parquet(folder/'parquet/curation_masks.parquet',index=False)
    counts=[]
    for split,part in audit.groupby('split'):
        unique=part.drop_duplicates('interval_start_utc')
        counts.append({'farm_id':farm_id,'split':split,'physical_quarters':len(unique),
            'outage_quarters':int(unique.outage_excluded.sum()),
            'dubious_quarters':int(unique.dubious_excluded.sum()),
            'plateau_quarters':int(unique.plateau_flag.sum()),
            'retained_daylight_quarters':int((unique.eligible&unique.daylight).sum()),
            'interpolated_ghi_quarters':int(unique.satellite_ghi_interpolated.sum())})
    pd.DataFrame(counts).to_csv(folder/'csv/exclusion_counts.csv',index=False)
    write(folder/'json/input_provenance.json',{'source_sha256':sha(source/'parquet/corpus.parquet'),
        'settings':settings,'weather_gap_fill':weather_report,'counts':counts,
        'raw_source_targets_unchanged':True,'excluded_modeling_instances_removed':True,
        'modeling_rows':len(frame),'mask_sha256':sha(folder/'parquet/curation_masks.parquet')})
    return frame.reset_index(drop=True),forecast.reset_index(drop=True),satellite.reset_index(drop=True),predictions,farm,settings


def fit_models(job):
    """Fit one cleaned neural model and select train-only physical fits on validation."""
    from threadpoolctl import threadpool_limits
    record=_worker(job['neural_job'])
    directory=Path(job['neural_job']['output'])
    physical_path=directory/'json/physical_fits.json'
    if job['neural_job']['resume'] and physical_path.exists():
        fits=json.loads(physical_path.read_text())
        if fits['identity']!=job['neural_job']['identity']:raise ValueError('Physical resume identity mismatch')
        return {'farm_id':job['farm_id'],'regime':job['regime'],'neural':record,'physical':fits}
    frame=pd.read_parquet(directory/'parquet/corpus.parquet')
    train=frame.loc[frame.eligible&frame.daylight&frame.split.eq('train')]
    validation=frame.loc[frame.split.eq('validation')].copy()
    candidates=[]
    with threadpool_limits(limits=1):
        for method in available_methods(job['physical_section'],job['farm']):
            fit=fit_one(train,job['farm'],job['settings'],job['physical_section'],method,
                job['settings']['transposition_model'],job['regularization'],job['seed'])
            prediction=predict_fit(validation,job['farm'],job['settings'],job['night'],fit)
            fit['validation_nrmse']=validation_score(validation,prediction,job['farm'].ac_capacity_mw)
            candidates.append(fit)
    selected=min(candidates,key=lambda x:x['validation_nrmse'])
    fits={'identity':job['neural_job']['identity'],'selected':selected,'candidates':candidates,
        'selection':'2026 Jan-Apr validation; each candidate fitted on cleaned 2025 only'}
    write(physical_path,fits)
    return {'farm_id':job['farm_id'],'regime':job['regime'],'neural':record,'physical':fits}


def metrics(frame,predictions,rating,farm_id):
    """Report observed-only and filled-inclusive common-mask errors in AC-rating percent."""
    rows=[];monthly=[];periods=[]
    local=pd.DatetimeIndex(frame.interval_start_utc).tz_convert('Europe/Rome')
    hour=local.hour+local.minute/60
    period=np.select([hour<9,hour<12,hour<15],['before_09','09_to_12','12_to_15'],default='after_15')
    for name,prediction in predictions.items():
        for split in ['train','validation','test']:
            for coverage in ['observed_only','including_interpolation']:
                base=(frame.eligible&frame.daylight&frame.split.eq(split)).to_numpy(copy=True)
                if coverage=='observed_only':base&=~frame.satellite_ghi_interpolated.to_numpy()
                if not base.any():raise ValueError('Empty common evaluation mask')
                rows.append({'farm_id':farm_id,'model':name,'split':split,'coverage':coverage,
                    **score(frame,prediction,rating,base)})
                if coverage=='observed_only':
                    for label in np.unique(period):
                        mask=base&(period==label)
                        if mask.any():periods.append({'farm_id':farm_id,'model':name,'split':split,
                            'period':label,**score(frame,prediction,rating,mask)})
        for (month,auction),positions in frame.groupby([frame.delivery_day.str[:7],frame.auction]).indices.items():
            mask=np.zeros(len(frame),dtype=bool);mask[positions]=True
            mask&=(frame.eligible&frame.daylight&~frame.satellite_ghi_interpolated).to_numpy()
            if mask.any():monthly.append({'farm_id':farm_id,'model':name,'month':month,'auction':auction,
                **score(frame,prediction,rating,mask)})
    return rows,monthly,periods


def publish(config,root,prepared,results):
    """Replay clean models across input regimes and export comparable tables and plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_rows=[];all_monthly=[];all_periods=[];audits=[]
    for farm_id,(raw,forecast,satellite,old,farm,settings) in prepared.items():
        folder=root/f'farm_{farm_id}'
        predictions={'pvlib_configured_forecast':old[0],'pvlib_configured_satellite':old[1]}
        for regime,target in [('forecast',forecast),('satellite',satellite)]:
            directory=root/regime/f'farm_{farm_id}'
            record=results[(farm_id,regime)]
            saved=np.load(directory/'arrays/neural_clean.npy')
            predictions[f'neural_trained_{regime}']=saved
            export_bundle(directory,'neural_clean',directory/'bundle')
            replay=predict_bundle(target,directory/'bundle',farm_id=farm_id)
            difference=float(np.max(np.abs(replay-saved)))
            if difference>config['pv_satellite_benchmark']['replay_tolerance_mw']:
                raise ValueError(f'Farm {farm_id}/{regime}: bundle replay mismatch {difference}')
            audits.append({'farm_id':farm_id,'regime':regime,'bundle_replay_max_error_mw':difference})
            fit=record['physical']['selected']
            predictions[f'pvlib_refit_{regime}']=predict_fit(target,farm,settings,config['policy']['night'],fit)
            if regime=='forecast':
                predictions['neural_forecast_trained_satellite_input']=predict_bundle(satellite,directory/'bundle',farm_id=farm_id)
                predictions['pvlib_forecast_fit_satellite_input']=predict_fit(satellite,farm,settings,config['policy']['night'],fit)
        rows,monthly,periods=metrics(forecast,predictions,farm.ac_capacity_mw,farm_id)
        all_rows+=rows;all_monthly+=monthly;all_periods+=periods
        saved=raw[['interval_start_utc','delivery_day','auction','split','measured_ac_mw']].copy()
        saved['eligible']=forecast.eligible;saved['daylight']=forecast.daylight
        saved['satellite_ghi_interpolated']=forecast.satellite_ghi_interpolated
        for name,value in predictions.items():saved[f'prediction_{name}_mw']=value
        saved.to_parquet(folder/'parquet/predictions.parquet',index=False)
        pd.DataFrame(rows).to_csv(folder/'csv/metrics.csv',index=False)
        # Use validation's median-error and hardest days, plus the median test day.
        active=saved.eligible&saved.daylight&saved.auction.eq('DA')
        errors=saved.loc[active,['delivery_day','split']].copy()
        errors['error']=np.abs(predictions['neural_trained_satellite'][active]-saved.loc[active,'measured_ac_mw'])
        daily=errors.groupby(['split','delivery_day']).error.mean().reset_index()
        chosen=[]
        for split in ['validation','test']:
            ordered=daily.loc[daily.split.eq(split)].sort_values('error')
            chosen.extend([ordered.iloc[len(ordered)//2].delivery_day,ordered.iloc[-1].delivery_day])
        for day in dict.fromkeys(chosen):
            part=saved.loc[saved.delivery_day.eq(day)&saved.auction.eq('DA')]
            start=pd.Timestamp(day).tz_localize('Europe/Rome')
            times=pd.date_range(start,start+pd.DateOffset(days=1),freq='15min',inclusive='left')
            part=part.set_index('interval_start_utc').reindex(times.tz_convert('UTC'))
            fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
            for axis,family in zip(axes,['neural','pvlib']):
                axis.plot(times,part.measured_ac_mw,color='black',label='Measured AC')
                for regime,color in [('forecast','tab:orange'),('satellite','tab:blue')]:
                    column=f'prediction_{family}_{"trained" if family=="neural" else "refit"}_{regime}_mw'
                    axis.plot(times,part[column],color=color,label=f'{family}: trained with {regime} GHI')
                excluded=~part.eligible
                axis.scatter(times[excluded],part.loc[excluded,'measured_ac_mw'],marker='x',color='red',label='Excluded target',s=16)
                axis.axhline(farm.ac_capacity_mw,color='gray',linestyle=':',label='AC rating')
                axis.set(ylabel='AC MW');axis.legend(fontsize=8);axis.grid(alpha=.2)
            axes[-1].set_xlabel('Europe/Rome delivery time')
            fig.suptitle(f'Farm {farm_id}, {day}: cleaned diagnostic benchmark')
            fig.savefig(folder/'plots'/f'{day}.png',dpi=135);plt.close(fig)
    table=pd.DataFrame(all_rows)
    (root/'csv').mkdir(exist_ok=True)
    table.to_csv(root/'csv/metrics.csv',index=False)
    pd.DataFrame(all_monthly).to_csv(root/'csv/monthly_auction_metrics.csv',index=False)
    pd.DataFrame(all_periods).to_csv(root/'csv/day_period_metrics.csv',index=False)
    selected=['pvlib_refit_forecast','pvlib_refit_satellite','neural_trained_forecast','neural_trained_satellite']
    for split in ['validation','test']:
        subset=table.loc[table.split.eq(split)&table.coverage.eq('observed_only')&table.model.isin(selected)]
        pivot=subset.pivot(index='farm_id',columns='model',values='nrmse_percent')
        pivot.to_csv(root/'csv'/f'{split}_comparison.csv')
        ax=pivot.plot.bar(figsize=(12,5),ylabel='Daylight RMSE / AC rating (%)',rot=0)
        ax.set_title(f'Cleaned {split}: observed satellite-GHI evaluation intervals')
        ax.figure.tight_layout();(root/'plots').mkdir(exist_ok=True)
        ax.figure.savefig(root/'plots'/f'{split}_comparison.png',dpi=140);plt.close(ax.figure)
    write(root/'json/replay_audit.json',audits)
    return table


def run(config):
    """Run eight paired cleaned training jobs under the configured CPU reserve."""
    section=config['pv_satellite_benchmark']
    config=deepcopy(config)
    config['pv_forecast_search']['source_root']=section['source_root']
    config['pv_neural']['train_end_day_exclusive']=section['validation_start']
    config['pv_neural']['validation_end_day_exclusive']=section['test_start']
    workers=int(section['workers'])
    available=len(os.sched_getaffinity(0))
    if not 1<=workers<=min(42,max(1,available-section['reserved_cpu_cores'])):
        raise ValueError('Study CPU workers must leave six reserved cores and never exceed 42')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]='1'
    root=config_path(config,section['output_root'])/section['run_id']
    source=config_path(config,section['source_root'])
    inputs={str(farm):sha(source/f'farm_{farm}/parquet/corpus.parquet') for farm in section['farm_ids']}
    exclusions=config_path(config,config['pv_data_quality']['exclusions_csv'])
    identity_config=deepcopy(config);identity_config['pv_satellite_benchmark'].pop('resume',None)
    contract={'configuration':identity_config,'sources':inputs,'exclusions_sha256':sha(exclusions),
        'code_sha256':sha(__file__),'shared_code':{name:sha(Path(__file__).parents[1]/'src/forecasting'/name)
            for name in ['pv_outages.py','pv_residual_networks.py','pv_satellite_inputs.py','pv_forecast_search.py']}}
    identity=digest(contract)
    if section['resume'] and (root/'json/identity.json').exists():
        if json.loads((root/'json/identity.json').read_text())['identity']!=identity:
            raise ValueError('Benchmark inputs/config/code changed; disable resume explicitly')
    prepare_experiment_output(config_path(config,section['output_root']),root,
        overwrite=config['experiment_outputs']['overwrite_existing'],resume=section['resume'])
    write(root/'json/resolved_config.json',config);write(root/'json/identity.json',{'identity':identity,**contract})
    prepared={};jobs=[]
    for farm_id in section['farm_ids']:
        print(f'Preparing common curated mask and GHI regimes for farm {farm_id}',flush=True)
        raw,forecast,satellite,old,farm,settings=prepare_inputs(config,farm_id,root)
        if not (raw.delivery_day.lt(section['validation_start'])==raw.split.eq('train')).all():
            raise ValueError('Configured split disagrees with frozen corpus')
        prepared[farm_id]=(raw,forecast,satellite,old,farm,settings)
        for regime,target in [('forecast',forecast),('satellite',satellite)]:
            variant=root/regime
            write(variant/'json/resolved_config.json',config)
            frame,_,directory,cache,rating,last=prepare_search(config,farm_id,variant,identity,frame_override=target)
            spec=deepcopy(section['neural_by_farm'][str(farm_id)])
            trial={'name':'neural_clean','family':'neural','features':spec.pop('features'),
                'parameters':spec,'seed':section['seed']+farm_id}
            neural_job={'farm_id':farm_id,'trial':trial,'directory':str(cache),'output':str(directory),
                'rating':rating,'train_last_target_utc':last,'identity':identity,'resume':section['resume'],
                'neural_settings':config['pv_forecast_search']['neural_training']}
            physical={**config['pv_calibration'],'methods':section['physical_methods'],
                'night_policy':config['policy']['night']}
            jobs.append({'neural_job':neural_job,'farm_id':farm_id,'regime':regime,'farm':farm,
                'settings':settings,'physical_section':physical,'regularization':section['physical_regularization'],
                'seed':section['seed']+farm_id,'night':config['policy']['night']})
    results={}
    print(f'Launching {len(jobs)} paired model jobs on {workers} single-threaded CPU workers',flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(fit_models,job) for job in jobs]
        for future in as_completed(futures):
            result=future.result();results[(result['farm_id'],result['regime'])]=result
            print(f'Completed farm {result["farm_id"]}/{result["regime"]}: neural and physical fits',flush=True)
    table=publish(config,root,prepared,results)
    write(root/'json/manifest.json',{'status':'complete','identity':identity,
        'interpretation':'Retrospectively curated production subset; observed-GHI diagnostic, not operational forecast.',
        'train':'2025','validation':'2026 Jan-Apr','test':'2026 May-Aug reused holdout',
        'sources':inputs,'exclusions_sha256':sha(exclusions),'model_jobs':len(jobs),
        'raw_source_measurements_unchanged':True,'excluded_feature_target_instances_removed':True,'metrics_rows':len(table)})
    print(root,flush=True)
    return root


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    run(load_config(args.config))

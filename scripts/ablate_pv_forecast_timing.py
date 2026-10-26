"""Matched lead/cycle feature ablation and sixteen-trial neural HPO per candidate."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from benchmark_satellite_ghi import sha, write
from stage_pv_timing_inputs import stage
from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path
from bess_pv_trading.forecasting.pv_artifacts import digest
from bess_pv_trading.forecasting.pv_forecast_search import prepare_search, _worker
from bess_pv_trading.forecasting.pv_forecast_bundle import export_bundle, predict_bundle
from bess_pv_trading.forecasting.pv_ensemble_study import score
from bess_pv_trading.reporting.artifacts import prepare_experiment_output


def report(config,root,parent,contexts,identity,total):
    """Compare paired and tuned treatments on the original common evaluation masks."""
    section=config['pv_timing_ablation'];rows=[];comparisons=[];monthly=[];periods=[];audits=[]
    for farm,(frame,folder,rating,cases) in contexts.items():
        saved=frame[['interval_start_utc','delivery_day','auction','split','eligible','daylight',
            'satellite_ghi_interpolated','measured_ac_mw']].copy()
        local=pd.DatetimeIndex(frame.interval_start_utc).tz_convert(config['pv_neural']['delivery_timezone'])
        hour=local.hour+local.minute/60
        labels=np.select([hour<9,hour<12,hour<15],['before_09','09_to_12','12_to_15'],default='after_15')
        for baseline,trials in cases:
            family=baseline['parameters']['architecture']
            predictions={'without_timing':np.load(parent/f'farm_{farm}/references/{baseline["name"]}.npy')}
            predictions.update({trial['name']:np.load(folder/'arrays'/f'{trial["name"]}.npy') for trial in trials})
            for name,prediction in predictions.items():
                for split in ['train','validation','test']:
                    mask=(frame.eligible&frame.daylight&frame.split.eq(split)&~frame.satellite_ghi_interpolated).to_numpy()
                    rows.append({'farm_id':farm,'family':family,'model':name,'split':split,
                        **score(frame,prediction,rating,mask)})
            table=pd.DataFrame(rows)
            ranking=table.loc[table.farm_id.eq(farm)&table.family.eq(family)&table.split.eq('validation')
                &table.model.ne('without_timing')].sort_values('nrmse_percent')
            winner=ranking.iloc[0].model
            export_bundle(folder,winner,folder/'bundles'/family)
            replay=predict_bundle(frame,folder/'bundles'/family,farm_id=farm)
            error=float(np.max(np.abs(replay-predictions[winner])))
            if error>section['replay_tolerance_mw']:raise ValueError('Exported timing model replay mismatch')
            audits.append({'farm_id':farm,'family':family,'max_replay_error_mw':error})
            for variant,name in {'without_timing':'without_timing','paired_timing':f'{family}_timing_paired','tuned_timing':winner}.items():
                prediction=predictions[name];saved[f'prediction_{family}_{variant}_mw']=prediction
                for split in ['train','validation','test']:
                    row=table.loc[table.farm_id.eq(farm)&table.family.eq(family)&table.model.eq(name)&table.split.eq(split)].iloc[0]
                    comparisons.append({**row.to_dict(),'variant':variant,'baseline_model':baseline['name']})
                    for label in np.unique(labels):
                        mask=(frame.eligible&frame.daylight&frame.split.eq(split)&~frame.satellite_ghi_interpolated).to_numpy()&(labels==label)
                        if mask.any():periods.append({'farm_id':farm,'family':family,'variant':variant,'split':split,
                            'period':label,**score(frame,prediction,rating,mask)})
                for (month,auction),indices in frame.groupby([frame.delivery_day.str[:7],frame.auction]).indices.items():
                    mask=np.zeros(len(frame),dtype=bool);mask[indices]=True
                    mask&=(frame.eligible&frame.daylight&~frame.satellite_ghi_interpolated).to_numpy()
                    if mask.any():monthly.append({'farm_id':farm,'family':family,'variant':variant,'month':month,
                        'auction':auction,**score(frame,prediction,rating,mask)})
        saved.to_parquet(folder/'parquet/predictions.parquet',index=False)
    (root/'csv').mkdir(exist_ok=True)
    for name,data in [('all_trial_metrics',rows),('paired_and_tuned_comparison',comparisons),
            ('monthly_auction_metrics',monthly),('day_period_metrics',periods)]:
        pd.DataFrame(data).to_csv(root/'csv'/f'{name}.csv',index=False)
    write(root/'json/replay_audit.json',audits)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (root/'plots').mkdir(exist_ok=True)
    result=pd.DataFrame(comparisons)
    for split in ['validation','test']:
        part=result.loc[result.split.eq(split)].copy();part['case']=part.farm_id.astype(str)+' / '+part.family
        pivot=part.pivot(index='case',columns='variant',values='nrmse_percent')
        ax=pivot.plot.bar(figsize=(12,5),ylabel='Daylight RMSE / installed AC rating (%)',rot=15)
        ax.set_title(f'Lead/cycle timing feature ablation: {split}')
        ax.figure.tight_layout();ax.figure.savefig(root/'plots'/f'{split}_comparison.png',dpi=140);plt.close(ax.figure)
    write(root/'json/manifest.json',{'status':'complete','identity':identity,'total_fits':total,
        'restored_features':['lead_hours','cycle_hour_sin','cycle_hour_cos'],
        'paired_control':'Original architecture, parameters and seed; three additional input channels',
        'selection':'Prior validation only; unchanged instances, splits, availability and night policy'})
    print(result.loc[result.split.eq('test'),['farm_id','family','variant','nrmse_percent']].to_string(index=False),flush=True)


def run(config):
    """Prepare identical retained instances and run the CPU-limited timing ablation."""
    config=deepcopy(config);section=config['pv_timing_ablation']
    if section['trials_per_model']<16:raise ValueError('At least 16 timing trials per candidate are required')
    parent,staged=stage(config)
    config['pv_forecast_search']['source_root']=str(parent)
    root=config_path(config,section['output_root'])/section['run_id'];workers=int(section['workers'])
    if not 1<=workers<=min(42,max(1,len(os.sched_getaffinity(0))-section['reserved_cpu_cores'])):
        raise ValueError('Invalid CPU budget')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    config['pv_forecast_search']['excluded_features']=section['excluded_features']
    signature={'section':{k:v for k,v in section.items() if k!='resume'},
        'parent_identity':staged['parent_identity'],
        'quality_csv_sha256':sha(config_path(config,config['pv_data_quality']['exclusions_csv'])),
        'training':config['pv_forecast_search']['neural_training'],
        'features':{key:config['pv_forecast_search'][key] for key in ['include_national_holiday','missing_weather_interpolation','include_missing_indicators','baseline_ridge_alpha']},
        'preprocessing':{key:config['pv_neural'][key] for key in ['delivery_timezone','standardization_floor','input_clip_standard_deviations']},
        'ratings':{str(farm):config['farms'][str(farm)]['ac_capacity_kw'] for farm in section['farm_ids']},
        'night_policy':config['policy']['night'],'staged_inputs':staged['files'],
        'worker_sha256':sha(Path(__file__).parents[1]/'src/forecasting/pv_forecast_search.py'),
        'network_sha256':sha(Path(__file__).parents[1]/'src/forecasting/pv_residual_networks.py'),
        'code_sha256':sha(__file__)}
    identity=digest(signature)
    if section['resume'] and (root/'json/identity.json').exists():
        if json.loads((root/'json/identity.json').read_text())['identity']!=identity:raise ValueError('Ablation identity changed')
    if section['resume'] and (root/'json/manifest.json').exists():return root
    prepare_experiment_output(config_path(config,section['output_root']),root,
        overwrite=config['experiment_outputs']['overwrite_existing'],resume=section['resume'])
    write(root/'json/identity.json',{'identity':identity,**signature});write(root/'json/resolved_config.json',config)
    (root/'source').mkdir(exist_ok=True)
    (root/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    jobs=[];contexts={}
    for farm in section['farm_ids']:
        contract=json.loads((parent/f'farm_{farm}/json/candidates.json').read_text())
        baselines=contract['specifications']
        frame=pd.read_parquet(parent/f'farm_{farm}/parquet/corpus.parquet')
        local=deepcopy(config);local['pv_forecast_search']['feature_sets']=list(dict.fromkeys(b['features'] for b in baselines))
        print(f'Preparing farm {farm}: adding lead_hours and UTC cycle-hour sine/cosine',flush=True)
        data,_,folder,cache,rating,last=prepare_search(local,farm,root,identity,frame_override=frame,sequence_length=contract['sequence_length'])
        keys=['interval_start_utc','delivery_day','auction','measured_ac_mw']
        if not data[keys].equals(frame[keys]):raise ValueError('Instance population changed')
        np.testing.assert_array_equal(np.load(cache/'train.npy'),
            np.load(parent/f'farm_{farm}/cache/train.npy'))
        cases=[]
        for baseline in baselines:
            family=baseline['parameters']['architecture'];rng=np.random.default_rng(section['seed']+farm)
            paired=deepcopy(baseline);paired['name']=f'{family}_timing_paired'
            trials=[paired];seen={digest(baseline['parameters'])}
            while len(trials)<section['trials_per_model']:
                params={key:values[int(rng.integers(len(values)))] for key,values in section['neural_space'].items()}
                params['architecture']=family
                if digest(params) in seen:continue
                seen.add(digest(params));number=len(trials)
                trials.append({'name':f'{family}_timing_{number:02d}','family':'neural',
                    'features':baseline['features'],'parameters':params,'seed':section['seed']+farm+number})
            cases.append((baseline,trials))
            for trial in trials:
                jobs.append({'farm_id':farm,'trial':trial,'directory':str(cache),'output':str(folder),
                    'rating':rating,'train_last_target_utc':last,'identity':identity,'resume':section['resume'],
                    'neural_settings':config['pv_forecast_search']['neural_training']})
        contexts[farm]=(data,folder,rating,cases)
    write(root/'json/trial_plan.json',{'total_fits':len(jobs),'workers':workers})
    print(f'Launching {len(jobs)} timing-ablation fits with {workers} single-threaded workers',flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(_worker,job):job for job in jobs}
        for number,future in enumerate(as_completed(futures),1):
            job=futures[future];future.result()
            print(f'[{number}/{len(jobs)}] Farm {job["farm_id"]}: {job["trial"]["name"]} complete',flush=True)
    report(config,root,parent,contexts,identity,len(jobs))
    print(root,flush=True)
    return root


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();run(load_config(args.config))

"""Broad reproducible HPO on paired, curated forecast/satellite-GHI corpora.

First run benchmark_satellite_ghi.py to materialize the common masks and caches.
This search reuses those numerical inputs and never downloads or installs data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import itertools
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
from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path, load_metadata
from bess_pv_trading.forecasting.pv_artifacts import digest
from bess_pv_trading.forecasting.pv_ensemble_study import score
from bess_pv_trading.forecasting.pv_fitting import fit_one, predict_fit, validation_score
from bess_pv_trading.forecasting.pv_forecast_search import _worker
from bess_pv_trading.forecasting.pv_forecast_bundle import export_bundle, predict_bundle
from bess_pv_trading.forecasting.pv_outages import apply_episode_exclusions
from bess_pv_trading.reporting.artifacts import prepare_experiment_output


def trials(config,farm_id):
    """Sample unique configurations, using at least sixteen per tunable family."""
    section=config['pv_cleaned_hpo'];count=int(section['trials_per_family'])
    if count<16:raise ValueError('This study requires at least 16 HPO trials per tunable family')
    rng=np.random.default_rng(section['seed']+farm_id)
    result=[]
    for family,space in section['statistical_spaces'].items():
        combinations=list(itertools.product(*space.values()))
        if len(combinations)<count:raise ValueError(f'{family}: fewer than 16 unique available trials')
        chosen=rng.choice(len(combinations),count,replace=False)
        for number,index in enumerate(chosen):
            result.append({'name':f'{family}_{number:02d}','family':family,
                'features':section['feature_sets'][number%len(section['feature_sets'])],
                'parameters':dict(zip(space,combinations[index])),'seed':section['seed']+farm_id+number})
    for architecture in section['neural_architectures']:
        space=section['neural_space'];combinations=list(itertools.product(*space.values()))
        chosen=rng.choice(len(combinations),count,replace=False)
        for number,index in enumerate(chosen):
            parameters=dict(zip(space,combinations[index]));parameters['architecture']=architecture
            result.append({'name':f'{architecture}_{number:02d}','family':'neural',
                'features':section['feature_sets'][number%len(section['feature_sets'])],
                'parameters':parameters,'seed':section['seed']+farm_id+number})
    result.append({'name':'ols','family':'ols','features':'weather','parameters':{},'seed':section['seed']+farm_id})
    return result


def physical_job(job):
    """One physical-family HPO trial, fitted on training and scored on validation."""
    from threadpoolctl import threadpool_limits
    directory=Path(job['output']);path=directory/'json/physical'/f'{job["name"]}.json'
    array=directory/'arrays'/f'{job["name"]}.npy'
    if job['resume'] and path.exists():
        record=json.loads(path.read_text())
        if record['identity']!=job['identity'] or record['prediction_sha256']!=sha(array):
            raise ValueError('Physical HPO resume mismatch')
        return record
    frame=pd.read_parquet(Path(job['source'])/'parquet/corpus.parquet')
    train=frame.loc[frame.eligible&frame.daylight&frame.split.eq('train')]
    with threadpool_limits(limits=1):
        fitted=fit_one(train,job['farm'],job['settings'],job['physical_section'],job['method'],
            job['transposition'],job['regularization'],job['seed'])
        evaluation=frame.copy()
        evaluation['eligible']=evaluation.eligible&evaluation.daylight
        prediction=predict_fit(evaluation,job['farm'],job['settings'],job['physical_section']['night_policy'],fitted)
        validation=validation_score(frame.loc[frame.split.eq('validation')],
            prediction[frame.split.eq('validation')],job['farm'].ac_capacity_mw)
    np.save(array,prediction)
    record={'name':job['name'],'family':job['family'],'identity':job['identity'],'fit':fitted,
        'validation_nrmse':validation,'prediction_sha256':sha(array),'status':'complete'}
    write(path,record)
    return record


def physical_trials(config,farm_id,farm,settings,source,directory,identity):
    """Search physical transposition, loss and regularization with unique configurations."""
    section=config['pv_cleaned_hpo'];count=section['trials_per_family']
    rng=np.random.default_rng(section['seed']+farm_id)
    jobs=[]
    for family in section['physical_families']:
        if family=='tracker' and farm.tracking_type_id not in {1,3}:continue
        if family=='bifacial' and farm.tracking_type_id!=3:continue
        models=[x for x in section['transposition_models'] if family!='bifacial' or x!='perez-driesse']
        grid=list(itertools.product(models,section['physical_losses'],section['physical_regularizations']))
        if len(grid)<count:raise ValueError(f'{family}: insufficient unique physical configurations')
        for number,index in enumerate(rng.choice(len(grid),count,replace=False)):
            transposition,loss,regularization=grid[index]
            jobs.append({'name':f'pvlib_{family}_{number:02d}','family':f'pvlib_{family}',
                'method':f'{family}_{loss}','transposition':transposition,'regularization':regularization})
    # Scale-only has no further equipment or architecture hyperparameters to tune.
    jobs.append({'name':'pvlib_scale','family':'pvlib_scale','method':'scale_mse',
        'transposition':settings['transposition_model'],'regularization':0.0})
    for job in jobs:
        job.update({'farm':farm,'settings':settings,'source':str(source),'output':str(directory),
            'identity':identity,'resume':section['resume'],'seed':section['seed']+farm_id,
            'physical_section':{**config['pv_calibration'],'night_policy':config['policy']['night']}})
    return jobs


def report(config,root,source,identity):
    """Select by prior validation, report every family and export its reusable winner."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    section=config['pv_cleaned_hpo'];rows=[];selections=[];monthly=[];daily=[];diagnostics=[];periods=[]
    for regime,farm_id in itertools.product(section['input_regimes'],section['farm_ids']):
        folder=root/regime/f'farm_{farm_id}';base=source/regime/f'farm_{farm_id}'
        frame=pd.read_parquet(base/'parquet/corpus.parquet')
        rating=config['farms'][str(farm_id)]['ac_capacity_kw']/1000
        candidates=[]
        for spec in trials(config,farm_id):
            candidates.append((spec['name'],spec['parameters'].get('architecture',spec['family'])))
        for path in (folder/'json/physical').glob('*.json'):
            record=json.loads(path.read_text());candidates.append((record['name'],record['family']))
        for name,family in candidates:
            prediction=np.load(folder/'arrays'/f'{name}.npy')
            for split in ['train','validation','test']:
                for coverage in ['observed_only','including_interpolation']:
                    mask=(frame.eligible&frame.daylight&frame.split.eq(split)).to_numpy(copy=True)
                    if coverage=='observed_only':mask&=~frame.satellite_ghi_interpolated.to_numpy()
                    scores=score(frame,prediction,rating,mask)
                    error=np.abs(prediction[mask]-frame.loc[mask,'measured_ac_mw'].to_numpy())/rating*100
                    rows.append({'regime':regime,'farm_id':farm_id,'model':name,'family':family,
                        'split':split,'coverage':coverage,**scores,
                        'absolute_error_min_percent':float(error.min()),'absolute_error_max_percent':float(error.max()),
                        'rmse_mw':scores['nrmse_percent']*rating/100,'mae_mw':scores['nmae_percent']*rating/100})
        current=pd.DataFrame(rows)
        ranking=current.loc[current.regime.eq(regime)&current.farm_id.eq(farm_id)
            &current.split.eq('validation')&current.coverage.eq('observed_only')].sort_values('nrmse_percent')
        ranking.to_csv(folder/'csv/validation_ranking.csv',index=False)
        family_winners=ranking.drop_duplicates('family')
        chosen=ranking.iloc[0].model
        predictions=frame[['interval_start_utc','delivery_day','auction','split','eligible','daylight',
            'measured_ac_mw','satellite_ghi_w_m2','satellite_ghi_input_w_m2','satellite_ghi_interpolated',
            'dubious_excluded','outage_excluded','plateau_flag']].copy()
        for row in family_winners.itertuples():
            name=row.model;prediction=np.load(folder/'arrays'/f'{name}.npy')
            predictions[f'prediction_{row.family}_mw']=prediction
            selection={'regime':regime,'farm_id':farm_id,'family':row.family,'model':name,
                'validation_nrmse_percent':row.nrmse_percent,'overall_winner':name==chosen}
            selections.append(selection)
            if row.family.startswith('pvlib_'):
                write(folder/'bundles'/row.family/'physical_parameters.json',{
                    **selection,**json.loads((folder/'json/physical'/f'{name}.json').read_text()),
                    'settings':json.loads((source/f'farm_{farm_id}/json/input_provenance.json').read_text())['settings']})
            else:
                export_bundle(folder,name,folder/'bundles'/row.family)
                replay=predict_bundle(frame,folder/'bundles'/row.family,farm_id=farm_id)
                difference=float(np.max(np.abs(replay-prediction)))
                if difference>section['replay_tolerance_mw']:raise ValueError('Selected bundle replay mismatch')
                diagnostics.append({**selection,'replay_max_error_mw':difference})
            mask=frame.eligible&frame.daylight&~frame.satellite_ghi_interpolated
            for (month,auction),positions in frame.groupby([frame.delivery_day.str[:7],frame.auction]).indices.items():
                eligible=np.zeros(len(frame),dtype=bool);eligible[positions]=True;eligible&=mask.to_numpy()
                if eligible.any():monthly.append({**selection,'month':month,'auction':auction,
                    **score(frame,prediction,rating,eligible)})
            data=frame.loc[mask,['delivery_day','auction','split']].copy()
            data['error_mw']=prediction[mask]-frame.loc[mask,'measured_ac_mw'].to_numpy()
            for (day,auction,split),part in data.groupby(['delivery_day','auction','split']):
                daily.append({**selection,'delivery_day':day,'auction':auction,'split':split,
                    'rmse_percent':float(np.sqrt(np.mean(part.error_mw**2))/rating*100),
                    'mae_percent':float(part.error_mw.abs().mean()/rating*100),
                    'energy_error_on_scored_intervals_mwh':float(part.error_mw.sum()*.25)})
        predictions.to_parquet(folder/'parquet/family_winner_predictions.parquet',index=False)
        local=pd.DatetimeIndex(frame.interval_start_utc).tz_convert(config['pv_neural']['delivery_timezone'])
        hour=local.hour+local.minute/60
        labels=np.select([hour<9,hour<12,hour<15],['before_09','09_to_12','12_to_15'],default='after_15')
        for row in family_winners.itertuples():
            values=predictions[f'prediction_{row.family}_mw'].to_numpy()
            for split,label in itertools.product(['train','validation','test'],np.unique(labels)):
                mask=(frame.eligible&frame.daylight&~frame.satellite_ghi_interpolated&frame.split.eq(split)).to_numpy()
                mask=mask&(labels==label)
                if mask.any():periods.append({'regime':regime,'farm_id':farm_id,'model':row.model,
                    'family':row.family,'split':split,'period':label,**score(frame,values,rating,mask)})
        # Representative DA curves are views only; removed quarters appear as visible gaps.
        for split in ['validation','test']:
            eligible=frame.eligible&frame.daylight&frame.split.eq(split)&frame.auction.eq('DA')
            days=frame.loc[eligible,'delivery_day'].drop_duplicates().to_list()
            for day in [days[len(days)//4],days[len(days)//2],days[3*len(days)//4]]:
                part=predictions.loc[predictions.delivery_day.eq(day)&predictions.auction.eq('DA')]
                start=pd.Timestamp(day).tz_localize(config['pv_neural']['delivery_timezone'])
                grid=pd.date_range(start,start+pd.DateOffset(days=1),freq='15min',inclusive='left')
                part=part.set_index('interval_start_utc').reindex(grid.tz_convert('UTC'))
                fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
                ax.plot(grid,part.measured_ac_mw,color='black',label='Measured AC (excluded quarters omitted)')
                winner_family=ranking.iloc[0].family
                for family in dict.fromkeys([winner_family,'pvlib_thermal','direct_tcn','ridge','extra_trees','boosting_direct']):
                    column=f'prediction_{family}_mw'
                    if column in part:ax.plot(grid,part[column],label=family,alpha=.8)
                ax.axhline(rating,color='gray',linestyle=':',label='Installed AC rating')
                ax.set(ylabel='AC power (MW)',xlabel=config['pv_neural']['delivery_timezone'],
                    title=f'Farm {farm_id}, {regime} GHI, {day}: validation-selected family models')
                ax.legend(fontsize=8);ax.grid(alpha=.2)
                fig.savefig(folder/'plots'/f'{split}_{day}.png',dpi=140);plt.close(fig)
        write(folder/'json/selected.json',{'model':chosen,'family':ranking.iloc[0].family,
            'selection':'minimum observed-only cleaned validation daylight nRMSE; never test',
            'mask_identity':identity,'training':'2025','validation':'2026 January-April',
            'test':'2026 May-August reused and retrospectively curated'})
        test=current.loc[current.regime.eq(regime)&current.farm_id.eq(farm_id)
            &current.split.eq('test')&current.coverage.eq('observed_only')&current.model.isin(family_winners.model)]
        test=test.set_index('family').sort_values('nrmse_percent')
        ax=test.nrmse_percent.plot.barh(figsize=(10,6),xlabel='Daylight RMSE / installed AC rating (%)')
        ax.set_title(f'Farm {farm_id}, {regime} GHI: validation-selected family winners on May–August')
        ax.figure.tight_layout();ax.figure.savefig(folder/'plots/family_test_comparison.png',dpi=140);plt.close(ax.figure)
        print(f'Selected farm {farm_id}/{regime}: {chosen} ({ranking.iloc[0].nrmse_percent:.3f}% validation nRMSE)',flush=True)
    for directory in ['csv','json','plots']:(root/directory).mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(root/'csv/all_trial_metrics.csv',index=False)
    pd.DataFrame(selections).to_csv(root/'csv/family_selections.csv',index=False)
    pd.DataFrame(monthly).to_csv(root/'csv/monthly_auction_metrics.csv',index=False)
    pd.DataFrame(daily).to_csv(root/'csv/daily_metrics.csv',index=False)
    pd.DataFrame(periods).to_csv(root/'csv/day_period_metrics.csv',index=False)
    write(root/'json/bundle_replay_audit.json',diagnostics)
    summary=pd.DataFrame(rows).merge(pd.DataFrame(selections),on=['regime','farm_id','model','family'],suffixes=('','_selection'))
    summary=summary.loc[summary.overall_winner&summary.coverage.eq('observed_only')]
    summary.to_csv(root/'csv/overall_winners.csv',index=False)
    paired=[];daily_table=pd.DataFrame(daily)
    for farm_id in section['farm_ids']:
        selected=pd.DataFrame(selections)
        op=selected.loc[selected.farm_id.eq(farm_id)&selected.regime.eq('forecast')&selected.overall_winner].iloc[0]
        sat=selected.loc[selected.farm_id.eq(farm_id)&selected.regime.eq('satellite')&selected.overall_winner].iloc[0]
        left=daily_table.loc[daily_table.farm_id.eq(farm_id)&daily_table.regime.eq('forecast')
            &daily_table.model.eq(op.model)&daily_table.split.eq('test')]
        right=daily_table.loc[daily_table.farm_id.eq(farm_id)&daily_table.regime.eq('satellite')
            &daily_table.model.eq(sat.model)&daily_table.split.eq('test')]
        common=left.merge(right,on=['delivery_day','auction'],suffixes=('_forecast','_satellite'))
        days=sorted(common.delivery_day.unique());rng=np.random.default_rng(section['seed']+farm_id)
        blocks=[common.loc[common.delivery_day.eq(day)] for day in days]
        draws=[];length=int(section['bootstrap_block_days'])
        for _ in range(section['bootstrap_repetitions']):
            starts=rng.integers(0,len(days),int(np.ceil(len(days)/length)))
            indices=[(start+j)%len(days) for start in starts for j in range(length)][:len(days)]
            sample=pd.concat([blocks[index] for index in indices],ignore_index=True)
            draws.append(float(np.sqrt(np.mean(sample.rmse_percent_forecast**2))-
                np.sqrt(np.mean(sample.rmse_percent_satellite**2))))
        paired.append({'farm_id':farm_id,'forecast_model':op.model,'satellite_model':sat.model,
            'metric':'forecast minus satellite daylight nRMSE, percentage points of AC rating',
            'point_difference':float(np.sqrt(np.mean(common.rmse_percent_forecast**2))-
                np.sqrt(np.mean(common.rmse_percent_satellite**2))),
            'lower_95':float(np.quantile(draws,.025)),'upper_95':float(np.quantile(draws,.975)),
            'block_days':length,'repetitions':section['bootstrap_repetitions']})
    pd.DataFrame(paired).to_csv(root/'csv/paired_forecast_vs_satellite.csv',index=False)
    return summary


def run(config,plan_only=False):
    """Launch independent HPO trials while enforcing the shared CPU budget."""
    section=config['pv_cleaned_hpo'];workers=int(section['workers'])
    available=len(os.sched_getaffinity(0))
    if not 1<=workers<=min(42,max(1,available-section['reserved_cpu_cores'])):
        raise ValueError('HPO must use at most 42 CPUs and leave the configured reserve')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    source=config_path(config,section['prepared_source_root']);root=config_path(config,section['output_root'])/section['run_id']
    if plan_only:
        print(json.dumps({'workers':workers,'trial_counts':{farm:len(trials(config,farm)) for farm in section['farm_ids']},
            'regimes':section['input_regimes'],'physical_trials_per_family':section['trials_per_family']},indent=2));return
    if not (source/'json/manifest.json').exists():raise ValueError('Run the paired cleaned benchmark first')
    identity_config=deepcopy(config);identity_config['pv_cleaned_hpo'].pop('resume',None)
    contract={'config':identity_config,'prepared_identity':json.loads((source/'json/identity.json').read_text())['identity'],
        'exclusions_csv_sha256':sha(config_path(config,config['pv_data_quality']['exclusions_csv'])),
        'code':sha(__file__),'worker_code':sha(Path(__file__).parents[1]/'src/forecasting/pv_forecast_search.py'),
        'network_code':sha(Path(__file__).parents[1]/'src/forecasting/pv_residual_networks.py')}
    identity=digest(contract)
    previous=root/'json/identity.json'
    if section['resume'] and previous.exists() and json.loads(previous.read_text())['identity']!=identity:
        raise ValueError('HPO identity changed; disable resume explicitly')
    if section['resume'] and (root/'json/manifest.json').exists():
        print(f'Completed matching study already exists: {root}');return
    prepare_experiment_output(config_path(config,section['output_root']),root,
        overwrite=config['experiment_outputs']['overwrite_existing'],resume=section['resume'])
    write(previous,{'identity':identity,**contract});write(root/'json/resolved_config.json',config)
    jobs=[];counts=[];population_checks=[]
    farms=load_metadata(config_path(config,config['inputs']['metadata_parquet']),section['farm_ids'])
    for regime,farm_id in itertools.product(section['input_regimes'],section['farm_ids']):
        base=source/regime/f'farm_{farm_id}';folder=root/regime/f'farm_{farm_id}'
        for name in ['json/trials','json/physical','models','arrays','csv','parquet','plots','bundles']:(folder/name).mkdir(parents=True,exist_ok=True)
        # Reuse immutable prepared arrays. Exported bundles copy all their dependencies.
        if not (folder/'cache').exists():(folder/'cache').symlink_to((base/'cache').resolve(),target_is_directory=True)
        write(root/regime/'json/resolved_config.json',config)
        write(folder/'json/input_provenance.json',json.loads((base/'json/input_provenance.json').read_text()))
        provenance=json.loads((source/f'farm_{farm_id}/json/input_provenance.json').read_text())
        settings=provenance['settings'];rating=farms[farm_id].ac_capacity_mw
        audit=pd.read_parquet(source/f'farm_{farm_id}/parquet/curation_masks.parquet')
        marked=apply_episode_exclusions(audit,config,farm_id=farm_id)
        keys=['interval_start_utc','delivery_day','auction']
        actual=pd.read_parquet(base/'parquet/corpus.parquet',columns=keys)
        expected=marked.loc[~marked.episode_excluded,keys].reset_index(drop=True)
        if not actual.equals(expected):raise ValueError(f'{regime}/{farm_id}: prepared instances disagree with authoritative CSV')
        cache_rows=len(np.load(base/'cache/y.npy',mmap_mode='r'))
        if cache_rows!=len(actual):raise ValueError('Neural/regression cache still contains excluded instances')
        population_checks.append({'farm_id':farm_id,'regime':regime,'retained_instances':len(actual),
            'excluded_instances':int(marked.episode_excluded.sum()),'csv_is_sole_quality_source':True})
        frame=pd.read_parquet(base/'parquet/corpus.parquet',columns=['split','eligible','target_available_at_utc'])
        last=str(frame.loc[frame.eligible&frame.split.eq('train'),'target_available_at_utc'].max())
        specifications=trials(config,farm_id)
        for trial in specifications:
            jobs.append(('learned',{'farm_id':farm_id,'trial':trial,'directory':str(base/'cache'),'output':str(folder),
                'rating':rating,'train_last_target_utc':last,'identity':identity,'resume':section['resume'],
                'neural_settings':config['pv_forecast_search']['neural_training']}))
        physical=physical_trials(config,farm_id,farms[farm_id],settings,base,folder,identity)
        jobs.extend(('physical',job) for job in physical)
        counts.append({'farm_id':farm_id,'regime':regime,'learned_trials':len(specifications),'physical_trials':len(physical)})
    write(root/'json/trial_plan.json',{'counts':counts,'total_trials':len(jobs),'workers':workers})
    write(root/'json/csv_population_validation.json',population_checks)
    print(f'Launching {len(jobs)} fits with {workers} single-threaded workers',flush=True)
    # Interleave farms and regimes so all comparisons progress together.
    rng=np.random.default_rng(section['seed']);rng.shuffle(jobs)
    failures=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(_worker if kind=='learned' else physical_job,job):(kind,job) for kind,job in jobs}
        completed=0
        for future in as_completed(futures):
            kind,job=futures[future];completed+=1
            try:record=future.result()
            except Exception as exc:
                failures.append({'kind':kind,'output':job['output'],'name':job.get('name',job.get('trial',{}).get('name')),'error':str(exc)})
                write(root/'json/failures.json',failures)
                print(f'[{completed}/{len(jobs)}] FAILED: {failures[-1]}',flush=True)
                continue
            print(f'[{completed}/{len(jobs)}] {job["output"]}: {job.get("name",job.get("trial",{}).get("name"))} complete',flush=True)
    if failures:raise RuntimeError(f'{len(failures)} HPO trials failed; inspect failures.json and resume')
    summary=report(config,root,source,identity)
    write(root/'json/manifest.json',{'status':'complete','identity':identity,'total_trials':len(jobs),
        'counts':counts,'winner_rows':len(summary),'exclusions':'Approved dubious episodes, plateaus and sustained-zero runs across all splits.',
        'selection':'2026 Jan-Apr only; 2026 May-Aug reused curated holdout reported separately',
        'operational_regime':'forecast','satellite_regime':'retrospective diagnostic'})
    print(root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    run(load_config(args.config),plan_only=args.plan_only)

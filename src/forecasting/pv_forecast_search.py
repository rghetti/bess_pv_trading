"""Reproducible statistical/neural search using weather and time, without proxy geometry."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from ..data import config_path, night_window_mask
from ..reporting.artifacts import prepare_experiment_output
from .pv_ensemble_study import score
from .pv_fitting import balanced_weights
from .pv_neural_data import feature_frame, fit_preprocessor, transform_features, pack_sequences


# Keep cached research artefacts independent of filesystem modification timestamps.
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n');temporary.replace(path)


def weather_features(frame,feature_set,timezone,*,include_national_holiday=False,
                     excluded_features=()):
    """Use forecast weather and known calendar, never observed or proxy inputs."""
    features=feature_frame(frame,feature_set,timezone,include_solar_geometry=False,
        include_national_holiday=include_national_holiday)
    unknown=set(excluded_features)-set(features.columns)
    if unknown:
        raise ValueError(f'Unknown excluded PV features: {sorted(unknown)}')
    return features.drop(columns=list(excluded_features))


def interpolate_forecast_gaps(frame,method='linear_per_view'):
    """Interpolate short weather gaps within each delivery-day forecast vintage.

    Interior gaps use linear interpolation on UTC interval times. At the start
    or end of a view, linear extrapolation from the nearest two finite values
    fills a boundary gap. A single available value is held constant. Original
    input frames are never mutated; remaining all-missing fields are reported.
    """
    if method not in {'linear_per_view','none'}:
        raise ValueError(f'Unsupported forecast gap interpolation: {method}')
    result=frame.copy()
    columns=[name for name in ['forecast_ghi','forecast_bhi','temperature_c','dew_point_c',
        'wind_u_10m','wind_v_10m','cloud_total_pct','cloud_low_pct','cloud_mid_pct',
        'cloud_high_pct','precipitation_mm','snow_water_equivalent_m','visibility_km',
        'wind_u_100m','wind_v_100m'] if name in result]
    group_columns=[name for name in ['delivery_day','auction','forecast_cycle_utc'] if name in result]
    before={name:int((~np.isfinite(result[name].to_numpy(dtype=float))).sum()) for name in columns}
    report={'method':method,'rows_before':before,'filled_by_feature':{name:0 for name in columns},
        'boundary_extrapolated_by_feature':{name:0 for name in columns},
        'unresolved_by_feature':{name:0 for name in columns},'view_count':0}
    if method=='none' or not columns:
        report['rows_after']=before.copy()
        return result,report
    grouped=result.groupby(group_columns,sort=False,dropna=False).indices if group_columns else {'all':np.arange(len(result))}
    report['view_count']=len(grouped)
    weather_values={name:result[name].to_numpy(dtype=float,copy=True) for name in columns}
    utc_ns=pd.DatetimeIndex(result.interval_start_utc).asi8
    for positions in grouped.values():
        positions=np.asarray(positions,dtype=int)
        if not any((~np.isfinite(weather_values[name][positions])).any() for name in columns):
            continue
        order=np.argsort(utc_ns[positions],kind='stable')
        positions=positions[order]
        x=(utc_ns[positions]-utc_ns[positions][0]).astype(np.float64)/1e9
        for name in columns:
            values=weather_values[name][positions].copy()
            finite=np.isfinite(values)
            missing=~finite
            if not missing.any():
                continue
            valid=np.flatnonzero(finite)
            if not len(valid):
                report['unresolved_by_feature'][name]+=int(missing.sum())
                continue
            filled=np.interp(x[missing],x[valid],values[valid])
            left=missing & (np.arange(len(values))<valid[0])
            right=missing & (np.arange(len(values))>valid[-1])
            if len(valid)>1:
                left_positions=np.flatnonzero(left)
                right_positions=np.flatnonzero(right)
                missing_positions=np.flatnonzero(missing)
                if len(left_positions):
                    slope=(values[valid[1]]-values[valid[0]])/(x[valid[1]]-x[valid[0]])
                    filled[np.searchsorted(missing_positions,left_positions)]=values[valid[0]]+slope*(x[left_positions]-x[valid[0]])
                if len(right_positions):
                    slope=(values[valid[-1]]-values[valid[-2]])/(x[valid[-1]]-x[valid[-2]])
                    filled[np.searchsorted(missing_positions,right_positions)]=values[valid[-1]]+slope*(x[right_positions]-x[valid[-1]])
            report['filled_by_feature'][name]+=int(missing.sum())
            report['boundary_extrapolated_by_feature'][name]+=int(left.sum()+right.sum())
            weather_values[name][positions[missing]]=filled
    for name in columns:
        result[name]=weather_values[name]
    report['rows_after']={name:int((~np.isfinite(result[name].to_numpy(dtype=float))).sum()) for name in columns}
    return result,report


def generate_trials(section,farm_id):
    """Sample finite YAML grids deterministically; do not inspect any model scores."""
    rng=np.random.default_rng(section['seed']+farm_id);trials=[]
    for family,space in (section['statistical_spaces'] or {}).items():
        for number in range(section['statistical_trials_per_family']):
            parameters={key:values[int(rng.integers(len(values)))] for key,values in space.items()}
            trials.append({'name':f'{family}_{number:02}', 'family':family,
                'features':section['feature_sets'][number%len(section['feature_sets'])],
                'parameters':parameters,'seed':section['seed']+farm_id+number})
    for architecture in section['neural_architectures']:
        space=section['neural_space']
        size_depth=[(width,depth) for width in space['width'] for depth in space['depth']]
        rng.shuffle(size_depth)
        for number in range(section['neural_trials_per_architecture']):
            spec={key:values[int(rng.integers(len(values)))] for key,values in space.items()
                if key not in {'width','depth'}}
            if size_depth:
                spec['width'],spec['depth']=size_depth[number%len(size_depth)]
            spec['architecture']=architecture
            trials.append({'name':f'{architecture}_{number:02}','family':'neural',
                'features':section['feature_sets'][number%len(section['feature_sets'])],
                'parameters':spec,'seed':section['seed']+farm_id+number})
    return trials


def _fit_statistical(x,y,train,weight,baseline,trial):
    """Fit native statistical models; residual models correct a train-only linear reference."""
    from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
    from sklearn.preprocessing import SplineTransformer
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.pipeline import make_pipeline
    family=trial['family'];parameters=dict(trial['parameters']);transform=None
    target=y-baseline if family.endswith('_residual') else y
    if family=='ridge':
        model=Ridge(**parameters)
    elif family=='ols':
        model=LinearRegression(**parameters)
    elif family.startswith('boosting'):
        model=HistGradientBoostingRegressor(random_state=trial['seed'],early_stopping=False,**parameters)
    elif family=='extra_trees':
        model=ExtraTreesRegressor(random_state=trial['seed'],n_jobs=1,**parameters)
    elif family=='huber':model=HuberRegressor(**parameters)
    else:
        alpha=parameters.pop('alpha')
        if family.startswith('spline'):transform=SplineTransformer(**parameters)
        elif family.startswith('rbf'):transform=RBFSampler(random_state=trial['seed'],**parameters)
        else:raise ValueError(f'Unknown statistical family {family}')
        transformed=transform.fit_transform(x[train]);model=Ridge(alpha=alpha)
        model.fit(transformed,target[train],sample_weight=weight)
        predictor=make_pipeline(transform,model)
        raw=predictor.predict(x)
        return predictor,raw+baseline if family.endswith('_residual') else raw
    model.fit(x[train],target[train],sample_weight=weight)
    raw=model.predict(x)
    return model,raw+baseline if family.endswith('_residual') else raw


def _worker(job):
    """One isolated, single-threaded CPU fit with atomic reusable outputs."""
    import sklearn  # Load the conda C++ runtime before optional PyTorch.
    from threadpoolctl import threadpool_limits
    started=time.perf_counter();trial=job['trial'];directory=Path(job['directory']);root=Path(job['output'])
    status=root/'json/trials'/f"{trial['name']}.json"
    prediction_path=root/'arrays'/f"{trial['name']}.npy"
    if job['resume'] and status.exists():
        record=json.loads(status.read_text())
        if record['identity']!=job['identity']:raise ValueError('Trial identity mismatch on resume')
        if _sha(prediction_path)!=record['prediction_sha256']:raise ValueError('Prediction checksum mismatch')
        if _sha(record['artifact'])!=record['artifact_sha256']:raise ValueError('Model checksum mismatch')
        return record
    x=np.load(directory/f"x_{trial['features']}.npy",mmap_mode='r')
    y=np.load(directory/'y.npy');train=np.load(directory/'train.npy');weight=np.load(directory/'weights.npy')
    baseline=np.load(directory/f"baseline_{trial['features']}.npy")
    metadata={'specification':trial,'identity':job['identity'],'farm_id':job['farm_id'],
        'features':json.loads((directory/f"preprocessor_{trial['features']}.json").read_text()),
        'input_contract':'weather_time_known_holiday_no_proxy_geometry_no_auction_no_age',
        'train_last_target_utc':job['train_last_target_utc'],'reference_regression':str(directory/f"baseline_{trial['features']}.joblib")}
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        if trial['family']=='neural':
            import torch
            from .pv_residual_networks import fit_residual_network
            packed=dict(np.load(directory/f"packed_{trial['features']}.npz",allow_pickle=False))
            model,prediction,training=fit_residual_network(packed,baseline*job['rating'],
                np.load(directory/f"train_groups_{trial['features']}.npy"),
                np.load(directory/f"validation_groups_{trial['features']}.npy"),trial['parameters'],
                job['neural_settings'],trial['seed'],job['rating'])
            artifact=root/'models'/f"{trial['name']}.pt"
            torch.save(model.state_dict(),artifact);metadata['training']=training
            metadata['torch_version']=str(torch.__version__)
        else:
            model,fraction=_fit_statistical(x,y,train,weight,baseline,trial)
            prediction=np.clip(fraction,0,1)*job['rating']
            artifact=root/'models'/f"{trial['name']}.joblib"
            joblib.dump({'model':model,**metadata},artifact)
        prediction[np.load(directory/'night.npy')]=0
    if not np.isfinite(prediction).all() or (prediction<0).any() or (prediction>job['rating']+1e-7).any():
        raise ValueError('Model prediction violates the AC output contract')
    np.save(prediction_path,prediction,allow_pickle=False)
    metadata.update(status='complete',runtime_seconds=time.perf_counter()-started,
        artifact=str(artifact),artifact_sha256=_sha(artifact),prediction_sha256=_sha(prediction_path),
        warnings=[str(w.message) for w in caught],sklearn_version=sklearn.__version__)
    metadata['converged_without_warning']=not any('converg' in str(w.message).lower() for w in caught)
    _write(status,metadata)
    return metadata


def prepare_search(config,farm_id,root,identity,*,frame_override=None,sequence_length=None):
    """Reuse frozen vintages and masks; fit feature transforms on 2025 targets only."""
    from sklearn.linear_model import Ridge
    section=config['pv_forecast_search'];source=config_path(config,section['source_root'])/f'farm_{farm_id}'
    from .pv_outages import modeling_instances
    frame=modeling_instances(
        pd.read_parquet(source/'parquet/corpus.parquet') if frame_override is None else frame_override,
        config,farm_id=farm_id)
    # A supplied numerical corpus is self-contained; historical experiment
    # predictions are not required to fit its features or unchanged AC targets.
    prior=(pd.read_parquet(source/'parquet/predictions.parquet')
        if frame_override is None else frame.copy())
    keys=['interval_start_utc','delivery_day','auction','forecast_cycle_utc']
    native_length=(int(sequence_length) if sequence_length is not None
        else int(prior.groupby(['delivery_day','auction']).size().max()))
    prior=frame[keys].merge(prior,on=keys,how='left',validate='one_to_one')
    frame=frame.reset_index(drop=True)
    if not frame[keys+['measured_ac_mw']].equals(prior[keys+['measured_ac_mw']]):
        raise ValueError('Frozen corpus/reference keys or raw targets disagree')
    if (frame.forecast_available_at_utc>frame.decision_cutoff_utc).any():raise ValueError('Unavailable forecast')
    rating=config['farms'][str(farm_id)]['ac_capacity_kw']/1000
    # All valid training quarters, including genuine night zeros, are retained.
    train=(frame.eligible & frame.split.eq('train')).to_numpy()
    validation=(frame.eligible & frame.daylight & frame.split.eq('validation')).to_numpy()
    test=(frame.eligible & frame.daylight & frame.split.eq('test')).to_numpy()
    if frame.loc[train,'target_available_at_utc'].max()>=frame.loc[validation,'decision_cutoff_utc'].min():
        raise ValueError('Training availability crosses the validation boundary')
    if frame.loc[validation,'target_available_at_utc'].max()>=frame.loc[test,'decision_cutoff_utc'].min():
        raise ValueError('Validation availability crosses the test boundary')
    directory=root/f'farm_{farm_id}';cache=directory/'cache'
    for folder in ['json/trials','arrays','models','parquet','csv','plots','cache']:(directory/folder).mkdir(parents=True,exist_ok=True)
    night=night_window_mask(pd.DatetimeIndex(frame.interval_start_utc),config['policy']['night'])
    weight=balanced_weights(frame.loc[train]);weight*=len(weight)
    y=frame.measured_ac_mw.to_numpy()/rating
    for name,value in [('y',y),('train',train),('weights',weight),('night',night)]:np.save(cache/f'{name}.npy',value)
    # This feature population intentionally never consults proxy geometry.
    neural_frame=frame.copy();neural_frame['daylight']=~night
    interpolated_frame,interpolation_report=interpolate_forecast_gaps(
        frame,section['missing_weather_interpolation'])
    for feature in section['feature_sets']:
        values=weather_features(interpolated_frame,feature,config['pv_neural']['delivery_timezone'],
            include_national_holiday=section['include_national_holiday'],
            excluded_features=section['excluded_features'])
        pre=fit_preprocessor(values,train,config['pv_neural'],
            include_missing_indicators=section['include_missing_indicators'])
        pre['missing_weather_interpolation']=section['missing_weather_interpolation']
        x=transform_features(values,pre).astype(np.float32)
        _write(cache/f'preprocessor_{feature}.json',pre);np.save(cache/f'x_{feature}.npy',x)
        baseline=Ridge(alpha=section['baseline_ridge_alpha']).fit(x[train],y[train],sample_weight=weight)
        base=np.clip(baseline.predict(x),0,1);base[night]=0
        np.save(cache/f'baseline_{feature}.npy',base);joblib.dump(baseline,cache/f'baseline_{feature}.joblib')
        packed=pack_sequences(neural_frame,x,rating,minimum_length=native_length)
        np.savez(cache/f'packed_{feature}.npz',**{key:packed[key] for key in ['x','y','mask','daylight','indices','view_ids']})
        np.save(cache/f'train_groups_{feature}.npy',np.flatnonzero((packed['split']=='train') & packed['mask'].any(1)))
        # Validate on the unchanged shared daylight mask, rather than a proxy input.
        valid_mask=np.zeros(packed['mask'].shape,dtype=bool);valid=packed['indices']>=0
        valid_mask[valid]=validation[packed['indices'][valid]]
        packed['mask'][np.asarray(packed['split'])=='validation']=valid_mask[np.asarray(packed['split'])=='validation']
        np.savez(cache/f'packed_{feature}.npz',**{key:packed[key] for key in ['x','y','mask','daylight','indices','view_ids']})
        np.save(cache/f'validation_groups_{feature}.npy',np.flatnonzero((packed['split']=='validation') & packed['mask'].any(1)))
    source_corpus=source/'parquet/corpus.parquet'
    _write(directory/'json/input_provenance.json',{'corpus_sha256':(_sha(source_corpus) if source_corpus.exists()
        else hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()),
        'prepared_override':frame_override is not None,
        'prepared_frame_sha256':hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest(),
        'prior_predictions_sha256':(_sha(source/'parquet/predictions.parquet') if frame_override is None else None),
        'identity':identity,'rating_mw':rating,'weather_geometry_used':False,'auction_features_used':False,
        'aging_features_used':False,'national_holiday_feature_used':bool(section['include_national_holiday']),
        'missing_weather_interpolation':interpolation_report,
        'missing_indicator_channels_used':bool(section['include_missing_indicators']),
        'training_intervals':int(train.sum()),
        'validation_daylight_intervals':int(validation.sum()),'test_daylight_intervals':int(test.sum()),
        'test_status':'reused holdout; previously inspected, never used by fitting/selection code'})
    frame.to_parquet(directory/'parquet/corpus.parquet',index=False)
    return frame,prior,directory,cache,rating,str(frame.loc[train,'target_available_at_utc'].max())


def run_forecast_search(config,progress=print):
    """Search both farms with a shared chronology, bounded CPU use and resumable fits."""
    section=config['pv_forecast_search'];workers=int(section['workers'])
    available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    if not 1<=workers<=min(42,max(1,available-int(section['reserved_cpu_cores']))):
        raise ValueError('Worker count must leave the configured CPU reserve and be at most 42')
    output_root=config_path(config,section['output_root']);root=output_root/section['run_id']
    comparable=deepcopy(config);comparable['pv_forecast_search'].pop('resume',None)
    sources={str(farm):{name:_sha(config_path(config,section['source_root'])/f'farm_{farm}'/'parquet'/name)
        for name in ['corpus.parquet','predictions.parquet']} for farm in section['farm_ids']}
    contract={'configuration':comparable,'inputs':sources,'code_sha256':_sha(__file__),
        'network_code_sha256':_sha(Path(__file__).with_name('pv_residual_networks.py'))}
    identity=hashlib.sha256(json.dumps(contract,sort_keys=True,default=str).encode()).hexdigest()
    previous=root/'json/identity.json'
    if section['resume'] and previous.exists() and json.loads(previous.read_text())['identity']!=identity:
        raise ValueError('Search configuration/code changed; disable resume or choose another run ID')
    prepare_experiment_output(output_root,root,overwrite=config['experiment_outputs']['overwrite_existing'],resume=section['resume'])
    _write(previous,{'identity':identity,**contract});_write(root/'json/resolved_config.json',config)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]='1'
    jobs=[];farms={}
    for farm_id in section['farm_ids']:
        progress(f'Preparing farm {farm_id} without geometry, auction identity or aging inputs')
        frame,prior,directory,cache,rating,last=prepare_search(config,farm_id,root,identity)
        farms[farm_id]=(frame,prior,directory,rating)
        for trial in generate_trials(section,farm_id):
            jobs.append({'farm_id':farm_id,'trial':trial,'directory':str(cache),'output':str(directory),
                'rating':rating,'train_last_target_utc':last,'identity':identity,
                'resume':section['resume'],'neural_settings':section['neural_training']})
    progress(f'Launching {len(jobs)} model fits across {workers} single-threaded CPU workers')
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(_worker,job):job for job in jobs}
        for future in as_completed(pending):
            job=pending[future];record=future.result()
            progress(f"Farm {job['farm_id']} {job['trial']['name']}: {record['runtime_seconds']:.1f}s")
    reports=[]
    for farm_id,(frame,prior,directory,rating) in farms.items():
        reports.append(report_search(config,farm_id,frame,prior,directory,rating))
    _write(root/'json/manifest.json',{'status':'complete','identity':identity,'farms':reports})
    return {'output':str(root),'farms':reports}


def report_search(config,farm_id,frame,prior,directory,rating):
    """Publish common-mask scores, monthly residuals and training-only-model artifacts."""
    predictions={name.removeprefix('prediction_').removesuffix('_mw'):prior[name].to_numpy() for name in prior if name.startswith('prediction_')}
    for trial in generate_trials(config['pv_forecast_search'],farm_id):
        predictions[trial['name']]=np.load(directory/'arrays'/f"{trial['name']}.npy")
    rows=[];monthly=[]
    for name,prediction in predictions.items():
        for split in ['train','validation','test']:
            mask=(frame.eligible & frame.daylight & frame.split.eq(split)).to_numpy()
            rows.append({'model':name,'split':split,**score(frame,prediction,rating,mask)})
        for (month,auction),positions in frame.groupby([frame.delivery_day.str[:7],frame.auction]).indices.items():
            mask=np.zeros(len(frame),dtype=bool);mask[positions]=True
            mask &= (frame.eligible & frame.daylight).to_numpy()
            if mask.any():monthly.append({'model':name,'month':month,'auction':auction,**score(frame,prediction,rating,mask)})
    table=pd.DataFrame(rows);table.to_csv(directory/'csv/metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(directory/'csv/monthly_auction_metrics.csv',index=False)
    candidates={t['name'] for t in generate_trials(config['pv_forecast_search'],farm_id)}
    ranking=table.loc[table.split.eq('validation') & table.model.isin(candidates)].sort_values('nrmse_percent')
    ranking.to_csv(directory/'csv/validation_ranking.csv',index=False)
    selected=ranking.iloc[0].model
    saved=frame[['interval_start_utc','delivery_day','auction','split','eligible','daylight','measured_ac_mw']].copy()
    for name,prediction in predictions.items():saved[f'prediction_{name}_mw']=prediction
    saved.to_parquet(directory/'parquet/predictions.parquet',index=False)
    report={'farm_id':farm_id,'selected_new_model':selected,'selection':'Jan-Apr 2026 validation balanced daylight NRMSE only',
        'test_used_for_selection':False,'test_is_reused_holdout':True,'output':str(directory)}
    _write(directory/'json/selection.json',report)
    return report

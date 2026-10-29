"""Chronological tree-ensemble benchmark with reusable physical PV features."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from ...data import config_path, load_metadata
from ...reporting.artifacts import prepare_experiment_output
from ..training.pv_fitting import balanced_weights
from ..data.pv_neural_data import feature_frame, fit_preprocessor, transform_features
from ..models.pv_physical import PhysicalPVModel
from ..data.pv_study_inputs import farm_settings
from ..data.pv_outages import modeling_instances


def physical_features(frame, farm, settings, night):
    """Use the production physical engine, without reading measured AC or satellite GHI."""
    result = pd.DataFrame(index=frame.index)
    for rear in (False, True) if farm.tracking_type_id == 3 else (False,):
        declared = deepcopy(settings)
        declared['bifacial']['enabled'] = rear
        if rear and declared['transposition_model'] == 'perez-driesse':
            declared['transposition_model'] = 'haydavies'
        model = PhysicalPVModel(frame.set_index('interval_start_utc'), farm, declared, night)
        values = model.evaluate(diagnostics=True)
        prefix = 'rear' if rear else 'front'
        for key in ('surface_tilt_deg', 'surface_azimuth_deg', 'poa_w_m2',
                    'poa_front_w_m2', 'poa_back_w_m2', 'cell_temperature_c', 'pv_ac_mw'):
            result[f'{prefix}_{key}'] = values[key]
    return result


def score(frame, prediction, rating, mask):
    """Use identical finite daylight targets and balanced day/auction weights."""
    part = frame.loc[mask]
    error = (prediction[mask]-part.measured_ac_mw.to_numpy())/rating
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite predictions on the common evaluation mask")
    weights = balanced_weights(part)
    return {'rows': len(part), 'days': part.delivery_day.nunique(),
        'nrmse_percent': float(np.sqrt(weights @ (error**2))*100),
        'nmae_percent': float(weights @ np.abs(error)*100),
        'bias_percent': float(weights @ error*100)}


def run_ensemble_study(config, progress=print):
    """Train only on frozen training rows; select on validation, report test separately."""
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    section = config['pv_ensemble_study']
    farm_id = int(config['farm_id'])
    source = config_path(config, section['source_root']) / f'farm_{farm_id}'
    frame = modeling_instances(
        pd.read_parquet(source/'parquet/corpus.parquet'),config,farm_id=farm_id)
    prior = pd.read_parquet(source/'parquet/predictions.parquet').loc[frame.index]
    keys = ['interval_start_utc', 'delivery_day', 'auction', 'forecast_cycle_utc']
    if not frame[keys].equals(prior[keys]):
        raise ValueError('Reference predictions do not match the frozen corpus')
    if not frame.measured_ac_mw.equals(prior.measured_ac_mw):
        raise ValueError('Reference and corpus targets differ')
    if (frame.forecast_available_at_utc > frame.decision_cutoff_utc).any():
        raise ValueError('An input forecast was unavailable at submission')
    farm = load_metadata(config_path(config,config['inputs']['metadata_parquet']),[farm_id])[farm_id]
    settings = farm_settings(config,farm)
    physical = physical_features(frame,farm,settings,config['policy']['night'])
    base = feature_frame(frame,'weather',config['pv_neural']['delivery_timezone'])
    # Auction context is known at submission and does not depend on observations.
    for auction in sorted(frame.auction.unique()):
        base[f'auction_{auction}'] = frame.auction.eq(auction).astype(float)
    views = {'weather':base,
        'tracker':pd.concat([base,physical.filter(regex='^front_')],axis=1),
        'bifacial':pd.concat([base,physical],axis=1)}
    train = (frame.eligible & frame.daylight & frame.split.eq('train')).to_numpy()
    validation = (frame.eligible & frame.daylight & frame.split.eq('validation')).to_numpy()
    test = (frame.eligible & frame.daylight & frame.split.eq('test')).to_numpy()
    if any(not mask.any() for mask in (train,validation,test)):
        raise ValueError('Every chronological split needs eligible daylight targets')
    if frame.loc[train,'target_available_at_utc'].max() >= frame.loc[validation,'decision_cutoff_utc'].min():
        raise ValueError('Training targets are not available before validation begins')
    if frame.loc[validation,'target_available_at_utc'].max() >= frame.loc[test,'decision_cutoff_utc'].min():
        raise ValueError('Validation targets are not available before test begins')
    output_root = config_path(config, section['output_root'])
    output = output_root / section['run_id'].replace('${farm_id}',str(farm_id))
    prepare_experiment_output(output_root, output,
        overwrite=config['experiment_outputs']['overwrite_existing'])
    for folder in ('json','csv','parquet','models','plots'):(output/folder).mkdir(parents=True,exist_ok=True)
    digest = hashlib.sha256((source/'parquet/corpus.parquet').read_bytes()).hexdigest()
    write = lambda name,value: (output/'json'/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
    write('resolved_config.json',config)
    write('provenance.json',{'corpus_sha256':digest,'source':str(source),'farm':farm.__dict__,
        'physical_settings':settings,'sklearn_version':sklearn.__version__,
        'test_used_for_selection':False,'unchanged_measured_targets':True,
        'geometry_status':'configured proxy location and tracker assumptions, not measured tracker telemetry'})
    prepared={};preprocessors={}
    for name in section['feature_sets']:
        pre = fit_preprocessor(views[name],train,config['pv_neural'])
        prepared[name] = transform_features(views[name],pre);preprocessors[name]=pre
    write('preprocessors.json',preprocessors)
    y = frame.measured_ac_mw.to_numpy()/farm.ac_capacity_mw
    weight=balanced_weights(frame.loc[train]);weight *= len(weight)
    predictions={name.removeprefix('prediction_'):prior[name].to_numpy() for name in prior if name.startswith('prediction_')}
    trials=[(feature,model) for feature in section['feature_sets'] for model in section['models']]

    def fit(trial):
        feature,spec=trial;started=time.perf_counter();options=dict(spec['parameters'])
        name=f"{spec['name']}_{feature}"
        classes={'extra_trees':ExtraTreesRegressor,'random_forest':RandomForestRegressor,'hist_gradient_boosting':HistGradientBoostingRegressor}
        if spec['family'] != 'hist_gradient_boosting':options['n_jobs']=1
        else: options['early_stopping']=False
        model=classes[spec['family']](random_state=section['seed'],**options)
        model.fit(prepared[feature][train],y[train],sample_weight=weight)
        prediction=np.clip(model.predict(prepared[feature]),0,1)*farm.ac_capacity_mw
        prediction[~frame.daylight.to_numpy(bool)]=0
        joblib.dump({'model':model,'preprocessor':preprocessors[feature], 'feature_set':feature,
            'farm_id':farm_id,'ac_rating_mw':farm.ac_capacity_mw,'physical_settings':settings,
            'corpus_sha256':digest,'train_last_target_utc':str(frame.loc[train,'target_available_at_utc'].max())},output/'models'/f'{name}.joblib')
        progress(f'Completed {name}: {time.perf_counter()-started:.1f}s')
        return name,prediction

    # Limit native BLAS/OpenMP pools globally; each tree job owns one CPU thread.
    workers=int(section['workers'])
    if not 1 <= workers <= 42:raise ValueError('pv_ensemble_study.workers must be between 1 and 42')
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        for name,prediction in pool.map(fit,trials):predictions[name]=prediction
    table=[]
    for name,prediction in predictions.items():
        for split,mask in [('validation',validation),('test',test),('january',validation & frame.delivery_day.str.startswith(section['focus_month']).to_numpy())]:
            table.append({'model':name,'split':split,**score(frame,prediction,farm.ac_capacity_mw,mask)})
    table=pd.DataFrame(table)
    candidates=[f"{m['name']}_{f}" for f,m in trials]
    selected=table.loc[table.split.eq('validation') & table.model.isin(candidates)].sort_values('nrmse_percent').iloc[0].model
    write('selection.json',{'selected_new_model':selected,'criterion':'validation balanced daylight nrmse','test_used_for_selection':False})
    table.to_csv(output/'csv/metrics.csv',index=False)
    saved=frame[keys+['split','eligible','daylight','measured_ac_mw']].copy()
    for name,prediction in predictions.items():saved[f'prediction_{name}_mw']=prediction
    saved.to_parquet(output/'parquet/predictions.parquet',index=False)
    _plot(table,output)
    write('manifest.json',{'status':'complete','farm_id':farm_id,'selected_new_model':selected,
        'corpus_sha256':digest,'metrics':table.to_dict('records')})
    return {'output':str(output),'selected_new_model':selected}


def _plot(table,output):
    """Compare the selected split without using its scores to choose candidates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(18,8),layout='constrained')
    order=table.loc[table.split.eq('validation')].sort_values('nrmse_percent').model.tolist()
    for ax,split in zip(axes,['validation','january','test']):
        rows=table[table.split.eq(split)].set_index('model').loc[order]
        ax.barh(rows.index,rows.nrmse_percent);ax.invert_yaxis();ax.set_title(split);ax.set_xlabel('Daylight RMSE / AC rating (%)');ax.grid(axis='x',alpha=.2)
    fig.savefig(output/'plots/model_comparison.png',dpi=150);plt.close(fig)

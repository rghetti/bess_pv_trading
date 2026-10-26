"""Portable, checksum-validated inference for the market-independent search models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from ..data import night_window_mask
from .pv_forecast_search import weather_features
from .pv_neural_data import transform_features,pack_sequences,unpack_predictions


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_bundle(farm_directory,model_name,destination):
    """Copy all prediction dependencies, keeping learned transforms frozen."""
    farm_directory=Path(farm_directory);destination=Path(destination)
    metadata=json.loads((farm_directory/'json/trials'/f'{model_name}.json').read_text())
    artifact=Path(metadata['artifact'])
    if _sha(artifact)!=metadata['artifact_sha256']:raise ValueError('Source model checksum mismatch')
    provenance=json.loads((farm_directory/'json/input_provenance.json').read_text())
    config=json.loads((farm_directory.parent/'json/resolved_config.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    model_path=destination/('model'+artifact.suffix);shutil.copy2(artifact,model_path)
    baseline_path=destination/'baseline.joblib';shutil.copy2(metadata['reference_regression'],baseline_path)
    manifest={'schema':'weather_pv_bundle_v1','farm_id':metadata['farm_id'],
        'rating_mw':provenance['rating_mw'],'specification':metadata['specification'],
        'preprocessor':metadata['features'],'night_policy':config['policy']['night'],
        'include_national_holiday':config['pv_forecast_search']['include_national_holiday'],
        'timezone':config['pv_neural']['delivery_timezone'],'model':model_path.name,
        'baseline':baseline_path.name,'training_identity':metadata['identity'],
        'train_last_target_utc':metadata['train_last_target_utc'],
        'corpus_sha256':provenance['corpus_sha256'],
        'feature_contract':'weather/time; no proxy geometry, auction identity or elapsed age',
        'software':{'sklearn':metadata['sklearn_version'],'torch':metadata.get('torch_version')},
        'checksums':{model_path.name:_sha(model_path),baseline_path.name:_sha(baseline_path)}}
    if metadata['specification']['family']=='neural':
        # Symmetric convolutions must retain the training tensor boundary,
        # including padding up to the longest DST delivery day in the corpus.
        cache=farm_directory/'cache'/f"packed_{metadata['specification']['features']}.npz"
        with np.load(cache,allow_pickle=False) as packed:
            manifest['sequence_length']=int(packed['indices'].shape[1])
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def predict_bundle(frame,bundle_directory,*,farm_id):
    """Predict AC MW from issue-time weather only; never read observed production.

    Sequence routing uses day/view identifiers only to prevent mixing forecast
    curves. An auction ID is never a feature, embedding, or calibration input.
    """
    bundle_directory=Path(bundle_directory)
    manifest=json.loads((bundle_directory/'manifest.json').read_text())
    if int(farm_id)!=manifest['farm_id']:raise ValueError('Forecast bundle belongs to another farm')
    for name,expected in manifest['checksums'].items():
        if _sha(bundle_directory/name)!=expected:raise ValueError(f'Forecast bundle checksum mismatch: {name}')
    spec=manifest['specification'];rating=manifest['rating_mw']
    features=weather_features(frame,spec['features'],manifest['timezone'],
        include_national_holiday=manifest.get('include_national_holiday',False))
    x=transform_features(features,manifest['preprocessor']).astype(np.float32)
    night=night_window_mask(pd.DatetimeIndex(frame.interval_start_utc),manifest['night_policy'])
    baseline_model=joblib.load(bundle_directory/manifest['baseline'])
    baseline=np.clip(baseline_model.predict(x),0,1);baseline[night]=0
    if spec['family']=='neural':
        import torch
        from .pv_residual_networks import WeatherResidualNetwork
        model=WeatherResidualNetwork(x.shape[-1],spec['parameters'])
        model.load_state_dict(torch.load(bundle_directory/manifest['model'],map_location='cpu',weights_only=True));model.eval()
        # Dummy target fields provide routing scaffolding; incoming measured values are ignored.
        inputs=frame.copy();inputs['measured_ac_mw']=np.nan;inputs['eligible']=False
        inputs['daylight']=~night;inputs['split']='prediction'
        if 'auction' not in inputs:inputs['auction']='DA'
        packed=pack_sequences(inputs,x,rating)
        packed=_pad_sequence_contract(packed,manifest['sequence_length'])
        valid=packed['indices']>=0;base=np.zeros(valid.shape,dtype=np.float32)
        base[valid]=baseline[packed['indices'][valid]]
        with torch.no_grad():
            values=model(torch.as_tensor(packed['x']),torch.as_tensor(base),
                torch.as_tensor(packed['daylight']),torch.as_tensor(valid)).numpy()
        prediction=unpack_predictions(packed,values,len(frame),rating)
    else:
        artifact=joblib.load(bundle_directory/manifest['model']);fraction=artifact['model'].predict(x)
        if spec['family'].endswith('_residual'):fraction+=baseline
        prediction=np.clip(fraction,0,1)*rating
    prediction[night]=0
    if not np.isfinite(prediction).all():raise ValueError('Bundle inference produced nonfinite AC power')
    return prediction


def _pad_sequence_contract(packed,sequence_length):
    """Preserve trained convolution boundaries for single-day inference."""
    padding=int(sequence_length)-packed['indices'].shape[1]
    if padding<0:raise ValueError('Forecast sequence exceeds the trained sequence length')
    if padding:
        packed=dict(packed)
        for name in ['x','y','mask','daylight','indices']:
            widths=[(0,0),(0,padding)]+[(0,0)]*(packed[name].ndim-2)
            packed[name]=np.pad(packed[name],widths,constant_values=-1 if name=='indices' else 0)
    return packed


def export_composite(definitions,name,destination,*,rating,timezone):
    """Export selected averages/corrections with their constituent prediction contracts."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    definition=definitions[name]
    if definition['kind']=='static':
        return export_bundle(definition['source'],definition['model'],destination)
    manifest={'schema':'weather_pv_composite_v1','name':name,'kind':definition['kind'],
        'rating_mw':rating,'timezone':timezone,'checksums':{}}
    if definition['kind']=='static_mean':
        members=[]
        for number,member in enumerate(definition['members']):
            child=f'member_{number}';export_composite(definitions,member,destination/child,rating=rating,timezone=timezone)
            members.append(child);manifest['checksums'][f'{child}/manifest.json']=_sha(destination/child/'manifest.json')
        manifest['members']=members
    elif definition['kind']=='online_gain':
        export_composite(definitions,definition['base'],destination/'base',rating=rating,timezone=timezone)
        shutil.copy2(definition['history'],destination/'history.parquet')
        manifest.update(window_days=definition['window_days'],minimum_days=definition['minimum_days'],
            gain_bounds=definition['gain_bounds'],base='base',history='history.parquet')
        manifest['checksums']={'base/manifest.json':_sha(destination/'base/manifest.json'),
            'history.parquet':_sha(destination/'history.parquet')}
        manifest['history_note']='Snapshot for reproducibility. Only rows observed by cutoff are read. Supply updated history for future operation.'
    else:raise ValueError('Cannot export a reference-only predictor as a new model')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def predict_composite(frame,bundle_directory,*,farm_id,history=None):
    """Target-free prediction; optional observed history updates an adaptive bundle."""
    bundle_directory=Path(bundle_directory)
    manifest=json.loads((bundle_directory/'manifest.json').read_text())
    if manifest['schema']=='weather_pv_bundle_v1':return predict_bundle(frame,bundle_directory,farm_id=farm_id)
    for name,expected in manifest['checksums'].items():
        if _sha(bundle_directory/name)!=expected:raise ValueError(f'Composite checksum mismatch: {name}')
    if manifest['kind']=='static_mean':
        return np.mean([predict_composite(frame,bundle_directory/child,farm_id=farm_id)
            for child in manifest['members']],axis=0)
    from .pv_online_bias import causal_gain_prediction
    base=predict_composite(frame,bundle_directory/manifest['base'],farm_id=farm_id)
    observed=pd.read_parquet(bundle_directory/manifest['history']) if history is None else history
    prediction,_=causal_gain_prediction(frame,base,observed,window_days=manifest['window_days'],
        minimum_days=manifest['minimum_days'],gain_bounds=manifest['gain_bounds'])
    return np.clip(prediction,0,manifest['rating_mw'])

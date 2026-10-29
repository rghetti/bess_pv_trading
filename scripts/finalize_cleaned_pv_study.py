"""Validate completed HPO, publish portable operational winners and result summaries."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from benchmark_satellite_ghi import sha, write
from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path
from bess_pv_trading.forecasting.data.pv_outages import apply_episode_exclusions
from bess_pv_trading.forecasting.inference.pv_forecast_bundle import predict_bundle
from bess_pv_trading.forecasting.training.pv_fitting import predict_fit
from bess_pv_trading.forecasting.studies.pv_ensemble_study import score
from bess_pv_trading.data import load_metadata


def run(config):
    """Check the sole CSV policy, freeze model provenance and copy operational winners."""
    section=config['pv_cleaned_hpo']
    root=config_path(config,section['output_root'])/section['run_id']
    manifest=json.loads((root/'json/manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('HPO is not complete')
    quality=config_path(config,config['pv_data_quality']['exclusions_csv'])
    identity=json.loads((root/'json/identity.json').read_text())
    if sha(quality)!=identity['exclusions_csv_sha256']:raise ValueError('Authoritative CSV changed during HPO')
    counts=[];checks=[];selected_rows=[];published=[];swap_scores=[]
    metrics=pd.read_csv(root/'csv/all_trial_metrics.csv')
    families=pd.read_csv(root/'csv/family_selections.csv')
    prepared=config_path(config,section['prepared_source_root'])
    farms=load_metadata(config_path(config,config['inputs']['metadata_parquet']),section['farm_ids'])
    for regime in section['input_regimes']:
        for farm in section['farm_ids']:
            folder=root/regime/f'farm_{farm}'
            source=prepared/regime/f'farm_{farm}'
            frame=pd.read_parquet(source/'parquet/corpus.parquet')
            marked=apply_episode_exclusions(frame,config,farm_id=farm)
            if marked.episode_excluded.any():raise ValueError('An excluded feature/target instance survived HPO')
            packed=np.load(source/'cache/packed_core.npz')
            ix=packed['indices'][packed['indices']>=0]
            assert len(ix)==len(frame) and len(np.unique(ix))==len(frame)
            checks.append({'farm_id':farm,'regime':regime,'modeling_rows':len(frame),
                'excluded_instances_retained':0,'neural_instances':len(ix)})
            for family,part in metrics.loc[metrics.regime.eq(regime)&metrics.farm_id.eq(farm)
                    &metrics.split.eq('validation')&metrics.coverage.eq('observed_only')].groupby('family'):
                count=part.model.nunique()
                if family not in {'ols','pvlib_scale'} and count<section['trials_per_family']:
                    raise ValueError(f'{farm}/{regime}/{family} has fewer than sixteen trials')
                counts.append({'farm_id':farm,'regime':regime,'family':family,'trials':count})
            selection=json.loads((folder/'json/selected.json').read_text())
            for record in families.loc[families.farm_id.eq(farm)&families.regime.eq(regime)].itertuples():
                bundle=folder/'bundles'/record.family
                path=bundle/('physical_parameters.json' if record.family.startswith('pvlib_') else 'manifest.json')
                content=json.loads(path.read_text())
                content.update(input_regime=regime,diagnostic_only=regime=='satellite',
                    exclusions_csv_sha256=sha(quality),excluded_feature_target_instances_removed=True,
                    training_period='2025',selection_period='2026 January-April',
                    report_period='2026 May-August reused retrospectively curated holdout')
                content['prepared_modeling_corpus_sha256']=sha(source/'parquet/corpus.parquet')
                content['night_policy']=config['policy']['night']
                content['farm_metadata']=asdict(farms[farm])
                write(path,content)
            row={'farm_id':farm,'regime':regime,'family':selection['family'],'model':selection['model']}
            for split in ['train','validation','test']:
                value=metrics.loc[metrics.farm_id.eq(farm)&metrics.regime.eq(regime)
                    &metrics.model.eq(selection['model'])&metrics.split.eq(split)&metrics.coverage.eq('observed_only')].iloc[0]
                for metric in ['nrmse_percent','nmae_percent','bias_percent','rmse_mw','mae_mw',
                        'absolute_error_min_percent','absolute_error_max_percent']:
                    row[f'{split}_{metric}']=value[metric]
            selected_rows.append(row)
            if regime=='forecast':
                destination=root/'selected_models'/f'farm_{farm}'
                destination.mkdir(parents=True,exist_ok=True)
                bundle=folder/'bundles'/selection['family']
                shutil.copytree(bundle,destination,dirs_exist_ok=True)
                write(destination/'selection.json',{**selection,'scores':row,
                    'study_root':str(root),'quality_csv':str(quality),'quality_csv_sha256':sha(quality)})
                published.append({'farm_id':farm,'family':selection['family'],'model':selection['model'],
                    'directory':str(destination),'input_regime':'forecast'})
                # Isolate input replacement without changing the selected model's fitted parameters.
                satellite=pd.read_parquet(prepared/'satellite'/f'farm_{farm}/parquet/corpus.parquet')
                keys=['interval_start_utc','delivery_day','auction','measured_ac_mw']
                if not frame[keys].equals(satellite[keys]):raise ValueError('Input-swap populations differ')
                if selection['family'].startswith('pvlib_'):
                    physical=json.loads((destination/'physical_parameters.json').read_text())
                    metadata=load_metadata(config_path(config,config['inputs']['metadata_parquet']),[farm])[farm]
                    evaluation=satellite.copy();evaluation['eligible']=evaluation.eligible&evaluation.daylight
                    swapped=predict_fit(evaluation,metadata,physical['settings'],config['policy']['night'],physical['fit'])
                else:
                    swapped=predict_bundle(satellite,destination,farm_id=farm)
                np.save(folder/'arrays/selected_forecast_model_satellite_input.npy',swapped)
                rating=config['farms'][str(farm)]['ac_capacity_kw']/1000
                for split in ['train','validation','test']:
                    mask=(frame.eligible&frame.daylight&frame.split.eq(split)&~frame.satellite_ghi_interpolated).to_numpy()
                    swap_scores.append({'farm_id':farm,'model':selection['model'],'family':selection['family'],
                        'split':split,'input':'satellite GHI with forecast-trained parameters unchanged',
                        **score(frame,swapped,rating,mask)})
    pd.DataFrame(counts).to_csv(root/'csv/completed_trials_by_family.csv',index=False)
    pd.DataFrame(selected_rows).to_csv(root/'csv/selected_model_summary.csv',index=False)
    pd.DataFrame(swap_scores).to_csv(root/'csv/selected_forecast_model_satellite_input.csv',index=False)
    # Describe cross-season family behavior without using test data to change selection.
    leaders=metrics.loc[metrics.split.eq('test')&metrics.coverage.eq('observed_only')].merge(
        families[['regime','farm_id','model','family']],on=['regime','farm_id','model','family'])
    leaders=leaders.sort_values('nrmse_percent').drop_duplicates(['regime','farm_id'])
    leaders['interpretation']='Descriptive test-period family leader; not used for model or hyperparameter selection'
    leaders.to_csv(root/'csv/descriptive_test_family_leaders.csv',index=False)
    write(root/'json/acceptance_audit.json',{'status':'passed','total_fits':sum(x['trials'] for x in counts),
        'checks':checks,'quality_csv':str(quality),'quality_csv_sha256':sha(quality),
        'single_csv_is_sole_quality_source':True,'published_operational_models':published})
    # Keep a compact review package beside the configurable study output.
    destination=root/'reports'
    destination.mkdir(parents=True,exist_ok=True)
    for name in ['selected_model_summary.csv','completed_trials_by_family.csv','paired_forecast_vs_satellite.csv']:
        shutil.copy2(root/'csv'/name,destination/name)
    shutil.copy2(root/'json/acceptance_audit.json',destination/'acceptance_audit.json')
    print(pd.DataFrame(selected_rows)[['farm_id','regime','family','validation_nrmse_percent','test_nrmse_percent']].to_string(index=False))
    return root


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    run(load_config(args.config))

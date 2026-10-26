"""Preserve minimal reusable timing-study inputs outside disposable experiments."""
from __future__ import annotations
import argparse
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


def stage(config):
    """Copy cleaned weather/AC pairs, five baseline prediction curves and small contracts."""
    section=config['pv_timing_ablation'];target=config_path(config,section['data_root'])
    manifest=target/'json/manifest.json'
    if manifest.exists():
        record=json.loads(manifest.read_text())
        for name,checksum in record['files'].items():
            if sha(target/name)!=checksum:raise ValueError('Staged timing input changed')
        if sha(config_path(config,config['pv_data_quality']['exclusions_csv']))!=record['quality_csv_sha256']:
            raise ValueError('Quality CSV changed; explicitly rebuild the staged modeling dataset')
        return target,record
    parent=config_path(config,section['parent_study_root'])
    prepared=config_path(config,section['prepared_source_root']);files={}
    for farm in section['farm_ids']:
        source=prepared/'forecast'/f'farm_{farm}';output=target/f'farm_{farm}'
        for sub in ['parquet','json','cache','references']:(output/sub).mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'parquet/corpus.parquet',output/'parquet/corpus.parquet')
        shutil.copy2(source/'cache/train.npy',output/'cache/train.npy')
        shutil.copy2(source/'json/input_provenance.json',output/'json/input_provenance.json')
        with np.load(source/'cache/packed_core.npz') as packed:
            length=int(packed['indices'].shape[1])
        folder=parent/'forecast'/f'farm_{farm}'
        ranking=pd.read_csv(folder/'csv/validation_ranking.csv')
        ranking=ranking.loc[ranking.family.isin(['direct_tcn','residual_tcn','residual_mlp'])]
        chosen=[ranking.iloc[0]]
        for family in section['extra_families_by_farm'].get(str(farm),[]):
            row=ranking.loc[ranking.family.eq(family)].iloc[0]
            if row.model not in [r.model for r in chosen]:chosen.append(row)
        specs=[]
        for row in chosen:
            spec=json.loads((folder/'json/trials'/f'{row.model}.json').read_text())['specification']
            specs.append(spec)
            shutil.copy2(folder/'arrays'/f'{row.model}.npy',output/'references'/f'{row.model}.npy')
        write(output/'json/candidates.json',{'specifications':specs,'sequence_length':length})
        for path in output.rglob('*'):
            if path.is_file():files[str(path.relative_to(target))]=sha(path)
    record={'status':'complete','files':files,
        'quality_csv_sha256':sha(config_path(config,config['pv_data_quality']['exclusions_csv'])),
        'parent_identity':json.loads((parent/'json/identity.json').read_text())['identity'],
        'purpose':'Reusable cleaned feature/AC-target instances and frozen neural comparison predictions',
        'not_needed_after_staging':'Old experiment directories, caches, trial models and plots'}
    write(manifest,record);return target,record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();print(stage(load_config(args.config))[0])

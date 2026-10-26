"""Rebuild the approved PV episode list from frozen review/diagnostic tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import pandas as pd
import numpy as np
from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path


def run(config):
    """Collect production-disconnect/plateau intervals; do not exclude weather disagreements."""
    section=config['pv_episode_curation'];paths=[]
    disconnect=config_path(config,section['disconnect_source_root'])/'csv/all_episodes.csv'
    paths.append(disconnect)
    table=pd.read_csv(disconnect)
    table=table.loc[table.kind.isin(section['disconnect_kinds']),['farm_id','start_utc','end_utc','kind']]
    frames=[table.rename(columns={'kind':'reason'})]
    for farm in section['farm_ids']:
        path=config_path(config,section['review_source_root'])/f'farm_{farm}/csv/candidate_episodes.csv'
        paths.append(path);review=pd.read_csv(path)
        frames.append(review.loc[review.issue.isin(section['review_issues']),
            ['farm_id','start_utc','end_utc','issue']].rename(columns={'issue':'reason'}))
        physical_path=config_path(config,section['review_source_root'])/f'farm_{farm}/parquet/physical_quarters.parquet'
        paths.append(physical_path)
        physical=pd.read_parquet(physical_path).sort_values('interval_start_utc')
        # Materialize all existing zero/plateau/review findings, including low-GHI plateaus.
        # Model readers subsequently need only the resulting CSV, never these flags/rules.
        for reason,column in section['source_flags'].items():
            times=pd.DatetimeIndex(physical.loc[physical[column],'interval_start_utc'])
            if not len(times):continue
            boundaries=np.flatnonzero(np.diff(times.as_unit('ns').asi8)!=
                pd.Timedelta(minutes=config['source']['historical_interval_minutes']).value)+1
            records=[{'farm_id':farm,'start_utc':str(group[0]),'end_utc':str(group[-1]),'reason':reason}
                for group in np.split(times,boundaries)]
            frames.append(pd.DataFrame(records))
    result=pd.concat(frames,ignore_index=True).drop_duplicates().sort_values(['farm_id','start_utc','reason'])
    result['decision']='exclude'
    destination=config_path(config,config['pv_data_quality']['exclusions_csv'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(destination,index=False)
    manifest={'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'rows':len(result),'interval_end':'inclusive UTC quarter start',
        'interpretation':'Retrospective exclusions explicitly requested by user; not confirmed faults.',
        'runtime_policy':'This CSV is the sole exclusion source; no additional runtime classification rules.',
        'configuration':section}
    destination.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    run(load_config(args.config))

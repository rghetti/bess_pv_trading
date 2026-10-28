#!/usr/bin/env python
"""Audit saved online PV forecasts and delayed feedback without rerunning them."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bess_pv_trading.forecasting.pv_probabilistic_data import file_hash, write_json
from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path


def audit(root, settings):
    root=Path(root); results=[]
    for farm in settings['farms']:
        base=root/f'farm_{farm}'; manifest=json.loads((base/'prepared_manifest.json').read_text())
        path=base/'prepared.parquet'
        if file_hash(path)!=manifest['prepared_sha256']: raise ValueError(f'Prepared checksum changed for farm {farm}')
        prepared=pd.read_parquet(path)
        for auction in settings['auctions']:
            folder=base/auction; marker=json.loads((folder/'manifest.json').read_text())
            for name, checksum in marker['files'].items():
                if file_hash(folder/f'{name}.parquet')!=checksum: raise ValueError(f'Checksum changed: {folder/name}')
            f=pd.read_parquet(folder/'forecasts.parquet'); u=pd.read_parquet(folder/'updates.parquet')
            if (f.deterministic_available_utc>f.decision_cutoff_utc).any(): raise ValueError('Deterministic vintage after gate')
            if (u.target_available_utc>u.update_processed_at_utc).any(): raise ValueError('Observation used before availability')
            if (u.forecast_decision_utc>=u.update_processed_at_utc).any(): raise ValueError('Feedback used at its issue')
            source=prepared.loc[prepared.auction.eq(auction),['interval_start_utc','retrospective_model']]
            u=u.merge(source,left_on='forecast_interval_utc',right_on='interval_start_utc',validate='many_to_one')
            if u.retrospective_model.any(): raise ValueError('Retrospective model used to update weights')
            for weight in f.weights.dropna():
                w=np.asarray(weight,float)
                if (w<0).any() or not np.isclose(w.sum(),1): raise ValueError('Invalid issued weight')
            for raw in f.det_quantile_corrected:
                q=np.asarray(raw,float)
                if np.isfinite(q).all() and ((np.diff(q)<0).any() or (q<0).any() or (q>manifest['rating_mw']).any()):
                    raise ValueError('Invalid deterministic quantiles')
            if auction=='IDA3':
                local=pd.DatetimeIndex(f.interval_start_utc).tz_convert('Europe/Rome')
                if (local.hour<12).any(): raise ValueError('IDA3 includes pre-noon products')
            prior=f.delivery_day.eq(f.delivery_day.min())
            results.append({'farm_id':farm,'auction':auction,'issued_forecast_rows':len(f),'feedback_updates':len(u),
                            'first_day_retrospective_rows':int(f.loc[prior,'retrospective_model'].sum()),'passed':True})
    output={'status':'passed','checks':results,'scope':'saved artifacts and timestamp invariants; quantile history logic covered by tests'}
    write_json(root/'audit.json',output)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args(); config=load_config(args.config); settings=config['pv_online_crps_study']
    path=config_path(config,settings['output_root'])/settings['run_id']
    print(json.dumps(audit(path,settings),indent=2))

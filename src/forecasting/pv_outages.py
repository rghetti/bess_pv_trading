"""Flag sustained zero AC production during measured daylight for PV studies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib
import json
from pathlib import Path
import yaml

from ..data import config_path
from ..reporting.artifacts import prepare_experiment_output


def sustained_daylight_zero_mask(times,power,ghi,*,minimum_intervals,
                                 minimum_ghi_w_m2,interval_minutes,
                                 daylight=None,day_keys=None,expand_full_daylight_run=False):
    """Anchor a contiguous zero-AC run in measured high GHI, then mask daylight."""
    if (isinstance(minimum_intervals,bool) or int(minimum_intervals)!=minimum_intervals
            or minimum_intervals<2):
        raise ValueError('minimum_intervals must be an integer of at least two')
    if interval_minutes<=0 or minimum_ghi_w_m2<0:
        raise ValueError('Outage cadence and GHI threshold must be nonnegative')
    index=pd.DatetimeIndex(times)
    if index.tz is None or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError('Outage detection needs sorted unique timezone-aware UTC times')
    if str(index.tz) not in {'UTC','datetime.timezone.utc'}:
        index=index.tz_convert('UTC')
    power=np.asarray(power,dtype=float);ghi=np.asarray(ghi,dtype=float)
    if len(index)!=len(power) or len(index)!=len(ghi):
        raise ValueError('Outage timestamps, measured AC and satellite GHI must align')
    if daylight is None:daylight=np.isfinite(ghi)&(ghi>=minimum_ghi_w_m2)
    daylight=np.asarray(daylight,dtype=bool)
    if len(daylight)!=len(index):raise ValueError('Outage daylight mask must align with timestamps')
    if day_keys is None:day_keys=index.date.astype(str)
    day_keys=np.asarray(day_keys)
    if len(day_keys)!=len(index):raise ValueError('Outage delivery-day keys must align with timestamps')
    zero=(power==0)
    anchor=np.isfinite(ghi)&(ghi>=minimum_ghi_w_m2)
    marked=np.zeros(len(index),dtype=bool)
    cadence=pd.Timedelta(minutes=interval_minutes)
    start=None
    def close_run(end):
        if (start is not None and end-start>=minimum_intervals
                and anchor[start:end].sum()>=minimum_intervals):
            marked[start:end]=(daylight[start:end] if expand_full_daylight_run
                else anchor[start:end])
    for position,active in enumerate(zero):
        continuous=(position>0 and index[position]-index[position-1]==cadence
            and day_keys[position]==day_keys[position-1])
        if active:
            if start is None or not continuous:
                close_run(position)
                start=position
        elif start is not None:
            close_run(position)
            start=None
    close_run(len(index))
    return marked


def apply_sustained_zero_exclusion(frame,config,*,farm_id=None):
    """Keep raw targets and rows, but exclude confirmed runs from model masks."""
    result=frame.copy()
    settings=config['pv_data_quality']['sustained_zero_outage']
    keys=['interval_start_utc','delivery_day','measured_ac_mw','satellite_ghi_w_m2','daylight']
    if not set(keys).issubset(result):raise ValueError('PV corpus lacks outage evidence columns')
    grouped=result.groupby('interval_start_utc',sort=True)
    for name in ['delivery_day','measured_ac_mw','satellite_ghi_w_m2','daylight']:
        if (grouped[name].nunique(dropna=False)>1).any():
            raise ValueError(f'Auction views disagree on physical-quarter {name}')
    unique=grouped[keys[1:]].first().reset_index()
    physical=sustained_daylight_zero_mask(unique.interval_start_utc,
        unique.measured_ac_mw,unique.satellite_ghi_w_m2,
        minimum_intervals=settings['minimum_consecutive_intervals'],
        minimum_ghi_w_m2=settings['minimum_satellite_ghi_w_m2'],
        interval_minutes=config['source']['historical_interval_minutes'],
        daylight=unique.daylight,day_keys=unique.delivery_day,
        expand_full_daylight_run=settings['expand_full_daylight_run'])
    lookup=pd.Series(physical,index=pd.DatetimeIndex(unique.interval_start_utc))
    result['outage_excluded']=lookup.reindex(pd.DatetimeIndex(result.interval_start_utc)).to_numpy()
    if 'review_excluded' not in result:result['review_excluded']=False
    review=config.get('review',{}).get('exclusions_csv')
    if review:
        if farm_id is None:raise ValueError('farm_id is required when applying manual PV review')
        decisions=pd.read_csv(config_path(config,review))
        required={'farm_id','start_utc','end_utc','decision'}
        if not required.issubset(decisions):
            raise ValueError('PV review CSV requires farm_id,start_utc,end_utc,decision')
        starts=pd.DatetimeIndex(result.interval_start_utc)
        excluded={str(value).lower() for value in config['review']['excluded_decisions']}
        for row in decisions.itertuples(index=False):
            if (str(row.farm_id) not in {str(farm_id),'portfolio'}
                    or str(row.decision).lower() not in excluded):continue
            lower=pd.Timestamp(row.start_utc);upper=pd.Timestamp(row.end_utc)
            if lower.tzinfo is None or upper.tzinfo is None or upper<lower:
                raise ValueError('Reviewed PV exclusion timestamps must be ordered and timezone-aware')
            selected=(starts>=lower)&(starts<=upper)
            result.loc[selected,'review_excluded']=True
        result['eligible']=result.eligible.to_numpy(dtype=bool)&~result.review_excluded.to_numpy(dtype=bool)
    if settings['enabled']:
        result['eligible']=result.eligible.to_numpy(dtype=bool)&~result.outage_excluded.to_numpy(dtype=bool)
    return result


def run_outage_curation(config,progress=print):
    """Materialize eligible modeling views and an audit index from frozen inputs."""
    section=config['pv_outage_export']
    source=config_path(config,section['source_root'])
    output_root=config_path(config,section['output_root']);output=output_root/section['run_id']
    if source.resolve().is_relative_to(output.resolve()):
        raise ValueError('Outage export source cannot reside inside its output')
    prepare_experiment_output(output_root,output,
        overwrite=config['experiment_outputs']['overwrite_existing'])
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    summaries=[]
    for farm in section['farm_ids']:
        path=source/f'farm_{farm}/parquet/corpus.parquet'
        if not path.exists():raise FileNotFoundError(path)
        original=pd.read_parquet(path)
        curated=apply_sustained_zero_exclusion(original,config,farm_id=farm)
        folder=output/f'farm_{farm}';folder.mkdir(parents=True,exist_ok=True)
        kept=curated.loc[curated.eligible].copy()
        kept.to_parquet(folder/'modeling_views.parquet',index=False)
        unique=curated.loc[curated.outage_excluded,
            ['interval_start_utc','delivery_day','measured_ac_mw',
             'satellite_ghi_w_m2']].drop_duplicates('interval_start_utc')
        unique.to_parquet(folder/'excluded_outage_intervals.parquet',index=False)
        reviewed=curated.loc[curated.review_excluded,
            ['interval_start_utc','delivery_day','measured_ac_mw',
             'satellite_ghi_w_m2']].drop_duplicates('interval_start_utc')
        reviewed.to_parquet(folder/'excluded_review_intervals.parquet',index=False)
        days=unique.groupby('delivery_day').size().rename('excluded_quarters').reset_index()
        days.to_csv(folder/'excluded_days.csv',index=False)
        summary={'farm_id':farm,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'raw_views':len(original),'modeling_views':len(kept),
            'excluded_outage_views':int(curated.outage_excluded.sum()),
            'excluded_physical_quarters':len(unique),'outage_days':len(days),
            'excluded_review_views':int(curated.review_excluded.sum()),
            'excluded_review_physical_quarters':len(reviewed),
            'raw_targets_preserved':True,'source_dataset_unchanged':True}
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        progress(f"Farm {farm}: excluded {len(unique)} physical quarters across {len(days)} days")
        summaries.append(summary)
    review=config.get('review',{}).get('exclusions_csv')
    manifest={'status':'complete','rule':config['pv_data_quality']['sustained_zero_outage'],
        'review_csv_sha256':hashlib.sha256(config_path(config,review).read_bytes()).hexdigest()
            if review else None,
        'summaries':summaries,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {'output':str(output),'summaries':summaries}

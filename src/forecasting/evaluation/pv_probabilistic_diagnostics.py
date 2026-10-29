"""Reliability and hourly diagnostics for saved paired PV predictions."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..data.pv_probabilistic_data import study_root
from ..plotting.pv_probabilistic_reporting import comparable
from ...data import config_path, load_metadata


def analyze(frame, ids, levels, farm_id, label, output, rating):
    """Report observed quantile frequencies with explicit zero/tie intervals."""
    qcols = [f'calibrated_q{round(100*q):02d}' for q in levels]
    valid = comparable(frame, ids) & frame.split.eq('validation').to_numpy()
    valid &= np.isfinite(frame[qcols].to_numpy(float)).all(axis=1)
    data = frame.loc[valid].copy()
    if data.empty: return [], []
    members = data[ids].to_numpy(float)
    target = data.measured_ac_mw.to_numpy(float)
    raw = np.quantile(members, levels, axis=1).T
    calibrated = data[qcols].to_numpy(float)
    records=[]
    for method, forecast in [('raw_ensemble', raw), ('calibrated_quantiles', calibrated)]:
        for j, level in enumerate(levels):
            strict = float(np.mean(target < forecast[:, j]))
            inclusive = float(np.mean(target <= forecast[:, j]))
            records.append({'farm_id':farm_id, 'model':label, 'method':method, 'quantile':level,
                'n':len(target), 'observed_strict_fraction':strict, 'observed_inclusive_fraction':inclusive,
                'ties_fraction':inclusive-strict})
    hours = pd.DatetimeIndex(data.interval_start_utc).tz_convert('Europe/Rome').hour.to_numpy()
    hourly=[]
    for hour in range(24):
        select = hours==hour
        if not select.any(): continue
        mean = members[select].mean(axis=1); median = np.median(members[select],axis=1)
        det = data.deterministic_prediction.to_numpy(float)[select]; actual=target[select]
        for method, prediction in [('deterministic',det),('member_mean',mean),('member_median',median)]:
            error=prediction-actual
            hourly.append({'farm_id':farm_id,'model':label,'local_hour':hour,'method':method,'n':int(select.sum()),
                           'mae_mw':float(np.mean(abs(error))),'rmse_mw':float(np.sqrt(np.mean(error**2))),
                           'bias_mw':float(np.mean(error)),
                           'nmae_percent_capacity':float(100*np.mean(abs(error))/rating),
                           'nrmse_percent_capacity':float(100*np.sqrt(np.mean(error**2))/rating),
                           'bias_percent_capacity':float(100*np.mean(error)/rating)})
    output.mkdir(parents=True, exist_ok=True)
    plot = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(7,5),layout='constrained')
    ax.plot([0,1],[0,1],color='black',linestyle='--',label='Nominal quantile')
    for method, color in [('raw_ensemble','#1b6ca8'),('calibrated_quantiles','#a43787')]:
        subset=plot.loc[plot.method.eq(method)]
        lo=subset.observed_strict_fraction.to_numpy(); hi=subset.observed_inclusive_fraction.to_numpy()
        mid=(lo+hi)/2
        ax.errorbar(subset['quantile'],mid,yerr=np.vstack([mid-lo,hi-mid]),marker='o',capsize=3,color=color,
                    label=f'{method}: strict–inclusive range')
    first, last = data.delivery_day.min(), data.delivery_day.max()
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Nominal quantile',ylabel='Observed fraction below forecast',
           title=f'Farm {farm_id} — {label} — {first} through {last}')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.savefig(output / f'{label}_reliability.png',dpi=140);plt.close(fig)
    return records,hourly


def summarize_diagnostics(config, include_winners=False):
    """Create reproducible validation reliability tables and comparison graphics."""
    root=study_root(config); levels=list(map(float,config['pv_probabilistic_study']['quantiles']))
    records=[]; hourly=[]
    for farm in config['pv_probabilistic_study']['farm_ids']:
        folder=root/f'farm_{farm}'; baseline=folder/'pvlib_paired_predictions.parquet'
        if not baseline.exists(): continue
        rating = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [int(farm)])[int(farm)].ac_capacity_mw
        items=[('pvlib',baseline)]
        ranking=folder/'validation_rankings.csv'
        if include_winners and ranking.exists():
            ranks=pd.read_csv(ranking)
            for row in ranks.loc[ranks['rank'].eq(1)].drop_duplicates('trial').itertuples():
                items.append((row.trial,Path(row.trial_path)/'predictions.parquet'))
        for label,path in items:
            data=pd.read_parquet(path)
            ids=sorted([x for x in data.columns if str(x).isdigit()],key=int)
            if len(ids)!=50: raise ValueError(f'{path}: expected 50 member columns')
            r,h=analyze(data,ids,levels,int(farm),label,folder/'plots'/'calibration',rating)
            records.extend(r);hourly.extend(h)
    pd.DataFrame(records).to_csv(root/'quantile_reliability.csv',index=False)
    pd.DataFrame(hourly).to_csv(root/'validation_error_by_local_hour.csv',index=False)
    section = config['pv_probabilistic_study']
    (root/'calibration_notes.md').write_text(
        f'# Calibration diagnostics\n\nReliability uses matched, eligible validation quarters after {section["train_end_day"]} '
        f'through {section.get("validation_end_day") or section["end_day"]}. '
        'For each quantile the lower endpoint is the fraction of outcomes strictly below it, '
        'and the upper endpoint includes ties. Ties matter at zero PV production. '
        'The small validation sample and validation-based model selection make these exploratory diagnostics. '
        'Percent error columns use each farm AC nameplate as denominator.\n')
    print(f'Calibration diagnostics saved under {root}',flush=True)

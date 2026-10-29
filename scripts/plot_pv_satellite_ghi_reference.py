"""Replay frozen PV models with observed GHI for retrospective diagnostic plots."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import torch

from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path, load_metadata
from bess_pv_trading.forecasting.inference.pv_neural_predict import predict_neural_frame
from bess_pv_trading.forecasting.models.pv_physical import PhysicalPVModel
from bess_pv_trading.forecasting.data.pv_satellite_inputs import interpolate_satellite, satellite_inputs
from bess_pv_trading.reporting.artifacts import prepare_experiment_output


def sha(path):
    """Identify the exact frozen input or model dependency used in this replay."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot_day(frame, episodes, day, farm, rating, path, timezone):
    """Separate neural and physical predictions, retaining quarter-hour boundaries."""
    sample=frame.loc[frame.delivery_day.eq(day)]
    times=pd.DatetimeIndex(sample.interval_start_utc).tz_convert(timezone)
    boundary=frame.loc[frame.interval_start_utc.eq(
        sample.interval_start_utc.iloc[-1]+pd.Timedelta(minutes=15))]
    # Connect the quarter-hour samples directly; append a real boundary point only.
    if not boundary.empty:
        sample=pd.concat([sample,boundary.iloc[:1]],ignore_index=True)
        times=pd.DatetimeIndex(sample.interval_start_utc).tz_convert(timezone)
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,layout='constrained')
    for axis,model in zip(axes[:2],['neural','pvlib']):
        for column,label,color,style in [
            ('measured_ac_mw','Measured AC','black','-'),
            (f'{model}_forecast_mw',f'{model}: forecast weather','tab:orange','--'),
            (f'{model}_satellite_ghi_mw',f'{model}: satellite-GHI input','tab:blue','-')]:
            values=sample[column].to_numpy()
            axis.plot(times,values,label=label,color=color,linestyle=style)
        axis.axhline(rating,color='gray',linestyle=':',label='Installed AC rating')
        filled=sample.satellite_ghi_interpolated.to_numpy(dtype=bool)
        if filled.any():
            axis.scatter(times[filled],sample.loc[filled,f'{model}_satellite_ghi_mw'],
                color='tab:purple',s=16,zorder=4,label='Interpolated GHI input')
        axis.set(ylabel='AC power (MW)',ylim=(0,rating*1.08))
    for column,label,color in [('satellite_ghi_input_w_m2','Satellite GHI (interior gaps interpolated)','black'),
        ('forecast_ghi','Forecast GHI (mean across auction views)','tab:orange')]:
        values=sample[column].to_numpy()
        axes[2].plot(times,values,label=label,color=color)
    filled=sample.satellite_ghi_interpolated.to_numpy(dtype=bool)
    if filled.any():
        axes[2].scatter(times[filled],sample.loc[filled,'satellite_ghi_input_w_m2'],
            color='tab:purple',s=18,zorder=4,label='Interpolated satellite GHI')
    axes[2].set(ylabel='GHI (W/m²)',xlabel=f'Delivery time ({timezone})')
    for row in episodes.loc[episodes.delivery_day.eq(day)].itertuples():
        start=pd.Timestamp(row.start_utc).tz_convert(timezone)
        end=pd.Timestamp(row.end_utc).tz_convert(timezone)+pd.Timedelta(minutes=15)
        for axis in axes:axis.axvspan(start,end,color='tab:red',alpha=.09)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(loc='upper right',fontsize=8)
    fig.suptitle(f'Farm {farm} · {day}\nFrozen models; observed GHI, forecast direct fraction and remaining weather')
    fig.savefig(path,dpi=140)
    plt.close(fig)


def run(config):
    """Regenerate candidate-day plots without fitting models or reclassifying episodes."""
    section=config['pv_satellite_ghi_reference']
    if section['plots_only']:
        return replot(config)
    torch.set_num_threads(int(section['cpu_threads']))
    root=config_path(config,section['output_root']);output=root/section['run_id']
    prepare_experiment_output(root,output,overwrite=config['experiment_outputs']['overwrite_existing'])
    (output/'json').mkdir(exist_ok=True)
    (output/'json/resolved_config.json').write_text(json.dumps(config,indent=2,default=str)+'\n')
    source=config_path(config,section['prediction_source_root'])
    scan=config_path(config,section['episode_source_root'])
    candidates=pd.read_csv(scan/'csv/all_episodes.csv')
    farms=load_metadata(config_path(config,config['inputs']['metadata_parquet']),section['farm_ids'])
    reports=[]
    for farm_id in section['farm_ids']:
        directory=source/f'farm_{farm_id}'
        prediction_path=directory/'parquet/predictions.parquet'
        provenance_path=directory/'json/input_provenance.json'
        selection_path=directory/'json/neural_selection.json'
        selection=json.loads(selection_path.read_text())['selected']['candidate']['name']
        artifact=directory/'models/trials'/f'{selection}.pt'
        provenance=json.loads(provenance_path.read_text())
        settings=deepcopy(provenance['settings']);settings['parameter_registry']=None
        frame=pd.read_parquet(prediction_path)
        # Replay the exact trained neural and physical models before changing GHI.
        metadata=json.loads(artifact.with_suffix('.json').read_text())
        night=metadata['night_policy'] if 'night_policy' in metadata else config['policy']['night']
        original=predict_neural_frame(frame,artifact,farm_id=farm_id)
        neural_difference=float(np.max(np.abs(original-frame.prediction_selected_mw.to_numpy())))
        physical=frame.set_index('interval_start_utc',drop=False)
        baseline=PhysicalPVModel(physical,farms[farm_id],settings,night).evaluate()
        baseline_difference=float(np.nanmax(np.abs(baseline-frame.prediction_pvlib_mw.to_numpy())))
        tolerance=float(section['replay_tolerance_mw'])
        if neural_difference>tolerance or baseline_difference>tolerance:
            raise ValueError(f'Farm {farm_id}: saved forecast replay mismatch: {neural_difference}, {baseline_difference}')
        frame=interpolate_satellite(frame,section['satellite_ghi_gap_fill'])
        modified=satellite_inputs(frame,settings)
        neural=predict_neural_frame(modified,artifact,farm_id=farm_id)
        neural[~modified.satellite_ghi_valid.to_numpy()]=np.nan
        pv=PhysicalPVModel(modified.set_index('interval_start_utc',drop=False),farms[farm_id],settings,night).evaluate()
        frame['neural_satellite_ghi_mw']=neural
        frame['pvlib_satellite_ghi_mw']=pv
        frame['satellite_ghi_valid']=modified.satellite_ghi_valid
        frame['direct_fraction_erbs_fallback']=modified.direct_fraction_erbs_fallback
        folder=output/f'farm_{farm_id}'
        for name in ['parquet','plots','json']:(folder/name).mkdir(parents=True,exist_ok=True)
        frame.to_parquet(folder/'parquet/auction_view_predictions.parquet',index=False)
        # Average complete model predictions across auction views only after inference.
        compact=frame.groupby('interval_start_utc',sort=True).agg(
            delivery_day=('delivery_day','first'),measured_ac_mw=('measured_ac_mw','first'),
            satellite_ghi_w_m2=('satellite_ghi_w_m2','first'),forecast_ghi=('forecast_ghi','mean'),
            satellite_ghi_input_w_m2=('satellite_ghi_input_w_m2','first'),
            satellite_ghi_interpolated=('satellite_ghi_interpolated','first'),
            neural_forecast_mw=('prediction_selected_mw','mean'),
            pvlib_forecast_mw=('prediction_pvlib_mw','mean'),
            neural_satellite_ghi_mw=('neural_satellite_ghi_mw','mean'),
            pvlib_satellite_ghi_mw=('pvlib_satellite_ghi_mw','mean')).reset_index()
        compact.to_parquet(folder/'parquet/physical_reference.parquet',index=False)
        episodes=candidates.loc[candidates.farm_id.eq(farm_id)]
        top=episodes.sort_values('indicative_gap_mwh',ascending=False).drop_duplicates('delivery_day')
        days=list(top.head(int(section['plot_top_per_farm'])).delivery_day)
        focus=section['focus_days_by_farm'].get(str(farm_id),section['focus_days_by_farm'].get(farm_id,[]))
        for day in dict.fromkeys(days+focus):
            plot_day(compact,episodes,day,farm_id,farms[farm_id].ac_capacity_mw,
                folder/'plots'/f'{day}.png',section['timezone'])
        report={'farm_id':farm_id,'neural_model':selection,'neural_artifact':str(artifact),
            'neural_weights_sha256':sha(artifact),'neural_contract_sha256':sha(artifact.with_suffix('.json')),
            'source_predictions_sha256':sha(prediction_path),'source_provenance_sha256':sha(provenance_path),
            'neural_original_replay_max_error_mw':neural_difference,
            'pvlib_original_replay_max_error_mw':baseline_difference,
            'missing_satellite_view_rows':int((~modified.satellite_ghi_valid).sum()),
            'satellite_ghi_interpolated_quarters':int(compact.satellite_ghi_interpolated.sum()),
            'satellite_ghi_interpolated_view_rows':int(frame.satellite_ghi_interpolated.sum()),
            'erbs_fallback_view_rows':int(modified.direct_fraction_erbs_fallback.sum()),
            'plotted_days':list(dict.fromkeys(days+focus))}
        (folder/'json/provenance.json').write_text(json.dumps(report,indent=2)+'\n')
        reports.append(report)
        print(f'Farm {farm_id}: saved satellite-GHI references and {len(report["plotted_days"])} plots',flush=True)
    manifest={'status':'complete','farms':reports,'models_retrained':False,
        'radiation':'Observed GHI with forecast BHI/GHI fraction; Erbs where that fraction is undefined.',
        'satellite_ghi_gap_fill':section['satellite_ghi_gap_fill'],
        'remaining_inputs':'Forecast temperature, wind, cloud and other weather; frozen geometry and parameters.',
        'plot_aggregation':'Arithmetic mean of per-auction predictions; not an operational forecast vintage.',
        'candidate_windows':'Original statistical-screen windows retained solely for visual context; not reclassified.',
        'episode_source_sha256':sha(scan/'csv/all_episodes.csv'),
        'code_sha256':sha(__file__),'raw_data_modified':False}
    (output/'json/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output


def replot(config):
    """Render saved numerical results again without changing inference or its provenance."""
    section=config['pv_satellite_ghi_reference']
    output=config_path(config,section['output_root'])/section['run_id']
    manifest=json.loads((output/'json/manifest.json').read_text())
    if manifest['status']!='complete':
        raise ValueError('Plot-only rendering requires a completed inference run')
    if manifest.get('satellite_ghi_gap_fill','none')!=section['satellite_ghi_gap_fill']:
        raise ValueError('Satellite interpolation changed; rerun inference before plotting')
    episode_path=config_path(config,section['episode_source_root'])/'csv/all_episodes.csv'
    if sha(episode_path)!=manifest['episode_source_sha256']:
        raise ValueError('Candidate episode inputs changed; rerun inference explicitly')
    episodes=pd.read_csv(episode_path)
    for report in manifest['farms']:
        farm=report['farm_id']
        if farm not in section['farm_ids']:continue
        metadata=json.loads(Path(report['neural_artifact']).with_suffix('.json').read_text())
        rating=metadata['farm_metadata']['ac_capacity_kw']/1000
        folder=output/f'farm_{farm}'
        frame=pd.read_parquet(folder/'parquet/physical_reference.parquet')
        selected=episodes.loc[episodes.farm_id.eq(farm)]
        for day in report['plotted_days']:
            plot_day(frame,selected,day,farm,rating,folder/'plots'/f'{day}.png',section['timezone'])
        print(f'Farm {farm}: regenerated {len(report["plotted_days"])} connected-line plots',flush=True)
    (output/'json/plot_rendering.json').write_text(json.dumps({
        'style':'connected lines between quarter-hour samples',
        'boundary':'Next interval sample included where available; no synthetic endpoint.',
        'code_sha256':sha(__file__),'config':section},indent=2,default=str)+'\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--plots-only',action='store_true',help='Redraw completed saved results without inference')
    args=parser.parse_args()
    configuration=load_config(args.config)
    if args.plots_only:configuration['pv_satellite_ghi_reference']['plots_only']=True
    print(run(configuration))

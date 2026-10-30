"""Rank retrained PV errors and inspect irradiance/production discrepancies; never curate."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmark_satellite_ghi import sha, write
from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path,load_metadata,open_store
from bess_pv_trading.forecasting.training.pv_fitting import balanced_weights
from bess_pv_trading.forecasting.inference.pv_forecast_bundle import predict_bundle
from bess_pv_trading.forecasting.training.pv_forecast_search import weather_features,interpolate_forecast_gaps
from bess_pv_trading.forecasting.data.pv_neural_data import transform_features
from bess_pv_trading.forecasting.models.pv_physical import PhysicalPVModel
from bess_pv_trading.forecasting.data.pv_study_inputs import farm_settings
from bess_pv_trading.forecasting.data.pv_satellite_inputs import satellite_inputs
from bess_pv_trading.forecasting.data.pv_outages import apply_episode_exclusions
from bess_pv_trading.reporting.artifacts import prepare_experiment_output


def episodes(frame,mask,reason,minimum):
    """Summarize contiguous candidate quarters separately by day and forecast vintage."""
    rows=[]
    # Summarize contiguous candidate quarters separately by day and forecast vintage for each `((_, auction), positions)` in `frame.groupby(['delivery_day', 'auction'], sort=False).indices.items()`.
    for (_,auction),positions in frame.groupby(['delivery_day','auction'],sort=False).indices.items():
        selected=np.asarray(positions)[np.asarray(mask)[positions]]
        # Skip the current item when `not len(selected)`.
        if not len(selected):continue
        times=pd.DatetimeIndex(frame.iloc[selected].interval_start_utc)
        cuts=np.flatnonzero(np.diff(times.as_unit('ns').asi8)!=pd.Timedelta(minutes=15).value)+1
        # Summarize contiguous candidate quarters separately by day and forecast vintage for each `group` in `np.split(selected, cuts)`.
        for group in np.split(selected,cuts):
            # Skip the current item when `len(group) < minimum`.
            if len(group)<minimum:continue
            p=frame.iloc[group]
            rows.append({'farm_id':int(frame.attrs['farm_id']),'reason':reason,'status':'review_candidate_only',
                'delivery_day':p.delivery_day.iloc[0],'auction':auction,'start_utc':str(p.interval_start_utc.iloc[0]),
                'end_utc':str(p.interval_start_utc.iloc[-1]),'quarters':len(p),'duration_hours':len(p)*.25,
                'actual_ac_mw':float(p.measured_ac_mw.mean()),'neural_satellite_mw':float(p.neural_satellite_mw.mean()),
                'pvlib_satellite_mw':float(p.pvlib_satellite_mw.mean()),
                'satellite_ghi_w_m2':float(p.satellite_ghi_input_w_m2.mean()),
                'forecast_ghi_mae_w_m2':float(p.ghi_error.abs().mean()),
                'max_input_change_sigma':float(p.weather_delta_sigma.max()),
                'max_ac_jump_percent':float(p.ac_delta_percent.abs().max())})
    return rows


def plot_profile(frame,raw,profile,folder,rating,config,pre,feature_columns):
    """Compare one exact forecast vintage with raw evidence, plus all weather-input changes."""
    day=profile['delivery_day'];auction=profile['auction']
    sample=frame.loc[frame.delivery_day.eq(day)&frame.auction.eq(auction)]
    start=pd.Timestamp(day).tz_localize(config['pv_neural']['delivery_timezone'])
    grid=pd.date_range(start,start+pd.DateOffset(days=1),freq='15min')
    utc=grid.tz_convert('UTC');s=sample.set_index('interval_start_utc').reindex(utc)
    r=raw.reindex(utc);base=folder/'plots'/f'{day}_{auction}'
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,layout='constrained')
    # Compare one exact forecast vintage with raw evidence, plus all weather-input changes for each `axis` in `axes[:2]`.
    for axis in axes[:2]:
        axis.plot(grid,r.measured_ac_mw,color='black',label='Measured AC (raw review context)')
        axis.axhline(rating,color='gray',linestyle=':',label='AC rating')
        axis.set_ylabel('AC MW')
    axes[0].plot(grid,s.neural_mw,color='tab:blue',label='Retrained neural, forecast GHI')
    axes[0].plot(grid,s.neural_satellite_mw,color='tab:green',linestyle='--',label='Same neural, satellite GHI')
    axes[0].plot(grid,s.without_timing_mw,color='tab:orange',alpha=.65,label='Previous neural without timing')
    axes[1].plot(grid,s.pvlib_forecast_mw,color='tab:orange',label='Configured pvlib, forecast GHI')
    axes[1].plot(grid,s.pvlib_satellite_mw,color='tab:green',linestyle='--',label='Configured pvlib, satellite GHI')
    axes[2].plot(grid,r.satellite_ghi_w_m2,color='black',alpha=.45,label='Raw satellite GHI')
    axes[2].plot(grid,s.satellite_ghi_input_w_m2,color='tab:green',label='Satellite GHI used by model (gaps interpolated)')
    axes[2].plot(grid,s.forecast_ghi,color='tab:orange',label=f'{auction} forecast GHI')
    filled=s.satellite_ghi_interpolated.fillna(False).to_numpy(dtype=bool)
    axes[2].scatter(grid[filled],s.loc[filled,'satellite_ghi_input_w_m2'],color='purple',s=12,label='Interpolated GHI')
    axes[2].set(ylabel='GHI W/m²',xlabel=config['pv_neural']['delivery_timezone'])
    excluded=r.episode_excluded.fillna(False).to_numpy(dtype=bool)
    # Compare one exact forecast vintage with raw evidence, plus all weather-input changes for each `axis` in `axes`.
    for axis in axes:
        axis.fill_between(grid,0,1,where=excluded,transform=axis.get_xaxis_transform(),color='gray',alpha=.12,
            label='CSV-excluded: context only, no predictions or scoring')
        axis.grid(alpha=.2);axis.legend(fontsize=7,loc='upper right')
    fig.suptitle(f'Farm {frame.attrs["farm_id"]} · {day} · {auction} · {frame.attrs["model"]}\n'
        f'Cycle UTC {sample.forecast_cycle_utc.iloc[0]}; lead {sample.lead_hours.min():.2f}–{sample.lead_hours.max():.2f} h')
    fig.savefig(str(base)+'.png',dpi=140);plt.close(fig)
    # Signed changes of every actual meteorological input, in training-standard-deviation units.
    delta=s[[f'delta_{name}' for name in feature_columns]].to_numpy().T
    delta=np.vstack([delta,s.satellite_ghi_input_w_m2.diff().to_numpy()/100,
        r.measured_ac_mw.diff().to_numpy()/(rating*.1)])
    labels=feature_columns+['Observed GHI change / 100 Wm-2','Observed AC change / 10% rating']
    fig,ax=plt.subplots(figsize=(13,max(4,len(labels)*.34)),layout='constrained')
    im=ax.imshow(delta,aspect='auto',vmin=-3,vmax=3,cmap='RdBu_r',interpolation='nearest')
    ax.set_yticks(range(len(labels)),labels,fontsize=8)
    ticks=np.linspace(0,len(grid)-1,9,dtype=int);ax.set_xticks(ticks,[grid[i].strftime('%H:%M') for i in ticks])
    ax.set(title=f'Farm {frame.attrs["farm_id"]}, {day}, {auction}: input changes versus observed AC/GHI',
        xlabel='Italian delivery time; gray/blank columns have no retained modeling instance')
    fig.colorbar(im,ax=ax,label='Signed change (weather: training SD; last two rows: stated scales)')
    fig.savefig(str(base)+'_features.png',dpi=140);plt.close(fig)


def farm_job(config,farm,root):
    """Diagnose fixed model predictions and observed-GHI substitutions without fitting."""
    import sklearn  # Load the environment's C++ runtime before optional PyTorch.
    import torch
    torch.set_num_threads(1)
    section=config['pv_retrained_error_diagnostics'];study=config_path(config,section['study_root'])
    data=config_path(config,config['pv_timing_ablation']['data_root'])
    frame=pd.read_parquet(data/f'farm_{farm}/parquet/corpus.parquet')
    published=config_path(config,section['models_root'])/f'farm_{farm}'/section['model_folder']
    # When `(published / 'selection.json').exists()`, set `selection` using `json.loads((published / 'selection.json').read_text())`.
    if (published/'selection.json').exists():
        selection=json.loads((published/'selection.json').read_text())
        row=pd.Series({'family':selection['primary_family'],'model':selection['model']})
        bundle=published/row.family
        prediction=predict_bundle(frame,bundle,farm_id=farm)
        previous=np.load(data/f'farm_{farm}/references'/f'{selection["baseline_model"]}.npy')
    else:
        saved=pd.read_parquet(study/f'farm_{farm}/parquet/predictions.parquet')
        keys=['interval_start_utc','delivery_day','auction','measured_ac_mw']
        # Enforce the input contract: Diagnostic instance population changed.
        if not frame[keys].equals(saved[keys]):raise ValueError('Diagnostic instance population changed')
        table=pd.read_csv(study/'csv/paired_and_tuned_comparison.csv')
        row=table.loc[table.farm_id.eq(farm)&table.split.eq('validation')&table.variant.eq('tuned_timing')].sort_values('nrmse_percent').iloc[0]
        bundle=study/f'farm_{farm}/bundles'/row.family
        prediction=saved[f'prediction_{row.family}_tuned_timing_mw'].to_numpy()
        previous=saved[f'prediction_{row.family}_without_timing_mw'].to_numpy()
    manifest=json.loads((bundle/'manifest.json').read_text())
    metadata=load_metadata(config_path(config,config['inputs']['metadata_parquet']),[farm])[farm]
    settings=farm_settings(config,metadata);rating=metadata.ac_capacity_mw
    changed=satellite_inputs(frame,settings)
    frame['neural_mw']=prediction
    frame['without_timing_mw']=previous
    frame['neural_satellite_mw']=predict_bundle(changed,bundle,farm_id=farm)
    frame.loc[~changed.satellite_ghi_valid,'neural_satellite_mw']=np.nan
    frame['pvlib_forecast_mw']=PhysicalPVModel(frame.set_index('interval_start_utc'),metadata,settings,config['policy']['night']).evaluate()
    frame['pvlib_satellite_mw']=PhysicalPVModel(changed.set_index('interval_start_utc'),metadata,settings,config['policy']['night']).evaluate()
    frame['ghi_error']=frame.forecast_ghi-frame.satellite_ghi_input_w_m2
    # Use exactly the weather features received by the selected model, with its frozen transform.
    interpolated,_=interpolate_forecast_gaps(frame,manifest['missing_weather_interpolation'])
    features=weather_features(interpolated,manifest['specification']['features'],manifest['timezone'],
        include_national_holiday=manifest['include_national_holiday']).loc[:,manifest['preprocessor']['columns']]
    transformed=transform_features(features,manifest['preprocessor'])
    deterministic=['hour_sin','hour_cos','annual_sin','annual_cos','month_sin','month_cos',
        'italian_national_holiday','lead_hours','cycle_hour_sin','cycle_hour_cos']
    weather=[name for name in features if name not in deterministic]
    frame['weather_delta_sigma']=np.nan;frame['ac_delta_percent']=np.nan;frame['ghi_delta']=np.nan
    # Diagnose fixed model predictions and observed-GHI substitutions without fitting for each `name` in `weather`.
    for name in weather:frame[f'delta_{name}']=np.nan
    # Diagnose fixed model predictions and observed-GHI substitutions without fitting for each `(_, positions)` in `frame.groupby(['delivery_day', 'auction'], sort=False).indices.items()`.
    for _,positions in frame.groupby(['delivery_day','auction'],sort=False).indices.items():
        positions=np.asarray(positions);times=pd.DatetimeIndex(frame.iloc[positions].interval_start_utc)
        adjacent=np.r_[False,np.diff(times.as_unit('ns').asi8)==pd.Timedelta(minutes=15).value]
        changes=np.diff(transformed[positions],axis=0,prepend=np.full((1,transformed.shape[1]),np.nan))
        # Diagnose fixed model predictions and observed-GHI substitutions without fitting for each `name` in `weather`.
        for name in weather:frame.loc[positions,f'delta_{name}']=np.where(adjacent,changes[:,features.columns.get_loc(name)],np.nan)
        frame.loc[positions,'weather_delta_sigma']=np.max(np.abs(changes[:,[features.columns.get_loc(n) for n in weather]]),axis=1)
        # Process each `(source, target, scale)` in this collection.
        for source,target,scale in [('measured_ac_mw','ac_delta_percent',100/rating),('satellite_ghi_input_w_m2','ghi_delta',1)]:
            value=frame.iloc[positions][source].to_numpy();difference=np.diff(value,prepend=np.nan)*scale
            frame.loc[positions,target]=np.where(adjacent,difference,np.nan)
    frame.attrs.update(farm_id=farm,model=f'{row.family}/{row.model}')
    threshold=section['weather_ghi_disagreement_w_m2'];ghi=frame.satellite_ghi_input_w_m2
    low=(ghi>=section['minimum_ghi_w_m2'])&(frame.ghi_error.abs()<=threshold)
    low&=(frame.measured_ac_mw<=np.minimum(frame.neural_satellite_mw,frame.pvlib_satellite_mw)*section['low_power_ratio'])
    low&=(np.minimum(frame.neural_satellite_mw,frame.pvlib_satellite_mw)-frame.measured_ac_mw)>=rating*section['gap_fraction_of_rating']
    jump=(ghi>=section['minimum_ghi_w_m2'])&(frame.ac_delta_percent.abs()>=section['power_jump_percent'])
    jump&=(frame.ghi_delta.abs()<=section['stable_ghi_change_w_m2'])&(frame.weather_delta_sigma<=section['stable_weather_delta_sigma'])
    jump&=~frame.satellite_ghi_interpolated
    frame['low_power_candidate']=low;frame['stable_weather_jump_candidate']=jump
    candidate=episodes(frame,low,'persistent_low_power_despite_both_satellite_input_models',section['minimum_episode_quarters'])
    candidate+=episodes(frame,jump,'ac_jump_with_stable_available_weather',1)
    profiles=[];contributions=[]
    frame['score_eligible']=frame.eligible&frame.daylight&~frame.satellite_ghi_interpolated
    frame['squared_error']=(frame.neural_mw-frame.measured_ac_mw)**2/rating**2
    # Diagnose fixed model predictions and observed-GHI substitutions without fitting for each `split` in `section['splits']`.
    for split in section['splits']:
        part=frame.loc[frame.score_eligible&frame.split.eq(split)].copy()
        weights=balanced_weights(part);error=part.squared_error.to_numpy();total=float(weights@error)
        # Process each `(label, mask)` in this collection.
        for label,mask in [('ghi_disagreement',part.ghi_error.abs().ge(threshold)),
            ('ghi_agreement',part.ghi_error.abs().lt(threshold)),('stable_weather_jump',part.stable_weather_jump_candidate),
            ('low_power_candidate',part.low_power_candidate)]:
            mask=mask.to_numpy()
            contributions.append({'farm_id':farm,'split':split,'group':label,'view_rows':int(mask.sum()),
                'unique_quarters':part.loc[mask,'interval_start_utc'].nunique(),'weighted_row_fraction':float(weights@mask),
                'weighted_squared_error_share':float(weights@(error*mask)/total),
                'overall_nrmse_percent':float(np.sqrt(total)*100)})
        part['weighted_squared_error']=weights*error
        # Diagnose fixed model predictions and observed-GHI substitutions without fitting for each `((day, auction), group)` in `part.groupby(['delivery_day', 'auction'])`.
        for (day,auction),group in part.groupby(['delivery_day','auction']):
            err=group.neural_mw-group.measured_ac_mw;oracle=group.neural_satellite_mw-group.measured_ac_mw
            profiles.append({'farm_id':farm,'split':split,'delivery_day':day,'auction':auction,
                'model':frame.attrs['model'],'cycle_utc':str(group.forecast_cycle_utc.iloc[0]),
                'lead_hours_mean':float(group.lead_hours.mean()),'quarters':len(group),
                'nrmse_percent':float(np.sqrt(np.mean(err**2))/rating*100),
                'satellite_input_nrmse_percent':float(np.sqrt(np.mean(oracle**2))/rating*100),
                'bias_percent':float(err.mean()/rating*100),'ghi_mae_w_m2':float(group.ghi_error.abs().mean()),
                'squared_error_share':float(group.weighted_squared_error.sum()/total),
                'stable_weather_jump_quarters':int(group.stable_weather_jump_candidate.sum()),
                'low_power_candidate_quarters':int(group.low_power_candidate.sum())})
    folder=root/f'farm_{farm}'
    # Diagnose fixed model predictions and observed-GHI substitutions without fitting for each `sub` in `['csv', 'json', 'plots', 'parquet']`.
    for sub in ['csv','json','plots','parquet']:(folder/sub).mkdir(parents=True,exist_ok=True)
    ranking=pd.DataFrame(profiles).sort_values('nrmse_percent',ascending=False)
    ranking.to_csv(folder/'csv/profile_ranking.csv',index=False)
    frame.to_parquet(folder/'parquet/diagnostic_instances.parquet',index=False)
    with open_store(config) as store:
        history=store[['power','ghi']].sel(farm_id=farm).load(scheduler='synchronous')
    times=pd.DatetimeIndex(pd.to_datetime(history.time.values,utc=True))
    raw=pd.DataFrame({'interval_start_utc':times,'measured_ac_mw':history.power.values*config['forecast']['realized_power_scale_to_mw'],
        'satellite_ghi_w_m2':history.ghi.values,'eligible':True})
    raw=apply_episode_exclusions(raw,config,farm_id=farm).set_index('interval_start_utc')
    chosen=[]
    # Diagnose fixed model predictions and observed-GHI substitutions without fitting for each `item` in `section['focus_profiles_by_farm'].get(str(farm), [])`.
    for item in section['focus_profiles_by_farm'].get(str(farm),[]):
        chosen+=ranking.loc[ranking.delivery_day.eq(item['delivery_day'])&ranking.auction.eq(item['auction'])].to_dict('records')
    # Diagnose fixed model predictions and observed-GHI substitutions without fitting for each `split` in `section['splits']`.
    for split in section['splits']:
        part=ranking.loc[ranking.split.eq(split)]
        chosen+=part.drop_duplicates('delivery_day').head(section['worst_profiles_per_split']).to_dict('records')
    # Process each `item` in this collection.
    for item in sorted(candidate,key=lambda r:r['max_ac_jump_percent'],reverse=True)[:section['candidate_plots_per_farm']]:
        match=ranking.loc[ranking.delivery_day.eq(item['delivery_day'])&ranking.auction.eq(item['auction'])]
        # When `len(match)`, set `chosen` using `match.head(1).to_dict('records')`.
        if len(match):chosen+=match.head(1).to_dict('records')
    seen=set()
    # Diagnose fixed model predictions and observed-GHI substitutions without fitting for each `profile` in `chosen`.
    for profile in chosen:
        key=(profile['delivery_day'],profile['auction'])
        # Skip the current item when `key in seen`.
        if key in seen:continue
        seen.add(key);plot_profile(frame,raw,profile,folder,rating,config,manifest['preprocessor'],weather)
    write(folder/'json/provenance.json',{'family':row.family,'model':row.model,'bundle_manifest_sha256':sha(bundle/'manifest.json'),
        'model_inputs_sha256':sha(data/f'farm_{farm}/parquet/corpus.parquet'),'candidate_flags_do_not_change_csv':True,
        'reference':'Same trained neural and configured pvlib with observed GHI; other weather remains forecast',
        'plotted_profiles':[list(x) for x in seen]})
    print(f'Farm {farm}: ranked {len(ranking)} profiles, {len(candidate)} candidate view-episodes, {len(seen)} profile/feature plot pairs',flush=True)
    return profiles,contributions,candidate


def run(config):
    """Run four independent post-fit diagnostic audits using local data only."""
    section=config['pv_retrained_error_diagnostics'];root=config_path(config,section['output_root'])/section['run_id']
    # Return `render_cached_profiles(config, root)` when `section['plots_only']`.
    if section['plots_only']:
        return render_cached_profiles(config,root)
    prepare_experiment_output(config_path(config,section['output_root']),root,overwrite=config['experiment_outputs']['overwrite_existing'])
    (root/'json').mkdir(exist_ok=True);write(root/'json/resolved_config.json',config)
    results=[]
    with ProcessPoolExecutor(max_workers=section['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(farm_job,config,farm,root) for farm in section['farm_ids']]
        # Run four independent post-fit diagnostic audits using local data only for each `future` in `as_completed(futures)`.
        for future in as_completed(futures):results.append(future.result())
    (root/'csv').mkdir(exist_ok=True)
    # Write the corresponding artifacts for `(index, name)` from `enumerate(['profile_ranking', 'error_contributions', 'review_candidates'])`.
    for index,name in enumerate(['profile_ranking','error_contributions','review_candidates']):
        pd.DataFrame([row for result in results for row in result[index]]).to_csv(root/'csv'/f'{name}.csv',index=False)
    write(root/'json/manifest.json',{'status':'complete','purpose':'Post-fit evidence, not exclusion or fault classification',
        'quality_csv_sha256':sha(config_path(config,config['pv_data_quality']['exclusions_csv'])),
        'known_limitation':'Satellite GHI alone is not perfect weather; unobserved direct/diffuse, operating settings and telemetry may matter.'})
    print(root,flush=True)


def render_cached_profiles(config,root,worst_days=0):
    """Render requested profile cases from cached diagnostics without rerunning inference."""
    section=config['pv_retrained_error_diagnostics']
    # Enforce the input contract: A completed diagnostic run is required.
    if not (root/'json/manifest.json').exists():raise ValueError('A completed diagnostic run is required')
    # Apply apply episode exclusions for `farm` from `section['farm_ids']`.
    for farm in section['farm_ids']:
        cases=list(section['focus_profiles_by_farm'].get(str(farm),[]))
        # Skip the current item when `not cases and (not worst_days)`.
        if not cases and not worst_days:continue
        folder=root/f'farm_{farm}';frame=pd.read_parquet(folder/'parquet/diagnostic_instances.parquet')
        provenance=json.loads((folder/'json/provenance.json').read_text())
        frame.attrs.update(farm_id=farm,model=f'{provenance["family"]}/{provenance["model"]}')
        published=config_path(config,section['models_root'])/f'farm_{farm}'/section['model_folder']/provenance['family']
        model=json.loads((published/'manifest.json').read_text())
        with open_store(config) as store:history=store[['power','ghi']].sel(farm_id=farm).load(scheduler='synchronous')
        raw=pd.DataFrame({'interval_start_utc':pd.to_datetime(history.time.values,utc=True),
            'measured_ac_mw':history.power.values*config['forecast']['realized_power_scale_to_mw'],
            'satellite_ghi_w_m2':history.ghi.values,'eligible':True})
        raw=apply_episode_exclusions(raw,config,farm_id=farm).set_index('interval_start_utc')
        ranking=pd.read_csv(folder/'csv/profile_ranking.csv')
        # When `worst_days`, set `test` using `ranking.loc[ranking.split.eq('test')].copy()`.
        if worst_days:
            test=ranking.loc[ranking.split.eq('test')].copy()
            daily=test.groupby('delivery_day').nrmse_percent.apply(
                lambda scores: float(np.sqrt(np.square(scores).mean()))
            ).rename('mean_auction_view_nrmse_percent').sort_values(ascending=False)
            chosen=daily.head(worst_days)
            selected_rows=[]
            # Render requested profile cases from cached diagnostics without rerunning inference for each `(day, score)` in `chosen.items()`.
            for day,score in chosen.items():
                profile=test.loc[test.delivery_day.eq(day)].sort_values('nrmse_percent',ascending=False).iloc[0]
                selected_rows.append({'farm_id':farm,'delivery_day':day,
                    'daily_mean_auction_view_nrmse_percent':score,
                    'worst_auction':profile.auction,
                    'worst_auction_nrmse_percent':profile.nrmse_percent,
                    'satellite_input_nrmse_percent':profile.satellite_input_nrmse_percent})
                cases.append({'delivery_day':day,'auction':profile.auction})
            pd.DataFrame(selected_rows).to_csv(folder/'csv/worst_test_days.csv',index=False)
        weather=[name.removeprefix('delta_') for name in frame if name.startswith('delta_')]
        rating=config['farms'][str(farm)]['ac_capacity_kw']/1000
        # Render requested profile cases from cached diagnostics without rerunning inference for each `case` in `cases`.
        for case in cases:
            selected=ranking.loc[ranking.delivery_day.eq(case['delivery_day'])&ranking.auction.eq(case['auction'])]
            # Enforce the input contract: No retained profile for `farm`: `case`.
            if selected.empty:raise ValueError(f'No retained profile for {farm}: {case}')
            plot_profile(frame,raw,selected.iloc[0].to_dict(),folder,rating,config,model['preprocessor'],weather)
        print(f'Farm {farm}: rendered {len(cases)} focused profile/feature plot pairs',flush=True)


# When `__name__ == '__main__'`, set `parser` using `argparse.ArgumentParser(description=__doc__)`.
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--worst-days',type=int,default=0,
        help='Additionally plot this many worst distinct held-out days per farm, selecting each day\'s worst auction view.')
    args=parser.parse_args();config=load_config(args.config)
    # When `args.plots_only`, set `config['pv_retrained_error_diagnostics']['plots_only']` using `True`.
    if args.plots_only:config['pv_retrained_error_diagnostics']['plots_only']=True
    # Process this part of the workflow when `args.worst_days < 0`.
    if args.worst_days < 0: parser.error('--worst-days must be nonnegative')
    # When `config['pv_retrained_error_diagnostics']['plots_only']`, set `section` using `config['pv_retrained_error_diagnostics']`.
    if config['pv_retrained_error_diagnostics']['plots_only']:
        section=config['pv_retrained_error_diagnostics']
        root=config_path(config,section['output_root'])/section['run_id']
        render_cached_profiles(config,root,worst_days=args.worst_days)
    else:
        run(config)

"""Read-only attribution diagnostics for frozen PV forecast errors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...data import config_path
from ...reporting.artifacts import prepare_experiment_output
from ..studies.pv_ensemble_study import score
from ..training.pv_fitting import balanced_weights
from ..data.pv_outages import apply_episode_exclusions


def _weighted_correlation(first,second,weights):
    """Measure association on the same balanced auction/day view population."""
    mass=weights.sum()
    # Return `float('nan')` when `mass <= 0`.
    if mass<=0:return float('nan')
    weights=weights/mass
    x=first-np.sum(weights*first);y=second-np.sum(weights*second)
    variance=np.sum(weights*x*x)*np.sum(weights*y*y)
    return float(np.sum(weights*x*y)/np.sqrt(variance)) if variance>0 else float('nan')


def diagnose_farm_errors(frame,previous,*,rating,timezone,section):
    """Retain raw AC, weather and flags while assigning weighted error shares."""
    mask=(frame.eligible&frame.daylight&frame.split.eq(section['evaluation_split'])).to_numpy()
    data=frame.loc[mask].copy()
    # Enforce the input contract: No eligible daylight forecast views in this split.
    if data.empty:raise ValueError('No eligible daylight forecast views in this split')
    power=np.asarray(previous,dtype=float)[mask]
    data['pv_error_mw']=power-data.measured_ac_mw.to_numpy(dtype=float)
    data['forecast_ac_mw']=power
    data['ghi_error_w_m2']=data.forecast_ghi-data.satellite_ghi_w_m2
    data['abs_ghi_error_w_m2']=data.ghi_error_w_m2.abs()
    data['weight']=balanced_weights(data)
    data['weighted_sq_error']=data.weight*(data.pv_error_mw/rating)**2
    # Enforce the input contract: Nonfinite PV error on the shared mask.
    if not np.isfinite(data.pv_error_mw).all():raise ValueError('Nonfinite PV error on the shared mask')
    total=data.weighted_sq_error.sum()
    local=pd.DatetimeIndex(data.interval_start_utc).tz_convert(timezone)
    data['local_hour']=local.hour
    summaries=[]
    def group(label,selected):
        portion=data.loc[selected]
        # Perform group when `portion.empty`.
        if portion.empty:return
        summaries.append({'group':label,'views':len(portion),
            'weight_fraction':float(portion.weight.sum()/data.weight.sum()),
            'squared_error_share':float(portion.weighted_sq_error.sum()/total),
            'signed_pv_bias_mw':float(np.average(portion.pv_error_mw,weights=portion.weight))})
    valid=data.abs_ghi_error_w_m2.notna()&np.isfinite(data.abs_ghi_error_w_m2)
    # Enforce the input contract: No satellite GHI values are available for the requested diagnostic split.
    if not valid.any():
        raise ValueError('No satellite GHI values are available for the requested diagnostic split')
    quantiles=np.quantile(data.loc[valid,'abs_ghi_error_w_m2'],
        np.linspace(0,1,section['ghi_quantile_count']+1))
    # Duplicate boundaries are allowed: zero-weather-error bins may collapse.
    edges=np.unique(quantiles)
    labels=[]
    # Retain raw AC, weather and flags while assigning weighted error shares for each `(position, (low, high))` in `enumerate(zip(edges[:-1], edges[1:]), 1)`.
    for position,(low,high) in enumerate(zip(edges[:-1],edges[1:]),1):
        selected=valid & data.abs_ghi_error_w_m2.ge(low) & (
            data.abs_ghi_error_w_m2.le(high) if position==len(edges)-1 else
            data.abs_ghi_error_w_m2.lt(high))
        label=f'GHI error bin {position}: [{low:.1f}, {high:.1f}] W/m²'
        group(label,selected);labels.append(label)
    group('Satellite GHI missing',~valid)
    # Retain raw AC, weather and flags while assigning weighted error shares for each `flag` in `section['operating_flags']`.
    for flag in section['operating_flags']:
        # Enforce the input contract: Missing operating diagnostic flag `flag`.
        if flag not in data:raise ValueError(f'Missing operating diagnostic flag {flag}')
        group(f'Flagged: {flag}',data[flag].fillna(False).astype(bool))
    by_hour=[]
    # Retain raw AC, weather and flags while assigning weighted error shares for each `(hour, part)` in `data.groupby('local_hour')`.
    for hour,part in data.groupby('local_hour'):
        by_hour.append({'local_hour':int(hour),'views':len(part),
            'squared_error_share':float(part.weighted_sq_error.sum()/total),
            'signed_pv_bias_mw':float(np.average(part.pv_error_mw,weights=part.weight))})
    by_auction=[]
    # Retain raw AC, weather and flags while assigning weighted error shares for each `(auction, part)` in `data.groupby('auction')`.
    for auction,part in data.groupby('auction'):
        by_auction.append({'auction':auction,'views':len(part),
            'squared_error_share':float(part.weighted_sq_error.sum()/total),
            'normalized_rmse_percent':float(100*np.sqrt(
                part.weighted_sq_error.sum()/part.weight.sum()))})
    both=valid&np.isfinite(data.pv_error_mw)
    correlation=_weighted_correlation(data.loc[both,'ghi_error_w_m2'].to_numpy(),
        data.loc[both,'pv_error_mw'].to_numpy(),data.loc[both,'weight'].to_numpy())
    result={'views':len(data),'days':int(data.delivery_day.nunique()),
        'capacity_normalized_rmse_percent':float(100*np.sqrt(total/data.weight.sum())),
        'satellite_ghi_coverage':float(data.loc[valid,'weight'].sum()/data.weight.sum()),
        'weighted_signed_pv_error_ghi_error_correlation':correlation,
        'ghi_error_quantile_edges_w_m2':edges.tolist(),
        'evaluation_split':section['evaluation_split'],
        'interpretation':'Associations only. Satellite GHI is diagnostic; flags are unconfirmed operating conditions.'}
    return result,pd.DataFrame(summaries),pd.DataFrame(by_hour),pd.DataFrame(by_auction),data,labels


def run_pv_error_diagnostics(config,progress=print):
    """Report where frozen forecast error accumulates without changing inputs."""
    section=config['pv_error_diagnostics'];source=config_path(config,section['source_root'])
    output_root=config_path(config,section['output_root']);output=output_root/section['run_id']
    # Enforce the input contract: ghi_quantile_count must be between 2 and 20.
    if not 2<=section['ghi_quantile_count']<=20:
        raise ValueError('ghi_quantile_count must be between 2 and 20')
    # Report where frozen forecast error accumulates without changing inputs for each `key` in `['top_flagged_days_for_sensitivity', 'plot_top_flagged_days']`.
    for key in ['top_flagged_days_for_sensitivity','plot_top_flagged_days']:
        # Enforce the input contract: `key` must be a nonnegative integer.
        if int(section[key])!=section[key] or section[key]<0:
            raise ValueError(f'{key} must be a nonnegative integer')
    # Enforce the input contract: Diagnostic input cannot be inside the output directory.
    if source.resolve().is_relative_to(output.resolve()):
        raise ValueError('Diagnostic input cannot be inside the output directory')
    prepare_experiment_output(output_root,output,
        overwrite=config['experiment_outputs']['overwrite_existing'])
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    summaries=[]
    # Apply apply episode exclusions for `farm` from `section['farm_ids']`.
    for farm in section['farm_ids']:
        directory=output/f'farm_{farm}'
        # Report where frozen forecast error accumulates without changing inputs for each `folder` in `['csv', 'json', 'plots']`.
        for folder in ['csv','json','plots']:(directory/folder).mkdir(parents=True,exist_ok=True)
        corpus=source/f'farm_{farm}/parquet/corpus.parquet'
        forecasts=source/f'farm_{farm}/parquet/comparison_predictions.parquet'
        frame=apply_episode_exclusions(pd.read_parquet(corpus),config,farm_id=farm)
        saved=pd.read_parquet(forecasts)
        keys=['interval_start_utc','delivery_day','auction','measured_ac_mw']
        # Enforce the input contract: Forecast and measured populations differ.
        if not frame[keys].equals(saved[keys]):raise ValueError('Forecast and measured populations differ')
        previous=saved[f"prediction_{section['forecast_model']}"].to_numpy()
        rating=config['farms'][str(farm)]['ac_capacity_kw']/1000
        result,groups,hours,auctions,data,labels=diagnose_farm_errors(
            frame,previous,rating=rating,timezone=config['pv_neural']['delivery_timezone'],
            section=section)
        result.update(farm_id=farm,input_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [corpus,forecasts]})
        flagged=np.zeros(len(data),dtype=bool)
        # Report where frozen forecast error accumulates without changing inputs for each `flag` in `section['operating_flags']`.
        for flag in section['operating_flags']:
            flagged|=data[flag].fillna(False).to_numpy(dtype=bool)
        flagged_data=data.loc[flagged].copy()
        flagged_data['forecast_ac_mw']=flagged_data.measured_ac_mw+flagged_data.pv_error_mw
        daily=flagged_data.groupby('delivery_day').agg(
            flagged_views=('pv_error_mw','size'),
            unique_physical_quarters=('interval_start_utc','nunique'),
            mean_measured_ac_mw=('measured_ac_mw','mean'),
            mean_forecast_ac_mw=('forecast_ac_mw','mean'),
            mean_satellite_ghi_w_m2=('satellite_ghi_w_m2','mean'),
            weighted_sq_error=('weighted_sq_error','sum')).reset_index()
        # When `not daily.empty`, set `daily['total_squared_error_share']` using `daily.weighted_sq_error / data.weighted_sq_error.sum()`.
        if not daily.empty:
            daily['total_squared_error_share']=daily.weighted_sq_error/data.weighted_sq_error.sum()
            daily=daily.sort_values('weighted_sq_error',ascending=False)
        daily.to_csv(directory/'csv/flagged_days.csv',index=False)
        selected_days=daily.head(section['top_flagged_days_for_sensitivity']).delivery_day.tolist()
        result['sensitivity_excluded_days']=selected_days
        # When `selected_days`, set `keep` using `~frame.delivery_day.isin(selected_days)`.
        if selected_days:
            keep=~frame.delivery_day.isin(selected_days)
            subset,subset_groups,*_=diagnose_farm_errors(frame.loc[keep].reset_index(drop=True),
                previous[keep.to_numpy()],rating=rating,
                timezone=config['pv_neural']['delivery_timezone'],section=section)
            quantile=subset_groups[subset_groups.group.str.startswith('GHI error bin')]
            result['without_top_flagged_days_weather_diagnostics']={
                'weighted_signed_pv_error_ghi_error_correlation':
                    subset['weighted_signed_pv_error_ghi_error_correlation'],
                'largest_absolute_ghi_error_bin_squared_error_share':
                    float(quantile.squared_error_share.iloc[-1]) if not quantile.empty else None}
        validation=json.loads((source/f'farm_{farm}/json/selection.json').read_text())
        models=list(dict.fromkeys([section['forecast_model'],validation['selected_static'],'old_pvlib']))
        baseline=(frame.eligible&frame.daylight&frame.split.eq(section['evaluation_split']))
        sensitivity=[]
        # Apply score for `model` from `models`.
        for model in models:
            curve=saved[f'prediction_{model}'].to_numpy()
            # Process each `(label, mask)` in this collection.
            for label,mask in [('all',baseline),
                               ('without_flagged_views',baseline&~frame[section['operating_flags']].any(axis=1)),
                               ('without_top_flagged_days',baseline&~frame.delivery_day.isin(selected_days))]:
                # Apply score when `mask.any()`.
                if mask.any():
                    sensitivity.append({'model':model,'sample':label,**score(frame,curve,rating,mask.to_numpy())})
        pd.DataFrame(sensitivity).to_csv(directory/'csv/sensitivity.csv',index=False)
        # Write the corresponding artifacts for `(name, table)` from `[('error_groups', groups), ('by_local_hour', hours), ('by_auction', auctions)]`.
        for name,table in [('error_groups',groups),('by_local_hour',hours),('by_auction',auctions)]:
            table.to_csv(directory/'csv'/f'{name}.csv',index=False)
        (directory/'json/summary.json').write_text(json.dumps(result,indent=2)+'\n')
        # Report where frozen forecast error accumulates without changing inputs when `section['save_plots']`.
        if section['save_plots']:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            rng=np.random.default_rng(section['scatter_seed'])
            finite=data.ghi_error_w_m2.notna()&np.isfinite(data.ghi_error_w_m2)
            rows=np.flatnonzero(finite.to_numpy())
            sample=rng.choice(rows,size=min(len(rows),section['scatter_sample_views']),replace=False)
            figure,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
            selected=groups[groups.group.isin(labels)]
            axes[0].bar(range(len(selected)),selected.squared_error_share*100,color='tab:blue')
            axes[0].set_xticks(range(len(selected)),[f'Q{i+1}' for i in range(len(selected))])
            axes[0].set_ylabel('Share of squared PV error (%)')
            axes[0].set_title('Absolute GHI disagreement quantiles')
            axes[1].scatter(data.ghi_error_w_m2.to_numpy()[sample],
                data.pv_error_mw.to_numpy()[sample],alpha=.12,s=5,color='tab:orange')
            axes[1].set_xlabel('Forecast minus satellite GHI (W/m²)')
            axes[1].set_ylabel('Forecast minus measured AC (MW)')
            axes[1].set_title('Signed errors; sampled forecast views')
            # Report where frozen forecast error accumulates without changing inputs for each `axis` in `axes`.
            for axis in axes:axis.grid(alpha=.2)
            figure.savefig(directory/'plots/error_diagnostics.png',dpi=150)
            plt.close(figure)
            import matplotlib.dates as mdates
            from zoneinfo import ZoneInfo
            # Report where frozen forecast error accumulates without changing inputs for each `day` in `daily.head(section['plot_top_flagged_days']).delivery_day`.
            for day in daily.head(section['plot_top_flagged_days']).delivery_day:
                views=data.loc[data.delivery_day.eq(day)]
                physical=views.groupby('interval_start_utc',sort=True).agg(
                    measured_ac_mw=('measured_ac_mw','first'),
                    satellite_ghi_w_m2=('satellite_ghi_w_m2','first')).reset_index()
                boundary=(pd.Timestamp(day).tz_localize(config['pv_neural']['delivery_timezone'])
                    +pd.DateOffset(days=1)).tz_convert('UTC')
                available=frame.loc[frame.interval_start_utc.eq(boundary)]
                # Update `physical` when this condition is met.
                if not available.empty:
                    physical=pd.concat([physical,pd.DataFrame([{
                        'interval_start_utc':boundary,
                        'measured_ac_mw':available.measured_ac_mw.iloc[0],
                        'satellite_ghi_w_m2':available.satellite_ghi_w_m2.iloc[0]}])],
                        ignore_index=True)
                figure,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,layout='constrained')
                axes[0].plot(physical.interval_start_utc,physical.measured_ac_mw,
                    color='black',lw=2,label='Measured AC')
                axes[1].plot(physical.interval_start_utc,physical.satellite_ghi_w_m2,
                    color='black',lw=2,label='Satellite GHI (diagnostic)')
                # Report where frozen forecast error accumulates without changing inputs for each `(auction, curve)` in `views.groupby('auction')`.
                for auction,curve in views.groupby('auction'):
                    axes[0].plot(curve.interval_start_utc,curve.forecast_ac_mw,
                        lw=1,label=f'{auction} AC forecast')
                    axes[1].plot(curve.interval_start_utc,curve.forecast_ghi,
                        lw=1,label=f'{auction} GHI forecast')
                axes[0].axhline(rating,color='gray',ls='--',lw=1,label='Farm AC rating')
                axes[0].set_ylabel('AC power (MW)')
                axes[1].set_ylabel('GHI (W/m²)')
                axes[1].set_xlabel('Delivery time, Europe/Rome')
                axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',
                    tz=ZoneInfo(config['pv_neural']['delivery_timezone'])))
                # Report where frozen forecast error accumulates without changing inputs for each `axis` in `axes`.
                for axis in axes:axis.grid(alpha=.2);axis.legend(ncol=3,fontsize=8)
                figure.suptitle(f'Farm {farm}, {day}: flagged production evidence')
                figure.savefig(directory/'plots'/f'flagged_day_{day}.png',dpi=150)
                plt.close(figure)
        progress(f"Farm {farm}: GHI/PV error correlation {result['weighted_signed_pv_error_ghi_error_correlation']:.3f}")
        summaries.append(result)
    (output/'manifest.json').write_text(json.dumps({'status':'complete','summaries':summaries,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    return {'output':str(output),'summaries':summaries}

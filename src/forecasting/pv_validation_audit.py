"""Attribute validation error before and after sustained-zero outage curation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data import config_path
from ..reporting.artifacts import prepare_experiment_output
from .pv_fitting import balanced_weights
from .pv_outages import apply_episode_exclusions


def _score(frame,prediction,rating,mask):
    """Score the same finite daylight rows with equal day/auction-view weight."""
    part=frame.loc[mask]
    error=(prediction[mask]-part.measured_ac_mw.to_numpy())/rating
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite predictions on the common evaluation mask')
    weights=balanced_weights(part)
    return {'rows':len(part),'days':part.delivery_day.nunique(),
        'nrmse_percent':float(np.sqrt(weights @ (error**2))*100),
        'nmae_percent':float(weights @ np.abs(error)*100),
        'bias_percent':float(weights @ error*100)}


def _day_table(frame,prediction,rating,*,split):
    """Give each available auction an equal contribution within its day."""
    selected=frame.loc[frame.eligible&frame.daylight&frame.split.eq(split)].copy()
    selected['forecast_ac_mw']=np.asarray(prediction)[selected.index]
    selected['normalized_squared_error']=((selected.forecast_ac_mw-selected.measured_ac_mw)/rating)**2
    selected['absolute_ghi_error_w_m2']=(selected.forecast_ghi-selected.satellite_ghi_w_m2).abs()
    selected['balance_weight']=balanced_weights(selected)
    selected['weighted_squared_error']=selected.balance_weight*selected.normalized_squared_error
    daily=selected.groupby('delivery_day').agg(
        views=('normalized_squared_error','size'),
        physical_quarters=('interval_start_utc','nunique'),
        mean_measured_ac_mw=('measured_ac_mw','mean'),
        mean_forecast_ac_mw=('forecast_ac_mw','mean'),
        mean_absolute_ghi_error_w_m2=('absolute_ghi_error_w_m2','mean'),
        balance_weight=('balance_weight','sum'),
        weighted_squared_error=('weighted_squared_error','sum'))
    daily['mean_normalized_squared_error']=daily.weighted_squared_error/daily.balance_weight
    daily['normalized_rmse_percent']=100*np.sqrt(daily.mean_normalized_squared_error)
    daily['share_of_total_squared_error']=daily.weighted_squared_error/daily.weighted_squared_error.sum()
    return daily.reset_index()


def _plot_day(frame,prediction,*,farm,day,rating,timezone,path,
              forecast_label='AC forecast',comparison_prediction=None,
              comparison_label='pvlib AC forecast'):
    """Compare AC predictors with measured power and weather with satellite GHI."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo
    views=frame.loc[frame.delivery_day.eq(day)].copy()
    views['forecast_ac_mw']=np.asarray(prediction)[views.index]
    if comparison_prediction is not None:
        views['comparison_ac_mw']=np.asarray(comparison_prediction)[views.index]
    physical=views.groupby('interval_start_utc',sort=True).agg(
        measured_ac_mw=('measured_ac_mw','first'),
        satellite_ghi_w_m2=('satellite_ghi_w_m2','first')).reset_index()
    boundary=(pd.Timestamp(day).tz_localize(timezone)+pd.DateOffset(days=1)).tz_convert('UTC')
    next_point=frame.loc[frame.interval_start_utc.eq(boundary)]
    if not next_point.empty:
        physical=pd.concat([physical,pd.DataFrame([{
            'interval_start_utc':boundary,
            'measured_ac_mw':next_point.measured_ac_mw.iloc[0],
            'satellite_ghi_w_m2':next_point.satellite_ghi_w_m2.iloc[0]}])],ignore_index=True)
    comparison=comparison_prediction is not None
    figure,axes=plt.subplots(3 if comparison else 2,1,
        figsize=(13,11 if comparison else 8),sharex=True,layout='constrained')
    power_axes=axes[:2] if comparison else axes[:1]
    for axis in power_axes:
        axis.plot(physical.interval_start_utc,physical.measured_ac_mw,
            color='black',lw=2,label='Measured AC')
        axis.axhline(rating,color='gray',ls='--',lw=1,label='Farm AC rating')
        axis.set_ylabel('AC power (MW)')
    axes[-1].plot(physical.interval_start_utc,physical.satellite_ghi_w_m2,
        color='black',lw=2,label='Satellite GHI (diagnostic)')
    for auction,curve in views.groupby('auction'):
        axes[0].plot(curve.interval_start_utc,curve.forecast_ac_mw,lw=1,
            label=f'{auction} {forecast_label}')
        if comparison:
            axes[1].plot(curve.interval_start_utc,curve.comparison_ac_mw,lw=1,
                label=f'{auction} {comparison_label}')
        axes[-1].plot(curve.interval_start_utc,curve.forecast_ghi,lw=1,
            label=f'{auction} GHI forecast')
    if comparison:
        upper=max(rating,physical.measured_ac_mw.max(),
            views.forecast_ac_mw.max(),views.comparison_ac_mw.max())
        for axis in power_axes:axis.set_ylim(-.03*upper,1.05*upper)
    axes[-1].set_ylabel('GHI (W/m²)')
    axes[-1].set_xlabel(f'Delivery time, {timezone}')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=ZoneInfo(timezone)))
    for axis in axes:axis.grid(alpha=.2);axis.legend(ncol=3,fontsize=8)
    title=f'Farm {farm}, {day}: measured AC and {forecast_label.lower()}'
    if comparison:title+=f' versus {comparison_label.lower()}'
    figure.suptitle(title)
    figure.savefig(path,dpi=150);plt.close(figure)


def run_validation_error_audit(config,progress=print):
    """Audit the configured frozen predictors without refitting or editing raw data."""
    section=config['pv_validation_audit'];split=section['split']
    output_root=config_path(config,section['output_root']);output=output_root/section['run_id']
    prepare_experiment_output(output_root,output,
        overwrite=config['experiment_outputs']['overwrite_existing'])
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    summaries=[]
    for farm in section['farm_ids']:
        source_spec=section['farm_sources'][str(farm)]
        source=config_path(config,source_spec['root'])/f'farm_{farm}/parquet'
        corpus=source/'corpus.parquet';prior=source/source_spec['predictions_file']
        frame=pd.read_parquet(corpus);saved=pd.read_parquet(prior)
        keys=['interval_start_utc','delivery_day','auction','measured_ac_mw']
        if not frame[keys].equals(saved[keys]):raise ValueError('Audit source keys or raw AC differ')
        prediction=saved[source_spec['prediction_column']].to_numpy(dtype=float)
        curated=apply_episode_exclusions(frame,config,farm_id=farm)
        rating=config['farms'][str(farm)]['ac_capacity_kw']/1000
        original_mask=(frame.eligible&frame.daylight&frame.split.eq(split)).to_numpy()
        curated_mask=(curated.eligible&curated.daylight&curated.split.eq(split)).to_numpy()
        previous=_score(frame,prediction,rating,original_mask)
        current=_score(curated,prediction,rating,curated_mask)
        weights=balanced_weights(frame.loc[original_mask])
        residual=((prediction[original_mask]-frame.measured_ac_mw.to_numpy()[original_mask])/rating)**2
        flagged=curated.outage_excluded.to_numpy()[original_mask]
        outage_share=float(weights[flagged]@residual[flagged]/(weights@residual))
        before=_day_table(frame,prediction,rating,split=split)
        after=_day_table(curated,prediction,rating,split=split)
        daily=before.merge(after,on='delivery_day',how='left',suffixes=('_before','_after'))
        counts=curated.loc[original_mask&curated.outage_excluded.to_numpy()].groupby('delivery_day').agg(
            excluded_auction_views=('auction','size'),
            excluded_physical_quarters=('interval_start_utc','nunique'))
        daily=daily.join(counts,on='delivery_day')
        for name in ['excluded_auction_views','excluded_physical_quarters']:
            daily[name]=daily[name].fillna(0).astype(int)
        folder=output/f'farm_{farm}'
        for name in ['csv','json','plots']:(folder/name).mkdir(parents=True,exist_ok=True)
        daily.sort_values('delivery_day').to_csv(folder/'csv/daily_error.csv',index=False)
        summary={'farm_id':farm,'split':split,'prediction_column':source_spec['prediction_column'],
            'before':previous,'after':current,
            'outage_excluded_views_on_split':int(flagged.sum()),
            'outage_squared_error_share_before':outage_share,
            'top_days_before':daily.nlargest(section['top_days_per_farm'],
                'share_of_total_squared_error_before').delivery_day.tolist(),
            'top_days_after':daily.nlargest(section['top_days_per_farm'],
                'share_of_total_squared_error_after').delivery_day.tolist(),
            'input_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [corpus,prior]},
            'interpretation':'Outage mask changes evaluation population; scores are from unchanged frozen models.'}
        (folder/'json/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if section['save_plots']:
            for day in dict.fromkeys(summary['top_days_before']+summary['top_days_after']):
                _plot_day(frame,prediction,farm=farm,day=day,rating=rating,
                    timezone=config['pv_neural']['delivery_timezone'],
                    forecast_label=f"{source_spec['prediction_column']} AC forecast",
                    path=folder/'plots'/f'validation_day_{day}.png')
        progress(f"Farm {farm}: validation RMSE {previous['nrmse_percent']:.3f}% -> {current['nrmse_percent']:.3f}%")
        summaries.append(summary)
    (output/'manifest.json').write_text(json.dumps({'status':'complete',
        'summaries':summaries,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        indent=2)+'\n')
    return {'output':str(output),'summaries':summaries}

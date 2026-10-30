"""Score matched auction forecasts and plot online PV CRPS diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..online.pv_online_crps import weighted_crps, weighted_quantile
from ..data.pv_probabilistic_data import file_hash, write_json
from ..evaluation.pv_probabilistic_metrics import quantile_crps


def _curve(values, levels, grid):
    return np.interp(grid, np.r_[0., levels, 1.], np.r_[values[0], values, values[-1]])


def score_one(row):
    """Summarize one issued distribution with explicit tail convention."""
    x = np.asarray(row['values'], float); y = float(row['observed_mw'])
    levels = row['levels']
    qs = np.arange(5,100,5,dtype=float)/100
    # When `levels is None or (isinstance(levels, float) and np.isnan(levels))`, set `w` using `np.asarray(row['weights'], float)`.
    if levels is None or (isinstance(levels, float) and np.isnan(levels)):
        w = np.asarray(row['weights'], float)
        mean = float(w @ x); q = weighted_quantile(x, w, qs)
        crps = weighted_crps(x, w, y)
        spread = float(np.sqrt(w @ (x-mean)**2))
    else:
        levels = np.asarray(levels, float)
        grid = np.linspace(0, 1, 201); values = _curve(x, levels, grid)
        mean = float(np.trapezoid(values, x=grid))
        q = _curve(x, levels, qs)
        crps = float(quantile_crps(x[None,:], levels, np.array([y]))[0])
        spread = float(np.sqrt(np.trapezoid((values-mean)**2, x=grid)))
    output = {'mean_mw': mean, 'median_mw': float(q[9]), 'crps_mw': crps, 'spread_mw': spread}
    # Summarize one issued distribution with explicit tail convention for each `(level, value)` in `zip(qs, q)`.
    for level, value in zip(qs,q):
        name = f'q{round(level*100):02d}'
        output[name] = float(value)
        residual = y-value
        output[f'pinball_{name}_mw'] = float(max(level*residual,(level-1)*residual))
        output[f'below_{name}'] = int(y < value)
        output[f'below_equal_{name}'] = int(y <= value)
    # Summarize one issued distribution with explicit tail convention for each `(lo, hi, width)` in `[(0, 18, 90), (1, 17, 80), (4, 14, 50)]`.
    for lo, hi, width in [(0,18,90),(1,17,80),(4,14,50)]:
        output[f'coverage_{width}'] = int(q[lo] <= y <= q[hi])
        output[f'width_{width}_mw'] = float(q[hi]-q[lo])
    return output


def _summary(frame, rating, scope):
    rows=[]
    # Perform summary for each `(method, part)` in `frame.groupby('method')`.
    for method, part in frame.groupby('method'):
        y = part.observed_mw.to_numpy(float)
        result = {'scope': scope, 'method': method, 'n': len(part)}
        # Perform summary for each `name` in `('mean', 'median')`.
        for name in ('mean', 'median'):
            error = part[f'{name}_mw'].to_numpy(float)-y
            # Process each `(label, value)` in this collection.
            for label, value in [('mae', np.mean(abs(error))), ('rmse', np.sqrt(np.mean(error**2))), ('bias', np.mean(error))]:
                result[f'{name}_{label}_mw'] = float(value)
                result[f'{name}_{label}_percent_capacity'] = float(100*value/rating)
        result['crps_mw'] = float(part.crps_mw.mean()); result['crps_percent_capacity'] = 100*result['crps_mw']/rating
        # Perform summary for each `width` in `(50, 80, 90)`.
        for width in (50,80,90):
            result[f'coverage_{width}'] = float(part[f'coverage_{width}'].mean())
            result[f'width_{width}_mw'] = float(part[f'width_{width}_mw'].mean())
            result[f'width_{width}_percent_capacity'] = 100*result[f'width_{width}_mw']/rating
        # Perform summary for each `level` in `range(5, 100, 5)`.
        for level in range(5,100,5):
            q = f'q{level:02d}'
            result[f'pinball_{q}_mw'] = float(part[f'pinball_{q}_mw'].mean())
            result[f'pinball_{q}_percent_capacity'] = 100*result[f'pinball_{q}_mw']/rating
            result[f'below_{q}'] = float(part[f'below_{q}'].mean())
            result[f'below_equal_{q}'] = float(part[f'below_equal_{q}'].mean())
        rows.append(result)
    return rows


def _plot_daily(part, destination, farm, auction, day):
    fig, ax = plt.subplots(figsize=(13,5))
    colors = {'equal_ensemble':'#1f77b4','online_ensemble_warm':'#36a3a3',
              'deterministic_quantiles':'#e69f00','online_combined_warm':'#d44739'}
    # Perform plot daily for each `(method, color)` in `colors.items()`.
    for method, color in colors.items():
        x = part.loc[part.method.eq(method)].sort_values('interval_start_utc')
        # Skip the current item when `x.empty`.
        if x.empty: continue
        t = pd.DatetimeIndex(x.interval_start_utc).tz_convert('Europe/Rome')
        ax.plot(t, x.mean_mw, label=method, color=color)
        # Perform plot daily when `method == 'online_combined_warm'`.
        if method == 'online_combined_warm': ax.fill_between(t, x.q10, x.q90, color=color, alpha=.15, label='combined warm p10–p90')
    observed = part.loc[part.method.eq('equal_ensemble')].sort_values('interval_start_utc')
    t = pd.DatetimeIndex(observed.interval_start_utc).tz_convert('Europe/Rome')
    ax.plot(t, observed.observed_mw, color='black', label='measured AC')
    ax.set(title=f'Farm {farm} — {auction} — {day} (exploratory validation)', ylabel='AC power (MW)', xlabel='Europe/Rome')
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8); fig.autofmt_xdate(); fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True); fig.savefig(destination, dpi=120); plt.close(fig)


def report(config, root):
    """Score only observations shared by every requested method, then publish diagnostics."""
    settings=config['pv_online_crps_study']; root=Path(root); summaries=[]; counts=[]
    expected={'equal_ensemble','online_ensemble_warm','online_ensemble_cold','deterministic_quantiles',
              'online_combined_warm','online_combined_cold','fixed_ensemble_quantiles'}
    # Write the corresponding artifacts for `farm` from `settings['farms']`.
    for farm in settings['farms']:
        farm_root=root/f'farm_{farm}'; manifest=json.loads((farm_root/'prepared_manifest.json').read_text()); rating=manifest['rating_mw']
        prepared=pd.read_parquet(farm_root/'prepared.parquet')
        # Write the corresponding artifacts for `auction` from `settings['auctions']`.
        for auction in settings['auctions']:
            folder=farm_root/auction; evaluation=json.loads((folder/'manifest.json').read_text())
            path=folder/'forecasts.parquet'
            # Enforce the input contract: Forecast artifact changed.
            if file_hash(path) != evaluation['files']['forecasts']: raise ValueError('Forecast artifact changed')
            frame=pd.read_parquet(path); frame['interval_start_utc']=pd.to_datetime(frame.interval_start_utc,utc=True)
            # Enforce the input contract: Missing methods for `farm` `auction`.
            if set(frame.method.unique()) != expected: raise ValueError(f'Missing methods for {farm} {auction}')
            # Enforce the input contract: Duplicate forecast score row.
            if frame.duplicated(['interval_start_utc','method']).any(): raise ValueError('Duplicate forecast score row')
            valid=frame.eligible & np.isfinite(frame.observed_mw)
            eligible=frame.loc[valid].copy()
            counts_by_time=eligible.groupby('interval_start_utc').method.nunique()
            matched=eligible.loc[eligible.interval_start_utc.isin(counts_by_time[counts_by_time.eq(len(expected))].index)].copy()
            # Enforce the input contract: No common observations for `farm` `auction`.
            if matched.empty: raise ValueError(f'No common observations for {farm} {auction}')
            lineage=prepared.loc[prepared.auction.eq(auction) & prepared['split'].eq('validation'),
                ['interval_start_utc','forecast_cycle_utc','forecast_available_at_utc','source_trial']].rename(
                columns={'forecast_cycle_utc':'ensemble_cycle_utc','forecast_available_at_utc':'ensemble_available_utc'})
            matched=matched.merge(lineage,on='interval_start_utc',how='left',validate='many_to_one')
            # Enforce the input contract: Ensemble provenance missing or unavailable at auction.
            if matched.ensemble_available_utc.isna().any() or (matched.ensemble_available_utc > matched.decision_cutoff_utc).any():
                raise ValueError('Ensemble provenance missing or unavailable at auction')
            scored=pd.DataFrame([score_one(row) for row in matched.to_dict('records')],index=matched.index)
            matched=pd.concat([matched,scored],axis=1)
            local=pd.DatetimeIndex(matched.interval_start_utc).tz_convert('Europe/Rome')
            clock=local.hour.to_numpy()*60+local.minute.to_numpy()
            start=pd.Timedelta(settings['daylight_start']+':00').total_seconds()/60
            end=pd.Timedelta(settings['daylight_end']+':00').total_seconds()/60
            matched['daytime']=(clock>=start)&(clock<end)
            # Score only observations shared by every requested method, then publish diagnostics for each `(scope, sub)` in `[('all_hours', matched), ('daytime', matched.loc[matched.daytime])]`.
            for scope, sub in [('all_hours',matched),('daytime',matched.loc[matched.daytime])]:
                # Score only observations shared by every requested method, then publish diagnostics for each `item` in `_summary(sub, rating, scope)`.
                for item in _summary(sub,rating,scope): summaries.append({'farm_id': int(farm),'auction':auction,**item})
            inputs=prepared.loc[prepared.auction.eq(auction) & prepared['split'].eq('validation')]
            counts.append({'farm_id':int(farm),'auction':auction,'matched_intervals':len(matched)//len(expected),
                'input_intervals':int(len(inputs)), 'forecast_intervals':int(frame.interval_start_utc.nunique()),
                'missing_forecast_intervals':int(len(inputs)-frame.interval_start_utc.nunique()),
                'excluded_target_intervals':int(inputs.episode_excluded.sum()),
                'unusable_target_intervals':int((~inputs.eligible).sum()),
                'eligible_intervals':int(eligible.interval_start_utc.nunique()),
                'retrospective_intervals':int(matched.loc[matched.method.eq('equal_ensemble'),'retrospective_model'].sum())})
            matched.to_parquet(folder/'scored_matched.parquet',index=False)
            # Score only observations shared by every requested method, then publish diagnostics for each `(day, part)` in `matched.groupby('delivery_day')`.
            for day, part in matched.groupby('delivery_day'):
                _plot_daily(part,folder/'plots'/str(day)/'forecast.png',farm,auction,day)
            weight_rows=[]
            # Score only observations shared by every requested method, then publish diagnostics for each `row` in `matched.loc[matched.method.str.startswith('online_')].itertuples()`.
            for row in matched.loc[matched.method.str.startswith('online_')].itertuples():
                w=np.asarray(row.weights,float)
                weight_rows.append({'delivery_day':row.delivery_day,'method':row.method,'block':row.block,
                    'ensemble_weight':float(w[:50].sum()), 'deterministic_weight':float(w[50]) if len(w)>50 else 0.,
                    'quantile_weight':float(w[51:].sum()) if len(w)>50 else 0.,
                    'effective_experts':float(1/(w@w)), 'largest_expert_weight':float(w.max())})
            weights=pd.DataFrame(weight_rows).groupby(['delivery_day','method','block'],as_index=False).mean()
            weights.to_csv(folder/'weight_summary.csv',index=False)
            fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
            # Process each `(method, group)` in this collection.
            for method,group in weights.loc[weights.method.str.startswith('online_combined')].groupby('method'):
                daily=group.groupby('delivery_day').mean(numeric_only=True)
                # Score only observations shared by every requested method, then publish diagnostics for each `column` in `('ensemble_weight', 'deterministic_weight', 'quantile_weight')`.
                for column in ('ensemble_weight','deterministic_weight','quantile_weight'):
                    axes[0].plot(daily.index,daily[column],label=f'{method}: {column}')
                axes[1].plot(daily.index,daily.effective_experts,label=method)
            axes[0].set_ylabel('Mean weight'); axes[1].set_ylabel('Effective experts')
            axes[0].legend(fontsize=7,ncol=2); axes[1].legend(fontsize=8)
            # Score only observations shared by every requested method, then publish diagnostics for each `ax` in `axes`.
            for ax in axes: ax.grid(alpha=.25)
            axes[0].set_title(f'Farm {farm} — {auction}: validation weight evolution')
            fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(folder/'weight_evolution.png',dpi=120); plt.close(fig)
            # Reliability and spread-skill are tabular to retain ties and sample counts.
            reliability=[]; spread=[]
            # Score only observations shared by every requested method, then publish diagnostics for each `(method, part)` in `matched.groupby('method')`.
            for method, part in matched.groupby('method'):
                # Score only observations shared by every requested method, then publish diagnostics for each `q` in `range(5, 100, 5)`.
                for q in range(5,100,5):
                    name=f'q{q:02d}'
                    reliability.append({'method':method,'quantile':q/100,'n':len(part),
                        'strict':float(part[f'below_{name}'].mean()),'inclusive':float(part[f'below_equal_{name}'].mean())})
                rank=pd.qcut(part.spread_mw.rank(method='first'),q=min(5,len(part)),labels=False,duplicates='drop')
                # Score only observations shared by every requested method, then publish diagnostics for each `(bin_id, group)` in `part.groupby(rank)`.
                for bin_id, group in part.groupby(rank):
                    spread.append({'method':method,'bin':int(bin_id),'n':len(group),
                        'mean_spread_mw':float(group.spread_mw.mean()),
                        'rmse_mean_mw':float(np.sqrt(np.mean((group.mean_mw-group.observed_mw)**2)))})
            pd.DataFrame(reliability).to_csv(folder/'reliability.csv',index=False)
            pd.DataFrame(spread).to_csv(folder/'spread_skill.csv',index=False)
    metrics=pd.DataFrame(summaries); metrics.to_csv(root/'metrics.csv',index=False)
    pd.DataFrame(counts).to_csv(root/'coverage.csv',index=False)
    write_json(root/'report_manifest.json',{'status':'complete','farms':list(map(int,settings['farms'])),
        'auctions':settings['auctions'],'methods':sorted(expected),'validation_selected':True,
        'first_validation_day':'DA/IDA1/IDA2 use retrospectively trained source model',
        'crps':'exact weighted empirical or quantile quadrature with constant tails',
        'daytime':'fixed local clock range; not a solar-position claim',
        'joint_trajectories':'not inferred from marginal interval forecasts'})
    lines=['# Online CRPS PV comparison','', 'Exploratory validation: 2026-08-20 to 2026-08-31. '
           'Candidate models were selected using these validation days; August 20 DA/IDA1/IDA2 use a retrospective source model.', '',
           '| Farm | Auction | Method | Matched intervals | CRPS (% AC) | Mean RMSE (% AC) |', '| --- | --- | --- | ---: | ---: | ---: |']
    # Process each `row` in this collection.
    for row in metrics.loc[metrics.scope.eq('all_hours')].sort_values(['farm_id','auction','crps_mw']).itertuples():
        lines.append(f'| {row.farm_id} | {row.auction} | {row.method} | {row.n} | {row.crps_percent_capacity:.3f} | {row.mean_rmse_percent_capacity:.3f} |')
    lines += ['', 'Each comparison uses matched eligible intervals. Quantile CRPS uses numerical integration with constant tails; '
              'empirical distributions use exact weighted CRPS. Full forecasts, updates, coverage and daily plots are saved locally.',
              '', 'The algorithm produces marginal distributions; no joint trajectory calibration or delayed-regret guarantee is asserted.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return f'Online CRPS report: {root}'

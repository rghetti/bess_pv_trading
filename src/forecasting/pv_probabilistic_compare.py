"""Compare prior-day and retrospective same-day PV experiments on identical targets."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .pv_probabilistic_data import study_root
from .pv_probabilistic_metrics import score_frame
from .pv_probabilistic_reporting import comparable, load_farm


def plot_comparison(a, b, ids, farm, day, prior_model, same_model, destination):
    """Show both vintage strategies and the assumed same-day issue/availability times."""
    local_start = pd.Timestamp(day).tz_localize('Europe/Rome')
    local_end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize('Europe/Rome')
    grid = pd.date_range(local_start, local_end, freq='15min', inclusive='left').tz_convert('UTC')
    p = a.loc[a.delivery_day.eq(day)].set_index('interval_start_utc').reindex(grid)
    s = b.loc[b.delivery_day.eq(day)].set_index('interval_start_utc').reindex(grid)
    time = grid.tz_convert('Europe/Rome')
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    ax.plot(time, p.measured_ac_mw, color='black', lw=1.7, label='Curated measured AC')
    for frame, color, label in ((p, '#1473a6', f'Previous day: {prior_model}'),
                                 (s, '#d06a20', f'Same day: {same_model}')):
        members = frame[ids].to_numpy(float)
        valid = np.isfinite(members).all(axis=1)
        mean = np.full(len(frame), np.nan); median = mean.copy(); lo = mean.copy(); hi = mean.copy()
        if valid.any():
            mean[valid] = members[valid].mean(axis=1); median[valid] = np.median(members[valid], axis=1)
            lo[valid], hi[valid] = np.quantile(members[valid], [.1, .9], axis=1)
        ax.fill_between(time, lo, hi, color=color, alpha=.12, label=f'{label} p10–p90')
        ax.plot(time, mean, color=color, lw=1.5, label=f'{label} mean')
        ax.plot(time, median, color=color, lw=1, ls='--', label=f'{label} median')
    issue = pd.Timestamp(day, tz='UTC').tz_convert('Europe/Rome')
    available = issue + pd.Timedelta(hours=6)
    ax.axvspan(local_start, issue, color='gray', alpha=.12, label='Previous-cycle midnight fill')
    ax.axvline(available, color='#6b2578', lw=1, ls=':', label='Same-day run assumed available (+6 h)')
    ax.set(title=f'Farm {farm} — {day} — matched vintages (same-day retrospective)',
           xlabel='Local time (Europe/Rome)', ylabel='AC power (MW)')
    ax.grid(alpha=.2); ax.legend(ncol=3, fontsize=7)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140); plt.close(fig)


def matched_frames(a, b, ids):
    """Use identical eligible intervals and verify measured targets and review decisions."""
    a = a.loc[a.split.eq('validation')].set_index('interval_start_utc').sort_index()
    b = b.loc[b.split.eq('validation')].set_index('interval_start_utc').sort_index()
    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError('Duplicate validation timestamps')
    common = a.index.intersection(b.index)
    a, b = a.loc[common].copy(), b.loc[common].copy()
    if not np.allclose(a.measured_ac_mw, b.measured_ac_mw, equal_nan=True):
        raise ValueError('The two experiments do not use identical measured AC targets')
    if not a.eligible.equals(b.eligible) or not a.episode_excluded.equals(b.episode_excluded):
        raise ValueError('The two experiments have different target/exclusion decisions')
    mask = comparable(a, ids) & comparable(b, ids)
    return a.loc[mask].reset_index(), b.loc[mask].reset_index()


def compare_experiments(prior_config, same_config):
    """Save same-sample pvlib and separately validation-selected model comparisons."""
    prior_root, same_root = study_root(prior_config), study_root(same_config)
    prior_section, same_section = prior_config['pv_probabilistic_study'], same_config['pv_probabilistic_study']
    if prior_section.get('vintage_strategy', 'preceding_day') != 'preceding_day' or same_section.get('vintage_strategy') != 'same_day_retrospective':
        raise ValueError('Expected preceding-day and retrospective same-day experiment configurations')
    for key in ('start_day','train_end_day','end_day','dataset_periods','calibration_folds','quantiles','farm_ids'):
        if prior_section[key] != same_section[key]: raise ValueError(f'Experiment definitions differ on {key}')
    levels = list(map(float, prior_section['quantiles']))
    rows=[]; detailed=[]
    output=prior_root/'vintage_comparison';output.mkdir(parents=True,exist_ok=True)
    for farm in prior_section['farm_ids']:
        p_root, _, _, ids, rating = load_farm(prior_config, int(farm))
        s_root, _, _, same_ids, same_rating = load_farm(same_config, int(farm))
        if ids != same_ids or not np.isclose(rating, same_rating): raise ValueError('Farm/member definitions differ')
        physical_a = pd.read_parquet(p_root/'pvlib_paired_predictions.parquet')
        physical_b = pd.read_parquet(s_root/'pvlib_paired_predictions.parquet')
        pairs=[('pvlib', 'pvlib', 'all', physical_a, physical_b)]
        p_rank = pd.read_csv(p_root/'validation_rankings.csv')
        s_rank = pd.read_csv(s_root/'validation_rankings.csv')
        for method in ('member_mean','member_median','calibrated_quantiles'):
            p = p_rank.loc[p_rank['rank'].eq(1)&p_rank.method.eq(method)].iloc[0]
            s = s_rank.loc[s_rank['rank'].eq(1)&s_rank.method.eq(method)].iloc[0]
            pairs.append((p.trial,s.trial,method,pd.read_parquet(Path(p.trial_path)/'predictions.parquet'),
                          pd.read_parquet(Path(s.trial_path)/'predictions.parquet')))
        for p_name,s_name,selection,p_frame,s_frame in pairs:
            p_frame,s_frame = matched_frames(p_frame,s_frame,ids)
            if not len(p_frame): raise ValueError(f'No comparable validation rows for farm {farm} / {selection}')
            if selection in {'all', 'member_mean'}:
                for day in sorted(set(p_frame.delivery_day)):
                    plot_comparison(p_frame, s_frame, ids, farm, day, p_name, s_name,
                                    output / f'farm_{farm}' / day / f'{selection}.png')
            scores=[]
            for strategy,model,frame in [('preceding_day',p_name,p_frame),('same_day_retrospective',s_name,s_frame)]:
                table=score_frame(frame,ids,levels,rating,'matched_validation')
                table.insert(0,'model',model);table.insert(0,'strategy',strategy);table.insert(0,'selection',selection);table.insert(0,'farm_id',int(farm))
                detailed.append(table);scores.append(table.set_index('method'))
            keys=[('member_mean','rmse_mw','nrmse_percent_capacity'),('member_median','mae_mw','nmae_percent_capacity'),
                  ('calibrated_quantiles','crps_mw','crps_percent_capacity')] if selection=='all' else [
                  {'member_mean':('member_mean','rmse_mw','nrmse_percent_capacity'),
                   'member_median':('member_median','mae_mw','nmae_percent_capacity'),
                   'calibrated_quantiles':('calibrated_quantiles','crps_mw','crps_percent_capacity')}[selection]]
            for method,raw,norm in keys:
                left,right=scores[0].loc[method],scores[1].loc[method]
                rows.append({'farm_id':int(farm),'selection':selection,'method':method,'metric':raw,
                    'preceding_model':p_name,'same_day_model':s_name,'n_common_quarters':len(p_frame),
                    'preceding_mw':float(left[raw]),'same_day_mw':float(right[raw]),
                    'preceding_percent_capacity':float(left[norm]),'same_day_percent_capacity':float(right[norm])})
    comparison=pd.DataFrame(rows); detail=pd.concat(detailed,ignore_index=True)
    comparison.to_csv(output/'paired_comparison.csv',index=False)
    detail.to_csv(output/'paired_detailed_metrics.csv',index=False)
    lines=['# Matched forecast-vintage comparison','',
        'Both studies use the same 49 training and 12 validation delivery days and matched measured AC targets. '
        'The preceding-day study uses the previous 00 UTC run throughout. The same-day study uses the current 00 UTC '
        'run retrospectively after 00 UTC and the previous run before 00 UTC.', '',
        'Same-day weather is unavailable at that delivery day’s DA and IDA1 gates under the six-hour availability assumption. '
        'The model for each objective is selected independently on the same validation days, so comparisons are exploratory.', '',
        '| Farm | Selection | Score | Prior model | Same-day model | Matched quarters | Prior (% AC) | Same day (% AC) |',
        '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    for row in comparison.itertuples():
        lines.append(f'| {row.farm_id} | {row.selection} | {row.metric} | {row.preceding_model} | {row.same_day_model} | {row.n_common_quarters} | {row.preceding_percent_capacity:.2f} | {row.same_day_percent_capacity:.2f} |')
    lines.extend(['', 'Scores use all matched eligible quarters, including nighttime. Percentages divide by installed farm AC capacity. '
        'Calibrated CRPS is calculated from marginal quantiles by numerical quadrature. '
        'Raw member labels stitched across runs do not establish joint trajectory identity.', '',
        'Daily plots show the curated observation in black, preceding-day mean/median and p10–p90 in blue, '
        'and same-day mean/median and p10–p90 in orange. Gray shading marks the preceding-cycle midnight fill; '
        'the dotted vertical line marks the assumed same-day cycle availability six hours after issue. '
        'The same-day values before that line are retrospective.'])
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'Matched vintage comparison: {output}',flush=True)
    return output

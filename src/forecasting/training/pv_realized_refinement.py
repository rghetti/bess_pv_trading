"""Refine leading retrospective physical candidates with additional optimizer budget."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json

import pandas as pd

from ...config import load_config
from ...data import config_path
from ..training.pv_realized_search import ranking_path, run_trial


def main():
    """Resume refined fits and append scores to the frozen screen ranking."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    section, refine = config['pv_realized_irradiance_search'], config['pv_realized_refinement']
    root = config_path(config, section['output_root']) / section['run_id']
    screen_path = ranking_path(root, section, screen=True)
    if not screen_path.exists():
        screen_path.write_bytes(ranking_path(root, section).read_bytes())
    ranking = pd.read_csv(screen_path)
    jobs = []
    for farm in section['farm_ids']:
        top = ranking.loc[(ranking.farm_id == farm) & ranking.loss.ne('fixed')].head(int(refine['top_candidates_per_farm']))
        for row in top.itertuples(index=False):
            path = root / f'farm_{farm}' / 'trials' / f'{int(row.trial_id):04d}.json'
            original = json.loads(path.read_text())
            tuned = deepcopy(config)
            tuned['pv_realized_irradiance_search']['optimization_max_evaluations'] = int(refine['optimization_max_evaluations'])
            tuned['pv_realized_irradiance_search']['multistarts'] = int(refine['multistarts'])
            farm_pv = tuned['farms'][str(farm)]['pv']
            params = original['parameters']
            farm_pv['temperature'].update(u0=params['u0'], u1=params['u1'])
            farm_pv['output'].update(method='calibrated_scale', calibrated_coefficient=params['coefficient'])
            farm_pv.setdefault('dc', {})['gamma_pdc_per_c'] = params['gamma']
            jobs.append((farm, 10000 + int(row.trial_id), original['candidate'], tuned, int(row.trial_id)))
    plan = {'refinement': refine, 'jobs': [{'farm': f, 'trial': i, 'parent_trial': parent, 'candidate': c}
                                          for f, i, c, _, parent in jobs],
            'selection_caveat': ('leading screen candidates selected on full-period in-sample error'
                                 if section.get('selection_scope', 'validation') == 'full_period'
                                 else 'leading screen candidates selected on August validation')}
    (root / 'refinement_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    results = []
    with ProcessPoolExecutor(max_workers=int(refine['workers'])) as pool:
        pending = {pool.submit(run_trial, tuned, farm, trial, candidate, root): (farm, trial)
                   for farm, trial, candidate, tuned, _ in jobs}
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(f"Refined {len(results)}/{len(jobs)}: farm {result['farm_id']} trial {result['trial_id']}", flush=True)
    rows = [dict(farm_id=x['farm_id'], trial_id=x['trial_id'], loss=x['candidate']['loss'],
                 location=x['candidate']['location'], transposition=x['candidate']['transposition'],
                 bifacial=x['candidate']['bifacial_enabled'], **x['metrics']['daytime']) for x in results]
    pd.concat([ranking, pd.DataFrame(rows)], ignore_index=True).sort_values(
        ['farm_id', 'rmse_percent_capacity']).to_csv(ranking_path(root, section), index=False)


if __name__ == '__main__':
    main()

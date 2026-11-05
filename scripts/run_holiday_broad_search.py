"""Run isolated, single-threaded holiday-feature comparisons in parallel."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def run_candidate(job):
    """Run one baseline/holiday pair and retain its outputs and error log."""
    root, output, study, candidate = job
    target = output / study['id'] / candidate['id']
    target.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, 'scripts/evaluate_italian_holiday_feature.py',
               '--matrix', str(root / study['matrix']), '--output', str(target),
               '--family', candidate['family'], '--parameters-json', json.dumps(candidate['parameters'])]
    if 'anchor_index' in study:
        command.extend(['--anchor-index', str(study['anchor_index']),
                        '--anchor-reference', str(root / study['anchor_reference'])])
    environment = dict(os.environ)
    environment.update({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                        'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''})
    with (target / 'stdout.log').open('w') as stdout, (target / 'stderr.log').open('w') as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, env=environment, check=False)
    summary = target / 'summary.json'
    return {'study': study['id'], 'candidate': candidate['id'], 'returncode': result.returncode,
            'summary': json.loads(summary.read_text()) if result.returncode == 0 and summary.exists() else None}


def main():
    """Validate inputs, then run the configured bounded CPU search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())['holiday_broad_search']
    root = Path(config['project_root'])
    output = Path(config['output_root'])
    jobs = []
    for study in config['studies']:
        # Every worker shares an immutable source matrix and writes a distinct directory.
        if not (root / study['matrix']).is_file():
            raise FileNotFoundError(study['matrix'])
        if 'anchor_reference' in study and not (root / study['anchor_reference']).is_file():
            raise FileNotFoundError(study['anchor_reference'])
        jobs.extend((root, output, study, candidate) for candidate in study['candidates'])
    if args.check_only:
        print(json.dumps({'jobs': len(jobs), 'workers': config['workers'], 'input_check': 'passed'}))
        return
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing study: {output}')
    output.mkdir(parents=True)
    (output / 'config.yaml').write_text(args.config.read_text())
    code_paths = [Path(__file__), Path('scripts/evaluate_italian_holiday_feature.py'),
                  Path('src/forecasting/price_search_models.py'), Path('src/forecasting/italian_holidays.py')]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        results = list(pool.map(run_candidate, jobs))
    manifest = {'implementation_sha256': hashes, 'workers': config['workers'], 'results': results,
                'failures': sum(result['returncode'] != 0 for result in results)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'jobs': len(results), 'failures': manifest['failures']}))
    if manifest['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

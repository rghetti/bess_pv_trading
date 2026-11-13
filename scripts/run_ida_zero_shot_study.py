"""Launch prepared own-gate IDA studies sequentially, with independent GPU models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from bess_pv_trading.config import load_config


def main():
    """Plan by default; explicit --run launches the configured historical replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/zero_shot_ida.yaml')
    parser.add_argument('--root')
    parser.add_argument('--python-root', default='outputs/zero_shot_envs')
    parser.add_argument('--devices', nargs='+', default=['auto'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    settings = load_config(args.config)['zero_shot_ida']
    root = Path(args.root or settings['output'])
    commands = []
    for market in settings['auctions']:
        for window in settings['windows']:
            for variant in settings['variants']:
                directory = root / market / f'{window}d' / variant
                manifest = directory / 'inputs/manifest.json'
                if not manifest.exists():
                    raise FileNotFoundError(f'Prepare inputs first: {manifest}')
                command = [sys.executable, '-m', 'scripts.da_zero_shot_probe', 'run',
                    '--bundle', str(directory / 'inputs'), '--output', str(directory / 'run'),
                    '--python-root', args.python_root, '--devices', *args.devices,
                    '--batch-size', str(args.batch_size), '--threads', str(args.threads)]
                if args.run:
                    command.append('--run')
                if args.allow_download:
                    command.append('--allow-download')
                if args.preflight_only:
                    command.append('--preflight-only')
                report = [sys.executable, '-m', 'scripts.da_zero_shot_probe', 'report',
                    '--bundle', str(directory / 'inputs'), '--forecast-directory', str(directory / 'run'),
                    '--output', str(directory / 'run/report')]
                commands.append((command, report))
    print(json.dumps({'mode': 'run' if args.run else 'plan_only', 'studies': len(commands),
                      'commands': commands}, indent=2), flush=True)
    if args.run:
        for command, report in commands:
            subprocess.run(command, check=True)
            if not args.preflight_only:
                subprocess.run(report, check=True)
        if not args.preflight_only:
            subprocess.run([sys.executable, '-m', 'scripts.report_ida_zero_shot',
                            '--config', args.config, '--root', str(root)], check=True)


if __name__ == '__main__':
    main()

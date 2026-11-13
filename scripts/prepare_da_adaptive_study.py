"""Prepare separate adaptive DA inputs without running historical forecasts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Build history and NORD variants for each requested calendar-window cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', nargs='+', type=int, choices=[7, 28, 60], default=[7, 28, 60])
    parser.add_argument('--output', default='outputs/da_zero_shot_adaptive')
    args = parser.parse_args()
    root = Path(args.output)
    for days in args.windows:
        config = 'configs/research/zero_shot_da_history.yaml' if days == 28 else f'configs/research/zero_shot_da_history_{days}d.yaml'
        history, nord = root / f'{days}d/history/inputs', root / f'{days}d/nord/inputs'
        for destination, command in [(history, [sys.executable, '-m', 'scripts.da_zero_shot_probe', 'prepare', '--config', config, '--output', str(history)]),
                                     (nord, [sys.executable, '-m', 'scripts.prepare_da_nord_covariates', '--history-bundle', str(history), '--output', str(nord)])]:
            if destination.exists():
                if not (destination / 'manifest.json').is_file():
                    raise ValueError(f'Incomplete preparation; choose a fresh output root: {destination}')
                subprocess.run([sys.executable, '-m', 'scripts.da_zero_shot_probe', 'audit', '--bundle', str(destination), '--output', str(destination)], check=True)
                continue
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()

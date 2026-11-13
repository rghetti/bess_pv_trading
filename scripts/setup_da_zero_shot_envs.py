"""Create separate locked model environments on another machine; never edit bess."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    """Install pinned packages only in newly created virtual environments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError('The supplied package locks target Python 3.14; use a Python 3.14 interpreter')
    destination = Path(args.root).absolute()
    commands = []
    for name in ('chronos', 't0', 'timesfm'):
        environment = destination / name
        if environment.exists():
            raise FileExistsError(f'Choose a new root; environment already exists: {environment}')
        commands.append([sys.executable, '-m', 'venv', str(environment)])
        commands.append([str(environment / 'bin/python'), '-m', 'pip', 'install',
                         '--extra-index-url', 'https://download.pytorch.org/whl/cu126',
                         '-r', str(ROOT / 'requirements/zero_shot' / f'da_{name}.txt')])
    print(json.dumps({'commands': commands, 'install': args.install}, indent=2), flush=True)
    if args.install:
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()

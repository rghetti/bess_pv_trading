"""Verify native IDA preparation without launching model experiments."""

import json
import sys
from pathlib import Path
import threading
import time

import pytest

from scripts import run_ida_zero_shot_study as launcher


MODELS = {'chronos2': {'repository': 'example/model', 'revision': 'pinned'}}
DEFINITIONS = {'auctions': {'IDA1': {'decision_time': '14:45', 'decision_day_offset': -1}}}


def study_settings(tmp_path):
    """Provide a new window and the same semantic settings as a real study."""
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'manifest.json').write_text('{}')
    return {'output': str(tmp_path / 'studies'), 'snapshot': str(snapshot),
            'auctions': ['IDA1'], 'windows': [56], 'variants': ['history'],
            'auction_config': 'auctions.yaml', 'model_config': 'models.yaml',
            'start': '2025-10-01', 'end_exclusive': None,
            'minimum_context_days': 7, 'models': ['chronos2'], 'seed': 1729,
            'quantiles': [0.1, 0.5, 0.9], 'timezone': 'Europe/Rome', 'precision': 'float32',
            'november_holdout_start': '2026-11-01'}


def mock_configs(monkeypatch, settings):
    """Resolve all study/auxiliary configuration without touching real data."""
    def load(path):
        if str(path) == 'auctions.yaml':
            return {'price_feature_search': DEFINITIONS}
        if str(path) == 'models.yaml':
            return {'zero_shot_price_study': {'models': MODELS}}
        return {'zero_shot_ida': settings}
    monkeypatch.setattr(launcher, 'load_config', load)


def saved_bundle(path, settings, window):
    """Create a minimal valid pinned manifest with a concrete inferred endpoint."""
    path.mkdir(parents=True)
    values = {**settings, 'context_days': window, 'market': 'IDA1', 'gate': 'IDA1',
              'decision_time': '14:45', 'decision_day_offset': -1, 'end_exclusive': '2026-11-11'}
    manifest = {'settings': values, 'models': MODELS, 'files': {},
                'source_manifest_sha256': launcher.digest(Path(settings['snapshot']) / 'manifest.json')}
    (path / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_plan_lists_missing_window_without_preparing_or_inference(tmp_path, monkeypatch, capsys):
    """A new 56-day window can be inspected without requiring prepared files."""
    settings = study_settings(tmp_path)
    mock_configs(monkeypatch, settings)
    monkeypatch.setattr(sys, 'argv', ['ida'])
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('Plan executed a process'))
    monkeypatch.setattr(launcher, 'IDAInputs', lambda *a: pytest.fail('Plan loaded source inputs'))
    launcher.main()
    assert not Path(settings['output']).exists()
    assert '56d' in capsys.readouterr().out


def test_preparation_creates_missing_bundles_once_and_reuses_source_reader(tmp_path, monkeypatch):
    """Multiple new windows share one expensive source reader and stay independent."""
    settings = study_settings(tmp_path)
    settings['windows'] = [56, 84]
    mock_configs(monkeypatch, settings)
    readers, prepared = [], []
    def reader(*args):
        readers.append(args)
        return object()
    def prepare(inputs, destination, auction, window, variant, models):
        prepared.append((auction, window, variant))
        saved_bundle(destination, settings, window)
    monkeypatch.setattr(launcher, 'IDAInputs', reader)
    monkeypatch.setattr(launcher, 'prepare_ida_bundle', prepare)
    actions = launcher.ensure_inputs('study.yaml', settings['output'], settings)
    assert len(readers) == 1
    assert sorted(prepared) == [('IDA1', 56, 'history'), ('IDA1', 84, 'history')]
    assert [action['action'] for action in actions] == ['prepared', 'prepared']
    assert all((Path(action['path']) / 'manifest.json').is_file() for action in actions)
    assert not list(Path(settings['output']).rglob('.inputs-preparing-*'))


def test_input_preparation_uses_at_most_configured_workers(tmp_path, monkeypatch):
    """Independent pinned bundles are prepared concurrently up to the worker cap."""
    settings = study_settings(tmp_path)
    settings['windows'] = [7, 28, 56, 84, 90, 120]
    mock_configs(monkeypatch, settings)
    active = 0
    maximum = 0
    lock = threading.Lock()
    monkeypatch.setattr(launcher, 'IDAInputs', lambda *args: object())

    def prepare(inputs, destination, auction, window, variant, models):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        saved_bundle(destination, settings, window)
        with lock:
            active -= 1

    monkeypatch.setattr(launcher, 'prepare_ida_bundle', prepare)
    launcher.ensure_inputs('study.yaml', settings['output'], settings, workers=5)
    assert maximum == 5


def test_adding_window_does_not_invalidate_existing_bundle(tmp_path, monkeypatch):
    """Global orchestration window lists and inferred end dates are not fit settings."""
    settings = study_settings(tmp_path)
    old = {**settings, 'windows': [7]}
    path = Path(settings['output']) / 'IDA1/7d/history/inputs'
    saved_bundle(path, old, 7)
    settings['windows'] = [7, 56]
    mock_configs(monkeypatch, settings)
    monkeypatch.setattr(launcher, 'IDAInputs', lambda *a: pytest.fail('Read sources in plan'))
    actions = launcher.ensure_inputs('study.yaml', settings['output'], settings, prepare=False)
    assert [action['action'] for action in actions] == ['reuse', 'prepare']
    assert not (Path(settings['output']) / 'IDA1/56d').exists()


def test_changed_settings_or_corrupted_source_request_are_rejected(tmp_path, monkeypatch):
    """A pinned directory cannot silently be reused for a changed experiment."""
    settings = study_settings(tmp_path)
    path = Path(settings['output']) / 'IDA1/56d/history/inputs'
    manifest = saved_bundle(path, settings, 56)
    mock_configs(monkeypatch, settings)
    settings['seed'] += 1
    with pytest.raises(ValueError, match='configuration mismatch'):
        launcher.ensure_inputs('study.yaml', settings['output'], settings, prepare=False)
    settings['seed'] -= 1
    manifest['files'] = {'request.json': 'wrong_checksum'}
    (path / 'manifest.json').write_text(json.dumps(manifest))
    (path / 'request.json').write_text('{}')
    with pytest.raises(ValueError, match='Changed pinned input'):
        launcher.ensure_inputs('study.yaml', settings['output'], settings, prepare=False)


def test_incomplete_directory_is_preserved_and_failed_preparation_is_atomic(tmp_path, monkeypatch):
    """Failures do not overwrite existing data or leave a reusable partial bundle."""
    settings = study_settings(tmp_path)
    path = Path(settings['output']) / 'IDA1/56d/history/inputs'
    path.mkdir(parents=True)
    (path / 'unfinished').write_text('preserve me')
    mock_configs(monkeypatch, settings)
    with pytest.raises(ValueError, match='Incomplete existing'):
        launcher.ensure_inputs('study.yaml', settings['output'], settings)
    assert (path / 'unfinished').read_text() == 'preserve me'
    path.rename(path.with_name('old-partial'))
    monkeypatch.setattr(launcher, 'IDAInputs', lambda *a: object())
    def failure(*args):
        raise RuntimeError('interrupted preparation')
    monkeypatch.setattr(launcher, 'prepare_ida_bundle', failure)
    with pytest.raises(RuntimeError, match='interrupted'):
        launcher.ensure_inputs('study.yaml', settings['output'], settings)
    assert not path.exists()
    assert not list(path.parent.glob('.inputs-preparing-*'))


def test_prepare_only_never_starts_model_subprocesses(tmp_path, monkeypatch):
    """The integrated CLI can stop after input preparation for user inspection."""
    settings = study_settings(tmp_path)
    mock_configs(monkeypatch, settings)
    calls = []
    monkeypatch.setattr(launcher, 'ensure_inputs', lambda *a, **k: calls.append(k['prepare']) or [])
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('Preparation launched inference'))
    monkeypatch.setattr(sys, 'argv', ['ida', '--prepare-only'])
    launcher.main()
    assert calls == [True]

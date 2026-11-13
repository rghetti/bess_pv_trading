"""Verify configurable DA windows are prepared and streamed in dependency order."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from scripts import run_da_zero_shot_studies as launcher


def test_new_context_window_prepares_history_then_nord_and_streams_both(tmp_path, monkeypatch):
    """A YAML-added window needs no hand-written per-window preparation command."""
    root = tmp_path / 'studies'
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'manifest.json').write_text('{}')
    model_specs = {'chronos2': {'revision': 'pinned'}}
    settings = {
        'root': str(root), 'snapshot': str(snapshot), 'base_config': 'base.yaml',
        'feature_search_config': 'features.yaml', 'model_config': 'models.yaml',
        'windows': [60, 84], 'baseline_variants': ['history', 'nord'],
        'models': ['chronos2'], 'cross_border_window': 60,
        'cross_border_base_variant': 'nord', 'cross_border_variants': {},
    }
    monkeypatch.setattr(launcher, 'load_config', lambda path: {
        'zero_shot_price_study': {'models': model_specs},
        'zero_shot_da_probe': {'models': ['chronos2']},
    })
    monkeypatch.setattr(launcher, 'ProcessPoolExecutor', ThreadPoolExecutor)

    for variant in ('history', 'nord'):
        path = root / '60d' / variant / 'inputs'
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text('{}')

    def verify(bundle):
        bundle = Path(bundle)
        days = int(bundle.parents[1].name.removesuffix('d'))
        return {'settings': {'context_days': days, 'market': 'DA'}, 'models': model_specs,
                'history_bundle_manifest_sha256': 'same'}

    def prepare(action, snapshot_path, base_config):
        output = Path(action['path'])
        output.mkdir(parents=True)
        (output / 'manifest.json').write_text(json.dumps({'variant': action['variant']}))
        return {'study': action['study'], 'path': action['path']}

    streamed = []
    monkeypatch.setattr(launcher, 'verify_bundle', verify)
    monkeypatch.setattr(launcher, '_prepare_core_action', prepare)
    monkeypatch.setattr(launcher, 'digest', lambda path: 'same')
    actions = launcher.prepare_study_inputs(settings, root, workers=2, on_ready=streamed.append)

    assert [Path(item).name for item in streamed[-2:]] == ['history', 'nord']
    assert [(item['window_days'], item['variant'], item['action']) for item in actions[-2:]] == [
        (84, 'history', 'prepared'), (84, 'nord', 'prepared')]

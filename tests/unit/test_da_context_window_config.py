"""Check the DA input preparer accepts context windows from study config."""

import sys

from scripts import da_zero_shot_probe


def test_prepare_cli_overrides_context_days_from_launcher_config(monkeypatch, tmp_path, capsys):
    """A configurable 84-day bundle reaches the shared input builder unchanged."""
    settings = {'minutes': 15, 'context_days': 28, 'precision': 'float32',
                'minimum_context_days': 7, 'model_config': 'models.yaml', 'models': ['chronos2']}
    monkeypatch.setattr(da_zero_shot_probe, 'load_config',
                        lambda path: {'zero_shot_da_probe': settings,
                                      'zero_shot_price_study': {'models': {'chronos2': {'revision': 'pinned'}}}})
    captured = {}

    def prepare(snapshot, output, received_settings, models):
        captured['context_days'] = received_settings['context_days']
        captured['models'] = models
        return {'eligible_days': 1, 'requested_days': 1, 'features': ['past_NORD_DA_price']}

    monkeypatch.setattr(da_zero_shot_probe, 'prepare_bundle', prepare)
    monkeypatch.setattr(sys, 'argv', ['probe', 'prepare', '--config', 'study.yaml',
        '--snapshot', 'pinned-snapshot', '--output', str(tmp_path / 'inputs'), '--context-days', '84'])
    da_zero_shot_probe.main()
    assert captured['context_days'] == 84
    assert captured['models'] == {'chronos2': {'revision': 'pinned'}}
    assert 'eligible_days' in capsys.readouterr().out

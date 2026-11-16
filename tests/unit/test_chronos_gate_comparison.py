"""Verify that gate-error comparisons use identical products and provenance."""

import json

import pandas as pd
import pytest

from scripts.report_chronos_gate_comparison import compare_target, match_gate_forecasts


def forecast_frame(gate, products):
    """Construct simple gate-labelled forecasts for one auction."""
    return pd.DataFrame([{'delivery_day': '2025-10-01',
                          'product_start': pd.Timestamp('2025-10-01T12:00Z') + pd.Timedelta(minutes=15 * product),
                          'auction': 'IDA3', 'gate': gate, 'variant': 'chronos2',
                          'period': 'retrospective_development', 'actual': 100.0 + product,
                          'prediction': 102.0 + product} for product in products])


def test_gate_comparison_uses_only_identical_products():
    """Different gate coverage must never masquerade as a performance change."""
    frames = {'DA': forecast_frame('DA', [0, 1]),
              'IDA1': forecast_frame('IDA1', [0, 1, 2]),
              'IDA2': forecast_frame('IDA2', [0, 2])}
    matched = match_gate_forecasts(frames)
    assert len(matched) == 3
    assert matched.product_start.nunique() == 1
    assert matched.groupby('gate').size().to_dict() == {'DA': 1, 'IDA1': 1, 'IDA2': 1}
    assert (matched.error == 2.0).all()


def test_gate_comparison_rejects_different_target_labels():
    """A price revision cannot be silently mixed across gate reports."""
    other = forecast_frame('IDA1', [0])
    other.loc[0, 'actual'] = 110.0
    with pytest.raises(ValueError, match='labels disagree'):
        match_gate_forecasts({'DA': forecast_frame('DA', [0]), 'IDA1': other})


def write_report(root, gate, products, calendar=False):
    """Write a minimal complete report with a consistent source contract."""
    folder = root / gate / 'IDA3'
    report = folder / 'run/report'
    report.mkdir(parents=True)
    (folder / 'inputs').mkdir()
    manifest = {'settings': {'market': 'IDA3', 'gate': gate, 'context_days': 28,
                             'timezone': 'Europe/Rome'},
                'calendar': {'italian_holiday': calendar}, 'models': {'chronos2': {}}}
    (folder / 'inputs/manifest.json').write_text(json.dumps(manifest))
    (report / 'status.json').write_text(json.dumps({'complete': True}))
    forecast_frame(gate, products).to_parquet(report / 'matched_forecasts.parquet')


def test_partial_gate_report_plots_matching_products_and_marks_missing_gate(tmp_path):
    """Reports retain row counts and cannot claim all requested gates completed."""
    write_report(tmp_path, 'DA', [0, 1])
    write_report(tmp_path, 'IDA1', [0])
    result = compare_target(tmp_path, 'IDA3', ['DA', 'IDA1', 'IDA2'], 'Europe/Rome')
    assert result['status'] == 'partial'
    output = tmp_path / 'gate_comparison/IDA3'
    coverage = pd.read_csv(output / 'coverage.csv').set_index('gate')
    assert coverage.loc['DA', 'available_products'] == 2
    assert coverage.loc['DA', 'matched_products'] == 1
    assert coverage.loc['IDA2', 'status'] == 'missing_report'
    assert (output / 'gate_mae_retrospective_development_chronos2.png').is_file()
    assert (output / 'daily_mae_retrospective_development_chronos2.png').is_file()
    assert (output / 'hourly_mae_retrospective_development_chronos2.png').is_file()
    assert (output / 'daily_errors/2025-10/2025-10-01_retrospective_development_chronos2.png').is_file()
    assert not json.loads((output / 'status.json').read_text())['complete']


def test_gate_comparison_rejects_mixed_feature_configuration(tmp_path):
    """An old calendar-enabled run cannot be compared with a no-calendar run."""
    write_report(tmp_path, 'DA', [0], calendar=False)
    write_report(tmp_path, 'IDA1', [0], calendar=True)
    with pytest.raises(ValueError, match='Incompatible model'):
        compare_target(tmp_path, 'IDA3', ['DA', 'IDA1'], 'Europe/Rome')


def test_completed_direct_report_triggers_gate_comparison(tmp_path, monkeypatch):
    """The existing runner generates the cross-gate report after each direct report."""
    from scripts import report_chronos_gate_comparison, run_timesfm_realized_study

    directory = tmp_path / 'DA' / 'IDA3'
    (directory / 'inputs').mkdir(parents=True)
    settings = {'market': 'IDA3', 'gate': 'DA', 'timezone': 'Europe/Rome',
                'gates': {'DA': ['IDA3'], 'IDA1': ['IDA3']}}
    (directory / 'inputs/manifest.json').write_text(json.dumps({
        'direct_contract_sha256': 'synthetic', 'settings': settings}))
    calls = []
    monkeypatch.setattr(run_timesfm_realized_study, 'report',
                        lambda *args: calls.append(('report', args)))
    monkeypatch.setattr(report_chronos_gate_comparison, 'compare_target',
                        lambda *args, **kwargs: calls.append(('compare', args, kwargs)))
    run_timesfm_realized_study.render(directory)
    assert [item[0] for item in calls] == ['report', 'compare']
    assert calls[1][1][:4] == (tmp_path, 'IDA3', ['DA', 'IDA1'], 'Europe/Rome')
    assert calls[1][2]['require_complete'] is True

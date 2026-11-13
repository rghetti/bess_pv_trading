"""Check six-way matched context-window scoring without model inference."""

import json
import sys

import pandas as pd
import pytest

from scripts.compare_da_context_windows import main


def test_six_way_comparison_excludes_unmatched_products(tmp_path, monkeypatch):
    """A price product absent from any variant cannot influence any score."""
    monkeypatch.chdir(tmp_path)
    paths = ['outputs/da_zero_shot_run_7d/report', 'outputs/da_zero_shot_run/report',
             'outputs/da_zero_shot_run_60d/report', 'outputs/da_zero_shot_nord_run_7d/report',
             'outputs/da_zero_shot_nord_run/report', 'outputs/da_zero_shot_nord_run_60d/report']
    for index, name in enumerate(paths):
        directory = tmp_path / name
        directory.mkdir(parents=True)
        (directory / 'status.json').write_text(json.dumps({'complete': True}))
        frame = pd.DataFrame([{'delivery_day': '2025-10-01', 'product_start': f'2025-10-01T00:{minute:02d}:00Z',
                               'auction': 'DA', 'gate': 'DA', 'variant': 'model', 'period': 'retrospective_development',
                               'actual': 10., 'prediction': 11.} for minute in ([0] if index == 0 else [0, 15])])
        frame.to_parquet(directory / 'matched_forecasts.parquet')
    monkeypatch.setattr(sys, 'argv', ['compare'])
    main()
    result = pd.read_csv('outputs/da_zero_shot_context_comparison/scores.csv')
    assert len(result) == 6
    assert result.products.eq(1).all()
    assert result.mae.eq(1).all()
    (tmp_path / paths[0] / 'status.json').write_text(json.dumps({'complete': False}))
    with pytest.raises(ValueError, match='incomplete'):
        main()

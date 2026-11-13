"""Add known Italian holiday indicators to pinned IDA model requests."""

import json
from pathlib import Path

import pandas as pd

from bess_pv_trading.forecasting.data.pv_calendar import italian_national_holiday
from bess_pv_trading.forecasting.studies.zero_shot_da import digest, write_json


def add_holidays(bundle, variant):
    """Add holiday flags for every historical and delivery product in a bundle."""
    root = Path(bundle)
    manifest = json.loads((root / 'manifest.json').read_text())
    timezone = manifest['settings']['timezone']
    # The holiday calendar is known for both the history and forecast horizon.
    for relative in list(manifest['files']):
        if not relative.startswith('requests/'):
            continue
        path = root / relative
        request = json.loads(path.read_text())
        history = request['context_product_starts_utc']
        horizon = request['product_starts_utc']
        starts = pd.DatetimeIndex(pd.to_datetime(history + horizon, utc=True))
        flags = italian_national_holiday(starts, timezone).astype(float).tolist()
        request['forecast_covariates']['italian_national_holiday'] = flags
        request['feature_names'].append('italian_national_holiday')
        write_json(path, request)
        manifest['files'][relative] = digest(path)
    manifest['features'].append('italian_national_holiday')
    manifest['feature_variant'] = variant
    manifest['holiday_definition'] = 'Italian national holiday indicator at each local delivery product; known calendar.'
    write_json(root / 'manifest.json', manifest)

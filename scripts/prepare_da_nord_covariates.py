"""Pin NORD load and generation forecast profiles for the DA zero-shot study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.price_feature_data import interval_values
from bess_pv_trading.forecasting.studies.zero_shot_da import digest, write_json
from bess_pv_trading.forecasting.studies.adaptive_price_context import adapt_request


FEATURES = ('load_forecast', 'generation_forecast', 'solar_forecast', 'wind_onshore_forecast')


def main():
    """Attach complete historical and delivery-day forecast profiles to DA requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-bundle', default='outputs/da_zero_shot_inputs')
    parser.add_argument('--snapshot', default='outputs/zero_shot_price_inputs/merged')
    parser.add_argument('--output', default='outputs/da_zero_shot_nord_inputs')
    args = parser.parse_args()
    history, snapshot, output = Path(args.history_bundle), Path(args.snapshot), Path(args.output)
    base = json.loads((history / 'manifest.json').read_text())
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    if digest(snapshot / 'manifest.json') != base['source_manifest_sha256']:
        raise ValueError('Forecast and history bundles must use the same pinned snapshot')
    entry = manifest['tables']['nord_forecasts']
    source = snapshot / entry['path']
    if digest(source) != entry['sha256']:
        raise ValueError('NORD forecast source checksum mismatch')
    if output.exists():
        raise FileExistsError(f'Choose a new bundle path: {output}')
    frame = pd.read_parquet(source)
    profiles = {}
    for name in FEATURES:
        part = frame.loc[frame.dataset.eq(name) & frame.zone.eq('NORD')].copy().sort_values('product_start')
        if part.empty or not part.unit.eq('MW').all():
            raise ValueError(f'Missing or unexpected units for {name}')
        profiles[name] = part
    output.mkdir(parents=True)
    (output / 'requests').mkdir()
    files, coverage = {}, []
    for name in ('labels.parquet', 'source_da.parquet', 'source_manifest.json'):
        shutil.copy2(history / name, output / name)
        files[name] = digest(output / name)
    # Align forecast values with actual UTC intervals, retaining DST and native source resolution.
    for relative, expected in base['files'].items():
        if not relative.startswith('requests/'):
            continue
        source_request = history / relative
        if digest(source_request) != expected:
            raise ValueError(f'History request checksum mismatch: {relative}')
        request = json.loads(source_request.read_text())
        start = pd.Timestamp(request['context_start'])
        target = pd.DatetimeIndex(pd.to_datetime(request['product_starts_utc'], utc=True))
        grid = pd.date_range(start, target[-1] + pd.Timedelta(minutes=15), freq='15min', inclusive='left')
        forecasts = {}
        missing = {}
        for feature, part in profiles.items():
            values = interval_values(part, grid, pd.to_numeric(part.value, errors='coerce').to_numpy(float))
            missing[feature] = int((~np.isfinite(values)).sum())
            forecasts[feature] = values.tolist()
        request['forecast_covariates'] = forecasts
        request['feature_names'] = ['past_NORD_DA_price', *FEATURES]
        request, adjustment = adapt_request(request, base['settings'].get('minimum_context_days', 7), base['settings']['timezone'], base['settings'].get('missing_context', 'exclude_day'))
        coverage.append({'delivery_day': Path(relative).stem, 'eligible': request is not None, **missing, **adjustment})
        if request is None:
            continue
        path = output / relative
        write_json(path, request)
        files[relative] = digest(path)
    pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
    files['coverage.csv'] = digest(output / 'coverage.csv')
    result = dict(base)
    result.update(files=files, eligible_days=sum(row['eligible'] for row in coverage),
        requested_days=len(coverage), features=['past_NORD_DA_price', *FEATURES],
        source_forecasts_sha256=entry['sha256'], history_bundle_manifest_sha256=digest(history / 'manifest.json'),
        forecast_availability_assumption='Archived ENTSO-E forecasts assumed available by previous-day DA gate; historical issue vintages are not verified.',
        forecast_missing_policy='Configured complete-case or adaptive full-day historical suffix; delivery covariates must be complete; no imputation.')
    write_json(output / 'manifest.json', result)
    print(json.dumps({'eligible_days': result['eligible_days'], 'requested_days': result['requested_days'],
                      'features': result['features']}, indent=2))


if __name__ == '__main__':
    main()

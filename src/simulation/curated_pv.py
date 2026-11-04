"""Load reviewed PV targets for counterfactual optimization replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import config_path, night_window_mask


def _sha(path):
    """Fingerprint source files used in the simulated production trajectory."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_curated_realized_pv(config, farm_id, grid):
    """Use reviewed labels and fill retained daylight exclusions with pvlib."""
    section = config['forecast']['curated_realized']
    root = config_path(config, section['dataset_root'])
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    farm_root = root / f'farm_{farm_id}'
    corpus_path = farm_root / 'corpus.parquet'
    view_path = farm_root / 'modeling_views.parquet'
    if _sha(corpus_path) != manifest['farms'][str(farm_id)]['corpus_sha256']:
        raise ValueError(f'Farm {farm_id}: curated corpus checksum mismatch')
    corpus = pd.read_parquet(corpus_path)
    view = pd.read_parquet(view_path).reset_index(drop=True)
    if not view.equals(corpus.loc[corpus['split'].ne('outside')].reset_index(drop=True)):
        raise ValueError(f'Farm {farm_id}: curated modeling view differs from its checked corpus')
    starts = pd.DatetimeIndex(grid.starts)
    source_times = pd.DatetimeIndex(view.interval_start_utc)
    if source_times.has_duplicates or not source_times.is_monotonic_increasing:
        raise ValueError(f'Farm {farm_id}: curated timestamps are not unique and ordered')
    positions = source_times.get_indexer(starts)
    if (positions < 0).any():
        raise ValueError(f'Farm {farm_id}: curated dataset does not cover every requested interval')
    rows = view.iloc[positions]
    policy = config['policy']['realized_pv_night']
    if not policy.get('enabled', True) or 'monthly_windows' not in policy:
        raise ValueError('Curated replay requires the enabled month-specific night policy')
    night = night_window_mask(starts, policy)
    if not np.array_equal(night, rows.night.to_numpy(dtype=bool)):
        raise ValueError('Curated and configured month-specific night windows disagree')
    actual = rows.model_target_mw.to_numpy(dtype=float).copy()
    sources = rows.model_target_source.to_numpy(dtype=object).copy()
    # The approved target already includes user-requested replacements. Only
    # previously retained exclusions need the additional pvlib reconstruction.
    excluded = rows.retained_prior_exclusion.to_numpy(dtype=bool) & ~night
    reference = rows.pvlib_historical_ghi_mw.to_numpy(dtype=float)
    actual[excluded] = reference[excluded]
    sources[excluded] = 'pvlib_retained_exclusion'
    actual[night] = 0.0
    sources[night] = 'monthly_night_zero'
    invalid = ~np.isfinite(actual) | (actual < 0)
    if invalid.any():
        missing = [str(grid.interval_ids[i]) for i in np.flatnonzero(invalid)]
        raise ValueError(f'Farm {farm_id}: curated daylight targets remain unresolved: {missing}')
    counts = pd.Series(sources).value_counts().to_dict()
    provenance = {'source': 'reviewed_curated_counterfactual', 'dataset_root': str(root),
        'manifest_sha256': _sha(manifest_path), 'corpus_sha256': _sha(corpus_path),
        'modeling_view_sha256': _sha(view_path), 'source_counts': counts,
        'interval_sources': [{'interval_id': str(identifier), 'source': str(source)}
                             for identifier, source in zip(grid.interval_ids, sources)],
        'interpretation': 'retrospective counterfactual; synthetic values are not observed production',
        'forecast_fallback_interval_ids': []}
    return actual, provenance

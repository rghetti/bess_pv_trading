"""Independently verify every completed December price-study execution ledger."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from bess_pv_trading.config import load_config
from bess_pv_trading.simulation.continuous_inputs import _fingerprint
from run_december_price_comparison import FARMS, MODES, OUTPUT, config_path, expected_profiles


def validate():
    """Reconstruct pinned input fingerprints and realized cash for every profile."""
    results = []
    for farm in FARMS:
        for mode in MODES:
            config = load_config(config_path(farm, mode))
            directory = OUTPUT / f'farm_{farm}' / f'december_2025_price_{mode}_v1'
            provenance = json.loads((directory / 'json/input_provenance.json').read_text())
            manifest = json.loads((directory / 'json/manifest.json').read_text())
            if manifest['status'] != 'complete':
                raise ValueError(f'Incomplete source arm: {directory}')
            for profile in expected_profiles(config):
                path = directory / 'strategies' / profile / 'csv'
                dispatch = pd.read_csv(path / 'realized_dispatch.csv', float_precision='round_trip')
                trades = pd.read_csv(path / 'accepted_trades.csv', float_precision='round_trip')
                if dispatch.interval_id.duplicated().any() or trades.duplicated(['auction', 'interval_id']).any():
                    raise ValueError(f'Duplicate interval or trade in {farm}/{mode}/{profile}')
                if _fingerprint(dispatch.imbalance_price_eur_mwh.to_numpy(float)) != provenance['settlement_values_sha256']:
                    raise ValueError(f'Settlement price fingerprint differs in {farm}/{mode}/{profile}')
                if not np.allclose(dispatch.imbalance_mwh * dispatch.imbalance_price_eur_mwh,
                                   dispatch.imbalance_cashflow_eur, rtol=1e-9, atol=1e-7):
                    raise ValueError(f'Incorrect actual imbalance settlement in {farm}/{mode}/{profile}')
                if not np.allclose(trades.quantity_mwh * trades.price_eur_per_mwh,
                                   trades.cashflow_eur, rtol=1e-9, atol=1e-7):
                    raise ValueError(f'Incorrect actual auction cashflow in {farm}/{mode}/{profile}')
                # Each accepted product price must reconstruct the preflighted
                # actual curve, including zero-valued noneligible IDA3 slots.
                auctions = ('DA', 'IDA1', 'IDA2', 'IDA3') if profile == 'sequential_all_auctions' else tuple(profile.split('+'))
                price_checks = 0
                for day, day_dispatch in dispatch.groupby('delivery_day', sort=True):
                    day_provenance = provenance['days'][day]
                    if _fingerprint(day_dispatch.pv_ac_mw.to_numpy(float)) != day_provenance['realized_pv_sha256']:
                        raise ValueError(f'Realized PV fingerprint differs on {day} in {farm}/{mode}/{profile}')
                    day_trades = trades.loc[trades.delivery_day.eq(day)]
                    expected = set(auctions) & set(day_provenance['auction_price_sha256'])
                    if set(day_trades.auction) != expected:
                        raise ValueError(f'Wrong participation on {day} in {farm}/{mode}/{profile}')
                    for auction, group in day_trades.groupby('auction'):
                        count = 48 if auction == 'IDA3' else 96
                        if len(group) != count:
                            raise ValueError(f'Incomplete {auction} products on {day} in {farm}/{mode}/{profile}')
                        prices = group.set_index('interval_id').price_eur_per_mwh.reindex(
                            day_dispatch.interval_id, fill_value=0.0).to_numpy(float)
                        if _fingerprint(prices) != day_provenance['auction_price_sha256'][auction]:
                            raise ValueError(f'Actual {auction} price fingerprint differs on {day} in {farm}/{mode}/{profile}')
                        price_checks += 1
                if len(dispatch) != 31 * 96 or len(set(dispatch.delivery_day)) != 31:
                    raise ValueError(f'Incomplete execution month in {farm}/{mode}/{profile}')
                results.append({'farm': farm, 'mode': mode, 'profile': profile,
                    'intervals': len(dispatch), 'trade_rows': len(trades),
                    'auction_price_curve_checks': price_checks,
                    'pv_daily_fingerprint_checks': 31, 'settlement_fingerprint_checks': 1})
            print(f'Validated all actual-price and PV ledgers: farm {farm}, {mode}', flush=True)
    totals = Counter()
    for row in results:
        for key in ('intervals', 'trade_rows', 'auction_price_curve_checks',
                    'pv_daily_fingerprint_checks', 'settlement_fingerprint_checks'):
            totals[key] += row[key]
    evidence = {'status': 'passed', 'trajectory_count': len(results), 'totals': dict(totals), 'trajectories': results}
    (OUTPUT / 'ledger_validation.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(f'Validated {len(results)} trajectories; evidence: {OUTPUT / "ledger_validation.json"}', flush=True)


if __name__ == '__main__':
    validate()

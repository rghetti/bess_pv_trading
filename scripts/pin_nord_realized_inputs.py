"""Pin actual NORD observations from a finished ENTSO-E catalog download."""

import argparse
from pathlib import Path

from bess_pv_trading.config import load_config
from bess_pv_trading.data.price_search_inputs import build_price_search_inputs


def main():
    """Export actual observations and snapshot identifiers without touching the database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/price_search_inputs_nord_realized.yaml')
    parser.add_argument('--database', default='data/api/market.duckdb')
    parser.add_argument('--output', default='outputs/price_realized_snapshot/merged')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Use a fresh immutable snapshot path: {output}')
    config = load_config(args.config)
    settings = config['price_search_inputs']
    settings['database_path'] = args.database
    settings['output_directory'] = str(output)
    settings['sources'] = [item for item in settings['sources'] if item['name'] == 'nord_actuals']
    settings['sources'][0]['required'] = True
    manifest = build_price_search_inputs(config)
    print(f"Pinned {manifest['tables']['nord_actuals']['rows']} rows to {output}", flush=True)


if __name__ == '__main__':
    main()

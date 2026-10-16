"""Regenerate a tiny synthetic price fixture; these are never market observations."""
from pathlib import Path
import numpy as np
import pandas as pd

from bess_pv_trading.markets import AuctionCalendar


def generate() -> Path:
    """Make 18 days with four selected noon/evening products for each auction."""
    calendar = AuctionCalendar()
    rng = np.random.default_rng(29)
    rows = []
    for day_number, day in enumerate(pd.date_range("2025-10-01", "2025-10-18")):
        date = day.date()
        grid = calendar.delivery_grid(date)
        products = [48, 49, 72, 73]
        prices = 65 + 0.3 * day_number + 10 * np.sin(np.array(products) * 2 * np.pi / 96)
        for event in calendar.schedule(date):
            if event.auction != "DA":
                prices = prices + {"IDA1": -2, "IDA2": 1, "IDA3": 3}[event.auction]
            prices = prices + rng.normal(0, 2, len(products))
            for t, price in zip(products, prices):
                rows.append({"provider": "synthetic_fixture", "dataset": "synthetic_conditional_auction_prices_seed29",
                    "zone": "NORD", "auction": event.auction, "product_start": grid.starts[t].isoformat(),
                    "product_end": grid.ends[t].isoformat(), "price_eur_mwh": round(price, 6),
                    "published_at": event.results_at.isoformat(),
                    "publication_provenance": "synthetic_clock_fixture_not_observed_publication", "revision": "fixture-v1"})
    path = Path(__file__).with_name("synthetic_market_prices.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print(generate())

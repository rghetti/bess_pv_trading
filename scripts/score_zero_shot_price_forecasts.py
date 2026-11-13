"""Score saved zero-shot forecasts, with optional strict variant matching."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bess_pv_trading.forecasting.studies.zero_shot_price_scoring import matched_predictions, score


def main():
    """Write aggregate, monthly and local-hour scores from a prediction table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Parquet table with realized labels and saved forecasts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--match-variants", nargs="+")
    parser.add_argument("--timezone", default="Europe/Rome")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    required = {"delivery_day", "product_start", "auction", "gate", "variant", "actual", "prediction"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing required forecast columns: {sorted(missing)}")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    before = len(frame)
    if args.match_variants:
        frame = matched_predictions(frame, args.match_variants)
    frame["month"] = pd.to_datetime(frame.delivery_day).dt.strftime("%Y-%m")
    frame["hour"] = pd.to_datetime(frame.product_start, utc=True).dt.tz_convert(args.timezone).dt.hour
    for name, dimensions in (("overall", []), ("monthly", ["month"]), ("hourly", ["hour"])):
        rows = []
        groups = ["auction", "gate", "variant", *dimensions]
        for values, group in frame.groupby(groups, dropna=False, sort=True):
            row = dict(zip(groups, values))
            row.update(score(group))
            rows.append(row)
        pd.DataFrame(rows).to_csv(destination / f"{name}.csv", index=False)
    print(f"Input rows: {before}; scored rows: {len(frame)}; output: {destination}")


if __name__ == "__main__":
    main()

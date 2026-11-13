"""Pin catalog sources for the rolling auction-price study without fitting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bess_pv_trading.config import load_config
from scripts.prepare_full_year_price_inputs import prepare_merged_inputs


def main():
    """Export one consistent catalog selection and include the city-weather archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/rolling_price_inputs.yaml")
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    options = load_config(args.config)["rolling_price_inputs"]
    root = Path(args.output).resolve()
    if not args.prepare:
        print(json.dumps({"action": "plan_only", "database": args.database,
            "output": str(root), "start": options["start"], "end_exclusive": options["end_exclusive"],
            "weather": "open_meteo city cycles; missing archive is recorded"}, indent=2))
        return
    config = load_config(options["base_inputs_config"])
    inputs = config["price_search_inputs"]
    inputs.update(database_path=str(Path(args.database).resolve()), start=options["start"],
                  end_exclusive=options["end_exclusive"])
    inputs["excluded_providers"] = [name for name in inputs["excluded_providers"] if name != "open_meteo"]
    # Commercial border spreads can embed the Italian target price and are not study features.
    inputs["sources"] = [source for source in inputs["sources"] if source["name"] != "swiss_commercial_border"]
    inputs["sources"].append({"name": "regional_weather", "provider": "open_meteo", "zone": "NORD",
        "datasets": ["weather_forecast"], "date_column": "valid_time", "required": False,
        "columns": ["provider", "dataset", "zone", "cycle_time", "valid_time", "location_id",
            "variable", "value", "unit", "temporal_semantics", "historical_available_at",
            "availability_provenance"], "role": "city_forecast_vintages"})
    result = prepare_merged_inputs(config, root / "raw", root / "merged")
    print(json.dumps({"status": result["status"], "manifest": str(root / "merged" / "manifest.json"),
        "weather_rows": result["tables"]["regional_weather"]["rows"],
        "weather_release_lag_hours": options["weather_release_lag_hours"]}, indent=2))


if __name__ == "__main__":
    main()

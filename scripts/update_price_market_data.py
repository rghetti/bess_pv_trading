"""Refresh configured price-study sources sequentially and report missing inputs.

The default action prints a plan. --run makes public API requests and updates
the specified DuckDB. Credentials are read only from configured environment
variables; absent credentials are reported, never saved or printed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from bess_pv_trading.config import load_config


def _summary(provider, result):
    """Keep the report small while retaining each provider's completeness evidence."""
    if isinstance(result, dict):
        return {"provider": provider, "status": result.get("status", "unknown"),
                "details": result}
    return {"provider": provider, "status": "unknown", "details": str(result)}


def _redacted(error, secret_names):
    """Prevent configured credentials from entering an error report."""
    message = f"{type(error).__name__}: {error}"
    for name in secret_names:
        value = os.environ.get(name)
        if value:
            message = message.replace(value, "<redacted>")
    return message[:1000]


def _load_entsoe_token(name, *, interactive):
    """Use a configured token or request it privately in an interactive terminal."""
    if os.environ.get(name):
        return True
    if interactive and sys.stdin.isatty():
        value = getpass.getpass(f"{name} (hidden; press Enter to skip): ").strip()
        if value:
            os.environ[name] = value
            return True
    return False


def _swiss_refresh_years(now, previous_year_refresh_months):
    """Refresh last year's delayed annual release around the year boundary."""
    years = [now.year]
    if now.month <= previous_year_refresh_months:
        years.insert(0, now.year - 1)
    return years


def main():
    """Run enabled acquisitions in sequence; a failed provider never masks another."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/daily_price_market_update.yaml")
    parser.add_argument("--database", help="Existing shared market DuckDB")
    parser.add_argument("--since", help="First local delivery date, YYYY-MM-DD")
    parser.add_argument("--until", help="Exclusive local delivery date, YYYY-MM-DD")
    parser.add_argument("--providers", nargs="+", help="Run only these configured providers")
    parser.add_argument("--non-interactive", action="store_true", help="Never prompt; report absent credentials")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    settings = load_config(args.config)["daily_price_market_update"]
    now = pd.Timestamp.now(tz=settings["timezone"])
    first = pd.Timestamp(args.since or (now.date() - timedelta(days=settings["lookback_days"])))
    stop = pd.Timestamp(args.until or (now.date() + timedelta(days=1)))
    if stop <= first:
        parser.error("--until must be after --since")
    database = str(Path(args.database or settings["database_path"]).resolve())
    providers = args.providers or settings["providers"]
    if set(providers) - set(settings["providers"]):
        parser.error("Unknown provider in --providers")
    # Use complete month boundaries for stable RTE request identities across daily runs.
    month_start = first.replace(day=1)
    month_stop = (stop - pd.Timedelta(days=1)).replace(day=1) + pd.offsets.MonthBegin(1)
    final_first = min(first, pd.Timestamp(now.date() - timedelta(days=settings["france_final_lookback_days"])))
    final_month_start = final_first.replace(day=1)
    plan = {"database": database, "start": str(first.date()), "end_exclusive": str(stop.date()),
            "rte_month_start": str(month_start.date()), "rte_final_month_start": str(final_month_start.date()),
            "rte_month_end_exclusive": str(month_stop.date()),
            "providers": providers, "run": args.run,
            "missing_automatic_sources": ["GME PSV gas index requires a supported source/import",
                                          "IDA requires GME credentials or manual workbook import"]}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        return
    if not Path(database).is_file():
        raise FileNotFoundError("Target database does not exist; refusing to create a partial catalog")
    from bess_pv_trading.data.api import ApiRequest, SecureTransport, download_from_config
    from bess_pv_trading.data.country_balances import download_french
    from bess_pv_trading.data.swiss_border import download_swiss_border
    from bess_pv_trading.data.swiss_reservoir import download_reservoir_levels
    from bess_pv_trading.data.swissgrid_energy import acquire_energy_statistics, discover_energy_workbook

    output = Path(settings["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f"update_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    results = []
    secrets = [settings["entsoe_token_env"], settings["gme_username_env"], settings["gme_password_env"]]
    # Sequential provider calls avoid concurrent writes to the embedded catalog.
    for provider in providers:
        print(f"Updating {provider}", flush=True)
        try:
            if provider == "energy_charts":
                command = [sys.executable, "-m", "scripts.cache_energy_charts_prices", "--fetch",
                    "--config", settings["energy_charts_config"], "--database", database,
                    "--start", str(first.date()), "--end-exclusive", str(stop.date())]
                completed = subprocess.run(command, text=True, capture_output=True, check=True)
                result = {"status": "complete", "acquisition": json.loads(completed.stdout)}
            elif provider in {"entsoe", "gme"}:
                required = ([settings["entsoe_token_env"]] if provider == "entsoe" else
                            [settings["gme_username_env"], settings["gme_password_env"]])
                if provider == "entsoe":
                    _load_entsoe_token(settings["entsoe_token_env"], interactive=not args.non_interactive)
                missing = [name for name in required if not os.environ.get(name)]
                if missing:
                    result = {"status": "missing_credentials", "environment_variables": missing}
                else:
                    config = load_config(settings[f"{provider}_config"])
                    config["api"].update(start=str(first.date()), end=str(stop.date()), database_path=database,
                                         database_threads=2, plots=False)
                    result = download_from_config(config, interactive=False, providers=[provider],
                        refresh_cache=(provider == "entsoe" and settings["entsoe_refresh_recent_requests"]),
                        progress=lambda message: print(message, flush=True))
            elif provider.startswith("france_"):
                config = load_config(settings["european_context_config"])
                config["country_balances"].update(database_path=database,
                    start=str((final_month_start if provider == "france_final" else month_start).date()),
                    end_exclusive=str(month_stop.date()))
                if provider == "france_realtime":
                    config["country_balances"]["france"].update(
                        endpoint=settings["france_realtime_endpoint"], dataset=settings["france_realtime_dataset"])
                result = download_french(config, refresh=True, progress=lambda message: print(message, flush=True))
            elif provider == "swissgrid_energy":
                config = load_config(settings["swissgrid_energy_config"])
                section = config["swissgrid_energy"]
                section["database_path"] = database
                years = _swiss_refresh_years(now, settings["swissgrid_energy_previous_year_refresh_months"])
                configured = {int(item["year"]): dict(item) for item in section["sources"]}
                client = SecureTransport(**section["http"])
                page = client.get(ApiRequest(section["source_page"]))
                sources = []
                discovery_errors = []
                # Keep source-year assumptions explicit and validate every discovered workbook on import.
                for year in years:
                    source = configured.get(year, {"year": year, "filename": f"EnergieUebersichtCH-{year}.xlsx",
                        **section["future_source_defaults"]})
                    try:
                        source["url"] = discover_energy_workbook(page, section["source_page"], year)
                        sources.append(source)
                    except Exception as error:
                        discovery_errors.append({"year": year, "error": _redacted(error, secrets)})
                if sources:
                    section["sources"] = sources
                    result = acquire_energy_statistics(config, refresh=True)
                else:
                    result = {"status": "partial", "sources": []}
                if discovery_errors:
                    result["status"] = "partial"
                    result["discovery_errors"] = discovery_errors
            else:
                config = load_config(settings["european_context_config"])
                if provider == "swissgrid_border":
                    years = _swiss_refresh_years(now, settings["swissgrid_border_previous_year_refresh_months"])
                    year_results = []
                    for year in years:
                        config["swiss_border"].update(database_path=database, year=year)
                        try:
                            year_results.append(download_swiss_border(config, refresh=True))
                        except Exception as error:
                            year_results.append({"status": "failed", "year": year, "error": _redacted(error, secrets)})
                    result = {"status": "complete" if all(item["status"] in {"complete", "cached"}
                        for item in year_results) else "partial", "years": year_results}
                else:
                    config["swiss_reservoir"]["database_path"] = database
                    result = download_reservoir_levels(config, refresh=True)
            results.append(_summary(provider, result))
        except Exception as error:
            results.append({"provider": provider, "status": "failed", "error": _redacted(error, secrets)})
        # Persist progress after each provider, including failures and missing credentials.
        report_path.write_text(json.dumps({"plan": plan, "results": results}, indent=2, default=str) + "\n")
        coverage = results[-1].get("details", {}).get("coverage_status")
        suffix = f"; coverage {coverage}" if coverage else ""
        print(f"{provider}: {results[-1]['status']}{suffix}", flush=True)
    print(f"Report: {report_path}", flush=True)
    if any(item["status"] not in {"complete", "cached"} for item in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

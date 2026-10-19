"""Explicit commands for data preparation, optimization, and evidence reports."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .config import load_config
from .simulation.synthetic import json_default, write_json


def _optimization_inputs(config: dict):
    """Construct validated inputs without creating a solver model."""
    from .domain import MarketState, OptimizationInput, PhysicalState
    from .optimization.degradation import degradation_settings_from_config
    from .simulation.synthetic import synthetic_inputs
    if "synthetic" not in config:
        if "forecast_inputs" in config.get("optimization", {}):
            from .optimization.forecast_inputs import load_forecast_inputs
            return load_forecast_inputs(config)
        from .optimization.prepared import load_prepared_inputs
        return load_prepared_inputs(config)
    fixture = synthetic_inputs(config)
    grid, trading = fixture["grid"], config["trading"]
    masks = {event.auction:tuple(t in event.eligible_intervals for t in range(len(grid.interval_ids)))
             for event in fixture["schedule"]}
    inputs = OptimizationInput(grid,fixture["plant"],PhysicalState(grid.starts[0],config["state"]["initial_energy_mwh"],
        cycle_segment_energy_mwh=config["state"].get("cycle_segment_energy_mwh")),
        MarketState(),fixture["scenarios"],fixture["tree"],masks,
        trading["trade_limit_mwh"],trading["position_limit_mwh"],trading["imbalance_penalty_eur_per_mwh"],
        config["state"]["terminal_target_mwh"],
        degradation_settings_from_config(config),
        trading["grid_import_fee_eur_per_mwh"],
        imbalance_limit_mwh=trading.get("imbalance_limit_mwh"),
        optimize_imbalances=trading.get("optimize_imbalances", True),
        planned_delivery_balance_resolution_minutes=trading.get("planned_delivery_balance_resolution_minutes"))
    return inputs


def _optimize(config: dict) -> dict:
    """Solve synthetic or prepared forecast inputs with one formulation builder."""
    return _solve_optimization(_optimization_inputs(config),config)


def _solve_optimization(inputs, config: dict) -> dict:
    """Keep solver extraction and output identical for both input routes."""
    from .domain import SolverSettings
    from .optimization import build_model, solve_model, validate_solution
    built = build_model(inputs)
    try:
        result = solve_model(built,SolverSettings(**config["solver"]))
    finally:
        built.model.dispose()
    if not result.has_incumbent:
        raise RuntimeError(f"Solver {result.status}: {result.message}")
    validation = validate_solution(inputs,result)
    report = {"status":result.status,"objective_eur":result.objective_eur,
              "validation":asdict(validation),"diagnostics":result.diagnostics,
              "components":result.components,"config":config,
              "scenario_provenance":inputs.scenarios.provenance}
    import numpy as np
    from datetime import datetime, timezone
    output = Path(config["outputs"]["root"])/(config["outputs"].get("run_id") or datetime.now(timezone.utc).strftime("optimize_%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/"solution.npz",**result.arrays)
    write_json(output/"result.json",report)
    report["output_dir"] = str(output)
    from .reporting.artifacts import organize_run_outputs
    organize_run_outputs(output)
    if not validation.valid:
        raise RuntimeError(f"Independent validation failed: {validation.violations}")
    return report


def dispatch(command: str, config: dict) -> dict:
    """Import optional workflows only when their commands are requested."""
    if command == "validate-config":
        if "inputs" in config:
            from .data.adapters import validate_source_config
            validate_source_config(config)
        if "synthetic" in config or "optimization" in config:
            _optimization_inputs(config)
        return {"status":"valid", "sections":list(config)}
    if command == "inspect-data":
        from .data import inspect_data
        return inspect_data(config)
    if command == "prepare-analysis-data":
        from .data import prepare_analysis_data
        return prepare_analysis_data(config)
    if command == "forecast-pv":
        from .forecasting.pv import forecast_pv_from_config
        return forecast_pv_from_config(config)
    if command == "import-market-data":
        from .data.market_prices import import_market_from_config
        return import_market_from_config(config)
    if command == "import-gme-workbooks":
        from .data.gme_workbooks import import_gme_workbooks
        return import_gme_workbooks(config)
    if command == "import-terna-workbooks":
        from .data.terna_workbooks import import_terna_workbooks
        return import_terna_workbooks(config)
    if command == "experiment-gme-oracle":
        from .simulation.gme_oracle import run_gme_oracle_experiment
        return run_gme_oracle_experiment(config)
    if command == "train-price-models":
        from .forecasting.prices import train_price_from_config
        return train_price_from_config(config)
    if command == "experiment-api-prices":
        from .forecasting.price_experiment import run_price_experiment
        return run_price_experiment(config)
    if command == "experiment-price-fundamentals":
        from .forecasting.fundamental_experiment import run_fundamental_experiment
        return run_fundamental_experiment(config)
    if command == "study-linear-prices":
        from .forecasting.price_pipeline import run_linear_price_pipeline
        return run_linear_price_pipeline(config,
            progress=lambda message: print(message, file=sys.stderr, flush=True))
    if command == "calibrate-price-scenarios":
        from .forecasting.gaussian_study import run_gaussian_study
        return run_gaussian_study(config,
            progress=lambda message: print(message, file=sys.stderr, flush=True))
    if command == "check-frozen-prices":
        from .forecasting.price_forward_check import run_frozen_price_check
        return run_frozen_price_check(config,
            progress=lambda message: print(message, file=sys.stderr, flush=True))
    if command == "api-cache-status":
        from .data.api_catalog import summarize_api_cache
        return summarize_api_cache(config)
    if command == "migrate-api-cache":
        from .data.api_migration import migrate_api_cache
        return migrate_api_cache(config,
            progress=lambda message: print(message, file=sys.stderr, flush=True))
    if command in {"run-synthetic","backtest"}:
        if "synthetic" not in config:
            raise ValueError("Historical market replay is blocked: import auction and settlement targets, confirm BESS/POI, and provide a historically validated settlement adapter. The synthetic event replay is available via configs/full_experiments/synthetic_small.yaml.")
        from .simulation.synthetic import run_synthetic
        return run_synthetic(config)
    if command == "optimize":
        return _optimize(config)
    if command == "report-implementation":
        from .reporting.implementation import report_implementation
        return report_implementation(config)
    if command == "download-data":
        from .data.api import download_from_config
        return download_from_config(config)
    raise ValueError(f"Unknown command {command}")


def main(argv: list[str] | None = None) -> int:
    """Print serializable results and actionable errors without credential prompts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("inspect-data","prepare-analysis-data","validate-config",
        "run-synthetic","forecast-pv","import-market-data","train-price-models","optimize",
        "backtest","report-implementation","download-data","experiment-api-prices","experiment-price-fundamentals",
        "study-linear-prices","calibrate-price-scenarios","check-frozen-prices",
        "api-cache-status","migrate-api-cache","diagnose-terna",
        "import-gme-workbooks","import-terna-workbooks","experiment-gme-oracle"))
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--interactive",action="store_true",help="Prompt invisibly for missing API credentials in a terminal")
    parser.add_argument("--dry-run",action="store_true",help="Plan API requests and inspect cache without authenticating")
    parser.add_argument("--refresh-cache",action="store_true",help="Retrieve API requests again even when cached")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command=="download-data":
            from .data.api import download_from_config
            result=download_from_config(config,interactive=args.interactive,dry_run=args.dry_run,
                refresh_cache=args.refresh_cache,progress=lambda message:print(message,file=sys.stderr,flush=True))
        elif args.command == "diagnose-terna":
            if args.dry_run or args.refresh_cache:
                raise ValueError("diagnose-terna does not support cache flags")
            from .data.terna_diagnostics import diagnose_terna
            result = diagnose_terna(config, interactive=args.interactive)
        else:
            if args.interactive or args.dry_run or args.refresh_cache:
                raise ValueError("API download flags apply only to download-data")
            result = dispatch(args.command,config)
        if args.command in {"run-synthetic", "backtest"}:
            result = {"run_id":result["run_id"],"kind":result["kind"],
                "intervals":result["intervals"],"scenario_count":result["scenario_count"],
                "output_dir":result["output_dir"],"strategies":[
                    {key:item[key] for key in ("strategy","realized_operating_profit_eur",
                        "terminal_error_mwh","max_validation_error","solves","runtime_seconds")}
                    for item in result["strategies"]]}
        elif args.command == "optimize":
            result = {"status":result["status"],"objective_eur":result["objective_eur"],
                "valid":result["validation"]["valid"],
                "max_validation_error":max(result["validation"]["max_errors"].values()),
                "diagnostics":result["diagnostics"],"output_dir":result["output_dir"]}
        elif args.command == "download-data":
            if args.dry_run:
                from collections import Counter
                groups=Counter((item["request"]["provider"],item["request"]["dataset"]) for item in result["items"])
                result={"status":result["status"],"requests":result["requests"],
                    "download_order":result["download_order"],
                    "start":config["api"]["start"],"end_exclusive":config["api"]["end"],
                    "cached_chunks":sum(item["cached"] for item in result["items"]),
                    "cached_no_data_chunks":sum(item["no_data_cached"] for item in result["items"]),
                    "groups":[{"provider":provider,"dataset":dataset,"chunks":count}
                              for (provider,dataset),count in groups.items()]}
            else:
                result={key:value for key,value in result.items() if key!="items"}
        elif args.command == "migrate-api-cache":
            result = {key: result[key] for key in
                      ("status", "database", "counts", "report", "coverage_plot")}
        elif args.command == "experiment-gme-oracle":
            result = {key: result[key] for key in
                      ("status", "study_id", "run_id", "study_date", "output_dir",
                       "active_auctions", "skipped_auctions", "optimizer_solves")}
        elif args.command == "check-frozen-prices":
            fields = ("status", "manifest", "database", "new_test_start", "new_test_end_exclusive",
                      "fresh_days", "fresh_rows", "missing_days", "reason", "overlap_verification",
                      "point_metrics", "adaptive_metrics", "static_metrics", "operational_eligible")
            result = {key: result[key] for key in fields if key in result}
        print(json.dumps(result,indent=2,default=json_default,allow_nan=False))
        return 0
    except (ValueError,KeyError,FileNotFoundError,FileExistsError,RuntimeError) as error:
        print(f"{args.command}: {error}",file=sys.stderr)
        return 2

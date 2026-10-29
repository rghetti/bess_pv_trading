#!/usr/bin/env python
"""Run an exported forecast model on prepared issue-time weather rows (Parquet).

  python scripts/predict_pv_forecast.py --bundle PATH --farm-id 901 --input weather.parquet --output pv_forecast.parquet

Required columns are the trained weather/time fields plus interval_start_utc,
forecast_cycle_utc, lead_hours and delivery_day. No measured power is required.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    import pandas as pd
    from bess_pv_trading.forecasting.inference.pv_forecast_bundle import predict_composite
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--farm-id',required=True,type=int)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--history',type=Path,help='Optional updated completed-day statistics for an online-gain bundle')
    args=parser.parse_args();frame=pd.read_parquet(args.input)
    history=pd.read_parquet(args.history) if args.history else None
    prediction=predict_composite(frame,args.bundle,farm_id=args.farm_id,history=history)
    output=frame[['interval_start_utc']].copy();output['pv_ac_forecast_mw']=prediction
    args.output.parent.mkdir(parents=True,exist_ok=True);output.to_parquet(args.output,index=False)


if __name__=='__main__':main()

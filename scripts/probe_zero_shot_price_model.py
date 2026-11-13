"""Run a synthetic, one-GPU compatibility probe without reading market data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
import traceback
from pathlib import Path


def main():
    """Load an immutable checkpoint and measure a small synthetic forecast."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["chronos2", "t0_gamma", "timeh2o", "timesfm3"], required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    result = {"model": args.model, "repository": args.repository, "revision": args.revision,
              "physical_gpu": args.gpu, "precision": "float32", "python": platform.python_version(),
              "purpose": "synthetic compatibility probe; no historical inference", "status": "failed"}
    started = time.monotonic()
    try:
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this environment")
        torch.set_num_threads(args.threads)
        torch.cuda.reset_peak_memory_stats()
        properties = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": properties.name, "total_bytes": properties.total_memory,
                         "compute_capability": [properties.major, properties.minor]}
        result["torch"] = torch.__version__
        result["cuda_runtime"] = torch.version.cuda
        context = np.sin(np.arange(28 * 96, dtype=np.float32) / 30.0)
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if args.model == "chronos2":
            from chronos import Chronos2Pipeline

            result["package"] = importlib.metadata.version("chronos-forecasting")
            model = Chronos2Pipeline.from_pretrained(args.repository, revision=args.revision,
                                                     device_map="cuda", torch_dtype=torch.float32)
            quantiles, _ = model.predict_quantiles([{"target": context}], prediction_length=96,
                                                    quantile_levels=levels)
            output = quantiles[0].detach().cpu().numpy().squeeze()
        elif args.model == "t0_gamma":
            from t0 import T0Forecaster

            result["package"] = importlib.metadata.version("tfc-t0")
            model = T0Forecaster.from_pretrained(args.repository, revision=args.revision).to("cuda").eval()
            with torch.inference_mode():
                forecast = model.predict(torch.as_tensor(context, device="cuda")[None, :],
                                         horizon=96, quantile_levels=levels)
            output = forecast.quantiles.detach().cpu().numpy().squeeze()
        elif args.model == "timeh2o":
            from h2ofm import TimeH2O

            result["package"] = importlib.metadata.version("h2ofm")
            model = TimeH2O.from_pretrained(args.repository, revision=args.revision, device="cuda")
            forecast = model.forecast(context.astype(np.float64), 96, quantile_levels=levels)
            output = np.asarray(forecast.quantiles).squeeze()
        else:
            from huggingface_hub import snapshot_download
            from timesfm3 import ModelConfig, TimesFM3Evaluator

            result["package"] = importlib.metadata.version("timesfm")
            checkpoint = snapshot_download(args.repository, revision=args.revision)
            config = ModelConfig(checkpoint_path=checkpoint, per_core_batch_size=1, device="cuda")
            model = TimesFM3Evaluator(config)
            forecast = list(model.predict_batch([context], horizon=96, return_quantiles=True,
                                                 use_symmetric_averaging=False))[0]
            output = np.asarray(forecast.quantiles).squeeze()
        torch.cuda.synchronize()
        if output.shape != (96, len(levels)) or not np.isfinite(output).all():
            raise ValueError(f"Unexpected forecast shape or non-finite values: {output.shape}")
        result["shape"] = list(output.shape)
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["status"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

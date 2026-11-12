"""Fit CPU temporal convolutions to frozen daily auction-price curves."""

import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.pilot_intraday_curve_ridge import metric

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class CurveNetwork(nn.Module):
    """Predict a whole price-spread curve with shared temporal convolutions."""

    def __init__(self, input_channels, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(width, 1, kernel_size=1),
        )

    def forward(self, inputs):
        """Return one normalized spread per ordered quarter-hour product."""
        return self.layers(inputs).squeeze(1)


def fit_scaling(values, rows):
    """Fit channel-wise missing-value replacement and scaling on training rows."""
    subset = values[rows]
    with np.errstate(all="ignore"):
        median = np.nanmedian(subset, axis=(0, 2))
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(subset), subset, median[None, :, None])
    mean = filled.mean(axis=(0, 2))
    std = filled.std(axis=(0, 2))
    std = np.where(std > 1e-8, std, 1.0)
    return median, mean, std


def transform(values, scaling):
    """Scale channel values and append a channel for each missingness flag."""
    median, mean, std = scaling
    missing = ~np.isfinite(values)
    filled = np.where(missing, median[None, :, None], values)
    normalized = (filled - mean[None, :, None]) / std[None, :, None]
    return np.concatenate([normalized, missing.astype(float)], axis=1).astype(np.float32)


def optimize(values, target, train_mask, valid_mask, width, fixed_epochs=None):
    """Select epochs on the chronological inner period, or refit fixed epochs."""
    seed = 20261112
    torch.manual_seed(seed)
    scale = fit_scaling(values, train_mask)
    inputs = torch.from_numpy(transform(values, scale))
    mean = float(target[train_mask].mean())
    deviation = float(target[train_mask].std())
    deviation = max(deviation, 1e-6)
    outputs = torch.from_numpy(((target - mean) / deviation).astype(np.float32))
    network = CurveNetwork(inputs.shape[1], width)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=0.001)
    train_ix = np.flatnonzero(train_mask)
    rng = np.random.default_rng(seed)
    best, best_epoch, no_improvement = float("inf"), 0, 0
    maximum = fixed_epochs if fixed_epochs is not None else 80
    for epoch in range(1, maximum + 1):
        network.train()
        for batch in np.array_split(rng.permutation(train_ix), max(1, int(np.ceil(len(train_ix) / 32)))):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(network(inputs[batch]), outputs[batch])
            loss.backward()
            optimizer.step()
        if fixed_epochs is None:
            network.eval()
            with torch.no_grad():
                prediction = network(inputs[valid_mask]).numpy() * deviation + mean
            mae = float(np.abs(prediction - target[valid_mask]).mean())
            if mae < best - 1e-6:
                best, best_epoch, no_improvement = mae, epoch, 0
            else:
                no_improvement += 1
            if no_improvement >= 10:
                break
    return network, scale, mean, deviation, best, best_epoch


def main():
    """Run paired core and external-feature variants for every auction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", default="outputs/price_staged_search_full_year")
    parser.add_argument("--output", default="outputs/price_temporal_curve_pilot")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("Use --run to fit this study")
    root = Path(args.search)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    prepared_path = root / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    report = {"prepared_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
        "model": "Two Conv1d layers with kernel 5 and dilations 1,2; ReLU; shared quarter-hour output head",
        "sequence": "ordered within-day quarter-hour products; static previous daily curves and calendar broadcast to each product",
        "optimization": "Adam learning rate 0.001, weight decay 0.001, standardized-spread MSE, batch 32, seed 20261112",
        "selection": "width 16 or 32; up to 80 epochs; patience 10 on chronological inner MAE; refit full annual data for selected epoch count",
        "availability": "pinned gate-aware price inputs and user-assumed previous-day features; no same-day realized features",
        "split": "2025-10-01 to 2026-09-30, labels published by first October gate; October is development; November untouched",
        "missing": "training channel median imputation and flags; no target imputation",
        "threads": 1, "auctions": {}}
    for market, count in (("da", 96), ("ida1", 96), ("ida2", 96), ("ida3", 48)):
        group = prepared["groups"][f"{market}_quarter_hourly"]
        def read(name):
            item = group["files"][name]
            path = root / item["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Changed prepared array: {path}")
            return np.load(path, mmap_mode="r")
        x, y, anchor, reference, days, starts, publication, decisions = (read(n) for n in
            ("x", "y", "anchor", "reference", "days", "starts", "publication", "decisions"))
        names = group["feature_names"]
        daily_names = [n for n in names if n.startswith("own_curve_") or n in
            {"weekday_sin", "weekday_cos", "weekend", "annual_sin", "annual_cos",
             "italian_public_holiday", "day_before_holiday", "day_after_holiday",
             "weekday_bridge_between_holiday_and_weekend", "earlier_IDA1_cancelled", "earlier_IDA2_cancelled"}]
        curve_names = [n for n in ("hour_sin", "hour_cos", "cleared_DA", "cleared_IDA1", "cleared_IDA2",
            "own_price_lag_24h", "own_price_lag_48h", "own_price_lag_168h") if n in names]
        extra_names = [n for n in names if n.startswith(("nord_", "psv_", "fr_")) and "same_day" not in n]
        daily_cols = [names.index(n) for n in daily_names]
        curve_cols = [names.index(n) for n in curve_names]
        extra_cols = [names.index(n) for n in extra_names]
        inner_descriptor = group["cohorts"]["stage_core"]["indices"]["inner_validation"]
        inner_start = min(days[np.load(root / inner_descriptor["path"])].tolist())
        october = (days >= "2026-10-01") & (days < "2026-11-01")
        cutoff = int(np.min(decisions[october]))
        samples, excluded = [], []
        for day in np.unique(days[(days >= "2025-10-01") & (days < "2026-11-01")]):
            ix = np.flatnonzero(days == day)
            ix = ix[np.argsort(starts[ix])]
            if len(ix) != count:
                excluded.append({"day": str(day), "reason": "nonstandard_or_missing_product_count", "rows": len(ix)})
                continue
            if not np.isfinite(y[ix]).all() or not np.isfinite(anchor[ix]).all() or not np.isfinite(reference[ix]).all():
                excluded.append({"day": str(day), "reason": "missing_target_anchor_or_reference", "rows": len(ix)})
                continue
            if day < "2026-10-01" and ((publication[ix] == pd.NaT.value).any() or (publication[ix] > cutoff).any()):
                excluded.append({"day": str(day), "reason": "label_not_published_by_october_gate", "rows": len(ix)})
                continue
            # Broadcast already-published day-level curves and calendar across the delivery grid.
            static = np.repeat(np.asarray(x[ix[0], daily_cols])[:, None], count, axis=1)
            curves = np.asarray(x[np.ix_(ix, curve_cols)]).T
            extra = np.asarray(x[np.ix_(ix, extra_cols)]).T
            samples.append({"day": str(day), "core": np.vstack([static, curves]), "extra": extra,
                "actual": np.asarray(y[ix]), "anchor": np.asarray(anchor[ix]),
                "reference": np.asarray(reference[ix]), "starts": np.asarray(starts[ix])})
        core = np.stack([s["core"] for s in samples])
        extra = np.stack([s["extra"] for s in samples])
        actual = np.stack([s["actual"] for s in samples])
        anchors = np.stack([s["anchor"] for s in samples])
        refs = np.stack([s["reference"] for s in samples])
        spread = actual - anchors
        dates = np.asarray([s["day"] for s in samples])
        train = dates < inner_start
        valid = (dates >= inner_start) & (dates < "2026-10-01")
        annual = dates < "2026-10-01"
        test = dates >= "2026-10-01"
        variants = {}
        for variant, features in (("core", core), ("core_plus_available", np.concatenate([core, extra], axis=1))):
            results = []
            for width in (16, 32):
                _, _, _, _, best, best_epoch = optimize(features, spread, train, valid, width)
                results.append({"width": width, "inner_mae": best, "best_epoch": best_epoch})
            chosen = min(results, key=lambda item: (item["inner_mae"], item["width"]))
            net, scaling, mean, std, _, _ = optimize(features, spread, annual, None, chosen["width"], chosen["best_epoch"])
            net.eval()
            with torch.no_grad():
                prediction = net(torch.from_numpy(transform(features[test], scaling))).numpy() * std + mean + anchors[test]
            weights_path = output / f"{market}_{variant}.pt"
            torch.save({"state_dict": net.state_dict(), "width": chosen["width"], "scale": scaling,
                "target_mean": mean, "target_std": std, "daily_features": daily_names,
                "curve_features": curve_names, "extra_features": extra_names if variant != "core" else []}, weights_path)
            records = []
            for sample, forecast in zip(np.asarray(samples, dtype=object)[test], prediction):
                for start, actual_value, baseline, value in zip(sample["starts"], sample["actual"], sample["reference"], forecast):
                    records.append({"local_day": sample["day"], "product_start": pd.Timestamp(int(start), tz="UTC"),
                        "actual_eur_per_mwh": actual_value, "reference_eur_per_mwh": baseline, "forecast_eur_per_mwh": value})
            pd.DataFrame(records).to_parquet(output / f"{market}_{variant}_october.parquet", index=False)
            variants[variant] = {"trials": results, "selected": chosen,
                "october": metric(actual[test], prediction, refs[test]),
                "annual_missing_input_values": int((~np.isfinite(features[annual])).sum()),
                "october_missing_input_values": int((~np.isfinite(features[test])).sum())}
            print(market, variant, json.dumps(variants[variant]), flush=True)
        report["auctions"][market] = {"daily_features": daily_names, "curve_features": curve_names,
            "extra_features": extra_names, "inner_start": inner_start,
            "inner_train_days": int(train.sum()), "inner_validation_days": int(valid.sum()),
            "annual_days": int(annual.sum()), "october_days": int(test.sum()),
            "excluded_days": excluded, "variants": variants}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

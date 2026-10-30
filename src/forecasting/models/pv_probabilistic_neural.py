"""Small member-wise MLP and temporal convolution predictors for PV power."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn


class MemberNetwork(nn.Module):
    """Predict a full member path; mask channels identify unavailable weather."""
    def __init__(self, inputs, family, width, kernel):
        super().__init__()
        self.family = family
        # Update `self.body` when this condition is met.
        if family == 'mlp':
            self.body = nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Dropout(.05),
                                      nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        # Update `self.body` when this condition is met.
        elif family == 'tcn':
            self.body = nn.Sequential(nn.Conv1d(inputs, width, kernel, padding=kernel//2), nn.GELU(),
                nn.Dropout(.05), nn.Conv1d(width, width, kernel, padding=kernel//2), nn.GELU(), nn.Conv1d(width, 1, 1))
        else: raise ValueError(f'Unknown neural family {family}')

    def forward(self, x):
        # Return `self.body(x.transpose(1, 2)).squeeze(1)` when `self.family == 'tcn'`.
        if self.family == 'tcn': return self.body(x.transpose(1, 2)).squeeze(1)
        return self.body(x).squeeze(-1)


def _pack(frame, features, scaler, target=None, weights=None):
    """Pack full daily member trajectories; invalid rows are masked, not targets."""
    raw = frame[features].to_numpy(dtype=float)
    finite = np.isfinite(raw).all(axis=1)
    standardized = np.zeros_like(raw)
    standardized[finite] = scaler.transform(raw[finite])
    standardized = np.column_stack([standardized, (~finite).astype(float)])
    groups = frame.groupby(['delivery_day', 'member'], sort=True).indices
    positions = [np.asarray(v, dtype=int) for v in groups.values()]
    length = max(map(len, positions))
    x = np.zeros((len(positions), length, standardized.shape[1]), np.float32)
    y = np.zeros((len(positions), length), np.float32)
    w = np.zeros((len(positions), length), np.float32)
    days = []
    # Pack full daily member trajectories; invalid rows are masked, not targets for each `(j, idx)` in `enumerate(positions)`.
    for j, idx in enumerate(positions):
        order = np.argsort(pd.DatetimeIndex(frame.iloc[idx].interval_start_utc).asi8)
        idx = idx[order]; positions[j] = idx
        x[j, :len(idx)] = standardized[idx]
        # When `target is not None`, set `valid` using `finite[idx] & np.isfinite(target[idx]) & (weights[idx] > 0)`.
        if target is not None:
            valid = finite[idx] & np.isfinite(target[idx]) & (weights[idx] > 0)
            y[j, :len(idx)] = np.where(valid, target[idx], 0)
            w[j, :len(idx)] = np.where(valid, weights[idx], 0)
        days.append(str(frame.iloc[idx[0]].delivery_day))
    return x, y, w, positions, np.asarray(days), finite


def fit_neural(frame, features, scaler, target, weights, trial, settings, seed):
    """Select epochs within training days, then refit on all fit-period days."""
    torch.set_num_threads(1)
    x, y, w, _, days, _ = _pack(frame, features, scaler, target, weights)
    selected = np.flatnonzero(w.sum(axis=1) > 0)
    unique_days = sorted(set(days[selected]))
    # Enforce the input contract: Neural fitting needs at least four training days.
    if len(unique_days) < 4: raise ValueError('Neural fitting needs at least four training days')
    early_days = set(unique_days[-2:])
    train = selected[~np.isin(days[selected], list(early_days))]
    validation = selected[np.isin(days[selected], list(early_days))]
    tensor_x, tensor_y, tensor_w = map(torch.from_numpy, (x, y, w))
    spec = trial['parameters']
    def make_model():
        torch.manual_seed(seed)
        return MemberNetwork(x.shape[-1], trial['family'], int(spec['width']), int(spec.get('kernel', 3)))
    def run_epochs(model, indices, epochs, validate=None):
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec.get('learning_rate', .001)),
                                      weight_decay=float(spec.get('weight_decay', .0001)))
        rng = np.random.default_rng(seed); best_epoch=1; best=float('inf'); stale=0
        # Perform run epochs for each `epoch` in `range(1, epochs + 1)`.
        for epoch in range(1, epochs+1):
            model.train()
            order = rng.permutation(indices)
            # Perform run epochs for each `start` in `range(0, len(order), int(settings['batch_size']))`.
            for start in range(0, len(order), int(settings['batch_size'])):
                pos = order[start:start+int(settings['batch_size'])]
                # Skip the current item when `tensor_w[pos].sum() <= 0`.
                if tensor_w[pos].sum() <= 0: continue
                prediction = model(tensor_x[pos])
                loss = ((prediction-tensor_y[pos])**2*tensor_w[pos]).sum()/tensor_w[pos].sum()
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            # Perform run epochs when `validate is not None`.
            if validate is not None:
                model.eval()
                with torch.no_grad():
                    prediction = model(tensor_x[validate])
                    score = float((((prediction-tensor_y[validate])**2*tensor_w[validate]).sum()/tensor_w[validate].sum()).item())
                # When `score < best - 1e-06`, set `best` using `score`.
                if score < best - 1e-6: best=score; best_epoch=epoch; stale=0
                else: stale += 1
                # Stop processing when `stale >= int(settings['early_stopping_patience'])`.
                if stale >= int(settings['early_stopping_patience']): break
        return best_epoch if validate is not None else epochs
    model = make_model(); best_epoch = run_epochs(model, train, int(settings['max_epochs']), validation)
    model = make_model(); run_epochs(model, selected, best_epoch); model.eval()
    return model, {'best_epoch': best_epoch, 'early_stopping_days': sorted(early_days),
                   'training_days': len(unique_days), 'mask_channel': 'missing weather input'}


def predict_neural(model, frame, features, scaler):
    """Predict each available quarter in each member's own daily trajectory."""
    x, _, _, positions, _, finite = _pack(frame, features, scaler)
    result = np.full(len(frame), np.nan)
    model.eval()
    with torch.no_grad():
        # Predict each available quarter in each member's own daily trajectory for each `start` in `range(0, len(x), 32)`.
        for start in range(0, len(x), 32):
            output = model(torch.from_numpy(x[start:start+32])).numpy()
            # Predict each available quarter in each member's own daily trajectory for each `(local, idx)` in `enumerate(positions[start:start + 32])`.
            for local, idx in enumerate(positions[start:start+32]):
                result[idx] = output[local, :len(idx)]
    result[~finite] = np.nan
    return result

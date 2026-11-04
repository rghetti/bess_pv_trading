"""Direct statistical and neural weather-to-AC models without a physical PV baseline."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from ..data.pv_curated_review import contiguous_positions
from ..training.pv_reviewed_hpo_common import day_balanced_weights, score_predictions


def fit_statistical(family, parameters, seed, x, y, train_mask, train_days):
    """Fit one single-threaded statistical candidate on balanced training quarters."""
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

    options = dict(parameters)
    weights = day_balanced_weights(train_days)
    weights *= len(weights)
    train_x = np.asarray(x[train_mask], dtype=float)
    train_y = np.asarray(y[train_mask], dtype=float)
    if family == 'ridge':
        model = Ridge(**options)
    elif family == 'elastic_net':
        model = ElasticNet(max_iter=5000, random_state=seed, **options)
    elif family == 'huber':
        model = HuberRegressor(max_iter=1000, **options)
    elif family == 'extra_trees':
        model = ExtraTreesRegressor(n_jobs=1, random_state=seed, **options)
    elif family == 'random_forest':
        model = RandomForestRegressor(n_jobs=1, random_state=seed, **options)
    elif family == 'hist_gradient_boosting':
        model = HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **options)
    elif family == 'lightgbm':
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_jobs=1, random_state=seed, deterministic=True,
                              force_col_wise=True, verbosity=-1, **options)
    elif family == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(thread_count=1, random_seed=seed, verbose=False,
                                  allow_writing_files=False, **options)
    elif family in {'spline_ridge', 'polynomial_ridge', 'rbf_ridge'}:
        alpha = options.pop('alpha')
        if family == 'spline_ridge':
            transformer = SplineTransformer(**options)
        elif family == 'polynomial_ridge':
            transformer = PolynomialFeatures(include_bias=False, **options)
        else:
            transformer = RBFSampler(random_state=seed, **options)
        # Fit transformations on training rows; weighted Ridge estimates the target mapping.
        transformed = transformer.fit_transform(train_x)
        regressor = Ridge(alpha=alpha)
        regressor.fit(transformed, train_y, sample_weight=weights)
        return make_pipeline(transformer, regressor)
    else:
        raise ValueError(f'Unsupported statistical family {family}')
    model.fit(train_x, train_y, sample_weight=weights)
    return model


def build_sequences(frame, features, masks, rating_mw):
    """Split local days at prior exclusions while keeping eligible nighttime targets."""
    times = pd.DatetimeIndex(frame.interval_start_utc)
    days = frame.delivery_day.to_numpy()
    context = (~frame.retained_prior_exclusion.to_numpy(dtype=bool)
               | masks['train'] | masks['validation'] | masks['test'])
    segments = contiguous_positions(times, days, context)
    if not segments:
        raise ValueError('No contiguous weather sequences are available')
    max_length = max(map(len, segments))
    shape = (len(segments), max_length)
    x = np.zeros((*shape, features.shape[1]), dtype=np.float32)
    y = np.zeros(shape, dtype=np.float32)
    indices = np.full(shape, -1, dtype=np.int64)
    train = np.zeros(shape, dtype=bool)
    validation = np.zeros(shape, dtype=bool)
    test = np.zeros(shape, dtype=bool)
    splits = []
    # Each sequence remains inside a delivery day, split and selected forecast vintage.
    for number, positions in enumerate(segments):
        sample = frame.iloc[positions]
        if sample['split'].nunique() != 1 or sample.forecast_cycle_utc.nunique() != 1:
            raise ValueError('A sequence crosses a split or forecast-vintage boundary')
        size = len(positions)
        x[number, :size] = features[positions]
        target = sample.model_target_mw.to_numpy(dtype=float) / rating_mw
        y[number, :size] = np.where(np.isfinite(target), target, 0)
        indices[number, :size] = positions
        train[number, :size] = masks['train'][positions]
        validation[number, :size] = masks['validation'][positions]
        test[number, :size] = masks['test'][positions]
        splits.append(sample['split'].iloc[0])
    return {'x': x, 'y': y, 'indices': indices, 'train': train,
            'validation': validation, 'test': test, 'split': np.asarray(splits)}


def build_neural_network(family, feature_count, parameters):
    """Construct a direct bounded network; all architectures share the same input contract."""
    import torch
    from torch import nn
    from .pv_neural_models import TemporalBlock

    class Network(nn.Module):
        """Predict normalized AC power directly from known weather and calendar inputs."""

        def __init__(self):
            super().__init__()
            width = int(parameters['width'])
            depth = int(parameters['depth'])
            dropout = float(parameters['dropout'])
            self.family = family
            self.stem = nn.Sequential(nn.Linear(feature_count, width), nn.GELU())
            if family == 'mlp':
                self.body = nn.Sequential(*[layer for _ in range(depth) for layer in
                    (nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))])
                head_width = width
            elif family == 'tcn':
                self.body = nn.ModuleList([TemporalBlock(width, int(parameters['kernel_size']),
                    2 ** number, dropout) for number in range(depth)])
                head_width = width
            elif family in {'bigru', 'bilstm'}:
                recurrent = nn.GRU if family == 'bigru' else nn.LSTM
                self.body = recurrent(width, width, num_layers=depth, bidirectional=True,
                    dropout=dropout if depth > 1 else 0, batch_first=True)
                head_width = width * 2
            elif family == 'transformer':
                layer = nn.TransformerEncoderLayer(width, int(parameters['heads']),
                    dim_feedforward=2 * width, dropout=dropout, batch_first=True, activation='gelu')
                self.body = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
                head_width = width
            else:
                raise ValueError(f'Unsupported neural family {family}')
            self.output = nn.Linear(head_width, 1)

        def forward(self, inputs, valid):
            """Prevent padded samples from contributing to temporal model inputs."""
            hidden = self.stem(inputs) * valid.unsqueeze(-1)
            if self.family == 'tcn':
                for block in self.body:
                    hidden = block(hidden.transpose(1, 2)).transpose(1, 2)
                    hidden = hidden * valid.unsqueeze(-1)
            elif self.family in {'bigru', 'bilstm'}:
                lengths = valid.sum(1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(hidden, lengths, batch_first=True,
                                                            enforce_sorted=False)
                hidden, _ = self.body(packed)
                hidden, _ = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True,
                    total_length=inputs.shape[1])
            elif self.family == 'transformer':
                hidden = self.body(hidden, src_key_padding_mask=~valid)
            else:
                hidden = self.body(hidden)
            return torch.sigmoid(self.output(hidden).squeeze(-1)) * valid

    return Network()


def predict_neural(model, packed, groups, batch_size, row_count, rating_mw):
    """Predict only requested sequences, returning NaN for unrequested quarters."""
    import torch

    prediction = np.full(row_count, np.nan)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(groups), batch_size):
            chosen = groups[start:start + batch_size]
            indices = packed['indices'][chosen]
            valid = indices >= 0
            values = model(torch.as_tensor(packed['x'][chosen]), torch.as_tensor(valid)).numpy()
            output = valid if 'output' not in packed else valid & packed['output'][chosen]
            prediction[indices[output]] = values[output] * rating_mw
    return prediction


def fit_neural(family, parameters, section, seed, packed, frame, masks, rating_mw):
    """Train on all eligible quarters and select the epoch using forecast-GHI validation."""
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_neural_network(family, packed['x'].shape[-1], parameters)
    target = packed['y'][packed['train']]
    if not len(target):
        raise ValueError('No eligible neural training targets')
    mean = float(np.clip(target.mean(), 1.0e-5, 1 - 1.0e-5))
    with torch.no_grad():
        model.output.bias.fill_(np.log(mean / (1 - mean)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['learning_rate'],
                                  weight_decay=parameters['weight_decay'])
    train_groups = np.flatnonzero(packed['train'].any(axis=1))
    validation_groups = np.flatnonzero(packed['validation'].any(axis=1))
    if not len(validation_groups):
        raise ValueError('No eligible neural validation targets')
    sample_weights = np.zeros(packed['indices'].shape, dtype=np.float32)
    mask = packed['train']
    positions = packed['indices'][mask]
    sample_weights[mask] = day_balanced_weights(frame.delivery_day.to_numpy()[positions])
    batch_size = int(parameters['batch_size'])
    best_score = float('inf')
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, int(section['max_epochs']) + 1):
        model.train()
        order = rng.permutation(train_groups)
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            valid = packed['indices'][chosen] >= 0
            predicted = model(torch.as_tensor(packed['x'][chosen]), torch.as_tensor(valid))
            residual = predicted - torch.as_tensor(packed['y'][chosen])
            if parameters['loss'] == 'huber':
                delta = float(section['huber_delta'])
                loss = torch.where(residual.abs() <= delta, residual.square(),
                                   2 * delta * residual.abs() - delta ** 2)
            else:
                loss = residual.square()
            weight = torch.as_tensor(sample_weights[chosen])
            objective = (loss * weight).sum() / weight.sum().clamp_min(1.0e-12)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(section['gradient_clip_norm']))
            optimizer.step()
        validation = predict_neural(model, packed, validation_groups, batch_size, len(frame), rating_mw)
        score = score_predictions(frame, validation, masks['validation'], rating_mw)
        current = score['day_balanced_nrmse_percent']
        history.append({'epoch': epoch, 'validation_nrmse_percent': current})
        if current < best_score - float(section['minimum_improvement']):
            best_score = current
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(section['patience']):
            break
    if best_state is None or not np.isfinite(best_score):
        raise ValueError('Neural training did not produce a finite validation checkpoint')
    model.load_state_dict(best_state)
    return model, {'epochs': epoch, 'best_epoch': best_epoch, 'validation_nrmse_percent': best_score,
                   'parameters': sum(parameter.numel() for parameter in model.parameters()),
                   'history': history}

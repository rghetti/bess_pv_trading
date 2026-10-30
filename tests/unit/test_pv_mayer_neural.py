"""Verify the paper-inspired neural scores, constraints and data split."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bess_pv_trading.forecasting.models.pv_mayer_neural import BernsteinQuantileNetwork
from bess_pv_trading.forecasting.models.pv_mayer_neural import NonCrossingQuantileNetwork
from bess_pv_trading.forecasting.models.pv_mayer_neural import censored_normal_crps
from bess_pv_trading.forecasting.models.pv_mayer_neural import quantile_huber_loss
from bess_pv_trading.forecasting.training.pv_mayer_neural import input_matrix
from bess_pv_trading.forecasting.training.pv_mayer_neural import fit_farm
from bess_pv_trading.forecasting.training.pv_mayer_neural import load_checkpoint
from bess_pv_trading.forecasting.training.pv_mayer_neural import prepare_examples
from bess_pv_trading.forecasting.training.pv_mayer_neural import prepare_search_examples
from bess_pv_trading.forecasting.training.pv_mayer_neural import score_trial
from bess_pv_trading.forecasting.training.pv_mayer_neural import search_family
from bess_pv_trading.forecasting.training.pv_mayer_neural import train_network
from bess_pv_trading.forecasting.training.pv_mayer_neural import trial_settings


def test_censored_crps_matches_r_reference_and_has_gradients():
    """Cover both boundary masses and an interior target against scoringRules."""
    rscript = Path("/home/rghetti/miniforge3/envs/r-censored/bin/Rscript")
    if not rscript.exists():
        pytest.skip("The separate R reference environment is unavailable")
    command = "\n".join([
        "y <- c(0, .3, 1)",
        "mu <- c(-.2, .4, 1.1)",
        "s <- c(.3, .2, .4)",
        "cat(scoringRules::crps_cnorm(y, mu, s, lower=0, upper=1), sep='\\n')",
    ])
    result = subprocess.run([str(rscript), "-e", command], capture_output=True, text=True, check=True)
    reference = np.fromstring(result.stdout, sep="\n")
    target = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
    location = torch.tensor([-0.2, 0.4, 1.1], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([0.3, 0.2, 0.4], dtype=torch.float64, requires_grad=True)
    values = censored_normal_crps(target, location, scale)
    np.testing.assert_allclose(values.detach().numpy(), reference, rtol=1e-5)
    assert torch.autograd.gradcheck(censored_normal_crps, (target, location, scale))


def test_quantile_huber_approaches_pinball():
    """For a small threshold, the smoothed loss agrees with quantile loss."""
    target = torch.tensor([0.3, 0.7])
    quantiles = torch.tensor([[0.1, 0.2], [0.8, 0.9]])
    levels = torch.tensor([0.25, 0.75])
    residual = target[:, None] - quantiles
    expected = torch.maximum(levels * residual, (levels - 1.0) * residual).mean()
    assert quantile_huber_loss(target, quantiles, levels, 1e-8).item() == pytest.approx(expected.item())


@pytest.mark.parametrize("family", ["bernstein", "noncrossing"])
def test_monotonic_networks_produce_ordered_quantiles(family):
    """Both constrained architectures retain ordered outputs and backpropagate."""
    levels = np.arange(1, 8) / 8
    if family == "bernstein":
        model = BernsteinQuantileNetwork(4, [8], "relu", 5, levels)
    else:
        model = NonCrossingQuantileNetwork(4, [8], "relu", len(levels), 9)
    features = torch.randn(20, 4)
    predictions = model(features)
    assert torch.all(torch.diff(predictions, dim=1) >= -1e-6)
    predictions.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def synthetic_frame():
    """Create fit and outer-evaluation days without real farm data."""
    dates = pd.date_range("2026-01-01", periods=15, freq="D")
    rows = []
    for index, day in enumerate(dates):
        split = "train" if index < 10 else "evaluation"
        rows.append({
            "delivery_day": str(day.date()),
            "split": split,
            "eligible": True,
            "fit_label_available": split == "train",
            "control": 0.4,
            "member_1": 0.5,
            "member_2": 0.3,
            "y": 0.45,
        })
    return pd.DataFrame(rows)


def test_split_is_by_day_and_evaluation_never_enters_training():
    """The fifth fit day is reserved for early stopping, not for gradients."""
    training, stopping, evaluation, members = prepare_examples(synthetic_frame(), 2, 5)
    assert set(training.delivery_day).isdisjoint(stopping.delivery_day)
    assert set(training.delivery_day).isdisjoint(evaluation.delivery_day)
    assert stopping.delivery_day.tolist() == ["2026-01-05", "2026-01-10"]
    matrix = input_matrix(training, members)
    np.testing.assert_allclose(matrix[0], [0.4, 0.3, 0.5])


def test_search_uses_three_one_one_blocks_without_outer_evaluation():
    """Keep hyperparameter selection within the previously available fit period."""
    training, stopping, selection = prepare_search_examples(synthetic_frame(), 5)
    assert len(training) == 6
    assert stopping.delivery_day.tolist() == ["2026-01-04", "2026-01-09"]
    assert selection.delivery_day.tolist() == ["2026-01-05", "2026-01-10"]
    assert set(training.delivery_day).isdisjoint(selection.delivery_day)
    assert not selection.split.eq("evaluation").any()


def test_hyperparameter_draw_and_failed_candidate_are_recorded():
    """Draw bounded seeded settings and retain failures without fabricating scores."""
    settings = {
        "search_hidden_width": [5, 200],
        "search_learning_rate": [0.0005, 0.05],
        "search_patience": [5, 50],
        "search_batch_size": [200, 20000],
        "search_bernstein_degree": [6, 15],
        "search_noncrossing_extra_width": [0, 9],
        "quantile_count": 3,
    }
    selected = trial_settings(settings, "ncqrnn", np.random.default_rng(12))
    assert 3 <= selected["noncrossing_width"] <= 12
    assert 0.0005 <= selected["learning_rate"] <= 0.05
    training, stopping, selection = prepare_search_examples(synthetic_frame(), 5)
    job = ("unknown", training, stopping, selection, ["member_1", "member_2"], {}, np.array([0.5]), 1)
    result = score_trial(job)
    assert result["status"] == "failed"
    assert "error" in result and "selection_crps" not in result


def test_search_ranks_synthetic_candidates_without_outer_targets(monkeypatch):
    """Exercise the candidate-selection workflow using a synchronous test pool."""
    class TestPool:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def map(self, function, jobs):
            return map(function, jobs)

    monkeypatch.setattr("bess_pv_trading.forecasting.training.pv_mayer_neural.ProcessPoolExecutor", TestPool)
    settings = {
        "early_stopping_every_n_days": 5,
        "hyperparameter_trials": 2,
        "search_workers": 1,
        "search_hidden_width": [4, 4],
        "search_learning_rate": [0.001, 0.001],
        "search_patience": [1, 1],
        "search_batch_size": [2, 2],
        "search_bernstein_degree": [3, 3],
        "search_noncrossing_extra_width": [0, 0],
        "quantile_count": 3,
        "drn_hidden_sizes": [4],
        "minimum_scale": 1e-6,
        "max_epochs": 1,
        "huber_epsilon": 1e-8,
        "seed": 42,
    }
    result = search_family(synthetic_frame(), ["member_1", "member_2"], "qrnn", settings, np.array([0.25, 0.5, 0.75]), 1)
    assert len(result) == 2
    assert all(row["status"] == "complete" for row in result)
    assert result[0]["selection_crps"] <= result[1]["selection_crps"]


@pytest.mark.parametrize("family", ["cn_drn", "qrnn", "bqn", "ncqrnn"])
def test_network_training_uses_only_synthetic_cases(family):
    """Exercise the training loop and early stopping without a historical run."""
    training, stopping, _, members = prepare_examples(synthetic_frame(), 2, 5)
    settings = {
        "drn_hidden_sizes": [4],
        "quantile_hidden_sizes": [4],
        "quantile_activation": "relu",
        "minimum_scale": 1e-6,
        "bernstein_degree": 3,
        "noncrossing_width": 3,
        "max_epochs": 2,
        "batch_size": 4,
        "patience": 2,
        "learning_rate": 0.001,
        "huber_epsilon": 1e-8,
    }
    model, record = train_network(family, training, stopping, members, settings, np.array([0.25, 0.5, 0.75]), 42)
    assert model is not None
    assert 1 <= record["best_epoch"] <= 2


def test_training_artifacts_and_checkpoint_reload(monkeypatch, tmp_path):
    """Fit tiny synthetic networks, persist them and reload a checked checkpoint."""
    import json
    from bess_pv_trading.config import load_config
    from bess_pv_trading.forecasting.data.pv_probabilistic_data import file_hash

    config = load_config(Path("configs/pv_experiments/mayer_neural_2026.yaml"))
    settings = config["pv_mayer_neural"]
    settings.update({
        "member_count": 2,
        "quantile_count": 3,
        "drn_hidden_sizes": [4],
        "quantile_hidden_sizes": [4],
        "bernstein_degree": 3,
        "noncrossing_width": 3,
        "drn_ensemble_size": 1,
        "max_epochs": 2,
        "batch_size": 4,
        "patience": 2,
        "hyperparameter_trials": 1,
    })
    source = tmp_path / "source" / "farm_1"
    source.mkdir(parents=True)
    frame = synthetic_frame()
    frame["row_id"] = np.arange(len(frame))
    frame["hour_start_utc"] = pd.to_datetime(frame["delivery_day"], utc=True)
    frame["lead_hours"] = 32.0
    frame["measured_ac_mw"] = frame["y"]
    frame.to_parquet(source / "hourly_inputs_normalized.parquet", index=False)

    # The search itself is checked separately; use one fixed candidate in this fit test.
    def fake_search(frame, members, family, settings, levels, farm_id):
        return [{"status": "complete", "settings": dict(settings), "selection_crps": 0.1}]
    monkeypatch.setattr("bess_pv_trading.forecasting.training.pv_mayer_neural.search_family", fake_search)
    destination = tmp_path / "output"
    fit_farm(config, 1, source.parent, destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert len(manifest["model_records"]) == 4
    assert set(manifest["training_days"]).isdisjoint(manifest["evaluation_days"])
    checkpoint = destination / "cn_drn_00.pt"
    restored = load_checkpoint("cn_drn", checkpoint, 3, settings, np.array([0.25, 0.5, 0.75]), file_hash(checkpoint))
    assert restored is not None

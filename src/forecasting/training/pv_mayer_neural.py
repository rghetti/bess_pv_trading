"""Train paper-inspired neural PV post-processors on prepared hourly cases.

This module does not run an experiment at import time. Each training example is
one forecast hour, with a control proxy, sorted member predictions and one
measured hourly AC-power target. The caller supplies an approved data split.
"""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from ...data import config_path
from ..data.pv_probabilistic_data import file_hash, write_json
from ..evaluation.pv_probabilistic_metrics import quantile_crps, raw_crps
from ..models.pv_mayer_neural import BernsteinQuantileNetwork
from ..models.pv_mayer_neural import CensoredGaussianNetwork
from ..models.pv_mayer_neural import NonCrossingQuantileNetwork
from ..models.pv_mayer_neural import QuantileNetwork
from ..models.pv_mayer_neural import censored_normal_crps, quantile_huber_loss
from ..studies.pv_mayer import censored_normal_mean, censored_normal_quantiles, summarize_scores


def prepare_examples(frame, member_count, every_n_days):
    """Separate fitting, early stopping and outer evaluation without target leakage."""
    member_columns = [name for name in frame.columns if name.startswith("member_")]
    member_columns = sorted(member_columns, key=lambda name: int(name.split("_")[1]))
    # Enforce the input contract: Expected `member_count` explicit member predictions.
    if len(member_columns) != member_count:
        raise ValueError(f"Expected {member_count} explicit member predictions")
    # Enforce the input contract: The early-stopping interval must be at least two days.
    if every_n_days < 2:
        raise ValueError("The early-stopping interval must be at least two days")

    # Reuse the earlier study's explicit target and forecast eligibility masks.
    fitting = frame.loc[
        frame["split"].eq("train")
        & frame["eligible"]
        & frame["fit_label_available"]
    ].copy()
    evaluation = frame.loc[frame["split"].eq("evaluation") & frame["eligible"]].copy()
    # Enforce the input contract: Both eligible fitting and outer evaluation cases are required.
    if fitting.empty or evaluation.empty:
        raise ValueError("Both eligible fitting and outer evaluation cases are required")

    # Every fifth calendar day is reserved for early stopping, following the paper.
    days = pd.to_datetime(fitting["delivery_day"])
    first_day = days.min().normalize()
    day_offsets = (days - first_day).dt.days
    stopping_mask = (day_offsets % every_n_days) == (every_n_days - 1)
    training = fitting.loc[~stopping_mask].copy()
    stopping = fitting.loc[stopping_mask].copy()
    # Enforce the input contract: The configured fitting window has no training or early-stopping cases.
    if training.empty or stopping.empty:
        raise ValueError("The configured fitting window has no training or early-stopping cases")

    # Fail on incomplete forecast rows rather than silently imputing members.
    for label, subset in (("training", training), ("early stopping", stopping), ("evaluation", evaluation)):
        columns = ["control", "y", *member_columns]
        # Enforce the input contract: Nonfinite `label` inputs or target.
        if not np.isfinite(subset[columns].to_numpy(float)).all():
            raise ValueError(f"Nonfinite {label} inputs or target")

    return training, stopping, evaluation, member_columns


def input_matrix(frame, member_columns):
    """Keep the deterministic proxy distinct from exchangeable sorted members."""
    control = frame["control"].to_numpy(np.float32)
    members = frame[member_columns].to_numpy(np.float32)
    sorted_members = np.sort(members, axis=1)
    return np.column_stack([control, sorted_members]).astype(np.float32)


def prepare_search_examples(frame, every_n_days):
    """Divide five-day blocks into 3 fit, 1 stopping and 1 trial-selection day."""
    # The paper's search protocol is defined on five-day calendar blocks.
    if every_n_days != 5:
        raise ValueError("The paper-style search requires five-day blocks")
    fitting = frame.loc[
        frame["split"].eq("train")
        & frame["eligible"]
        & frame["fit_label_available"]
    ].copy()
    # Hyperparameter selection cannot proceed without eligible historical targets.
    if fitting.empty:
        raise ValueError("No eligible fitting cases for hyperparameter search")
    days = pd.to_datetime(fitting["delivery_day"])
    offsets = (days - days.min().normalize()).dt.days % 5
    training = fitting.loc[offsets < 3].copy()
    stopping = fitting.loc[offsets == 3].copy()
    selection = fitting.loc[offsets == 4].copy()

    # Empty partitions cannot provide an honest early-stop or selection score.
    if training.empty or stopping.empty or selection.empty:
        raise ValueError("The paper-style search needs cases in all three block partitions")
    return training, stopping, selection


def trial_settings(settings, family, rng):
    """Draw one seeded configuration from the paper's reported search ranges."""
    trial = deepcopy(settings)
    low_width, high_width = settings["search_hidden_width"]
    layer_count = int(rng.integers(1, 3))
    trial["quantile_hidden_sizes"] = [
        int(rng.integers(low_width, high_width + 1))
        for _ in range(layer_count)
    ]
    trial["quantile_activation"] = str(rng.choice(["relu", "softplus", "sigmoid", "tanh"]))
    low_rate, high_rate = settings["search_learning_rate"]
    trial["learning_rate"] = float(np.exp(rng.uniform(np.log(low_rate), np.log(high_rate))))
    low_patience, high_patience = settings["search_patience"]
    trial["patience"] = int(rng.integers(low_patience, high_patience + 1))
    low_batch, high_batch = settings["search_batch_size"]
    trial["batch_size"] = int(round(np.exp(rng.uniform(np.log(low_batch), np.log(high_batch)))))

    # These additional parameters apply only to their respective architectures.
    if family == "bqn":
        lower, upper = settings["search_bernstein_degree"]
        trial["bernstein_degree"] = int(rng.integers(lower, upper + 1))
    # The NCQRNN hidden width must remain at least as large as its output grid.
    if family == "ncqrnn":
        lower, upper = settings["search_noncrossing_extra_width"]
        extra = int(rng.integers(lower, upper + 1))
        trial["noncrossing_width"] = int(settings["quantile_count"]) + extra
    return trial


def score_trial(job):
    """Train one candidate and score its separate within-fit selection days."""
    family, training, stopping, selection, members, settings, levels, seed = job
    try:
        model, record = train_network(family, training, stopping, members, settings, levels, seed)
        features = torch.from_numpy(input_matrix(selection, members))

        # Treat the paper's 51 predictive quantiles as an equally weighted ensemble.
        with torch.no_grad():
            prediction = model(features).numpy().astype(float)
        score = raw_crps(prediction, selection["y"].to_numpy(float)).mean()
        # A nonfinite validation score cannot rank a candidate.
        if not np.isfinite(score):
            raise ValueError(f"Nonfinite candidate score for {family}")
        return {
            "status": "complete",
            "settings": settings,
            "seed": seed,
            "selection_crps": float(score),
            "fit": record,
        }
    except Exception as error:
        # Preserve unsuccessful candidate settings instead of inventing a score.
        return {"status": "failed", "settings": settings, "seed": seed, "error": str(error)}


def search_family(frame, members, family, settings, levels, farm_id):
    """Rank seeded candidates without inspecting outer evaluation targets."""
    training, stopping, selection = prepare_search_examples(
        frame,
        int(settings["early_stopping_every_n_days"]),
    )
    trials = int(settings["hyperparameter_trials"])
    # An empty search cannot select a network for the final fit.
    if trials < 1:
        raise ValueError("At least one candidate is required for neural search")
    rng = np.random.default_rng(int(settings["seed"]) + farm_id)
    jobs = []

    # All candidate draws are fixed before outcomes are examined.
    for index in range(trials):
        candidate = trial_settings(settings, family, rng)
        seed = int(settings["seed"]) + farm_id + index
        jobs.append((family, training, stopping, selection, members, candidate, levels, seed))

    # Each worker uses one PyTorch thread to avoid nested CPU oversubscription.
    with ProcessPoolExecutor(max_workers=int(settings["search_workers"])) as pool:
        results = list(pool.map(score_trial, jobs))
    results.sort(key=lambda record: (record.get("selection_crps", float("inf")), record["seed"]))
    return results


def build_network(family, input_size, settings, levels):
    """Construct one of the four neural architectures in the Mayer paper."""
    # DRN returns the location and scale of the censored Gaussian.
    if family == "cn_drn":
        return CensoredGaussianNetwork(
            input_size,
            hidden_sizes=settings["drn_hidden_sizes"],
            minimum_scale=settings["minimum_scale"],
        )
    # QRNN predicts each quantile independently and can therefore cross.
    if family == "qrnn":
        return QuantileNetwork(
            input_size,
            hidden_sizes=settings["quantile_hidden_sizes"],
            activation=settings["quantile_activation"],
            quantile_count=len(levels),
        )
    # BQN represents an ordered quantile function with Bernstein coefficients.
    if family == "bqn":
        return BernsteinQuantileNetwork(
            input_size,
            hidden_sizes=settings["quantile_hidden_sizes"],
            activation=settings["quantile_activation"],
            degree=settings["bernstein_degree"],
            levels=levels,
        )
    # NCQRNN uses nonnegative cumulative increments to enforce quantile order.
    if family == "ncqrnn":
        return NonCrossingQuantileNetwork(
            input_size,
            hidden_sizes=settings["quantile_hidden_sizes"],
            activation=settings["quantile_activation"],
            quantile_count=len(levels),
            noncrossing_width=settings["noncrossing_width"],
        )
    raise ValueError(f"Unknown neural architecture: {family}")


def model_loss(model, family, features, target, levels, epsilon):
    """Use censored-Gaussian CRPS for DRN and smooth pinball loss otherwise."""
    prediction = model(features)
    # DRN predicts distribution parameters; the other networks predict quantiles.
    if family == "cn_drn":
        location, scale = prediction
        return censored_normal_crps(target, location, scale).mean()
    return quantile_huber_loss(target, prediction, levels, epsilon)


def train_network(family, training, stopping, member_columns, settings, levels, seed):
    """Fit one seeded model and retain the best early-stopping checkpoint."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    features = torch.from_numpy(input_matrix(training, member_columns))
    targets = torch.from_numpy(training["y"].to_numpy(np.float32))
    stop_features = torch.from_numpy(input_matrix(stopping, member_columns))
    stop_targets = torch.from_numpy(stopping["y"].to_numpy(np.float32))
    level_tensor = torch.as_tensor(levels, dtype=torch.float32)
    model = build_network(family, features.shape[1], settings, levels)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    generator = torch.Generator()
    generator.manual_seed(seed)
    best_score = float("inf")
    best_epoch = None
    best_state = None
    stale_epochs = 0
    history = []

    # Shuffle only the fitting examples; the held-out days never supply gradients.
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(features), generator=generator)
        # Apply one optimizer update per shuffled minibatch.
        for start in range(0, len(order), int(settings["batch_size"])):
            indices = order[start : start + int(settings["batch_size"])]
            loss = model_loss(
                model,
                family,
                features[indices],
                targets[indices],
                level_tensor,
                float(settings["huber_epsilon"]),
            )
            # Enforce the input contract: Nonfinite `family` training loss at epoch `epoch`.
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {family} training loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Select the checkpoint using held-out fit-period days, never outer evaluation.
        model.eval()
        with torch.no_grad():
            score = model_loss(
                model,
                family,
                stop_features,
                stop_targets,
                level_tensor,
                float(settings["huber_epsilon"]),
            )
        validation_score = float(score.item())
        # Enforce the input contract: Nonfinite `family` early-stopping score at epoch `epoch`.
        if not np.isfinite(validation_score):
            raise ValueError(f"Nonfinite {family} early-stopping score at epoch {epoch}")
        history.append({"epoch": epoch, "early_stopping_loss": validation_score})
        # Keep the checkpoint with the lowest loss on the early-stopping days.
        if validation_score < best_score:
            best_score = validation_score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        # Stop processing when `stale_epochs >= int(settings['patience'])`.
        if stale_epochs >= int(settings["patience"]):
            break

    # Restore the best checkpoint before generating any outer evaluation forecast.
    if best_state is None:
        raise ValueError(f"No finite checkpoint for {family}")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"seed": seed, "best_epoch": best_epoch, "best_loss": best_score, "history": history}


def load_checkpoint(family, checkpoint, input_size, settings, levels, expected_sha256):
    """Reconstruct an architecture and verify its saved weights before inference."""
    checkpoint = Path(checkpoint)
    # Reject a checkpoint that no longer matches the trained-run manifest.
    if file_hash(checkpoint) != expected_sha256:
        raise ValueError(f"Neural checkpoint checksum mismatch: {checkpoint}")

    # Model structure and quantile levels must match those saved with the run.
    model = build_network(family, input_size, settings, levels)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def make_prediction_table(frame, family, quantiles, means, crps, rating, levels, raw_quantiles):
    """Record unmodified output, physical validity and hourly verification values."""
    raw_crossing = (np.diff(raw_quantiles, axis=1) < 0).any(axis=1)
    out_of_bounds = ((raw_quantiles < 0) | (raw_quantiles > 1)).any(axis=1)
    valid = np.isfinite(quantiles).all(axis=1)
    valid &= (np.diff(quantiles, axis=1) >= 0).all(axis=1)
    valid &= (quantiles >= 0).all(axis=1)
    valid &= (quantiles <= 1).all(axis=1)
    valid &= np.isfinite(crps) & np.isfinite(means)
    target = frame["y"].to_numpy(float)
    residual = target[:, None] - quantiles
    pinball = np.maximum(levels * residual, (levels - 1.0) * residual).mean(axis=1)

    # Preserve the raw network output separately from any approved rearrangement.
    output = frame[["row_id", "delivery_day", "hour_start_utc", "lead_hours", "measured_ac_mw"]].copy()
    output["method"] = family
    output["valid_distribution"] = valid
    output["raw_crossing"] = raw_crossing
    output["raw_out_of_bounds"] = out_of_bounds
    output["crps_fraction_capacity"] = np.where(valid, crps, np.nan)
    output["pinball_mean_fraction_capacity"] = pinball
    output["mean_fraction_capacity"] = means
    output["median_fraction_capacity"] = [np.interp(0.5, levels, row) for row in quantiles]
    output["lower_90_fraction_capacity"] = [np.interp(0.05, levels, row) for row in quantiles]
    output["upper_90_fraction_capacity"] = [np.interp(0.95, levels, row) for row in quantiles]
    output["observation_fraction_capacity"] = target
    # Store both the usable and original value of every predicted quantile.
    quantile_columns = {}
    # Name columns by quantile index so the raw and rearranged curves remain paired.
    for index in range(len(levels)):
        quantile_columns[f"q_{index + 1:02d}"] = quantiles[:, index]
        quantile_columns[f"raw_q_{index + 1:02d}"] = raw_quantiles[:, index]
    quantile_frame = pd.DataFrame(quantile_columns, index=output.index)
    output = pd.concat([output, quantile_frame], axis=1)
    return output


def fit_farm(config, farm_id, source, destination):
    """Train one farm's four networks from an already prepared PV ensemble run."""
    settings = config["pv_mayer_neural"]
    input_path = source / f"farm_{farm_id}" / "hourly_inputs_normalized.parquet"
    frame = pd.read_parquet(input_path)
    training, stopping, evaluation, members = prepare_examples(
        frame,
        int(settings["member_count"]),
        int(settings["early_stopping_every_n_days"]),
    )
    levels = np.arange(1, int(settings["quantile_count"]) + 1)
    levels = levels / (int(settings["quantile_count"]) + 1)
    rating = float(config["farms"][str(farm_id)]["ac_capacity_kw"]) / 1000.0
    evaluation_features = torch.from_numpy(input_matrix(evaluation, members))
    evaluation_target = evaluation["y"].to_numpy(dtype=float, copy=True)
    destination.mkdir(parents=True, exist_ok=True)
    predictions = []
    model_records = []

    # Ten independently seeded DRNs average location and scale, as in the paper.
    for family_index, family in enumerate(("cn_drn", "qrnn", "bqn", "ncqrnn")):
        family_settings = settings
        # Quantile-network hyperparameters are selected using only fit-period days.
        if family != "cn_drn":
            search = search_family(frame, members, family, settings, levels, farm_id)
            write_json(destination / f"{family}_search.json", search)
            successful = [record for record in search if record["status"] == "complete"]
            # Do not silently substitute default hyperparameters after every trial fails.
            if not successful:
                raise ValueError(f"All {family} hyperparameter trials failed")
            family_settings = successful[0]["settings"]
        count = int(settings["drn_ensemble_size"]) if family == "cn_drn" else 1
        family_outputs = []
        # Fit each independently seeded DRN member or the single quantile model.
        for member_index in range(count):
            seed = int(settings["seed"]) + farm_id + family_index * 1000 + member_index
            model, record = train_network(family, training, stopping, members, family_settings, levels, seed)
            checkpoint = destination / f"{family}_{member_index:02d}.pt"
            torch.save(model.state_dict(), checkpoint)
            record["family"] = family
            record["architecture_settings"] = family_settings
            record["checkpoint"] = checkpoint.name
            record["checkpoint_sha256"] = file_hash(checkpoint)
            model_records.append(record)
            with torch.no_grad():
                family_outputs.append(model(evaluation_features))

        # For DRN, average the parameters before calculating the censored distribution.
        if family == "cn_drn":
            location = torch.stack([output[0] for output in family_outputs]).mean(dim=0)
            scale = torch.stack([output[1] for output in family_outputs]).mean(dim=0)
            loc = location.numpy().astype(float)
            std = scale.numpy().astype(float)
            raw = censored_normal_quantiles(loc, std, levels)
            quantiles = raw.copy()
            means = censored_normal_mean(loc, std)
            target_tensor = torch.as_tensor(evaluation_target, dtype=torch.float64)
            score = censored_normal_crps(target_tensor, location.double(), scale.double()).numpy()
        else:
            raw = family_outputs[0].numpy().astype(float)
            quantiles = raw.copy()
            # Rearrange QRNN quantiles only when the experiment enables it.
            if family == "qrnn" and settings["rearrange_qrnn"]:
                quantiles = np.sort(quantiles, axis=1)
            anchors = np.r_[0.0, levels, 1.0]
            bounded_curve = np.column_stack([quantiles[:, 0], quantiles, quantiles[:, -1]])
            means = np.trapezoid(bounded_curve, x=anchors, axis=1)
            score = quantile_crps(quantiles, levels, evaluation_target)

        predictions.append(make_prediction_table(
            evaluation,
            family,
            quantiles,
            means,
            score,
            rating,
            levels,
            raw,
        ))

    # Include the raw member distribution on the same evaluation rows.
    raw_members = evaluation[members].to_numpy(float)
    ensemble_quantiles = np.quantile(raw_members, levels, axis=1).T
    predictions.append(make_prediction_table(
        evaluation,
        "raw_ensemble",
        ensemble_quantiles,
        raw_members.mean(axis=1),
        raw_crps(raw_members, evaluation_target),
        rating,
        levels,
        ensemble_quantiles,
    ))
    combined = pd.concat(predictions, ignore_index=True)

    # Compare every method only where all predicted distributions are valid.
    common = combined.groupby("row_id")["valid_distribution"].all()
    combined["common_score"] = combined["row_id"].map(common).astype(bool)
    combined.to_parquet(destination / "evaluation_predictions.parquet", index=False)
    has_scores = summarize_scores(combined, levels, rating, destination)
    summary = {
        "farm_id": farm_id,
        "source_sha256": file_hash(input_path),
        "training_days": sorted(training["delivery_day"].unique()),
        "early_stopping_days": sorted(stopping["delivery_day"].unique()),
        "evaluation_days": sorted(evaluation["delivery_day"].unique()),
        "members": members,
        "model_records": model_records,
        "matched_hours": int(common.sum()),
        "comparison_available": has_scores,
        "retrospective": True,
        "ncqrnn_note": "Triangular nonnegative construction; exact original equations not reproduced by Mayer et al.",
    }
    write_json(destination / "manifest.json", summary)
    return summary


def run(config):
    """Train only after an explicit experiment approval; never overwrite a run."""
    settings = config["pv_mayer_neural"]
    source = config_path(config, settings["input_root"])
    source_manifest = source / "manifest.json"
    # Enforce the input contract: Prepared source manifest is unavailable: `source_manifest`.
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Prepared source manifest is unavailable: {source_manifest}")
    destination = config_path(config, settings["output_root"]) / settings["run_id"]
    # Enforce the input contract: Choose a new neural run ID: `destination`.
    if destination.exists():
        raise FileExistsError(f"Choose a new neural run ID: {destination}")
    destination.mkdir(parents=True)
    write_json(destination / "resolved_config.json", config)
    write_json(destination / "manifest.json", {
        "status": "running",
        "settings": settings,
        "source_manifest_sha256": file_hash(source_manifest),
        "torch_version": torch.__version__,
    })
    farm_summaries = []

    # Isolate each farm so its own predictions and targets determine its models.
    try:
        # Child processes inherit one-thread numerical-library limits.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = "1"
        # Fit each farm independently to avoid pooling across installations.
        for farm_id in settings["farm_ids"]:
            print(f"Training neural post-processors for farm {farm_id}", flush=True)
            summary = fit_farm(config, int(farm_id), source, destination / f"farm_{farm_id}")
            farm_summaries.append(summary)
    except Exception as error:
        write_json(destination / "manifest.json", {
            "status": "failed",
            "settings": settings,
            "error": str(error),
        })
        raise

    # A completed fit without a common valid evaluation subset remains diagnostic only.
    status = "complete"
    # Make an empty matched comparison visible in the top-level run status.
    if any(not summary["comparison_available"] for summary in farm_summaries):
        status = "complete_without_common_comparison"
    write_json(destination / "manifest.json", {
        "status": status,
        "settings": settings,
        "source_manifest_sha256": file_hash(source_manifest),
        "torch_version": torch.__version__,
        "training_code_sha256": file_hash(Path(__file__)),
        "model_code_sha256": file_hash(
            Path(__file__).resolve().parents[1] / "models" / "pv_mayer_neural.py"
        ),
        "farms": farm_summaries,
        "retrospective": True,
    })

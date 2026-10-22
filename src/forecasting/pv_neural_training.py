"""Deterministic CPU training with prior-validation-only early stopping."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .pv_neural_models import PVNetwork


def configure_cpu(section: dict):
    """Limit each independent farm worker; never select or initialize a CUDA device."""
    torch.set_num_threads(int(section["threads_per_worker"]))
    torch.set_num_interop_threads(int(section["interop_threads_per_worker"]))
    torch.use_deterministic_algorithms(bool(section["deterministic"]))


def tensor_pack(packed):
    """Keep small full-farm tensors in CPU memory to avoid loader/process overhead."""
    return {name: torch.from_numpy(np.array(packed[name], copy=True))
            for name in ("x", "y", "mask", "daylight", "month", "auction")}


def view_mse(prediction, target, mask):
    """Give each valid day/auction equal aggregate weight, including IDA3's suffix."""
    count = mask.sum(dim=1)
    valid = count > 0
    squared = (prediction-target)**2
    per_view = (squared*mask).sum(dim=1)/count.clamp(min=1)
    return per_view[valid].mean()


def _forward(model, tensors, positions):
    return model(tensors["x"][positions], tensors["month"][positions],
                 tensors["auction"][positions], tensors["daylight"][positions])


def predict_network(model, tensors, batch_size: int) -> np.ndarray:
    """Predict all input views in deterministic order, without consulting targets."""
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(tensors["x"]), batch_size):
            positions = slice(start, start+batch_size)
            pieces.append(_forward(model, tensors, positions).numpy())
    return np.concatenate(pieces)


def _validation_mse(model, tensors, positions, batch_size):
    """Accumulate validation error without including padded or unobserved rows."""
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chosen = positions[start:start+batch_size]
            mask = tensors["mask"][chosen]
            valid = mask.sum(dim=1) > 0
            if valid.any():
                value = view_mse(_forward(model, tensors, chosen), tensors["y"][chosen], mask)
                total += float(value)*int(valid.sum())
                count += int(valid.sum())
    if not count:
        raise ValueError("No finite daylight validation targets")
    return total/count


def fit_network(packed: dict, candidate: dict, section: dict, *, seed: int, progress=None):
    """Select epochs on validation only and retain the best immutable CPU weights."""
    started = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tensors = tensor_pack(packed)
    model = PVNetwork(packed["x"].shape[-1], candidate, section)
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=float(candidate.get("learning_rate", section["learning_rate"])),
        weight_decay=float(candidate.get("weight_decay", section["weight_decay"])))
    train = np.flatnonzero((packed["split"] == "train") & packed["mask"].any(axis=1))
    validation = np.flatnonzero((packed["split"] == "validation") & packed["mask"].any(axis=1))
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, int(section["max_epochs"])+1):
        model.train()
        order = rng.permutation(train)
        total_loss, batches = 0., 0
        batch_size = int(candidate.get("batch_size", section["batch_size"]))
        for start in range(0, len(order), batch_size):
            positions = order[start:start+batch_size]
            target, mask = tensors["y"][positions], tensors["mask"][positions]
            prediction = _forward(model, tensors, positions)
            residual = prediction-target
            error = residual**2
            loss_name = candidate.get("loss", section["loss"])
            if loss_name == "huber":
                delta = float(candidate.get("huber_delta", section["huber_delta"]))
                error = torch.where(abs(residual) <= delta, error, 2*delta*abs(residual)-delta**2)
            loss = ((error*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(section["gradient_clip_norm"]))
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        score = _validation_mse(model, tensors, validation, int(section["prediction_batch_size"]))
        history.append({"epoch": epoch, "training_loss": total_loss/max(batches, 1),
                        "validation_nrmse": float(np.sqrt(score))})
        if score < best-float(section["minimum_improvement"]):
            best, best_epoch, stale = score, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if progress and (epoch == 1 or epoch % 20 == 0):
            progress(f"{candidate['name']}: epoch {epoch}, validation NRMSE={100*np.sqrt(score):.3f}% AC")
        if stale >= int(section["patience"]):
            break
    if best_state is None or not np.isfinite(best):
        raise ValueError("Neural training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, {"candidate": candidate, "seed": seed, "epochs_completed": epoch,
        "best_epoch": best_epoch, "validation_nrmse": float(np.sqrt(best)),
        "runtime_seconds": time.perf_counter()-started,
        "parameter_count": sum(p.numel() for p in model.parameters()), "device": "cpu",
        "torch_version": str(torch.__version__), "history": history,
        "selection": "prior_validation_daylight_day_auction_balanced_nrmse",
        "test_targets_used_for_training_or_selection": False,
        "hyperparameters": {name: candidate[name] for name in (
            "width", "depth", "kernel_size", "dropout", "learning_rate", "weight_decay",
            "batch_size", "loss", "huber_delta", "month_embedding_dim", "auction_embedding_dim")
            if name in candidate}}


def save_network(model, path: Path, metadata: dict):
    """Atomically save tensor weights and an inspectable JSON contract/checksum."""
    temporary = path.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)
    metadata = {**metadata, "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    json_path = path.with_suffix(".json")
    temporary_json = json_path.with_suffix(".tmp")
    temporary_json.write_text(json.dumps(metadata, indent=2, default=str)+"\n")
    temporary_json.replace(json_path)


def load_network(path: Path, *, farm_id=None):
    """Load only checked CPU tensors; farm-specific normalization cannot be mixed."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    if farm_id is not None and int(metadata["farm_metadata"]["farm_id"]) != int(farm_id):
        raise ValueError("Neural artifact belongs to a different farm")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["weights_sha256"]:
        raise ValueError("Neural weights checksum mismatch")
    network = PVNetwork(len(metadata["preprocessor"]["output_columns"]),
                        metadata["candidate"], metadata["neural_settings"])
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    network.eval()
    return network, metadata

"""Target-free inference using a saved farm neural model and forecast input frame."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.pv_neural_data import feature_frame, pack_sequences, transform_features, unpack_predictions
from ..training.pv_neural_training import load_network, predict_network, tensor_pack


def predict_neural_frame(frame: pd.DataFrame, artifact: str | Path, *, farm_id: int) -> np.ndarray:
    """Forecast MW using only weather, timestamps and deterministic solar geometry.

    Any measured-power column supplied by a benchmark is explicitly discarded.
    A production caller supplies the same quarter-hour weather frame as the
    benchmark, plus date/auction/vintage identifiers and the daylight policy.
    """
    model, metadata = load_network(Path(artifact), farm_id=farm_id)
    candidate, section = metadata["candidate"], metadata["neural_settings"]
    inputs = frame.copy()
    inputs["measured_ac_mw"] = np.nan
    inputs["eligible"] = False
    inputs["split"] = "prediction"
    features = feature_frame(inputs, candidate["features"], section["delivery_timezone"])
    rating = metadata["farm_metadata"]["ac_capacity_kw"]/1000
    packed = pack_sequences(inputs, transform_features(features, metadata["preprocessor"]), rating)
    prediction = predict_network(model, tensor_pack(packed), int(section["prediction_batch_size"]))
    return unpack_predictions(packed, prediction, len(inputs), rating)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model_bundle() -> Tuple[object, object, list[str]]:
    """Load the released Logistic Regression model, StandardScaler and feature order."""
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    return model, scaler, feature_names


def predict_from_features(
    features: Dict[str, float],
    model: object,
    scaler: object,
    feature_names: list[str],
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Return the model probability and threshold-based research label."""
    frame = pd.DataFrame([features]).reindex(columns=feature_names).fillna(0.0)
    scaled = scaler.transform(frame.values)
    probability = float(model.predict_proba(scaled)[0][1])
    is_suspect = probability >= threshold
    return {
        "probability": probability,
        "label": "Suspect" if is_suspect else "Normal",
        "is_suspect": is_suspect,
        "threshold": threshold,
        "model_used": "Logistic Regression + GLCM Texture",
    }

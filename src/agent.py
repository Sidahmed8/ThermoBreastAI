from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .feature_extraction import extract_all_features
from .prediction import predict_from_features
from .preprocessing import preprocess_image, simulate_thermal_colormap

MEDICAL_WARNING = (
    "Research prototype only. This system is not a medical diagnostic device and does not replace "
    "clinical examination, mammography, ultrasound, MRI, biopsy, or medical advice."
)


@dataclass(frozen=True)
class AgentConfig:
    name: str
    threshold: float


MODES = {
    "Sensitive screening (research)": AgentConfig("Sensitive screening (research)", 0.15),
    "Default threshold": AgentConfig("Default threshold", 0.50),
    "Balanced research threshold": AgentConfig("Balanced research threshold", 0.60),
}


class ThermoBreastAgent:
    """Deterministic orchestration layer for validation, preprocessing, features and inference."""

    def __init__(self, model: object, scaler: object, feature_names: list[str], threshold: float = 0.5):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.threshold = threshold

    def run(self, img_bgr: Optional[np.ndarray]) -> Dict[str, Any]:
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.ndim not in (2, 3):
            return {"status": "error", "message": "Invalid image.", "warning": MEDICAL_WARNING}

        height, width = img_bgr.shape[:2]
        if height < 32 or width < 32:
            return {"status": "error", "message": "Image is too small for this research pipeline.", "warning": MEDICAL_WARNING}

        gray, normalized = preprocess_image(img_bgr)
        thermal_view = simulate_thermal_colormap(gray)
        features = extract_all_features(gray, normalized)
        prediction = predict_from_features(
            features,
            self.model,
            self.scaler,
            self.feature_names,
            threshold=self.threshold,
        )

        return {
            "status": "success",
            "processed": {"gray": gray, "normalized": normalized, "thermal_colormap": thermal_view},
            "features": features,
            "prediction": prediction,
            "warning": MEDICAL_WARNING,
        }

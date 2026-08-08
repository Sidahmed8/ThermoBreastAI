from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops

# Exact 12 GLCM features expected by the released Logistic Regression model.
MODEL_FEATURE_NAMES = [
    "glcm_contrast_mean", "glcm_contrast_std",
    "glcm_dissimilarity_mean", "glcm_dissimilarity_std",
    "glcm_homogeneity_mean", "glcm_homogeneity_std",
    "glcm_correlation_mean", "glcm_correlation_std",
    "glcm_ASM_mean", "glcm_ASM_std",
    "glcm_energy_mean", "glcm_energy_std",
]


def extract_texture_features(img_gray: np.ndarray) -> Dict[str, float]:
    """Extract GLCM features using 256 gray levels and four directions."""
    img_uint8 = img_gray if img_gray.dtype == np.uint8 else (img_gray * 255).astype(np.uint8)
    glcm = graycomatrix(
        img_uint8,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    features: Dict[str, float] = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
        values = graycoprops(glcm, prop).flatten()
        features[f"glcm_{prop}_mean"] = float(values.mean())
        features[f"glcm_{prop}_std"] = float(values.std())
    return features


def extract_thermal_features(img_normalized: np.ndarray) -> Dict[str, float]:
    """Compute descriptive thermal/intensity features used for interpretation only."""
    values = img_normalized.flatten()
    hot_mask = values >= np.percentile(values, 90)
    return {
        "mean_temperature": float(np.mean(values)),
        "std_temperature": float(np.std(values)),
        "min_temperature": float(np.min(values)),
        "max_temperature": float(np.max(values)),
        "median_temperature": float(np.median(values)),
        "percentile_90": float(np.percentile(values, 90)),
        "iqr_temperature": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "hot_region_ratio": float(np.sum(hot_mask) / len(values)),
        "hot_region_mean_temp": float(np.mean(values[hot_mask])) if hot_mask.any() else 0.0,
        "skewness": float(skew(values)),
        "kurtosis": float(kurtosis(values)),
    }


def extract_all_features(
    img_gray: np.ndarray,
    img_normalized: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if img_normalized is None:
        img_normalized = img_gray.astype(np.float32) / 255.0
    return {
        **extract_texture_features(img_gray),
        **extract_thermal_features(img_normalized),
    }

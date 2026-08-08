from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE: Tuple[int, int] = (224, 224)


def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """Load a Streamlit upload as a BGR uint8 NumPy array."""
    try:
        image = Image.open(uploaded_file).convert("RGB")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def preprocess_image(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    apply_clahe: bool = True,
    apply_denoise: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the same public inference preprocessing used by the project app."""
    img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    if apply_denoise:
        img_gray = cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

    img_normalized = img_gray.astype(np.float32) / 255.0
    return img_gray, img_normalized


def simulate_thermal_colormap(img_gray: np.ndarray) -> np.ndarray:
    """Create an INFERNO visualization for the preprocessed grayscale image."""
    img_uint8 = img_gray if img_gray.dtype == np.uint8 else (img_gray * 255).astype(np.uint8)
    colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

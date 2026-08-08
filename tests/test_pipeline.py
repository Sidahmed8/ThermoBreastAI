import numpy as np

from src.feature_extraction import MODEL_FEATURE_NAMES, extract_all_features
from src.prediction import load_model_bundle, predict_from_features
from src.preprocessing import preprocess_image


def test_model_bundle_and_feature_contract():
    model, scaler, feature_names = load_model_bundle()
    assert feature_names == MODEL_FEATURE_NAMES
    assert hasattr(model, "predict_proba")
    assert hasattr(scaler, "transform")


def test_end_to_end_inference_smoke_test():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

    gray, normalized = preprocess_image(image)
    features = extract_all_features(gray, normalized)
    model, scaler, feature_names = load_model_bundle()
    result = predict_from_features(features, model, scaler, feature_names, threshold=0.5)

    assert gray.shape == (224, 224)
    assert all(name in features for name in MODEL_FEATURE_NAMES)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in {"Normal", "Suspect"}

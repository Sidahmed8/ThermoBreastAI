from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agent import MEDICAL_WARNING, MODES, ThermoBreastAgent
from src.feature_extraction import MODEL_FEATURE_NAMES
from src.prediction import load_model_bundle
from src.preprocessing import load_image_from_upload

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "data" / "project_results"

st.set_page_config(page_title="ThermoBreastAI", page_icon="🌡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 1.5rem;}
    .hero {padding: 1.5rem 1.7rem; border: 1px solid #dbe7f3; border-radius: 18px; background: linear-gradient(135deg,#f7fbff,#eef7fb); margin-bottom: 1rem;}
    .hero h1 {margin:0; font-size:2.2rem;}
    .hero p {margin:.5rem 0 0; color:#4b6478; max-width:900px;}
    .note {padding: .9rem 1rem; border-radius: 12px; background:#fff8e8; border:1px solid #f3d58c;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🌡️ ThermoBreastAI</h1>
      <p>Interpretable machine-learning research prototype for breast thermal-image analysis. The public inference model uses 12 GLCM texture features with a Logistic Regression classifier.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="note"><b>Medical-use limitation:</b> {MEDICAL_WARNING}</div>', unsafe_allow_html=True)

model, scaler, feature_names = load_model_bundle()

tab_demo, tab_results, tab_method = st.tabs(["Interactive research demo", "Evaluation results", "Methodology"])

with tab_demo:
    left, right = st.columns([1, 1])
    with left:
        mode_name = st.selectbox("Decision threshold mode", list(MODES.keys()), index=1)
        threshold = MODES[mode_name].threshold
        st.caption(f"Decision threshold: {threshold:.2f}")
        upload = st.file_uploader("Upload a thermal-image sample", type=["png", "jpg", "jpeg"])

    if upload is not None:
        image = load_image_from_upload(upload)
        agent = ThermoBreastAgent(model, scaler, feature_names, threshold=threshold)
        output = agent.run(image)

        if output["status"] != "success":
            st.error(output["message"])
        else:
            pred = output["prediction"]
            with right:
                st.metric("Research label", pred["label"])
                st.metric("Model probability", f"{pred['probability']:.1%}")
                st.caption(pred["model_used"])

            c1, c2 = st.columns(2)
            with c1:
                st.image(output["processed"]["gray"], caption="Preprocessed grayscale image", clamp=True)
            with c2:
                st.image(output["processed"]["thermal_colormap"], caption="Thermal-style visualization")

            texture = {name: output["features"][name] for name in MODEL_FEATURE_NAMES}
            st.subheader("GLCM features used by the model")
            st.dataframe(
                pd.DataFrame({"feature": texture.keys(), "value": texture.values()}),
                use_container_width=True,
                hide_index=True,
            )

with tab_results:
    comparison = pd.read_csv(RESULTS_DIR / "final_model_comparison.csv")
    bootstrap = pd.read_csv(RESULTS_DIR / "bootstrap_ci_best_ml.csv")
    threshold_curve = pd.read_csv(RESULTS_DIR / "threshold_curve_best_ml.csv")
    groupkfold = pd.read_csv(RESULTS_DIR / "groupkfold_summary.csv", header=[0, 1])

    a, b, c, d = st.columns(4)
    a.metric("Best test AUC", "0.9526", "LogReg + Texture")
    b.metric("Sensitivity @ 0.50", "97.78%")
    c.metric("Best F1-score", "0.8974", "Thermal + Texture")
    d.metric("Test images", "300")

    top = comparison.sort_values("auc", ascending=False).head(12).copy()
    top["configuration"] = top["model"] + " · " + top["feature_type"]
    fig = px.bar(top.sort_values("auc"), x="auc", y="configuration", orientation="h", title="Top model configurations by AUC")
    fig.update_layout(xaxis_range=[0.75, 1.0], height=520)
    st.plotly_chart(fig, use_container_width=True)

    t = threshold_curve[["threshold", "sensitivity", "specificity", "f1_score", "FP", "FN"]].copy()
    melted = t.melt("threshold", ["sensitivity", "specificity", "f1_score"], var_name="metric", value_name="score")
    fig2 = px.line(melted, x="threshold", y="score", color="metric", markers=True, title="Threshold trade-off for Logistic Regression + Texture")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Bootstrap 95% confidence intervals")
    st.dataframe(bootstrap, use_container_width=True, hide_index=True)

    with st.expander("Patient-group cross-validation summary"):
        st.dataframe(groupkfold, use_container_width=True)

with tab_method:
    st.markdown(
        """
        ### Public inference pipeline
        1. Resize the uploaded image to **224×224**.
        2. Convert to grayscale, apply bilateral denoising and CLAHE.
        3. Compute a **256-level GLCM** in four directions.
        4. Extract 12 texture statistics: mean and standard deviation of contrast, dissimilarity, homogeneity, correlation, ASM and energy.
        5. Apply the released `StandardScaler` and **Logistic Regression** model.
        6. Convert the predicted probability into a research label using the selected threshold.

        ### Evaluation design represented in this repository
        - 1,522 thermal images from 56 patient groups are summarized by the project artifacts.
        - The held-out evaluation set contains 300 images.
        - Patient overlap between train and test was controlled in the project workflow.
        - Patient-group cross-validation (`GroupKFold`) and bootstrap confidence intervals are included as robustness checks.
        - Machine Learning, feature ablations, Deep Learning baselines, threshold analysis and a Pennes bio-heat simulation were compared during the research work.

        ### Important scope note
        The Pennes simulation is an explanatory research component and is **not** used to train the released classifier. This repository is an academic/research demonstration, not a clinical product.
        """
    )

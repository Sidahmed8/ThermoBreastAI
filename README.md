# 🌡️ ThermoBreastAI

**Interpretable Machine Learning for Breast Thermal-Image Analysis**

ThermoBreastAI is my Master's research project in Data Science and Data Engineering. It investigates whether thermal-image features can support the **research analysis of breast abnormalities** through a rigorous Machine Learning pipeline, interpretable features, patient-aware validation and an interactive Streamlit application.

> **Medical-use limitation:** This repository is an academic research prototype. It is **not a medical diagnostic device** and does not replace clinical examination, mammography, ultrasound, MRI, biopsy or medical advice.

## Why this project matters

Thermal imaging is non-ionizing and contactless, but Machine Learning results can be misleading when images from the same patient leak across training and evaluation sets. A central objective of this work was therefore not only predictive performance, but also **methodological rigor and interpretability**.

The project compares classical Machine Learning, Deep Learning baselines and multiple feature families, while using patient-group validation, threshold analysis and uncertainty estimation.

## Main results

| Configuration | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression + **Texture** | 0.7967 | **0.9778** | 0.5250 | 0.8523 | **0.9526** |
| Logistic Regression + **Thermal + Texture** | **0.8667** | 0.9722 | 0.7083 | **0.8974** | 0.9497 |
| Logistic Regression + Hybrid | 0.8467 | 0.9667 | 0.6667 | 0.8832 | 0.9170 |

For the best-AUC ML configuration, bootstrap analysis gives an AUC mean of **0.9528** with a **95% interval of approximately [0.9308, 0.9723]**.

Threshold analysis also illustrates the screening trade-off: at a threshold of **0.15**, sensitivity reaches **1.00** on the held-out result table, with lower specificity; higher thresholds improve specificity while increasing false negatives.

## Dataset / evaluation summary

Project artifacts report:

- **1,522** thermal images
- **762** normal and **760** suspect images
- **56** patient groups
- **1,222** train images and **300** held-out test images
- **0 patient overlap** in the project train/test workflow
- Patient-group robustness analysis with **GroupKFold**
- Bootstrap confidence intervals for the best ML model

Raw medical-image data are intentionally **not published** in this repository.

## Public inference model

The released interactive demo uses:

- 224×224 preprocessing
- grayscale conversion, bilateral denoising and CLAHE
- 256-level GLCM computation in 4 directions
- 12 GLCM texture features
- StandardScaler
- Logistic Regression

The model bundle included in `models/` was serialized with **scikit-learn 1.6.1**, which is pinned in `requirements.txt` for reproducibility.

## Research pipeline

```text
Thermal image
    ↓
Patient-aware data organization
    ↓
Preprocessing
    ↓
Feature extraction
    ├── Thermal statistics
    ├── GLCM texture
    └── Graph / hybrid representations (experiments)
    ↓
Model comparison
    ├── Logistic Regression
    ├── SVM RBF
    ├── XGBoost
    ├── Simple CNN
    └── MobileNetV2 transfer learning
    ↓
Held-out evaluation + GroupKFold
    ↓
Threshold analysis + bootstrap confidence intervals
    ↓
Interpretation and Streamlit research demo
```

## Additional research component: Pennes bio-heat simulation

The project also contains an explanatory Pennes bio-heat simulation comparing normal and suspect thermal-source scenarios. In the saved simulation summary, the suspect-source scenario shows a higher maximum simulated temperature and a hotspot increase of approximately **0.95°C** versus the normal simulation.

This simulation is used as a **research/physical interpretation component** and is **not used to train the public classifier**.

## Repository structure

```text
ThermoBreastAI/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── prediction.py
│   └── agent.py
└── data/
    └── project_results/
        ├── final_model_comparison.csv
        ├── ablation_best_by_feature.csv
        ├── bootstrap_ci_best_ml.csv
        ├── groupkfold_summary.csv
        ├── threshold_curve_best_ml.csv
        └── pennes_*.csv
```

## Run locally

```bash
git clone https://github.com/Sidahmed8/ThermoBreastAI.git
cd ThermoBreastAI
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

## Reproducibility and privacy

The repository publishes the inference code, compact trained model and aggregate evaluation artifacts needed to demonstrate the pipeline. Raw medical images and row-level error-analysis files are excluded to keep the public repository focused and privacy-conscious.

## Author

**Sidahmed Ahmedou Emeihimid**  
MSc Data Science & Data Engineering  
[LinkedIn](https://www.linkedin.com/in/sid-ahmed-emeihimid-7208002b3/) · [GitHub](https://github.com/Sidahmed8)

---

If you work on **medical imaging, thermal imaging, interpretable ML, Computer Vision or healthcare AI research**, I am open to research discussions and collaboration.

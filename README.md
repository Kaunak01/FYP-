# AI-based Credit Card Fraud Detection with Velocity Feature Engineering and Behavioural Profiling 

**Author:** Kaunak (Student ID: 001354164)  
**Module:** COMP1682 Final Year Project  
**Supervisor:** Dr. Yasmine Arafa  
**University of Greenwich — BSc Computer Science**

---

## Overview

FraudLens is a Flask-based web application that detects fraudulent credit card transactions using a three-model staged study:

| Model | Architecture | F1 Score | Inference Latency |
|-------|-------------|----------|-------------------|
| Baseline | XGBoost (SMOTE + tuned) | 0.8646 | 0.75 ms |
| Comparator | LSTM + Random Forest | 0.7892 | 88.97 ms |
| Proposed | AE + BDS + GA + XGBoost | 0.8706 | 1.62 ms |

The proposed model enriches each transaction with autoencoder reconstruction error and four behavioural deviation scores (BDS), then classifies via a GA-optimised XGBoost model.

## Dataset

Sparkov synthetic dataset (Shenoy, 2020) from Kaggle:
- 1,296,675 training transactions
- 555,719 test transactions
- 0.58% fraud rate (extreme class imbalance)

## Key Contributions

- **Velocity feature engineering** — `velocity_1h`, `velocity_24h`, `amount_velocity_1h` via pandas groupby + rolling windows
- **Behavioural Deviation Scoring (BDS)** — per-cardholder profiling across amount, time, category, and location dimensions
- **Genetic Algorithm optimisation** — principled, reproducible hyperparameter tuning for BDS thresholds
- **SHAP explainability** — top feature: `amount_velocity_1h` (mean |SHAP| = 2.32)

## Project Structure

```
FYP-Fraud-Detection/
├── run.py                      # Flask entry point
├── requirements.txt            # Python dependencies
├── app/                        # Flask application
│   ├── main.py                 # App factory
│   ├── config.py               # Thresholds, paths, feature columns
│   ├── database.py             # SQLite manager
│   ├── api/                    # REST API (/api/predict, /api/health, etc.)
│   ├── dashboard/              # Web UI (6 pages)
│   ├── models/                 # Model manager + drift detector
│   ├── pipeline/               # Preprocessor, rule engine, postprocessor
│   ├── templates/              # Jinja2 HTML templates
│   └── static/                 # CSS, JS, sample CSVs
├── models/saved/               # Trained model artefacts
│   ├── 01_baseline/            # XGBoost baseline (.joblib)
│   ├── 02_comparator/          # LSTM + RF (.keras, .joblib)
│   ├── 03_proposed/            # AE (.pt), BDS profiles, GA params, XGBoost
│   └── supplementary/          # Additional model variants
├── data/                       # Raw and engineered datasets
├── notebooks/                  # Jupyter notebooks (model development)
│   ├── 01_EDA.ipynb            # Exploratory data analysis + feature engineering
│   ├── 02_Hybrid_Model.ipynb   # LSTM + Random Forest pipeline
│   ├── 03_Autoencoder_XGBoost.ipynb  # AE + XGBoost + SHAP
│   └── 04_BDS_GA.ipynb         # BDS + Genetic Algorithm optimisation
├── scripts/                    # Utility and verification scripts
├── tests/                      # Pytest unit tests (13 tests)
├── testing/                    # Comprehensive test suites (8 suites)
├── results/                    # Evaluation metrics and audit reports
├── figures/                    # Visualisations (EDA, BDS, SHAP, GA)
├── configs/                    # Model configuration JSONs
└── docs/                       # Project documentation
```

## Setup and Installation

```bash
git clone https://github.com/<username>/FYP-Fraud-Detection.git
cd FYP-Fraud-Detection
pip install -r requirements.txt
```

## Running the Application

```bash
python run.py
```

Open `http://localhost:5000` in your browser.

### Web UI Features

- **Dashboard** — overview statistics and active model info
- **Predict** — manual transaction entry with real-time fraud scoring
- **Analyse** — cardholder history, velocity analysis, category breakdown
- **Monitor** — real-time alert queue with three-tier triage (FRAUD ≥ 0.70, REVIEW ≥ 0.50, MONITOR ≥ 0.30)
- **Performance** — three-model comparison tables
- **Settings** — model switching and threshold configuration

## Running Tests

```bash
python -m pytest tests/ -v
```

All 13 unit tests cover: BDS scoring, Flask routes, velocity features, triage bands, and train-serve consistency.

## Evaluation Results (Proposed Model, threshold = 0.5)

| Metric | Value |
|--------|-------|
| F1 | 0.8706 |
| Precision | 0.9338 |
| Recall | 0.8154 |
| ROC-AUC | 0.9976 |
| PR-AUC | 0.9158 |

Operational threshold (0.70): Precision = 0.9629, Recall = 0.7869.

## Technologies

- Python 3.14, Flask, SQLite
- XGBoost, scikit-learn, PyTorch, Keras
- SHAP, DEAP (genetic algorithm)
- Bootstrap 5, Chart.js

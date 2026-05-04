# FraudLens — Hybrid AI Credit Card Fraud Detection System

**Author:** Kaunak Bhattacharya  
**Degree:** BSc (Hons) Computer Science, University of Greenwich  
**Supervisor:** Dr. Arafa  
**Year:** 2026  

## Overview

FraudLens is a hybrid machine-learning system for credit card fraud detection under extreme class imbalance (0.52% fraud rate), deployed as a Flask web application. It integrates personalised behavioural profiling, anomaly detection, and engineered transactional features within a unified feature-fusion pipeline.

The system implements a **three-model staged study**:

| Model | Role | F1 Score |
|-------|------|----------|
| XGBoost (SMOTE+Tuned) | Baseline | 0.8646 |
| LSTM + Random Forest | Comparator | 0.7892 |
| **AE + BDS + GA + XGBoost** | **Proposed** | **0.8706** |

Median inference latency: **1.62 ms** per transaction.

## Personal Contributions

1. **Three velocity features** — `velocity_1h`, `velocity_24h`, `amount_velocity_1h` — capturing short-window burst patterns using 1-hour and 24-hour rolling windows, validated through SHAP attribution, ablation, and McNemar's significance test.

2. **Parameterised Behavioural Deviation Score (BDS)** — a 4-dimension score (amount, time, frequency, merchant category) computed per cardholder, operationalising Bolton and Hand's (2002) conceptual framework into a concrete, deployable scoring mechanism.

3. **From-scratch Genetic Algorithm** — population-based optimisation (30 population, 20 generations, tournament selection, arithmetic crossover, Gaussian mutation) to tune BDS parameters using F1 as the fitness function.

4. **Hybrid feature-fusion pipeline** — autoencoder reconstruction error treated as one feature among many (not a standalone threshold), fused with velocity and BDS features for downstream XGBoost classification.

5. **FraudLens Flask application** — 7-page analyst dashboard with per-transaction SHAP explanations, live SSE streaming, rule-engine layer, PSI drift monitoring, and PDF report generation.

## Project Structure

```
FYP-Fraud-Detection/
├── run.py                      # Entry point (python run.py)
├── app/                        # Flask application
│   ├── main.py                 # App factory
│   ├── config.py               # Configuration & thresholds
│   ├── database.py             # SQLite (5 tables)
│   ├── api/
│   │   ├── routes.py           # /api/predict, /api/health, /api/model/switch
│   │   └── simulation.py       # SSE streaming for live monitoring
│   ├── dashboard/
│   │   └── routes.py           # 7 dashboard pages
│   ├── models/
│   │   ├── model_manager.py    # Loads & serves 5 models + SHAP
│   │   └── drift_detector.py   # PSI-based drift monitoring
│   ├── pipeline/
│   │   ├── preprocessor.py     # Raw transaction → 14 features
│   │   ├── rule_engine.py      # 5 hard fraud rules
│   │   └── postprocessor.py    # SHAP explanation + triage report
│   └── templates/              # 7 Jinja2 HTML pages
├── models/saved/               # Trained model artefacts
│   ├── 01_baseline/            # XGBoost (SMOTE+Tuned)
│   ├── 02_comparator/          # LSTM + Random Forest
│   ├── 03_proposed/            # AE + BDS + GA + XGBoost
│   └── supplementary/          # Ablation variants
├── data/
│   ├── raw/                    # Sparkov dataset (1.85M transactions)
│   └── engineered/             # Feature-engineered train/test CSVs
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis + feature engineering
│   ├── 02_Hybrid_Model.ipynb   # LSTM + Random Forest
│   ├── 03_Autoencoder_XGBoost.ipynb  # AE + XGBoost + SHAP
│   └── 04_BDS_GA.ipynb         # BDS + Genetic Algorithm
├── tests/                      # Pytest unit tests (13/13 passing)
├── testing/                    # 8 comprehensive test suites
├── scripts/                    # Utility & figure generation scripts
├── results/                    # Verified metrics & evaluation outputs
├── figures/                    # EDA & model visualisations
└── configs/                    # Model hyperparameter configs
```

## Dataset

**Sparkov synthetic transaction dataset** (Shenoy, 2020):
- 1,852,394 transactions (Jan 2019 – Dec 2020)
- 9,651 fraudulent (0.52% fraud rate)
- Chronological train/test split at 2020-06-21
- SMOTE applied to training fold only

## 14 Engineered Features

| Feature | Description |
|---------|-------------|
| `amt` | Transaction amount |
| `hour`, `month`, `day_of_week_encoded` | Temporal features |
| `is_night`, `is_weekend` | Binary flags |
| `distance_cardholder_merchant` | Haversine distance |
| `city_pop`, `age` | Demographic |
| `category_encoded`, `gender_encoded` | Encoded categoricals |
| **`velocity_1h`** | Transactions per card in last 1 hour |
| **`velocity_24h`** | Transactions per card in last 24 hours |
| **`amount_velocity_1h`** | Total spend per card in last 1 hour |

Bold = personal contribution (velocity features).

## Quick Start

```bash
# Install dependencies
pip install flask scikit-learn xgboost torch keras shap pandas numpy joblib

# Run the application
python run.py

# Open in browser
# http://localhost:5000

# Run unit tests
pytest tests/ -v

# Run comprehensive test suites
python testing/run_all_tests.py
```

## Dashboard Pages

| Page | URL | Purpose |
|------|-----|---------|
| Home | `/` | System stats, active model info |
| Predict | `/predict` | Manual transaction scoring |
| Monitor | `/monitor` | Live SSE stream + alert queue |
| Analyse | `/analyse` | Cardholder history + attack timeline |
| Performance | `/performance` | 3-model comparison + ablation tables |
| Settings | `/settings` | Model switching + threshold config |

## Evaluation

All models evaluated under the same protocol: F1, precision, recall, ROC-AUC, PR-AUC, and per-transaction inference latency. Complemented by McNemar's significance testing, SHAP-based interpretability, and controlled ablation.

| Metric | Baseline | Comparator | Proposed |
|--------|----------|------------|----------|
| F1 | 0.8646 | 0.7892 | **0.8706** |
| Precision | 0.9297 | 0.6770 | 0.9338 |
| Recall | 0.8079 | 0.9459 | 0.8154 |
| ROC-AUC | 0.9972 | 0.9981 | **0.9976** |
| PR-AUC | 0.9092 | 0.9517 | **0.9158** |
| Latency (median) | 0.75 ms | 88.97 ms | **1.62 ms** |

## Testing

- **13 pytest unit tests** — BDS scoring, velocity features, triage bands, Flask routes, train-serve consistency
- **8 comprehensive test suites** — single transaction, scenarios, boundary, adversarial, stress, temporal, model comparison, simulation
- **~85% coverage** on core ML pipeline

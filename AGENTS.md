# FYP: Credit Card Fraud Detection

## Project Overview
- **Student:** Kaunak, BSc Computer Science, University of Greenwich (graduating July 2026)
- **Supervisor:** Dr. Arafa (Yasmine)
- **Title:** "Hybrid Machine Learning for Credit Card Fraud Detection Using Transaction Behavioural Features and Synthetic Oversampling"
- **Dataset:** Sparkov synthetic dataset (Kartik2112) — 1.85M transactions, 0.58% fraud rate
- **Files:** fraudTrain.csv (1,296,675 rows), fraudTest.csv (555,719 rows)
- **Engineered files:** fraudTrain_engineered.csv, fraudTest_engineered.csv (16 columns, 14 used as features)

## Features (14 used for training, from engineered CSVs)
Original/derived features: amt, city_pop, age, hour, month, distance_cardholder_merchant, category_encoded, gender_encoded, day_of_week_encoded, is_weekend, is_night
Velocity features (personal contribution): velocity_1h, velocity_24h, amount_velocity_1h
Target: is_fraud
Drop column: unix_time
Note: lat, long, merch_lat, merch_long were dropped by EDA notebook and replaced by distance_cardholder_merchant. is_weekend, is_night, month, day_of_week_encoded were engineered in EDA notebook.

## Notebooks in This Folder

### 1. EDA_FYP_FINAL.ipynb
- 327 cells (111 code, 216 markdown), fully complete
- EDA + preprocessing + feature engineering
- Generates the engineered CSVs with velocity features
- 27+ visualisations
- No models trained — all modeling deferred to downstream notebooks

### 2. FYP_Hybrid_Model.ipynb
- 15 cells — Hybrid 1: LSTM + Random Forest
- LSTM (64 units, Dropout 0.3) trained for 5 epochs
- Best F1 = 0.47 (LSTM prob + RF v2)
- SMOTE on merged features made it worse (F1 = 0.28)

### 3. FYP_Autoencoder_XGBoost.ipynb
- 40 cells (27 code, 13 markdown) — Hybrid 2: Autoencoder + XGBoost
- PyTorch autoencoder (14→10→5→10→14) trained on normal transactions only
- Includes XGBoost-only baselines (no AE), AE+XGBoost (3 configs), ablation study, SHAP
- Best F1 = 0.87 (AE+XGBoost SMOTE+tuned)

### 4. FYP_BDS_GA.ipynb
- 37 cells (24 code, 13 markdown) — BDS algorithm + GA optimisation
- Behavioural Deviation Scoring: 4 per-cardholder deviation scores
- GA from scratch (pop=30, gen=20) optimises 10 BDS parameters
- Best F1 = 0.868 (AE+BDS(GA)+XGBoost SMOTE+tuned)
- Full SHAP analysis on 19-feature model

## Results Summary (All Verified)

### XGBoost Only (in Autoencoder notebook, no autoencoder)
- Class weights only → F1 = (see notebook output)
- SMOTE → F1 = (see notebook output)
- SMOTE + tuned → F1 = (see notebook output)

### Hybrid 1: LSTM + RF
- Best config (v2) → F1 = 0.47

### Hybrid 2: AE + XGBoost
- Class weights → F1 = 0.52, SMOTE → F1 = 0.84, SMOTE+tuned → **F1 = 0.87**

### Hybrid 2 + BDS(GA)
- Class weights → F1 = 0.53, SMOTE → F1 = 0.83, SMOTE+tuned → **F1 = 0.868**

### Ablation Study
- WITH velocity features: F1 = 0.8705
- WITHOUT velocity features: F1 = 0.8561
- Velocity contribution: +0.0144

### SHAP Top 5 (from AE+XGBoost)
1. is_night (2.86), 2. amt (1.95), 3. amount_velocity_1h (1.65), 4. hour (1.06), 5. category_encoded (1.02)

## Error Analysis (from AE+XGBoost best model)
- Missed frauds average $218 vs $600 for caught frauds
- 56.2% of missed frauds are under $50
- Missed frauds have lower reconstruction error (0.66 vs 1.79)

## What's Been Done
- ✅ Dataset selection and loading
- ✅ EDA with 27+ graphs
- ✅ Feature engineering (velocity features)
- ✅ Hybrid 1: LSTM + RF (F1 = 0.47)
- ✅ Hybrid 2: AE + XGBoost (F1 = 0.87)
- ✅ BDS algorithm + GA optimisation (F1 = 0.868)
- ✅ Ablation study (velocity features +0.014 F1)
- ✅ SHAP analysis (velocity feature ranked #3)
- ✅ Error analysis on best model

## What's Next
1. **Flask app** — Demo for the project
2. **Dissertation writeup**

## Hardware
- Windows laptop with RTX 3050 (8GB VRAM)
- Python 3.14 installed locally (TensorFlow not supported — using PyTorch instead)
- Packages: numpy, pandas, scikit-learn, xgboost, matplotlib, seaborn, shap, imbalanced-learn, torch

## Important Notes
- All notebooks were originally built in Google Colab — file paths referencing `/content/drive/MyDrive/FYP_Fraud_Detection/` need to be updated to local paths (same folder as notebooks)
- The Colab `drive.mount()` calls should be skipped/removed when running locally
- XGBoost runs on CPU — no GPU needed
- LSTM/TensorFlow can use GPU if needed
- The project is CPU-heavy (XGBoost, RF, SHAP) — GPU only matters for LSTM retraining

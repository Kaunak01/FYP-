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
- **Corrected pipeline** uses fraudTrain/Test_engineered_with_ids.csv (composite-key merge)
- LSTM (64 units, Dropout 0.3) trained for 5 epochs → RF on merged features
- **Corrected F1 = 0.7892** (LSTM_reproduced_baseline in verified_metrics.json)
- Historical broken F1 = 0.4747 preserved only in FYP_Hybrid_Model_BROKEN.ipynb (positional `.values` attach bug, fixed 2026-04-22)

### 3. FYP_Autoencoder_XGBoost.ipynb
- 47 cells — Hybrid 2: Autoencoder + XGBoost
- PyTorch autoencoder (14→10→5→10→14) trained on normal transactions only
- Includes XGBoost-only baselines (no AE), AE+XGBoost (3 configs), ablation study, SHAP
- **Best saved F1 = 0.8690** (AE+XGBoost SMOTE+tuned)
- Post-Codex-audit additions (code ready, pending rerun): true from-scratch no-velocity ablation; SMOTE-inside-CV Pipeline tuning

### 4. FYP_BDS_GA.ipynb
- 43 cells — BDS algorithm + GA optimisation
- Behavioural Deviation Scoring: 4 per-cardholder deviation scores, **train-only profiles** (test txns never contribute; streaming past-only is future work)
- GA from scratch (pop=30, gen=20) optimises 10 BDS parameters
- **Best saved F1 = 0.8706** (AE+BDS(GA)+XGBoost SMOTE+tuned) ← best model
- Full SHAP analysis on 19-feature model
- Post-Codex-audit addition (pending rerun): SMOTE-inside-CV Pipeline tuning

## Results Summary (authoritative — see verified_metrics.json)

### XGBoost Only (no AE)
- Class weights → F1 = 0.5215, SMOTE+tuned → F1 = 0.8646

### Hybrid 1: LSTM + RF
- Corrected → **F1 = 0.7892** | Broken historical → F1 = 0.4747 (not for citation)

### Hybrid 2: AE + XGBoost
- Class weights → F1 = 0.52 | SMOTE → F1 = 0.84 | SMOTE+tuned → **F1 = 0.8690**

### Hybrid 2 + BDS(GA)
- Class weights → F1 = 0.53 | SMOTE → F1 = 0.83 | SMOTE+tuned → **F1 = 0.8706 ← best**

### Velocity ablation (current implementation — input-layer only)
- WITH velocity features: F1 = 0.8705
- WITHOUT velocity features (input removed, AE recon_error still velocity-derived): F1 = 0.8561
- Velocity contribution: +0.0144
- A true from-scratch ablation (fresh AE retrained without velocity) is staged in the notebook and `run_audit_retrains.py --true-no-velocity`; results pending rerun.

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
- ✅ Hybrid 1: LSTM + RF — corrected pipeline F1 = 0.7892 (broken historical 0.4747 preserved)
- ✅ Hybrid 2: AE + XGBoost — F1 = 0.8690
- ✅ BDS algorithm + GA optimisation — F1 = 0.8706 (best)
- ✅ Ablation study (velocity features +0.014 F1, input-layer)
- ✅ SHAP analysis (velocity feature ranked #3)
- ✅ Error analysis on best model
- ✅ Codex audit fixes (P1/P3/P5/P7) — notebook source-of-truth reconciled
- ⏳ Pending Codex audit reruns (P2/P4) — true no-velocity ablation + Pipeline-CV tuning, runnable via `python run_audit_retrains.py --all`

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

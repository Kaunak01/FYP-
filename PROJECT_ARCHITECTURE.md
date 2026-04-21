# PROJECT ARCHITECTURE — Paste this into Claude Code

Here is the full project architecture and pipeline. Understand this before doing anything.

## DATASET FLOW
```
Raw Data (Sparkov/Kartik2112)
├── fraudTrain.csv (1,296,675 rows, 0.58% fraud)
├── fraudTest.csv (555,719 rows)
│
▼ EDA + Feature Engineering (EDA_FYP_FINAL.ipynb — 322 cells, DONE)
│   ├── 30+ visualisations
│   ├── Encoding (category, gender)
│   ├── Distance calculation (haversine)
│   ├── Time features (hour, day_of_week)
│   └── Velocity features ← PERSONAL CONTRIBUTION
│       ├── velocity_1h (tx count per card in last 1hr)
│       ├── velocity_24h (tx count per card in last 24hr)
│       └── amount_velocity_1h (total spend per card in last 1hr)
│
▼ Engineered Data (14 features + target)
├── fraudTrain_engineered.csv
└── fraudTest_engineered.csv
```

## MODEL PIPELINE (Total: 6 trained models expected)
```
ROUND 1 — Models WITHOUT velocity features (13 features)
┌─────────────────────────────────────────────────────┐
│ Model 1: XGBoost (Baseline — class weights only)    │ ← DONE (F1=0.52)
│ Model 2: XGBoost (SMOTE)                            │ ← DONE (F1=0.72)
│ Model 3: XGBoost (SMOTE + RandomizedSearchCV)       │ ← DONE (F1=0.83)
│ Model 4: Hybrid LSTM + Random Forest                │ ← DONE (F1=0.47)
└─────────────────────────────────────────────────────┘

ABLATION STUDY — Prove velocity features matter
┌─────────────────────────────────────────────────────┐
│ Model 5: XGBoost (SMOTE+tuned) WITHOUT velocity     │ ← TODO
│          features (only 11 features)                 │
│          Compare F1 to Model 3 (0.83)               │
│          Expected: F1 drops significantly            │
└─────────────────────────────────────────────────────┘

EXPLAINABILITY
┌─────────────────────────────────────────────────────┐
│ SHAP Analysis on Model 3 (best XGBoost)             │ ← TODO
│   ├── Summary plot (global feature importance)      │
│   ├── Bar plot (mean |SHAP|)                        │
│   ├── Dependence plots for top features             │
│   └── PROVE velocity features rank in top 5         │
└─────────────────────────────────────────────────────┘

COMPARISON TABLE (for report)
┌──────────────────────┬────────┬───────┬────────┬─────────┐
│ Model                │ F1     │ Prec  │ Recall │ ROC-AUC │
├──────────────────────┼────────┼───────┼────────┼─────────┤
│ XGBoost (baseline)   │ 0.52   │  ?    │  ?     │  ?      │
│ XGBoost (SMOTE)      │ 0.72   │  ?    │  ?     │  ?      │
│ XGBoost (SMOTE+tuned)│ 0.83   │  ?    │  ?     │  ?      │
│ XGBoost (NO velocity)│  TODO  │ TODO  │ TODO   │ TODO    │
│ Hybrid LSTM+RF       │ 0.47   │  ?    │  ?     │  ?      │
└──────────────────────┴────────┴───────┴────────┴─────────┘

OPTIONAL ROUND 2 (if time allows — from 9-step plan)
┌─────────────────────────────────────────────────────┐
│ Model 6: LSTM + FFNN (supervised hybrid)            │ ← OPTIONAL
│ Model 7: Autoencoder + XGBoost (anomaly hybrid)     │ ← OPTIONAL
│ + Spending deviation feature engineering            │ ← OPTIONAL
│ + Retrain Models 6&7 with new feature               │
│ + Compare Round 1 vs Round 2                        │
└─────────────────────────────────────────────────────┘

FLASK APP
┌─────────────────────────────────────────────────────┐
│ Simple web demo using best model (XGBoost F1=0.83)  │ ← TODO
│   ├── Input: transaction features                   │
│   ├── Output: fraud/not fraud + probability         │
│   └── Optional: SHAP explanation per prediction     │
└─────────────────────────────────────────────────────┘
```

## TASK PRIORITY ORDER
```
1. [NOW]      Ablation study — XGBoost without velocity features
2. [NEXT]     SHAP analysis on best XGBoost
3. [THEN]     Fill comparison table with all metrics
4. [THEN]     Flask app demo
5. [BONUS]    LSTM+FFNN hybrid (9-step plan)
6. [BONUS]    Autoencoder+XGBoost hybrid (9-step plan)
7. [BONUS]    Federated learning simulation
```

## ERROR ANALYSIS (already done on XGBoost)
- Missed frauds average $120 vs $599 for caught frauds
- 63.5% of missed frauds under $50 (small transactions slip through)
- Missed frauds skew toward daytime/weekends

## KEY RESEARCH FINDINGS
1. XGBoost (F1=0.83) significantly outperforms hybrid LSTM+RF (F1=0.47)
2. Velocity features capture sequential patterns — making LSTM redundant
3. SMOTE + hyperparameter tuning is critical for imbalanced fraud data
4. Small-amount frauds (<$50) remain hardest to detect

## FILE PATHS
- All CSVs and notebooks are in the SAME folder (current directory)
- DO NOT use Google Drive mount paths — everything is local
- Remove/skip any drive.mount() cells from notebooks

# FULL PROJECT CONTEXT — FYP Credit Card Fraud Detection
# Use this prompt to brief any AI (Codex, Gemini, GPT-4, etc.) on the full state of this project
# so it can make changes, verify correctness, or extend the codebase.

---

## WHO I AM & WHAT THIS IS

I am Kaunak, a BSc Computer Science student at the University of Greenwich (graduating July 2026).
My Final Year Project is titled:
**"Hybrid Machine Learning for Credit Card Fraud Detection Using Transaction Behavioural Features and Synthetic Oversampling"**
Supervised by Dr. Arafa (Yasmine).

This is a complete end-to-end fraud detection system: 4 Jupyter notebooks (research/training) +
a full Flask web app (demo). All code runs locally on Windows 11 with Python 3.14.
The working directory is: C:\Users\User\OneDrive\Desktop\FYP-Fraud-Detection\

---

## DATASET

- **Source:** Sparkov synthetic credit card dataset (Kaggle: Kartik2112)
- **Size:** 1.85M transactions, 0.58% fraud rate (very imbalanced)
- **Files:** fraudTrain.csv (1,296,675 rows), fraudTest.csv (555,719 rows)
- **Engineered files:** fraudTrain_engineered.csv, fraudTest_engineered.csv (produced by EDA notebook)

---

## FEATURES (14 used for training)

```
FEATURE_COLS = [
    'amt',                          # Transaction amount
    'city_pop',                     # City population of cardholder
    'hour',                         # Hour of transaction (0-23)
    'month',                        # Month (1-12)
    'distance_cardholder_merchant', # Haversine distance in km
    'age',                          # Cardholder age
    'is_weekend',                   # 1 if Saturday/Sunday, else 0
    'is_night',                     # 1 if 22:00-05:59, else 0
    'velocity_1h',                  # Number of transactions by this card in last 1h
    'velocity_24h',                 # Number of transactions by this card in last 24h
    'amount_velocity_1h',           # Total $ spent by this card in last 1h
    'category_encoded',             # Merchant category integer (0-13)
    'gender_encoded',               # M=1, F=0
    'day_of_week_encoded',          # Monday=0 ... Sunday=6
]
Target: is_fraud (0 or 1)
Dropped: unix_time, lat, long, merch_lat, merch_long
```

velocity_1h, velocity_24h, amount_velocity_1h are my original contribution (personal feature engineering).
lat/long replaced by distance_cardholder_merchant using Haversine formula.

---

## RESEARCH NOTEBOOKS (in project root)

### 1. EDA_FYP_FINAL.ipynb
- 327 cells (111 code, 216 markdown)
- Full EDA, preprocessing, feature engineering
- Generates the engineered CSVs with velocity features
- 27+ visualisations
- NO models trained here

### 2. FYP_Hybrid_Model.ipynb — Hybrid 1: LSTM + Random Forest
- LSTM architecture: 64 units, Dropout(0.3), trained 5 epochs on sequential data
- LSTM output probabilities concatenated with original 14 features → fed to RF
- Results: F1=0.47 (v2 config), SMOTE made it worse (F1=0.28)
- Underperformed due to class imbalance and LSTM not learning fraud sequences well

### 3. FYP_Autoencoder_XGBoost.ipynb — Hybrid 2: Autoencoder + XGBoost
- **Autoencoder architecture (PyTorch):** 14→10→5→10→14
  - Encoder: Linear(14,10), ReLU, Dropout(0.2), Linear(10,5), ReLU
  - Decoder: Linear(5,10), ReLU, Dropout(0.2), Linear(10,14)
  - Trained ONLY on normal (non-fraud) transactions
  - Reconstruction error = anomaly score (fraud = high recon error)
- **XGBoost trained on 15 features** (14 base + reconstruction error)
- Three configs tested:
  - Class weights only → F1=0.52
  - SMOTE → F1=0.84
  - SMOTE + hyperparameter tuning → **F1=0.87 (best)**
- Ablation study:
  - WITH velocity features: F1=0.8705
  - WITHOUT velocity features: F1=0.8561
  - Contribution of velocity features: **+0.0144 F1**
- SHAP analysis top 5 features (by mean |SHAP|):
  1. is_night (2.86)
  2. amt (1.95)
  3. amount_velocity_1h (1.65)
  4. hour (1.06)
  5. category_encoded (1.02)
- Error analysis on missed frauds:
  - Missed frauds average $218 vs $600 for caught frauds
  - 56.2% of missed frauds are under $50 (small amounts evade detection)
  - Missed frauds have lower recon error (0.66 vs 1.79 for caught)

### 4. FYP_BDS_GA.ipynb — BDS Algorithm + Genetic Algorithm Optimisation
- **BDS (Behavioural Deviation Scoring):** 4 per-cardholder deviation scores:
  1. Amount deviation: how much this transaction deviates from cardholder's mean spend
  2. Time deviation: how unusual the transaction hour is for this cardholder
  3. Frequency deviation: how unusual the velocity is compared to cardholder's normal pace
  4. Category deviation: how unusual the merchant category is for this cardholder
- These 4 BDS scores added as extra features → 19 features total
- **GA (Genetic Algorithm) — written from scratch:**
  - Population: 30 chromosomes, Generations: 20
  - Optimises 10 BDS parameters (thresholds + caps for each of the 4 deviation scores + smoothing)
  - Fitness function: F1 score on validation set
- Results:
  - AE+BDS(GA)+XGBoost class weights → F1=0.53
  - AE+BDS(GA)+XGBoost SMOTE → F1=0.83
  - AE+BDS(GA)+XGBoost SMOTE+tuned → **F1=0.868**
- Full SHAP analysis on 19-feature model

---

## FINAL RESULTS SUMMARY

| Model | F1 |
|---|---|
| LSTM + RF (Hybrid 1) | 0.47 |
| AE + XGBoost (class weights) | 0.52 |
| AE + XGBoost (SMOTE) | 0.84 |
| AE + XGBoost (SMOTE + tuned) | **0.87** ← best |
| AE + BDS(GA) + XGBoost (class weights) | 0.53 |
| AE + BDS(GA) + XGBoost (SMOTE) | 0.83 |
| AE + BDS(GA) + XGBoost (SMOTE + tuned) | **0.868** |

Best overall: **AE + XGBoost SMOTE+tuned, F1 = 0.87**

---

## FLASK WEB APP (app/ folder)

The Flask app is a production-grade fraud detection demo. It wraps all trained models and exposes
a dashboard + REST API. Entry point: `python run.py` → runs at http://localhost:5000

### File Structure
```
FYP-Fraud-Detection/
├── run.py                          # Entry point
├── app/
│   ├── __init__.py
│   ├── main.py                     # Flask app factory (create_app)
│   ├── config.py                   # All paths, feature list, thresholds, mappings
│   ├── database.py                 # SQLite database manager
│   ├── pipeline/
│   │   ├── preprocessor.py         # Raw input → 14 model features
│   │   ├── rule_engine.py          # Hard fraud rules (5 rules)
│   │   └── postprocessor.py        # SHAP + human-readable explanations
│   ├── models/
│   │   ├── model_manager.py        # Loads/switches all 4 models, SHAP, BDS, AE
│   │   └── drift_detector.py       # PSI-based data drift monitoring
│   ├── api/
│   │   ├── routes.py               # REST API endpoints
│   │   └── simulation.py          # SSE simulation endpoint
│   ├── dashboard/
│   │   └── routes.py               # HTML page routes
│   └── templates/
│       ├── base.html
│       ├── welcome.html
│       ├── dashboard.html          # Main dashboard
│       ├── predict.html            # Single transaction prediction
│       ├── batch.html              # Batch CSV upload
│       ├── analyse.html            # Dataset analysis
│       ├── monitor.html            # Live simulation monitor
│       ├── alerts.html             # Fraud alert management
│       ├── model_performance.html  # Model metrics + drift
│       └── settings.html          # Model switching
├── models/
│   └── saved/
│       ├── xgboost_baseline_cw.joblib          # XGBoost class weights only
│       ├── xgboost_smote_tuned.joblib          # XGBoost SMOTE+tuned (14 features)
│       ├── ae_model.pt                          # PyTorch Autoencoder state dict
│       ├── ae_scaler.joblib                     # StandardScaler for AE input
│       ├── ae_xgboost_smote_tuned.joblib        # AE+XGBoost (15 features)
│       ├── bds_profiles.joblib                  # BDS global stats + cardholder profiles
│       ├── ga_best_params.json                  # GA-optimised BDS parameters
│       └── ae_bds_xgboost_smote_tuned.joblib   # AE+BDS+XGBoost (19 features)
└── models/
    └── stats/
        ├── training_stats.json      # Per-feature stats (mean, std, p99, etc.) for normal vs fraud
        ├── category_mapping.json    # Category name ↔ integer code mapping
        └── category_aliases.json   # Bank term → Sparkov category name aliases
```

---

## KEY MODULE DETAILS

### app/config.py
- Defines all paths, FEATURE_COLS list, thresholds, risk level bands
- DEFAULT_THRESHOLD = 0.5, OPTIMAL_THRESHOLD = 0.53
- RISK_LEVELS = {LOW: 0-0.2, MEDIUM: 0.2-0.5, HIGH: 0.5-0.8, CRITICAL: 0.8-1.0}
- GENDER_MAP and DAY_MAP for encoding

### app/main.py — create_app()
Initialises in order: Database → ModelManager → Preprocessor → RuleEngine → Postprocessor → DriftDetector
Then registers blueprints: api_bp (/api/*), sim_bp (/api/simulation/*), dashboard_bp (/)

### app/database.py — SQLite (WAL mode)
Tables:
- transactions: full feature vector + prediction results per transaction
- alerts: flagged transactions (FRAUD or REVIEW), with status workflow
- feedback: analyst label corrections
- card_history: rolling window store for velocity computation
- model_metrics: periodic performance snapshots

Key methods:
- get_card_velocity(card_number, timestamp) → (velocity_1h, velocity_24h, amount_velocity_1h)
- store_transaction(txn_data) — INSERT OR REPLACE
- store_alert / get_alerts / update_alert_status
- get_stats() — summary counts for dashboard
- reset() — clears all tables before simulation

### app/pipeline/preprocessor.py — Preprocessor
Converts raw bank transaction JSON → 14-feature dict for the model.
Handles: amount sanitization, timestamp parsing, haversine distance, category encoding (2-step alias lookup),
gender encoding, velocity computation from DB or override.
Input field aliases supported: amount/amt, city_population/city_pop, timestamp/trans_date_trans_time,
cardholder_age/age, cardholder_gender/gender, merchant_category/category, card_number/cc_num, etc.
Falls back to training medians for missing fields.

### app/pipeline/rule_engine.py — RuleEngine
5 hard rules that run alongside the ML model:
1. VELOCITY_SPIKE: velocity_1h >= 5 AND amount_velocity_1h > $500 → HIGH risk
2. AMOUNT_ANOMALY: amt > cardholder_mean + 3*std → MEDIUM risk (needs cardholder profile)
3. FIRST_TXN_HIGH_VALUE: velocity_24h <= 1 AND amt > $300 → MEDIUM risk
4. NIGHTTIME_HIGH_VALUE: is_night=1 AND amt > $200 → HIGH risk
5. RAPID_ESCALATION: 3+ transactions with strictly increasing amounts → CRITICAL risk (sequence-based)

combine_decision() logic:
- model_probability >= threshold → FRAUD
- model says normal BUT rules HIGH/CRITICAL → REVIEW
- model says normal AND rules MEDIUM → MONITOR
- else → NORMAL

### app/models/model_manager.py — ModelManager
Loads all 4 XGBoost models + autoencoder + BDS profiles + GA params on startup.
Active model defaults to 'XGBoost (SMOTE+Tuned)'.

Autoencoder class (PyTorch):
```python
class Autoencoder(nn.Module):
    def __init__(self, d):
        self.encoder = Sequential(Linear(d,10), ReLU, Dropout(0.2), Linear(10,5), ReLU)
        self.decoder = Sequential(Linear(5,10), ReLU, Dropout(0.2), Linear(10,d))
```

Feature counts per model:
- XGBoost (Class Weights) or (SMOTE+Tuned): 14 features
- AE+XGBoost: 15 features (14 + recon_error)
- AE+BDS+XGBoost: 19 features (14 + recon_error + 4 BDS scores)

predict(features_dict) → builds correct feature array, runs model, returns probability + processing_time_ms
predict_with_shap(features_dict) → same but also runs SHAP TreeExplainer (slower, used for single txn UI)

compute_recon_error(features_array): scales input, passes through AE, returns MSE(original, reconstructed)
compute_bds_scores(features_dict): computes 4 deviation scores using GA-optimised params and global stats

### app/models/drift_detector.py — DriftDetector
PSI (Population Stability Index) — banking industry standard:
- PSI < 0.1: GREEN (no drift)
- 0.1 <= PSI < 0.25: YELLOW (moderate drift, monitor)
- PSI >= 0.25: RED (significant drift, retrain)

Baseline metrics stored: F1=0.8646, precision=0.9297, recall=0.8079, roc_auc=0.9972

Methods:
- compute_psi(expected, actual, n_bins=10): PSI from scratch using numpy histograms
- check_prediction_drift(recent_f1): compares to baseline
- check_feature_drift(recent_data): per-feature PSI vs training distribution
- check_prediction_distribution(recent_probabilities): checks flag rate abnormalities
- generate_report(): combines all checks into one status

### app/pipeline/postprocessor.py — Postprocessor
format_prediction() returns a full report dict with:
- risk_assessment: probability, risk_level, headline text
- toward_fraud / away_from_fraud: top SHAP factors with plain English explanations
- recommendation: action text by risk level (BLOCK / HOLD / MONITOR / PROCESS)
- rules: triggered rule reasons
- population_comparison: z-scores vs normal/fraud training distributions for all 14 features
- shap_chart_data: pre-formatted for Chart.js bar chart
- summary: plain text for alerts table

### app/api/routes.py — REST API Blueprint (/api/*)
Endpoints:
- POST /api/predict — single transaction prediction (with SHAP)
- POST /api/predict/batch — batch prediction (no SHAP, fast)
- GET  /api/health — system health check
- GET  /api/model/info — loaded models, active model, feature count
- POST /api/model/switch — switch active model by name
- GET  /api/model/performance — stats + drift report
- POST /api/feedback — analyst label feedback
- GET  /api/alerts — get alerts (optional ?status= filter)
- PUT  /api/alerts/<id> — update alert status (confirmed/false_alarm/dismissed)
- GET  /api/stats — dashboard counts
- GET  /api/recent — recent transactions
- POST /api/analyse — rich dataset analysis with performance metrics if labels provided

All endpoints have in-memory rate limiting (100 req/min per IP).

### app/api/simulation.py — Simulation Blueprint (/api/simulation/*)
Server-Sent Events (SSE) simulation:
- POST /api/simulation/start — streams demo transactions in real-time (configurable delay)
- GET  /api/simulation/status — current DB stats
- GET  /api/simulation/transactions — demo transaction list + act breakdown

Reads from app/demo_transactions.json — a curated set of transactions with:
- pre-computed velocity fields
- actual_is_fraud labels
- act (e.g. "Act 1: Normal Shopping") and narration text for storytelling
- resets DB before each run

### app/dashboard/routes.py — Page Routes
Pages:
- GET / → dashboard.html (stats + model info)
- GET /welcome → welcome.html
- GET /predict → predict.html (single transaction form)
- GET /batch → batch.html (CSV upload)
- GET /analyse → analyse.html
- GET /monitor → monitor.html (live simulation)
- GET /alerts → alerts.html
- GET /performance → model_performance.html
- GET /settings → settings.html (model switch)
- GET /api/sample/<normal|mixed|fraud> → downloads sample CSV from test set

---

## TECHNOLOGY STACK

- Python 3.14 (Windows 11)
- Flask (web framework)
- PyTorch (autoencoder — CPU inference)
- XGBoost (main classifier — CPU)
- scikit-learn (SMOTE via imbalanced-learn, StandardScaler, RandomForest)
- SHAP (TreeExplainer for XGBoost)
- SQLite (database via sqlite3, WAL mode)
- numpy, pandas, joblib
- NO TensorFlow (not supported on Python 3.14 — PyTorch used instead)
- Frontend: Jinja2 templates + Chart.js + plain JS (no React/Vue)

---

## WHAT IS COMPLETE vs WHAT MAY NEED WORK

### FULLY COMPLETE (research side):
- All 4 notebooks (EDA, Hybrid1, Hybrid2, BDS+GA)
- All model training, ablation study, SHAP, error analysis
- All models saved in models/saved/
- All stats saved in models/stats/

### FLASK APP — BACKEND COMPLETE:
- config.py, database.py, preprocessor.py, rule_engine.py, postprocessor.py
- model_manager.py (loads all models, SHAP, BDS, AE)
- drift_detector.py (PSI from scratch)
- api/routes.py (all REST endpoints)
- api/simulation.py (SSE simulation)
- dashboard/routes.py (all page routes)
- run.py (entry point)

### FLASK APP — FRONTEND (templates):
- app/templates/base.html, welcome.html, dashboard.html, predict.html,
  batch.html, analyse.html, monitor.html, alerts.html,
  model_performance.html, settings.html
- These templates exist but may need UI polish or bug fixes

---

## HOW PREDICTION WORKS (end to end)

1. Raw JSON arrives at POST /api/predict
2. Preprocessor.process(raw_input):
   - Parses timestamp → hour, month, day_of_week, is_weekend, is_night
   - Encodes category (bank term → alias → Sparkov code)
   - Computes velocity from DB card history (or override if provided)
   - Sanitizes/clamps all numeric fields
   - Returns feature dict (14 keys in FEATURE_COLS order) + metadata
3. ModelManager.predict_with_shap(features):
   - Builds base 14-feature array
   - If AE model: compute recon_error, append → 15 features
   - If BDS model: compute 4 BDS scores, append → 19 features
   - XGBoost.predict_proba() → fraud probability
   - SHAP TreeExplainer → shap_values array
4. RuleEngine.evaluate(features) → checks 5 hard rules
5. RuleEngine.combine_decision(probability, rule_result) → FRAUD / REVIEW / MONITOR / NORMAL
6. Postprocessor.format_prediction() → full human-readable report
7. Database: store transaction + card history + alert if flagged
8. Return JSON response

---

## IMPORTANT NOTES FOR MAKING CHANGES

1. FEATURE ORDER IS CRITICAL. FEATURE_COLS in config.py must always be:
   ['amt', 'city_pop', 'hour', 'month', 'distance_cardholder_merchant', 'age',
    'is_weekend', 'is_night', 'velocity_1h', 'velocity_24h', 'amount_velocity_1h',
    'category_encoded', 'gender_encoded', 'day_of_week_encoded']
   The saved XGBoost models were trained on exactly this order. Do NOT reorder.

2. AE model expects 14 inputs. Its input MUST be scaled with ae_scaler.joblib first.

3. BDS models need bds_profiles.joblib (global stats) AND ga_best_params.json (10 params).
   The GA params dict has key 'params' which is a dict of 10 float values used in order:
   at, ac, tt, tc, ft, fc, ct, cc_, mh, sm (amount_threshold, amount_cap, time_threshold, time_cap,
   freq_threshold, freq_cap, cat_threshold, cat_cap, max_hour_prob, smoothing)

4. Category encoding: 14 categories (codes 0-13). Category names from Sparkov dataset.
   category_mapping.json has name_to_code and code_to_name.
   category_aliases.json maps bank terms to Sparkov names.

5. Velocity features: velocity_1h and velocity_24h are COUNTS (including current transaction).
   amount_velocity_1h is cumulative $ (caller adds current transaction amount on top of DB query result).

6. The Flask app uses application factory pattern (create_app() returns app, db, model_manager).
   All route functions capture db/model_manager via closure from init_*() functions.

7. Thresholds: DEFAULT_THRESHOLD=0.5 is used for classification.
   OPTIMAL_THRESHOLD=0.53 is stored but not yet used everywhere — this is an area for improvement.

8. The simulation uses SSE (Server-Sent Events) streaming — requires Flask's stream_with_context.
   demo_transactions.json must exist in app/ folder for simulation to work.

9. Python 3.14 on Windows — no TensorFlow. PyTorch only for the autoencoder.
   XGBoost and scikit-learn run on CPU.

---

## WHAT I MIGHT ASK YOU TO DO

Possible tasks:
- Fix bugs in Flask templates or API endpoints
- Improve the dashboard UI
- Add a new feature or endpoint
- Review the code for correctness
- Verify the ML pipeline logic
- Help write dissertation sections based on the code
- Generate sample data or test cases
- Debug an error message I paste

When making changes:
- Do NOT change FEATURE_COLS order
- Do NOT change model loading logic in model_manager.py without understanding the feature counts (14/15/19)
- Prefer editing existing files over creating new ones
- The app must remain runnable with `python run.py` from the project root

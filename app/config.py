"""Application configuration."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- Paths ----
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')
STATS_DIR = os.path.join(BASE_DIR, 'models', 'stats')
DB_PATH = os.path.join(BASE_DIR, 'app', 'fraud_detection.db')
LOG_PATH = os.path.join(BASE_DIR, 'app', 'logs', 'fraud_detection.log')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'engineered')  # engineered CSVs
SIM_DATA_DIR = os.path.join(BASE_DIR, 'data')  # simulation datasets

# ---- Model files ----
MODEL_FILES = {
    'xgb_cw': os.path.join(MODELS_DIR, 'xgboost_baseline_cw.joblib'),
    'xgb_tuned': os.path.join(MODELS_DIR, 'xgboost_smote_tuned.joblib'),
    'ae_model': os.path.join(MODELS_DIR, 'ae_model.pt'),
    'ae_scaler': os.path.join(MODELS_DIR, 'ae_scaler.joblib'),
    'ae_xgb': os.path.join(MODELS_DIR, 'ae_xgboost_smote_tuned.joblib'),
    'bds_profiles': os.path.join(MODELS_DIR, 'bds_profiles.joblib'),
    'ga_params': os.path.join(MODELS_DIR, 'ga_best_params.json'),
    'bds_xgb': os.path.join(MODELS_DIR, 'ae_bds_xgboost_smote_tuned.joblib'),
}

STATS_FILES = {
    'training_stats': os.path.join(STATS_DIR, 'training_stats.json'),
    'category_mapping': os.path.join(STATS_DIR, 'category_mapping.json'),
    'category_aliases': os.path.join(STATS_DIR, 'category_aliases.json'),
}

# ---- Features ----
FEATURE_COLS = [
    'amt', 'city_pop', 'hour', 'month', 'distance_cardholder_merchant',
    'age', 'is_weekend', 'is_night', 'velocity_1h', 'velocity_24h',
    'amount_velocity_1h', 'category_encoded', 'gender_encoded',
    'day_of_week_encoded'
]

# ---- Thresholds ----
DEFAULT_THRESHOLD  = 0.70  # prob >= 0.70 → FRAUD
REVIEW_THRESHOLD   = 0.50  # prob >= 0.50 → REVIEW (or rules HIGH/CRITICAL)
MONITOR_THRESHOLD  = 0.30  # prob >= 0.30 → MONITOR (or rules MEDIUM)
OPTIMAL_THRESHOLD  = 0.70  # kept for backwards compatibility

# ---- Risk levels ----
RISK_LEVELS = {'LOW': (0.0, 0.2), 'MEDIUM': (0.2, 0.5), 'HIGH': (0.5, 0.8), 'CRITICAL': (0.8, 1.01)}

# ---- Decision outcomes ----
# FRAUD:   model prob >= threshold
# REVIEW:  model says normal BUT rules say HIGH or CRITICAL
# MONITOR: model says normal AND rules say MEDIUM
# NORMAL:  model says normal AND no rules (or rules say NONE)

# ---- Server ----
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

# ---- Rate limiting ----
RATE_LIMIT_PER_MINUTE = 100

# ---- Model name mappings ----
# code name (from frontend) → internal ModelManager name
MODEL_CODE_TO_INTERNAL = {
    'xgboost_baseline': 'XGBoost (Class Weights)',
    'xgboost_smote':    'XGBoost (SMOTE+Tuned)',
    'ae_xgboost':       'AE+XGBoost',
    'ae_bds_xgboost':   'AE+BDS+XGBoost',
}
# internal name → display name shown in UI
MODEL_INTERNAL_TO_DISPLAY = {
    'XGBoost (Class Weights)': 'XGBoost Baseline',
    'XGBoost (SMOTE+Tuned)':   'XGBoost + SMOTE',
    'AE+XGBoost':              'Autoencoder + XGBoost',
    'AE+BDS+XGBoost':          'AE + BDS + XGBoost',
}
# display name → F1 score label (4-decimal precision, per verified_metrics.json @ threshold=0.5)
MODEL_F1_LABELS = {
    'xgboost_baseline': 'F1: 0.5215',
    'xgboost_smote':    'F1: 0.8646',
    'ae_xgboost':       'F1: 0.8690',
    'ae_bds_xgboost':   'F1: 0.8706',
}

# ---- 3-Model Staged Study framing ----
# Category each .joblib model belongs to in the dissertation framing.
# 'comparator' (LSTM + RF) is reported as a comparison row only — not a runtime .joblib.
MODEL_CATEGORIES = {
    'XGBoost (Class Weights)': 'supplementary',
    'XGBoost (SMOTE+Tuned)':   'baseline',
    'AE+XGBoost':              'proposed_component',
    'AE+BDS+XGBoost':          'proposed',
}

MODEL_CATEGORY_LABELS = {
    'baseline':           'Baseline',
    'comparator':         'Hybrid Comparator',
    'proposed_component': 'Proposed Model — Component',
    'proposed':           'Proposed Model',
    'supplementary':      'Supplementary',
}

MODEL_DESCRIPTIONS = {
    'XGBoost (Class Weights)': 'Supplementary baseline variant — class-weighted, no SMOTE',
    'XGBoost (SMOTE+Tuned)':   'Strong tabular baseline using gradient boosting',
    'AE+XGBoost':              'Proposed Model component (without BDS) — adds anomaly signal',
    'AE+BDS+XGBoost':          'Proposed Model — anomaly + behavioural hybrid (default)',
}

# Headline 3-Model Staged Study metrics — sourced from results/verified_metrics.json @ threshold=0.5.
# Used for Table A (main comparison) and Table B (Proposed Model component analysis) on the
# Model Performance page. LSTM PR-AUC is "n/a (not recorded)" in verified_metrics.json.
STAGED_STUDY_TABLE_A = [
    {'category': 'Baseline',           'model': 'XGBoost (SMOTE+tuned)',      'f1': 0.8646, 'precision': 0.9297, 'recall': 0.8079, 'roc_auc': 0.9972, 'pr_auc': 0.9092},
    {'category': 'Hybrid Comparator',  'model': 'LSTM + Random Forest',       'f1': 0.7892, 'precision': 0.6770, 'recall': 0.9459, 'roc_auc': 0.9981, 'pr_auc': None},
    {'category': 'Proposed Model',     'model': 'AE + BDS(GA) + XGBoost',     'f1': 0.8706, 'precision': 0.9338, 'recall': 0.8154, 'roc_auc': 0.9976, 'pr_auc': 0.9158},
]

STAGED_STUDY_TABLE_B = [
    {'variant': 'XGBoost only',           'tests': 'Without anomaly/BDS',     'f1': 0.8646, 'precision': 0.9297, 'recall': 0.8079, 'roc_auc': 0.9972, 'pr_auc': 0.9092},
    {'variant': 'AE + XGBoost',           'tests': 'Adds autoencoder',        'f1': 0.8690, 'precision': 0.9369, 'recall': 0.8103, 'roc_auc': 0.9973, 'pr_auc': 0.9142},
    {'variant': 'AE + BDS(GA) + XGBoost', 'tests': 'Adds behavioural scores', 'f1': 0.8706, 'precision': 0.9338, 'recall': 0.8154, 'roc_auc': 0.9976, 'pr_auc': 0.9158},
]

# ---- Simulation datasets ----
import os as _os
SIMULATION_DATASETS = {
    'demo':   {'label': 'Demo Transactions (102 rows) — Quick Test',  'file': None,         'rows': 102},
    'sim10k': {'label': 'Simulation 10K (10,000 rows, ~41 frauds)',   'file': _os.path.join(SIM_DATA_DIR, 'simulation_data.csv'),     'rows': 10000},
    'sim20k': {'label': 'Simulation 20K (20,000 rows, ~120 frauds)',  'file': _os.path.join(SIM_DATA_DIR, 'simulation_20k.csv'),      'rows': 20000},
    'sim50k': {'label': 'Simulation 50K (50,000 rows, ~300 frauds)',  'file': _os.path.join(SIM_DATA_DIR, 'simulation_50k.csv'),      'rows': 50000},
}

# ---- SHAP feature display names ----
FEATURE_DISPLAY_NAMES = {
    'amt':                          'Transaction Amount ($)',
    'hour':                         'Hour of Day',
    'is_night':                     'Nighttime (10pm–6am)',
    'category_encoded':             'Merchant Category',
    'velocity_1h':                  'Transactions (Last Hour)',
    'velocity_24h':                 'Transactions (Last 24h)',
    'amount_velocity_1h':           'Amount Spent (Last Hour $)',
    'age':                          'Cardholder Age',
    'city_pop':                     'City Population',
    'distance_cardholder_merchant': 'Distance to Merchant (km)',
    'gender_encoded':               'Gender',
    'day_of_week_encoded':          'Day of Week',
    'is_weekend':                   'Weekend',
    'month':                        'Month',
    'recon_error':                  'Anomaly Score (Autoencoder)',
    'bds_amount':                   'Amount Deviation (BDS)',
    'bds_time':                     'Time Deviation (BDS)',
    'bds_freq':                     'Frequency Deviation (BDS)',
    'bds_category':                 'Category Novelty (BDS)',
}

# ---- Gender encoding ----
GENDER_MAP = {'M': 1, 'F': 0, 'MALE': 1, 'FEMALE': 0, 'm': 1, 'f': 0}

# ---- Day of week encoding ----
DAY_MAP = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
           'Friday': 4, 'Saturday': 5, 'Sunday': 6,
           'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
           'friday': 4, 'saturday': 5, 'sunday': 6,
           'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}

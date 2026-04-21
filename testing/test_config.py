"""Test configuration — all constants, paths, thresholds, feature definitions."""
import os
import json

# ============================================================
# PATHS (relative to project root)
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Data files
TRAIN_CSV = os.path.join(PROJECT_ROOT, 'fraudTrain_engineered.csv')
TEST_CSV = os.path.join(PROJECT_ROOT, 'fraudTest_engineered.csv')
TRAIN_RAW_CSV = os.path.join(PROJECT_ROOT, 'fraudTrain.csv')
TEST_RAW_CSV = os.path.join(PROJECT_ROOT, 'fraudTest.csv')

# Model files
MODEL_XGB_CW = os.path.join(PROJECT_ROOT, 'xgboost_baseline_cw.joblib')
MODEL_XGB_TUNED = os.path.join(PROJECT_ROOT, 'xgboost_smote_tuned.joblib')
MODEL_AE_PT = os.path.join(PROJECT_ROOT, 'ae_model.pt')
MODEL_AE_SCALER = os.path.join(PROJECT_ROOT, 'ae_scaler.joblib')
MODEL_AE_XGB = os.path.join(PROJECT_ROOT, 'ae_xgboost_smote_tuned.joblib')
MODEL_BDS_PROFILES = os.path.join(PROJECT_ROOT, 'bds_profiles.joblib')
MODEL_GA_PARAMS = os.path.join(PROJECT_ROOT, 'ga_best_params.json')
MODEL_BDS_XGB = os.path.join(PROJECT_ROOT, 'ae_bds_xgboost_smote_tuned.joblib')

# Config files
CATEGORY_MAPPING_FILE = os.path.join(PROJECT_ROOT, 'category_mapping.json')
TRAINING_STATS_FILE = os.path.join(PROJECT_ROOT, 'training_stats.json')

# ============================================================
# THRESHOLDS
# ============================================================
DEFAULT_THRESHOLD = 0.5
OPTIMAL_THRESHOLD = 0.53

# ============================================================
# RISK LEVELS
# ============================================================
RISK_LEVELS = {
    'LOW': (0.0, 0.2),
    'MEDIUM': (0.2, 0.5),
    'HIGH': (0.5, 0.8),
    'CRITICAL': (0.8, 1.01),
}

def get_risk_level(probability):
    for level, (lo, hi) in RISK_LEVELS.items():
        if lo <= probability < hi:
            return level
    return 'CRITICAL'

# ============================================================
# FEATURES — exact order the model expects
# ============================================================
FEATURE_COLS = [
    'amt', 'city_pop', 'hour', 'month', 'distance_cardholder_merchant',
    'age', 'is_weekend', 'is_night', 'velocity_1h', 'velocity_24h',
    'amount_velocity_1h', 'category_encoded', 'gender_encoded',
    'day_of_week_encoded'
]

FEATURE_DESCRIPTIONS = {
    'amt': 'Transaction amount in USD',
    'city_pop': 'Population of the city where the cardholder lives',
    'hour': 'Hour of the day (0-23)',
    'month': 'Month of the year (1-12)',
    'distance_cardholder_merchant': 'Distance between cardholder and merchant (km)',
    'age': 'Age of the cardholder',
    'is_weekend': 'Whether the transaction occurred on a weekend (0/1)',
    'is_night': 'Whether the transaction occurred at night, 10pm-6am (0/1)',
    'velocity_1h': 'Number of transactions by this card in the last 1 hour',
    'velocity_24h': 'Number of transactions by this card in the last 24 hours',
    'amount_velocity_1h': 'Total amount spent by this card in the last 1 hour (USD)',
    'category_encoded': 'Merchant category (0-13, alphabetical encoding)',
    'gender_encoded': 'Cardholder gender (0=Female, 1=Male)',
    'day_of_week_encoded': 'Day of the week (0=Monday, 6=Sunday)',
}

FEATURE_VALID_RANGES = {
    'amt': (0, None),
    'city_pop': (0, None),
    'hour': (0, 23),
    'month': (1, 12),
    'distance_cardholder_merchant': (0, None),
    'age': (1, 120),
    'is_weekend': (0, 1),
    'is_night': (0, 1),
    'velocity_1h': (1, None),
    'velocity_24h': (1, None),
    'amount_velocity_1h': (0, None),
    'category_encoded': (0, 13),
    'gender_encoded': (0, 1),
    'day_of_week_encoded': (0, 6),
}

# ============================================================
# CATEGORY MAPPING (loaded from JSON)
# ============================================================
def load_category_mapping():
    with open(CATEGORY_MAPPING_FILE) as f:
        return json.load(f)

CATEGORY_MAP = load_category_mapping()
CATEGORY_NAME_TO_CODE = CATEGORY_MAP['name_to_code']
CATEGORY_CODE_TO_NAME = {int(k): v for k, v in CATEGORY_MAP['code_to_name'].items()}

# ============================================================
# TRAINING STATS (loaded from JSON)
# ============================================================
def load_training_stats():
    with open(TRAINING_STATS_FILE) as f:
        return json.load(f)

TRAINING_STATS = load_training_stats()

def get_feature_median(feature_name):
    """Get median value for a feature from training data."""
    return TRAINING_STATS['stats'][feature_name]['all']['median']

def get_median_transaction():
    """Return a dict with median values for all features — the 'average' transaction."""
    return {col: get_feature_median(col) for col in FEATURE_COLS}

# ============================================================
# RANDOM SEEDS
# ============================================================
RANDOM_SEED = 42

# ============================================================
# SHAP EXPLANATION TEMPLATES
# ============================================================
SHAP_EXPLANATIONS = {
    'amt': {
        'positive': 'High transaction amount (${value:.2f}) — larger transactions are more associated with fraud',
        'negative': 'Low transaction amount (${value:.2f}) — small amounts are typical of normal spending',
    },
    'is_night': {
        'positive': 'Transaction occurred at night — nighttime transactions have higher fraud rates',
        'negative': 'Transaction occurred during the day — daytime transactions are typically legitimate',
    },
    'amount_velocity_1h': {
        'positive': 'Card spent ${value:.2f} in the last hour — unusually high spending velocity suggests fraud',
        'negative': 'Card spending velocity is low (${value:.2f}/hr) — consistent with normal behaviour',
    },
    'velocity_1h': {
        'positive': '{value:.0f} transactions in the last hour — rapid transaction frequency suggests card compromise',
        'negative': 'Only {value:.0f} transaction(s) in the last hour — normal transaction pace',
    },
    'velocity_24h': {
        'positive': '{value:.0f} transactions in the last 24 hours — elevated activity level',
        'negative': 'Normal transaction count ({value:.0f}) in the last 24 hours',
    },
    'hour': {
        'positive': 'Transaction at hour {value:.0f} — this time of day is associated with higher fraud risk',
        'negative': 'Transaction at hour {value:.0f} — this is a common time for legitimate purchases',
    },
    'category_encoded': {
        'positive': 'Merchant category ({cat_name}) is associated with higher fraud rates',
        'negative': 'Merchant category ({cat_name}) is a common, low-risk category',
    },
    'age': {
        'positive': 'Cardholder age ({value:.0f}) is in a demographic with higher fraud targeting',
        'negative': 'Cardholder age ({value:.0f}) — typical demographic, no elevated risk',
    },
    'city_pop': {
        'positive': 'City population ({value:,.0f}) — this city size is associated with more fraud',
        'negative': 'City population ({value:,.0f}) — no unusual risk from location',
    },
    'distance_cardholder_merchant': {
        'positive': 'High distance ({value:.1f} km) between cardholder and merchant — unusual shopping location',
        'negative': 'Cardholder is close to merchant ({value:.1f} km) — normal shopping pattern',
    },
    'gender_encoded': {
        'positive': 'Gender factor contributes to fraud risk for this transaction',
        'negative': 'Gender factor does not elevate risk for this transaction',
    },
    'is_weekend': {
        'positive': 'Weekend transaction — weekend spending patterns differ from weekday',
        'negative': 'Weekday transaction — consistent with regular spending patterns',
    },
    'day_of_week_encoded': {
        'positive': 'Day of week ({value:.0f}) is associated with elevated fraud risk',
        'negative': 'Day of week ({value:.0f}) has normal fraud rates',
    },
    'month': {
        'positive': 'Month {value:.0f} has historically higher fraud rates',
        'negative': 'Month {value:.0f} shows typical fraud patterns',
    },
    'recon_error': {
        'positive': 'High reconstruction error ({value:.4f}) — transaction pattern is anomalous',
        'negative': 'Low reconstruction error ({value:.4f}) — transaction follows normal patterns',
    },
    'bds_amount': {
        'positive': 'Amount deviation score ({value:.2f}) — spending amount is unusual for this cardholder',
        'negative': 'Amount deviation score ({value:.2f}) — spending amount is normal for this cardholder',
    },
    'bds_time': {
        'positive': 'Time deviation score ({value:.2f}) — unusual transaction time for this cardholder',
        'negative': 'Time deviation score ({value:.2f}) — typical transaction time for this cardholder',
    },
    'bds_freq': {
        'positive': 'Frequency deviation score ({value:.2f}) — unusual transaction rate for this cardholder',
        'negative': 'Frequency deviation score ({value:.2f}) — normal transaction rate for this cardholder',
    },
    'bds_category': {
        'positive': 'Category deviation score ({value:.2f}) — unusual merchant category for this cardholder',
        'negative': 'Category deviation score ({value:.2f}) — typical merchant category for this cardholder',
    },
}

def generate_explanation(feature_name, shap_value, feature_value):
    """Generate plain English explanation for a SHAP value."""
    templates = SHAP_EXPLANATIONS.get(feature_name, {
        'positive': f'{feature_name} = {{value}} pushes toward fraud',
        'negative': f'{feature_name} = {{value}} pushes toward normal',
    })
    direction = 'positive' if shap_value > 0 else 'negative'
    template = templates[direction]

    # Handle special formatting
    if 'cat_name' in template:
        cat_name = CATEGORY_CODE_TO_NAME.get(int(feature_value), f'code {int(feature_value)}')
        return template.format(value=feature_value, cat_name=cat_name)
    return template.format(value=feature_value)

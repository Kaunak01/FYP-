"""
Compute ROC-AUC and PR-AUC for all 4 saved models on the Sparkov test set.
Run from the FYP-Fraud-Detection folder:
    python compute_metrics.py
"""
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score)

# ── Config ────────────────────────────────────────────────────────────────────
TEST_CSV   = "fraudTest_engineered.csv"
MODELS_DIR = "models/saved"

# Must match the exact column order used in save_all_models.py (all cols except is_fraud, unix_time)
FEATURE_COLS = ['amt', 'city_pop', 'hour', 'month', 'distance_cardholder_merchant',
                'age', 'is_weekend', 'is_night', 'velocity_1h', 'velocity_24h',
                'amount_velocity_1h', 'category_encoded', 'gender_encoded', 'day_of_week_encoded']
TARGET = 'is_fraud'
THRESHOLD = 0.70  # same as training evaluation

# ── Load test data ────────────────────────────────────────────────────────────
print("Loading test data...")
df = pd.read_csv(TEST_CSV)
X_test = df[FEATURE_COLS].values
y_test = df[TARGET].values
print(f"  {len(df):,} rows | {y_test.sum():,} fraud ({y_test.mean()*100:.2f}%)\n")

# ── Autoencoder reconstruction error helper ───────────────────────────────────
def ae_reconstruction_error(X):
    """Returns per-row MSE reconstruction error from the PyTorch autoencoder."""
    scaler = joblib.load(f"{MODELS_DIR}/ae_scaler.joblib")
    X_scaled = scaler.transform(X).astype(np.float32)

    # Build same architecture as training
    import torch.nn as nn
    class Autoencoder(nn.Module):
        def __init__(self, n_features=14):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 10), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(10, 5),          nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(5, 10),  nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(10, n_features)
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    ae = Autoencoder(n_features=X.shape[1])
    ae.load_state_dict(torch.load(f"{MODELS_DIR}/ae_model.pt", map_location='cpu'))
    ae.eval()

    tensor = torch.tensor(X_scaled)
    with torch.no_grad():
        recon = ae(tensor).numpy()
    errors = np.mean((X_scaled - recon) ** 2, axis=1)
    return errors

# ── BDS features helper ───────────────────────────────────────────────────────
def add_bds_features(X, df_ref):
    """Compute 4 BDS deviation scores using saved profiles."""
    profiles = joblib.load(f"{MODELS_DIR}/bds_profiles.joblib")
    ga_params = __import__('json').load(open(f"{MODELS_DIR}/ga_best_params.json"))

    # BDS scores: amount_dev, velocity_dev, time_dev, category_dev
    # Replicate the same logic used during training
    amt       = df_ref['amt'].values
    vel_1h    = df_ref['velocity_1h'].values
    vel_24h   = df_ref['velocity_24h'].values
    hour      = df_ref['hour'].values
    cat_enc   = df_ref['category_encoded'].values

    n = len(df_ref)
    bds_amount   = np.zeros(n)
    bds_velocity = np.zeros(n)
    bds_time     = np.zeros(n)
    bds_category = np.zeros(n)

    for i, (_, row) in enumerate(df_ref.iterrows()):
        card = str(row.get('cc_num', row.get('card_number', 'unknown')))
        p = profiles.get(card, {})
        if p:
            mu_a, sd_a = p.get('mean_amt', amt[i]), max(p.get('std_amt', 1), 1e-6)
            bds_amount[i]   = abs(amt[i] - mu_a) / sd_a

            mu_v, sd_v = p.get('mean_vel', vel_1h[i]), max(p.get('std_vel', 1), 1e-6)
            bds_velocity[i] = abs(vel_1h[i] - mu_v) / sd_v

            usual_hours = p.get('usual_hours', [])
            bds_time[i] = 0 if (not usual_hours or hour[i] in usual_hours) else 1.0

            usual_cats = p.get('usual_cats', [])
            bds_category[i] = 0 if (not usual_cats or cat_enc[i] in usual_cats) else 1.0

    bds = np.column_stack([bds_amount, bds_velocity, bds_time, bds_category])
    return np.hstack([X, bds])

# ── Print results ─────────────────────────────────────────────────────────────
def print_metrics(name, y_true, y_proba):
    roc  = roc_auc_score(y_true, y_proba)
    pr   = average_precision_score(y_true, y_proba)
    pred = (y_proba >= THRESHOLD).astype(int)
    f1   = f1_score(y_true, pred, zero_division=0)
    prec = precision_score(y_true, pred, zero_division=0)
    rec  = recall_score(y_true, pred, zero_division=0)
    print(f"  {'Model':<35} ROC-AUC  PR-AUC   F1      Prec    Recall")
    print(f"  {name:<35} {roc:.4f}  {pr:.4f}  {f1:.4f}  {prec:.4f}  {rec:.4f}")
    print()
    return {'model': name, 'roc_auc': roc, 'pr_auc': pr, 'f1': f1,
            'precision': prec, 'recall': rec}

results = []

# ── 1. XGBoost Baseline (class weights) ──────────────────────────────────────
print("=" * 65)
print("1. XGBoost Baseline (class weights)")
xgb_cw = joblib.load(f"{MODELS_DIR}/xgboost_baseline_cw.joblib")
y_proba = xgb_cw.predict_proba(X_test)[:, 1]
results.append(print_metrics("XGBoost Baseline (CW)", y_test, y_proba))

# ── 2. XGBoost SMOTE + tuned ──────────────────────────────────────────────────
print("2. XGBoost SMOTE + tuned")
xgb_sm = joblib.load(f"{MODELS_DIR}/xgboost_smote_tuned.joblib")
y_proba = xgb_sm.predict_proba(X_test)[:, 1]
results.append(print_metrics("XGBoost SMOTE+tuned", y_test, y_proba))

# ── 3. AE + XGBoost SMOTE + tuned ────────────────────────────────────────────
print("3. AE + XGBoost SMOTE + tuned")
print("   (computing AE reconstruction errors — may take ~30s)")
ae_errors = ae_reconstruction_error(X_test)
X_ae = np.hstack([X_test, ae_errors.reshape(-1, 1)])
ae_xgb = joblib.load(f"{MODELS_DIR}/ae_xgboost_smote_tuned.joblib")
y_proba = ae_xgb.predict_proba(X_ae)[:, 1]
results.append(print_metrics("AE + XGBoost SMOTE+tuned", y_test, y_proba))

# ── 4. AE + BDS + XGBoost SMOTE + tuned ──────────────────────────────────────
print("4. AE + BDS + XGBoost SMOTE + tuned")
print("   (computing BDS scores — may take a few minutes on 555K rows)")
X_ae_bds = add_bds_features(X_ae, df)
ae_bds_xgb = joblib.load(f"{MODELS_DIR}/ae_bds_xgboost_smote_tuned.joblib")
y_proba = ae_bds_xgb.predict_proba(X_ae_bds)[:, 1]
results.append(print_metrics("AE + BDS + XGBoost SMOTE+tuned", y_test, y_proba))

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 65)
print("SUMMARY")
print(f"  {'Model':<35} ROC-AUC  PR-AUC   F1")
print(f"  {'-'*35} -------  ------   --")
for r in results:
    print(f"  {r['model']:<35} {r['roc_auc']:.4f}  {r['pr_auc']:.4f}  {r['f1']:.4f}")
print("=" * 65)

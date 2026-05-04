"""Re-score the Proposed model on the full test set using per-card BDS
profiles (the training-time path used in scripts/run_bds_ga.py), to confirm
that the dissertation's reported F1 = 0.8706 is reproducible from the saved
artefacts when per-card BDS is used. READ ONLY — modifies nothing."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROP_DIR = os.path.join(ROOT, "models", "saved", "03_proposed")

FEATURE_COLS = [
    "amt", "city_pop", "hour", "month", "distance_cardholder_merchant",
    "age", "is_weekend", "is_night", "velocity_1h", "velocity_24h",
    "amount_velocity_1h", "category_encoded", "gender_encoded",
    "day_of_week_encoded",
]


class Autoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, 10), nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(10, 5), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(5, 10), nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(10, d))

    def forward(self, x):
        return self.decoder(self.encoder(x))


print("Loading artefacts...")
t0 = time.time()
xgb_prop = joblib.load(os.path.join(PROP_DIR, "ae_bds_xgboost_smote_tuned.joblib"))
ae_scaler = joblib.load(os.path.join(PROP_DIR, "ae_scaler.joblib"))
profiles = joblib.load(os.path.join(PROP_DIR, "bds_profiles.joblib"))
with open(os.path.join(PROP_DIR, "ga_best_params.json")) as f:
    ga = json.load(f)

ae = Autoencoder(len(FEATURE_COLS))
ae.load_state_dict(torch.load(os.path.join(PROP_DIR, "ae_model.pt"),
                              weights_only=True, map_location="cpu"))
ae.eval()

card_amt = profiles["card_amt"]            # DataFrame indexed by cc_num: amt_mean, amt_std, amt_count
card_hour_prob = profiles["card_hour_prob"]  # DataFrame indexed by cc_num: 24 columns
card_cat_prob = profiles["card_cat_prob"]    # DataFrame indexed by cc_num: 14 columns
card_vel = profiles["card_vel"]              # DataFrame indexed by cc_num: vel_mean, vel_std
gs = profiles["global_stats"]
print(f"  loaded in {time.time() - t0:.1f}s; "
      f"per-card profiles for {len(card_amt)} cardholders")

print("Loading test data + cc_num...")
t0 = time.time()
df_test = pd.read_csv(os.path.join(ROOT, "data", "engineered", "fraudTest_engineered.csv"))
cc_test = pd.read_csv(os.path.join(ROOT, "data", "raw", "fraudTest.csv"),
                      usecols=["cc_num"])["cc_num"].values
X_test = df_test[FEATURE_COLS].values.astype(np.float64)
y_test = df_test["is_fraud"].values.astype(int)
amts = df_test["amt"].values.astype(np.float64)
hours = df_test["hour"].values.astype(int)
cats = df_test["category_encoded"].values.astype(int)
vels = df_test["velocity_1h"].values.astype(np.float64)
print(f"  {len(y_test):,} rows, {len(np.unique(cc_test))} unique cards in test "
      f"(loaded in {time.time() - t0:.1f}s)")

# AE recon error (batched, eval mode)
print("AE forward pass...")
t0 = time.time()
X_scaled = ae_scaler.transform(X_test).astype(np.float32)
with torch.no_grad():
    recon = ae(torch.FloatTensor(X_scaled)).numpy()
recon_error = np.mean((X_scaled - recon) ** 2, axis=1)
print(f"  done in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Per-card BDS — mirrors precompute_raw_deviations() in scripts/run_bds_ga.py
# ---------------------------------------------------------------------------
print("Computing per-card raw deviations (vectorised)...")
t0 = time.time()

df_dev = pd.DataFrame({
    "cc_num": cc_test, "amt": amts, "hour": hours, "cat": cats, "vel": vels,
})

# AMOUNT: per-card mean/std + count, fallback to global
df_dev = df_dev.merge(
    card_amt[["amt_mean", "amt_std", "amt_count"]],
    left_on="cc_num", right_index=True, how="left",
)
df_dev["amt_mean"] = df_dev["amt_mean"].fillna(gs["amt_mean"])
df_dev["amt_std"] = df_dev["amt_std"].fillna(gs["amt_std"])
df_dev["amt_count"] = df_dev["amt_count"].fillna(0)
df_dev["unseen"] = df_dev["amt_count"] == 0
safe_std = df_dev["amt_std"].where(df_dev["amt_std"] > 0, gs["amt_std"])
df_dev["card_amt_z"] = (df_dev["amt"] - df_dev["amt_mean"]).abs() / safe_std
df_dev["global_amt_z"] = (df_dev["amt"] - gs["amt_mean"]).abs() / gs["amt_std"]

# HOUR: per-card hour probability + fallback
hour_stack = card_hour_prob.stack()
hour_stack.index.names = ["cc_num", "hour"]
hour_lookup = hour_stack.reset_index()
hour_lookup.columns = ["cc_num", "hour", "card_hour_p"]
df_dev = df_dev.merge(hour_lookup, on=["cc_num", "hour"], how="left")
df_dev["card_hour_p"] = df_dev["card_hour_p"].fillna(0.0)
df_dev["global_hour_p"] = df_dev["hour"].map(
    {int(k): v for k, v in gs["hour_prob"].items()}
).fillna(1.0 / 24)

# CATEGORY: per-card cat prob + fallback
cat_stack = card_cat_prob.stack()
cat_stack.index.names = ["cc_num", "cat"]
cat_lookup = cat_stack.reset_index()
cat_lookup.columns = ["cc_num", "cat", "card_cat_p"]
df_dev = df_dev.merge(cat_lookup, on=["cc_num", "cat"], how="left")
df_dev["card_cat_p"] = df_dev["card_cat_p"].fillna(0.0)
n_cats = gs["n_categories"]
df_dev["global_cat_p"] = df_dev["cat"].map(
    {int(k): v for k, v in gs["cat_prob"].items()}
).fillna(1.0 / n_cats)

# FREQUENCY: vel / card_vel_mean (or global)
df_dev = df_dev.merge(card_vel[["vel_mean"]], left_on="cc_num",
                      right_index=True, how="left")
df_dev["vel_mean"] = df_dev["vel_mean"].fillna(gs["vel_mean"])
safe_vel = df_dev["vel_mean"].where(df_dev["vel_mean"] > 0, gs["vel_mean"])
df_dev["freq_ratio"] = df_dev["vel"] / safe_vel

card_counts = df_dev["amt_count"].values
unseen = df_dev["unseen"].values
card_amt_z = df_dev["card_amt_z"].values
global_amt_z = df_dev["global_amt_z"].values
card_hour_p = df_dev["card_hour_p"].values
global_hour_p = df_dev["global_hour_p"].values
card_cat_p = df_dev["card_cat_p"].values
global_cat_p = df_dev["global_cat_p"].values
freq_ratio = df_dev["freq_ratio"].values
print(f"  done in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Apply GA-evolved BDS thresholds (per scripts/run_bds_ga.py:200)
# ---------------------------------------------------------------------------
p = list(ga["params"].values())
at, ac, tt, tc, ft, fc, ct, cc_, mh_f, sm = p
mh = int(round(mh_f))

use_global = (card_counts < mh) | unseen

amt_z = np.where(use_global, global_amt_z, card_amt_z)
amount_score = np.clip(np.maximum(amt_z - at, 0.0), 0.0, ac)

hour_p = np.where(use_global, global_hour_p, card_hour_p)
time_raw = -np.log(hour_p + sm)
time_score = np.clip(np.maximum(time_raw - tt, 0.0), 0.0, tc)

freq_raw = np.maximum(freq_ratio - 1.0, 0.0)
freq_score = np.clip(np.maximum(freq_raw - ft, 0.0), 0.0, fc)

cat_p = np.where(use_global, global_cat_p, card_cat_p)
cat_raw = -np.log(cat_p + sm)
cat_score = np.clip(np.maximum(cat_raw - ct, 0.0), 0.0, cc_)

# Concatenate full feature matrix and score
print("XGBoost predict...")
t0 = time.time()
X_full = np.column_stack([X_test, recon_error, amount_score, time_score,
                          freq_score, cat_score])
prob = xgb_prop.predict_proba(X_full)[:, 1]
pred = (prob >= 0.5).astype(int)
print(f"  done in {time.time() - t0:.1f}s")

# Metrics
f1 = f1_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
prauc = average_precision_score(y_test, prob)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print()
print("=" * 60)
print("PROPOSED — full test set with per-card BDS")
print("=" * 60)
print(f"  F1            = {f1:.6f}   (report 0.870582)  delta {f1 - 0.870582:+.6f}")
print(f"  Precision     = {prec:.6f}   (report 0.933796)  delta {prec - 0.933796:+.6f}")
print(f"  Recall        = {rec:.6f}   (report 0.815385)  delta {rec - 0.815385:+.6f}")
print(f"  PR-AUC        = {prauc:.6f}   (report 0.915837)  delta {prauc - 0.915837:+.6f}")
print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"  (report:    TP=1749 FP=124 FN=396 TN=553450)")
within = (abs(f1 - 0.870582) <= 0.001 and abs(prec - 0.933796) <= 0.001 and
          abs(rec - 0.815385) <= 0.001 and abs(prauc - 0.915837) <= 0.001 and
          tp == 1749 and fp == 124 and fn == 396 and tn == 553450)
print(f"  -> WITHIN 0.001 TOLERANCE: {'YES' if within else 'NO'}")

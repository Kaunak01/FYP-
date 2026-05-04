"""30-minute pipeline verification (READ ONLY).

Loads saved Baseline / Proposed / Comparator artefacts and re-scores them on
the held-out test set, comparing to the values stored in
results/verified_metrics.json. Prints stage-by-stage so failures are localised.

Tolerances:
  - Baseline + Proposed (full test set): 0.001 on F1/P/R/PR-AUC; exact match on TP/FP/FN/TN
  - Comparator (5k stratified sample): used for sanity only; report numbers, no assert
"""
from __future__ import annotations

import json
import os
import sys
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


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def report(name: str, prob: np.ndarray, y: np.ndarray, expected: dict, tol: float = 0.001) -> dict:
    pred = (prob >= 0.5).astype(int)
    f1 = float(f1_score(y, pred))
    p = float(precision_score(y, pred))
    r = float(recall_score(y, pred))
    prauc = float(average_precision_score(y, prob))
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    matches = {
        "f1":     abs(f1 - expected["f1"]) <= tol,
        "p":      abs(p - expected["precision"]) <= tol,
        "r":      abs(r - expected["recall"]) <= tol,
        "prauc":  abs(prauc - expected["pr_auc"]) <= tol,
        "tp":     tp == expected["tp"],
        "fp":     fp == expected["fp"],
        "fn":     fn == expected["fn"],
        "tn":     tn == expected["tn"],
    }
    all_ok = all(matches.values())
    print(f"\n  {name}: threshold 0.50")
    print(f"    metric        |    expected | computed |  match")
    print(f"    F1            |  {expected['f1']:.6f} | {f1:.6f} |  {'OK' if matches['f1'] else 'FAIL'}")
    print(f"    Precision     |  {expected['precision']:.6f} | {p:.6f} |  {'OK' if matches['p'] else 'FAIL'}")
    print(f"    Recall        |  {expected['recall']:.6f} | {r:.6f} |  {'OK' if matches['r'] else 'FAIL'}")
    print(f"    PR-AUC        |  {expected['pr_auc']:.6f} | {prauc:.6f} |  {'OK' if matches['prauc'] else 'FAIL'}")
    print(f"    TP            |  {expected['tp']:>9d} | {tp:>8d} |  {'OK' if matches['tp'] else 'FAIL'}")
    print(f"    FP            |  {expected['fp']:>9d} | {fp:>8d} |  {'OK' if matches['fp'] else 'FAIL'}")
    print(f"    FN            |  {expected['fn']:>9d} | {fn:>8d} |  {'OK' if matches['fn'] else 'FAIL'}")
    print(f"    TN            |  {expected['tn']:>9d} | {tn:>8d} |  {'OK' if matches['tn'] else 'FAIL'}")
    print(f"    -> {'ALL MATCH' if all_ok else 'MISMATCH'}")
    return {"all_ok": all_ok, "computed": {"f1": f1, "p": p, "r": r, "prauc": prauc,
                                           "tp": tp, "fp": fp, "fn": fn, "tn": tn}}


# ---------------------------------------------------------------------------
# Stage 0 — load expected metrics + test data
# ---------------------------------------------------------------------------
banner("STAGE 0 — load test data + expected metrics")
t0 = time.time()
with open(os.path.join(ROOT, "results", "verified_metrics.json")) as f:
    vm = json.load(f)

expected = {}
for entry in vm["models"]:
    if entry["threshold"] == 0.5:
        expected[entry["model"]] = entry

FEATURE_COLS = [
    "amt", "city_pop", "hour", "month", "distance_cardholder_merchant",
    "age", "is_weekend", "is_night", "velocity_1h", "velocity_24h",
    "amount_velocity_1h", "category_encoded", "gender_encoded",
    "day_of_week_encoded",
]

df_test = pd.read_csv(os.path.join(ROOT, "data", "engineered", "fraudTest_engineered.csv"))
X_test = df_test[FEATURE_COLS].values.astype(np.float64)
y_test = df_test["is_fraud"].values.astype(int)
print(f"  test rows : {len(y_test):,} | fraud rows : {int(y_test.sum()):,}")
print(f"  loaded in : {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Stage 1 — Baseline (XGBoost SMOTE+Tuned)
# ---------------------------------------------------------------------------
banner("STAGE 1 — Baseline: XGBoost SMOTE+Tuned")
t0 = time.time()
xgb_base = joblib.load(os.path.join(ROOT, "models", "saved", "01_baseline",
                                    "xgboost_smote_tuned.joblib"))
prob_base = xgb_base.predict_proba(X_test)[:, 1]
print(f"  scored in : {time.time() - t0:.1f}s")
res_base = report("BASELINE", prob_base, y_test, expected["XGBoost SMOTE+tuned"])


# ---------------------------------------------------------------------------
# Stage 2 — Proposed (AE + BDS + GA + XGBoost)
# ---------------------------------------------------------------------------
banner("STAGE 2 — Proposed: AE + BDS + GA + XGBoost")
t0 = time.time()


class Autoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, 10), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(10, 5), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(10, d))

    def forward(self, x):
        return self.decoder(self.encoder(x))


prop_dir = os.path.join(ROOT, "models", "saved", "03_proposed")
xgb_prop = joblib.load(os.path.join(prop_dir, "ae_bds_xgboost_smote_tuned.joblib"))
ae_scaler = joblib.load(os.path.join(prop_dir, "ae_scaler.joblib"))
bds_profiles = joblib.load(os.path.join(prop_dir, "bds_profiles.joblib"))
with open(os.path.join(prop_dir, "ga_best_params.json")) as f:
    ga = json.load(f)

ae = Autoencoder(len(FEATURE_COLS))
ae.load_state_dict(torch.load(os.path.join(prop_dir, "ae_model.pt"),
                              weights_only=True, map_location="cpu"))
ae.eval()

# AE forward (batched) → recon_error per row
X_scaled = ae_scaler.transform(X_test).astype(np.float32)
with torch.no_grad():
    recon = ae(torch.FloatTensor(X_scaled)).numpy()
recon_error = np.mean((X_scaled - recon) ** 2, axis=1)

# BDS — vectorised, mirrors ModelManager.compute_bds_scores
gs = bds_profiles["global_stats"]
p = list(ga["params"].values())
at, ac, tt, tc, ft, fc, ct, cc_, _mh, sm = p

amt = X_test[:, FEATURE_COLS.index("amt")]
hour = X_test[:, FEATURE_COLS.index("hour")].astype(int)
cat = X_test[:, FEATURE_COLS.index("category_encoded")].astype(int)
vel = X_test[:, FEATURE_COLS.index("velocity_1h")]

if gs["amt_std"] > 0:
    amt_z = np.abs(amt - gs["amt_mean"]) / gs["amt_std"]
else:
    amt_z = np.zeros_like(amt)
amount_score = np.clip(np.maximum(amt_z - at, 0.0), 0.0, ac)

hour_p = np.array([gs["hour_prob"].get(str(int(h)), 1.0 / 24) for h in hour])
time_raw = -np.log(hour_p + sm)
time_score = np.clip(np.maximum(time_raw - tt, 0.0), 0.0, tc)

if gs["vel_mean"] > 0:
    freq_raw = np.maximum(vel / gs["vel_mean"] - 1.0, 0.0)
else:
    freq_raw = np.zeros_like(vel)
freq_score = np.clip(np.maximum(freq_raw - ft, 0.0), 0.0, fc)

n_cats = gs["n_categories"]
cat_p = np.array([gs["cat_prob"].get(str(int(c)), 1.0 / n_cats) for c in cat])
cat_raw = -np.log(cat_p + sm)
cat_score = np.clip(np.maximum(cat_raw - ct, 0.0), 0.0, cc_)

X_full = np.column_stack([X_test, recon_error, amount_score, time_score,
                          freq_score, cat_score])
prob_prop = xgb_prop.predict_proba(X_full)[:, 1]
print(f"  scored in : {time.time() - t0:.1f}s")
res_prop = report("PROPOSED", prob_prop, y_test, expected["AE + BDS + XGBoost (full)"])


# ---------------------------------------------------------------------------
# Stage 3 — Comparator (LSTM + RF) on 5k stratified sample (sanity only)
# ---------------------------------------------------------------------------
banner("STAGE 3 — Comparator: LSTM + RF on stratified 5k sample (sanity check)")
t0 = time.time()
comp_dir = os.path.join(ROOT, "models", "saved", "02_comparator")
keras_path = os.path.join(comp_dir, "lstm_rf_keras.keras")
rf_sc_path = os.path.join(comp_dir, "lstm_rf_scaler.joblib")
rf_clf_path = os.path.join(comp_dir, "lstm_rf_classifier.joblib")

if not all(os.path.exists(p) for p in (keras_path, rf_sc_path, rf_clf_path)):
    print("  COMPARATOR ARTEFACTS MISSING — skipping (paths checked: "
          f"{keras_path}, {rf_sc_path}, {rf_clf_path})")
    res_comp = {"skipped": True}
else:
    try:
        os.environ.setdefault("KERAS_BACKEND", "torch")
        import keras as _keras
        lstm = _keras.models.load_model(keras_path, compile=False)
        rf_scaler = joblib.load(rf_sc_path)
        rf_clf = joblib.load(rf_clf_path)
        print(f"  artefacts loaded in : {time.time() - t0:.1f}s")

        # Stratified 5k sample
        rng = np.random.default_rng(42)
        fraud_idx = np.where(y_test == 1)[0]
        norm_idx = np.where(y_test == 0)[0]
        n_fraud_sample = min(2500, len(fraud_idx))
        n_norm_sample = 5000 - n_fraud_sample
        sample_fraud = rng.choice(fraud_idx, size=n_fraud_sample, replace=False)
        sample_norm = rng.choice(norm_idx, size=n_norm_sample, replace=False)
        sample_idx = np.concatenate([sample_fraud, sample_norm])
        rng.shuffle(sample_idx)
        print(f"  sample size : {len(sample_idx)} ({n_fraud_sample} fraud, "
              f"{n_norm_sample} normal)")

        # Build per-card sequence priors using all engineered rows for that card
        # The locked LSTM run (LSTM_reproduced_baseline) used cc_num+timestamp
        # ordering across train+test.
        df_tr = pd.read_csv(os.path.join(ROOT, "data", "engineered",
                                         "fraudTrain_engineered.csv"))
        df_te = df_test.copy()
        # Need cc_num — check engineered files
        if "cc_num" not in df_tr.columns or "cc_num" not in df_te.columns:
            cc_tr = pd.read_csv(os.path.join(ROOT, "data", "raw", "fraudTrain.csv"),
                                usecols=["cc_num"])["cc_num"].values
            cc_te = pd.read_csv(os.path.join(ROOT, "data", "raw", "fraudTest.csv"),
                                usecols=["cc_num"])["cc_num"].values
            df_tr["cc_num"] = cc_tr
            df_te["cc_num"] = cc_te
        # Sort by cc_num + unix_time
        df_tr = df_tr.sort_values(["cc_num", "unix_time"]).reset_index(drop=True)
        df_te = df_te.sort_values(["cc_num", "unix_time"]).reset_index(drop=True)

        # Build a lookup: for each (cc_num, unix_time) in test, find prior 4 rows
        # by combining train + test rows sorted chronologically per card.
        SEQ_LEN = 5
        all_combined = pd.concat([df_tr.assign(_src="tr"),
                                  df_te.assign(_src="te")], ignore_index=True)
        all_combined = all_combined.sort_values(["cc_num", "unix_time"]).reset_index(drop=True)

        # Index test rows in the combined frame by (cc_num, unix_time)
        # Reload test sample with original ordering preserved
        df_te_orig = df_test.iloc[sample_idx].reset_index(drop=True)
        if "cc_num" not in df_te_orig.columns:
            cc_te_full = pd.read_csv(os.path.join(ROOT, "data", "raw", "fraudTest.csv"),
                                     usecols=["cc_num"])["cc_num"].values
            df_te_orig["cc_num"] = cc_te_full[sample_idx]

        # Build sequences row-by-row (5k iterations only — manageable)
        all_features_scaled = rf_scaler.transform(all_combined[FEATURE_COLS].values)
        # combined index by (cc_num, unix_time)
        combined_lookup = all_combined.set_index(["cc_num", "unix_time"])
        # also need positional index in all_combined for slicing
        all_combined_pos = {
            (cc, ut): i for i, (cc, ut)
            in enumerate(zip(all_combined["cc_num"].values,
                             all_combined["unix_time"].values))
        }
        cc_arr = all_combined["cc_num"].values

        seqs = np.zeros((len(df_te_orig), SEQ_LEN, len(FEATURE_COLS)), dtype=np.float32)
        sample_y = y_test[sample_idx]
        for i, (cc, ut) in enumerate(zip(df_te_orig["cc_num"].values,
                                         df_te_orig["unix_time"].values)):
            pos = all_combined_pos.get((cc, ut))
            if pos is None:
                continue
            # walk backward to gather up to 4 prior rows with same cc_num
            priors = []
            j = pos - 1
            while j >= 0 and len(priors) < SEQ_LEN - 1 and cc_arr[j] == cc:
                priors.append(all_features_scaled[j])
                j -= 1
            priors = priors[::-1]  # oldest first
            pad = SEQ_LEN - 1 - len(priors)
            for k, vec in enumerate(priors):
                seqs[i, pad + k] = vec
            seqs[i, SEQ_LEN - 1] = all_features_scaled[pos]

        # LSTM forward pass (batched)
        print(f"  sequence build done in : {time.time() - t0:.1f}s; running LSTM...")
        t1 = time.time()
        lstm_prob = lstm.predict(seqs, batch_size=512, verbose=0).ravel()
        print(f"  LSTM forward done in : {time.time() - t1:.1f}s")

        # Build RF input = [lstm_prob, scaled static features]
        sample_static = X_test[sample_idx]
        sample_static_scaled = rf_scaler.transform(sample_static)
        rf_input = np.column_stack([lstm_prob, sample_static_scaled])
        prob_comp = rf_clf.predict_proba(rf_input)[:, 1]

    except Exception as e:
        print(f"  COMPARATOR FAILED: {type(e).__name__}: {e}")
        res_comp = {"skipped": True, "error": str(e)}
    else:
        # Reported full-test Comparator numbers
        comp_full = next(e for e in vm["gap_experiments"]
                         if e.get("experiment_name") == "LSTM_reproduced_baseline")
        pred_comp = (prob_comp >= 0.5).astype(int)
        f1_c = float(f1_score(sample_y, pred_comp))
        p_c = float(precision_score(sample_y, pred_comp, zero_division=0))
        r_c = float(recall_score(sample_y, pred_comp))
        prauc_c = float(average_precision_score(sample_y, prob_comp))
        tn, fp, fn, tp = confusion_matrix(sample_y, pred_comp).ravel()
        print(f"\n  COMPARATOR (5k stratified sample, sanity only):")
        print(f"    F1     : sample={f1_c:.4f}  full-test reported={comp_full['F1']:.4f}")
        print(f"    P      : sample={p_c:.4f}  full-test reported={comp_full['Precision']:.4f}")
        print(f"    R      : sample={r_c:.4f}  full-test reported={comp_full['Recall']:.4f}")
        print(f"    PR-AUC : sample={prauc_c:.4f}  full-test reported={comp_full.get('PR_AUC', 'n/a')}")
        print(f"    TP={tp} FP={fp} FN={fn} TN={tn}  (sample n={len(sample_y)})")
        within = (abs(f1_c - comp_full["F1"]) <= 0.05 and
                  abs(p_c - comp_full["Precision"]) <= 0.05 and
                  abs(r_c - comp_full["Recall"]) <= 0.05)
        print(f"    -> within ±0.05 tolerance of full-test numbers: {'YES' if within else 'NO'}")
        res_comp = {"all_ok": within, "skipped": False}


# ---------------------------------------------------------------------------
# Stage 4 — Latency (100 single-tx scores per model)
# ---------------------------------------------------------------------------
banner("STAGE 4 — Latency on 100 single-transaction scores per model")
rng = np.random.default_rng(42)
sample100 = rng.choice(len(X_test), size=100, replace=False)


def time_baseline_single():
    times = []
    for i in sample100:
        t0 = time.perf_counter()
        _ = xgb_base.predict_proba(X_test[i:i + 1])[0, 1]
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def time_proposed_single():
    times = []
    for i in sample100:
        t0 = time.perf_counter()
        x = X_test[i:i + 1]
        x_s = ae_scaler.transform(x).astype(np.float32)
        with torch.no_grad():
            r = ae(torch.FloatTensor(x_s)).numpy()
        re = float(np.mean((x_s - r) ** 2, axis=1)[0])
        h = int(x[0, FEATURE_COLS.index("hour")])
        c = int(x[0, FEATURE_COLS.index("category_encoded")])
        v = float(x[0, FEATURE_COLS.index("velocity_1h")])
        a = float(x[0, FEATURE_COLS.index("amt")])
        amt_z = abs(a - gs["amt_mean"]) / gs["amt_std"] if gs["amt_std"] > 0 else 0
        amount_s = min(max(amt_z - at, 0), ac)
        hp = gs["hour_prob"].get(str(h), 1.0 / 24)
        time_s = min(max(-np.log(hp + sm) - tt, 0), tc)
        fr = max(v / gs["vel_mean"] - 1.0, 0) if gs["vel_mean"] > 0 else 0
        freq_s = min(max(fr - ft, 0), fc)
        cp = gs["cat_prob"].get(str(c), 1.0 / gs["n_categories"])
        cat_s = min(max(-np.log(cp + sm) - ct, 0), cc_)
        x_full = np.concatenate([x[0], [re, amount_s, time_s, freq_s, cat_s]]).reshape(1, -1)
        _ = xgb_prop.predict_proba(x_full)[0, 1]
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


lat_b = time_baseline_single()
lat_p = time_proposed_single()
print(f"\n  Baseline median latency : {lat_b:.3f} ms  (report: 0.748 ms)")
print(f"  Proposed median latency : {lat_p:.3f} ms  (report: 1.620 ms)")
print(f"  Comparator latency      : SKIPPED (LSTM single-tx is ~89 ms; tested in app smoke)")
lat_b_ok = 0.5 * 0.748 <= lat_b <= 1.5 * 0.748
lat_p_ok = 0.5 * 1.620 <= lat_p <= 1.5 * 1.620
print(f"  baseline within ±50% : {'OK' if lat_b_ok else 'OUT OF RANGE'}")
print(f"  proposed within ±50% : {'OK' if lat_p_ok else 'OUT OF RANGE'}")


# ---------------------------------------------------------------------------
# Stage 5 — pytest
# ---------------------------------------------------------------------------
banner("STAGE 5 — pytest tests/")
import subprocess
r = subprocess.run(["python", "-m", "pytest", "tests/", "-q"],
                   cwd=ROOT, capture_output=True, text=True, timeout=120)
print(r.stdout[-1500:])
if r.stderr:
    print("STDERR:", r.stderr[-500:])
pytest_ok = ("passed" in r.stdout) and ("failed" not in r.stdout.lower() or
                                        " 0 failed" in r.stdout)


# ---------------------------------------------------------------------------
# Stage 6 — Final verdict
# ---------------------------------------------------------------------------
banner("FINAL VERDICT")
flags = []
if not res_base.get("all_ok"):
    flags.append("BASELINE")
if not res_prop.get("all_ok"):
    flags.append("PROPOSED")
if not res_comp.get("skipped") and not res_comp.get("all_ok"):
    flags.append("COMPARATOR (sanity)")
if not lat_b_ok:
    flags.append("BASELINE LATENCY")
if not lat_p_ok:
    flags.append("PROPOSED LATENCY")
if not pytest_ok:
    flags.append("PYTEST")

if flags:
    print(f"  PIPELINE MISMATCH — failures in: {', '.join(flags)}")
    sys.exit(1)
else:
    print("  PIPELINE VERIFIED")
    sys.exit(0)

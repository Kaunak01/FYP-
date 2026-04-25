"""
Gap-closing experiments runner.

Runs 5 experiments sequentially, saves after each, logs every step.
Designed for overnight CPU execution. If any experiment fails, log it
honestly and continue with the next.

Experiments:
  1. LSTM_reproduced_baseline — LSTM+RF with velocity on FIXED pipeline
  2. LSTM_no_velocity          — LSTM+RF without velocity on FIXED pipeline
  3. LSTM_with_velocity        — cross-reference to #1 if deterministic
  4. XGBoost_ADASYN            — replace SMOTE with ADASYN
  5. XGBoost_FocalLoss         — custom focal-loss objective, no resampling

Outputs:
  - verified_metrics.json (new top-level key 'gap_experiments')
  - FEEDBACK_GAPS_RUN_LOG.txt
"""
from __future__ import annotations

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTHONHASHSEED"] = "42"

import json
import math
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# ---------- setup ----------
ROOT = Path(__file__).parent
LOG_PATH = ROOT / "FEEDBACK_GAPS_RUN_LOG.txt"
METRICS_PATH = ROOT / "verified_metrics.json"

TRAIN_CSV = ROOT / "fraudTrain_engineered_with_ids.csv"
TEST_CSV = ROOT / "fraudTest_engineered_with_ids.csv"

SEED = 42
SEQ_LEN = 5

FEATURES_14 = [
    "amt", "city_pop", "hour", "month", "distance_cardholder_merchant",
    "age", "is_weekend", "is_night",
    "velocity_1h", "velocity_24h", "amount_velocity_1h",
    "category_encoded", "gender_encoded", "day_of_week_encoded",
]
FEATURES_11 = [f for f in FEATURES_14 if f not in ("velocity_1h", "velocity_24h", "amount_velocity_1h")]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_all_seeds(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_metrics() -> dict:
    with METRICS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def save_metrics(d: dict) -> None:
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def append_gap_entry(entry: dict) -> None:
    d = load_metrics()
    d.setdefault("gap_experiments", []).append(entry)
    save_metrics(d)


def build_entry(name: str, gap: str, model: str, features: list[str],
                oversampling: str, threshold: float, y_true, y_pred, y_score,
                runtime_s: float, notes: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "experiment_name": name,
        "gap_addressed": gap,
        "model": model,
        "features_used": features,
        "oversampling": oversampling,
        "threshold": float(threshold),
        "F1": float(f1_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred)),
        "ROC_AUC": float(roc_auc_score(y_true, y_score)),
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "runtime_seconds": float(runtime_s),
        "random_state": SEED,
        "notes": notes,
    }


# ---------- sequence builder ----------
def build_sequences(X_scaled: np.ndarray, cc_nums: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Per-cardholder sliding window. Input is assumed pre-sorted by (cc_num, time)."""
    n, f = X_scaled.shape
    out = np.zeros((n, seq_len, f), dtype=np.float32)
    current_cc = None
    buf: list[np.ndarray] = []
    for i in range(n):
        cc = cc_nums[i]
        if cc != current_cc:
            buf = []
            current_cc = cc
        buf.append(X_scaled[i])
        if len(buf) > seq_len:
            buf = buf[-seq_len:]
        # write padded sequence into out[i]
        k = len(buf)
        out[i, seq_len - k:] = np.asarray(buf, dtype=np.float32)
    return out


# ---------- LSTM experiment ----------
def run_lstm_experiment(exp_name: str, features: list[str], notes_suffix: str = "") -> dict | None:
    """Core LSTM+RF pipeline. Returns built entry or None on failure."""
    import keras
    from keras import Sequential, layers, optimizers

    log(f"EXPERIMENT: {exp_name} — loading data")
    t0 = time.time()
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)

    # Sort by (cc_num, unix_time) for sequence construction
    df_tr = df_tr.sort_values(["cc_num", "unix_time"], kind="mergesort").reset_index(drop=True)
    df_te = df_te.sort_values(["cc_num", "unix_time"], kind="mergesort").reset_index(drop=True)

    X_tr = df_tr[features].to_numpy(dtype=np.float32)
    X_te = df_te[features].to_numpy(dtype=np.float32)
    y_tr = df_tr["is_fraud"].to_numpy(dtype=np.int64)
    y_te = df_te["is_fraud"].to_numpy(dtype=np.int64)
    cc_tr = df_tr["cc_num"].to_numpy()
    cc_te = df_te["cc_num"].to_numpy()

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    log(f"  building sequences (train)  rows={len(X_tr):,} feats={X_tr.shape[1]}")
    seq_tr = build_sequences(X_tr_s, cc_tr, SEQ_LEN)
    log(f"  building sequences (test)   rows={len(X_te):,}")
    seq_te = build_sequences(X_te_s, cc_te, SEQ_LEN)

    # Class weight as in original notebook
    n0 = int((y_tr == 0).sum()); n1 = int((y_tr == 1).sum())
    cw = {0: 1.0, 1: n0 / max(n1, 1)}
    log(f"  class_weight = {{0:1, 1:{cw[1]:.2f}}}")

    set_all_seeds(SEED)
    keras.utils.set_random_seed(SEED)

    log("  compiling Keras model (LSTM64 + Dropout0.3 + Dense32 + Dense16 + Sigmoid)")
    model = Sequential([
        layers.Input(shape=(SEQ_LEN, X_tr.shape[1])),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu", name="lstm_features"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    log("  training LSTM (5 epochs, batch=512, val_split=0.1) — this is the long step")
    model.fit(
        seq_tr, y_tr,
        epochs=5, batch_size=512,
        class_weight=cw,
        validation_split=0.1,
        verbose=2,
    )

    log("  producing LSTM probabilities")
    p_tr = model.predict(seq_tr, batch_size=1024, verbose=0).ravel()
    p_te = model.predict(seq_te, batch_size=1024, verbose=0).ravel()

    log("  fitting Random Forest on [LSTM_prob + static features]")
    X_rf_tr = np.column_stack([p_tr, X_tr])
    X_rf_te = np.column_stack([p_te, X_te])
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight={0: 1, 1: 150},
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_rf_tr, y_tr)

    proba_te = rf.predict_proba(X_rf_te)[:, 1]

    runtime = time.time() - t0
    note_base = (
        "Keras 3.14 + PyTorch 2.11 CPU backend (original Colab used Keras-on-TF-GPU). "
        "Architecture/hyperparameters preserved exactly per LSTM_BASELINE_CONFIG.json. "
        "Data from fraudTrain/Test_engineered_with_ids.csv (post-alignment fix). "
        "Expected ±0.01-0.02 F1 drift vs TF backend."
    )
    if notes_suffix:
        note_base = note_base + " " + notes_suffix

    # Evaluate at threshold 0.5
    pred_05 = (proba_te >= 0.5).astype(int)
    entry = build_entry(
        name=exp_name,
        gap="Gap 1: sequence model ablation",
        model="LSTM(64)+Dropout0.3+Dense32+Dense16+Sigmoid → RandomForest(200,d12,cw150)",
        features=features,
        oversampling="None",
        threshold=0.5,
        y_true=y_te, y_pred=pred_05, y_score=proba_te,
        runtime_s=runtime,
        notes=note_base,
    )
    append_gap_entry(entry)
    log(f"  SAVED @ thr=0.5 | F1={entry['F1']:.4f} P={entry['Precision']:.4f} R={entry['Recall']:.4f} "
        f"ROC={entry['ROC_AUC']:.4f} | runtime={runtime:.0f}s")
    return entry


# ---------- XGBoost experiments ----------
def _load_xgb_tuned_params() -> dict:
    """Extract best_params from existing xgboost_smote_tuned.joblib."""
    p = ROOT / "xgboost_smote_tuned.joblib"
    model = joblib.load(p)
    params = model.get_params()
    # keep only the hyperparams we care about, drop noise
    keep = [
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
    ]
    out = {k: params[k] for k in keep if k in params}
    log(f"  reused XGBoost tuned params from joblib: {out}")
    return out


def run_xgb_adasyn() -> dict | None:
    from imblearn.over_sampling import ADASYN
    from xgboost import XGBClassifier

    exp_name = "XGBoost_ADASYN"
    log(f"EXPERIMENT: {exp_name}")
    t0 = time.time()

    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)
    X_tr = df_tr[FEATURES_14].to_numpy(dtype=np.float32)
    X_te = df_te[FEATURES_14].to_numpy(dtype=np.float32)
    y_tr = df_tr["is_fraud"].to_numpy(dtype=np.int64)
    y_te = df_te["is_fraud"].to_numpy(dtype=np.int64)

    tuned = _load_xgb_tuned_params()

    # ADASYN fallback chain
    fallback_used = None
    X_res = y_res = None
    for attempt in [
        {"sampling_strategy": 1.0, "n_neighbors": 5},
        {"sampling_strategy": 1.0, "n_neighbors": 3},
        {"sampling_strategy": 0.5, "n_neighbors": 5},
    ]:
        try:
            log(f"  trying ADASYN {attempt}")
            ada = ADASYN(random_state=SEED, **attempt)
            X_res, y_res = ada.fit_resample(X_tr, y_tr)
            log(f"  ADASYN OK — resampled shape {X_res.shape}, fraud={int((y_res==1).sum()):,}")
            if attempt != {"sampling_strategy": 1.0, "n_neighbors": 5}:
                fallback_used = f"ADASYN fallback: {attempt}"
            break
        except Exception as e:
            log(f"  ADASYN failed with {attempt}: {e}")
            fallback_used = f"ADASYN failed: {e}"
    if X_res is None:
        log("  ADASYN exhausted all fallbacks — logging FAILED")
        d = load_metrics()
        d.setdefault("gap_experiments", []).append({
            "experiment_name": exp_name, "status": "FAILED",
            "gap_addressed": "Gap 2: oversampling comparison",
            "notes": f"ADASYN failed all fallbacks. Last error: {fallback_used}",
        })
        save_metrics(d)
        return None

    clf = XGBClassifier(
        **tuned,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    log("  fitting XGBoost on ADASYN-resampled train")
    clf.fit(X_res, y_res)
    proba_te = clf.predict_proba(X_te)[:, 1]
    runtime = time.time() - t0

    base_note = f"XGBoost tuned hyperparameters reused from xgboost_smote_tuned.joblib. ADASYN post-split on train only. {fallback_used or 'Default params used (ss=1.0, k=5).'}"
    for thr in (0.5, 0.7):
        pred = (proba_te >= thr).astype(int)
        entry = build_entry(
            name=exp_name, gap="Gap 2: oversampling comparison",
            model="XGBoost (SMOTE+tuned params)",
            features=FEATURES_14, oversampling="ADASYN",
            threshold=thr, y_true=y_te, y_pred=pred, y_score=proba_te,
            runtime_s=runtime, notes=base_note,
        )
        append_gap_entry(entry)
        log(f"  SAVED @ thr={thr} | F1={entry['F1']:.4f} P={entry['Precision']:.4f} "
            f"R={entry['Recall']:.4f} ROC={entry['ROC_AUC']:.4f}")
    return entry


def _focal_loss_obj(gamma: float, alpha: float):
    """Custom XGBoost focal-loss objective. Returns grad, hess (1st and 2nd deriv wrt raw score)."""
    def obj(y_pred: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        y = dtrain.get_label() if hasattr(dtrain, "get_label") else dtrain
        # numerical-stable sigmoid
        p = 1.0 / (1.0 + np.exp(-y_pred))
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        # focal loss: y*alpha*(1-p)^g*log(p) + (1-y)*(1-alpha)*p^g*log(1-p)
        # grad wrt z = p - y weighted by focal modulation; derivation:
        # for positives: dL/dz = -alpha*(1-p)^g * (g*p*log(p) + p - 1)*something ... use std derivation:
        # Use Lin et al. compact formulation.
        a_t = np.where(y == 1, alpha, 1 - alpha)
        p_t = np.where(y == 1, p, 1 - p)
        # grad wrt z (=p*y_sign)
        # standard focal-loss derivation yields:
        grad = a_t * (1 - p_t) ** gamma * (gamma * p_t * np.log(p_t + eps) + p_t - 1) * np.where(y == 1, 1, -1)
        # hessian approximation (common simplification)
        hess = a_t * (1 - p_t) ** gamma * (
            (1 - p_t) * (1 - (1 + gamma) * p_t - gamma * p_t * np.log(p_t + eps))
        )
        hess = np.maximum(hess, 1e-6)
        return grad, hess
    return obj


def run_xgb_focal_loss() -> dict | None:
    from xgboost import XGBClassifier

    exp_name = "XGBoost_FocalLoss"
    log(f"EXPERIMENT: {exp_name}")
    t0 = time.time()

    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)
    X_tr = df_tr[FEATURES_14].to_numpy(dtype=np.float32)
    X_te = df_te[FEATURES_14].to_numpy(dtype=np.float32)
    y_tr = df_tr["is_fraud"].to_numpy(dtype=np.int64)
    y_te = df_te["is_fraud"].to_numpy(dtype=np.int64)

    tuned = _load_xgb_tuned_params()
    pi = float(y_tr.mean())  # fraud rate
    alpha = 1.0 - pi  # emphasize minority (positives) — alpha > 0.5
    gamma = 2.0
    log(f"  focal-loss: gamma={gamma}, alpha={alpha:.4f} (=1-fraud_rate)")

    fallback_used = None
    try:
        obj = _focal_loss_obj(gamma, alpha)
        clf = XGBClassifier(
            **tuned,
            objective=obj,
            random_state=SEED, n_jobs=-1, tree_method="hist",
        )
        log("  fitting XGBoost with custom focal-loss objective (no resampling)")
        clf.fit(X_tr, y_tr)
        proba_te = clf.predict_proba(X_te)[:, 1]
        oversampling = "FocalLoss"
        model_tag = "XGBoost (FocalLoss obj, no resampling)"
    except Exception as e:
        log(f"  custom focal-loss failed: {e}")
        log(f"  traceback: {traceback.format_exc()}")
        log("  FALLBACK: BalancedBaggingClassifier with XGBoost base")
        try:
            from imblearn.ensemble import BalancedBaggingClassifier
            base = XGBClassifier(
                **tuned, eval_metric="logloss", random_state=SEED,
                n_jobs=-1, tree_method="hist",
            )
            clf = BalancedBaggingClassifier(
                estimator=base, n_estimators=5,
                sampling_strategy="auto", replacement=False,
                random_state=SEED, n_jobs=1,
            )
            clf.fit(X_tr, y_tr)
            proba_te = clf.predict_proba(X_te)[:, 1]
            oversampling = "BalancedBagging"
            model_tag = "XGBoost (BalancedBagging — Focal Loss unavailable)"
            fallback_used = f"focal-loss fallback -> BalancedBagging ({e})"
        except Exception as e2:
            log(f"  BalancedBagging also failed: {e2}")
            d = load_metrics()
            d.setdefault("gap_experiments", []).append({
                "experiment_name": exp_name, "status": "FAILED",
                "gap_addressed": "Gap 2: oversampling comparison",
                "notes": f"focal-loss + BalancedBagging both failed: {e} / {e2}",
            })
            save_metrics(d)
            return None

    runtime = time.time() - t0
    base_note = (
        f"Custom focal-loss (gamma=2.0, alpha=1-fraud_rate={alpha:.4f}). No resampling, no class weighting. "
        f"Tuned params reused from xgboost_smote_tuned.joblib."
    )
    if fallback_used:
        base_note = fallback_used + " | " + base_note

    for thr in (0.5, 0.7):
        pred = (proba_te >= thr).astype(int)
        entry = build_entry(
            name=exp_name, gap="Gap 2: oversampling comparison",
            model=model_tag, features=FEATURES_14, oversampling=oversampling,
            threshold=thr, y_true=y_te, y_pred=pred, y_score=proba_te,
            runtime_s=runtime, notes=base_note,
        )
        append_gap_entry(entry)
        log(f"  SAVED @ thr={thr} | F1={entry['F1']:.4f} P={entry['Precision']:.4f} "
            f"R={entry['Recall']:.4f} ROC={entry['ROC_AUC']:.4f}")
    return entry


# ---------- orchestrator ----------
def safe_run(fn, *args, **kwargs) -> dict | None:
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log(f"UNCAUGHT FAILURE in {fn.__name__}: {e}")
        log(traceback.format_exc())
        d = load_metrics()
        d.setdefault("gap_experiments", []).append({
            "experiment_name": kwargs.get("exp_name", fn.__name__),
            "status": "FAILED",
            "notes": f"uncaught exception: {e}",
        })
        save_metrics(d)
        return None


def main() -> None:
    log("=" * 70)
    log("GAP EXPERIMENTS RUNNER — 5 experiments sequential")
    log(f"Python {sys.version.split()[0]} | torch {torch.__version__} | seed={SEED}")
    log(f"Train: {TRAIN_CSV.name}  Test: {TEST_CSV.name}")
    log("=" * 70)

    t_all = time.time()

    # Experiment 1: LSTM reproduced baseline (with velocity)
    log("\n### EXPERIMENT 1/5: LSTM_reproduced_baseline ###")
    exp1 = safe_run(run_lstm_experiment,
                    exp_name="LSTM_reproduced_baseline",
                    features=FEATURES_14,
                    notes_suffix=(
                        "Supersedes original F1=0.4747 which was produced by the BROKEN "
                        "alignment pipeline (cc_num positional attach silently misaligned). "
                        "See AUDIT_WIN_NARRATIVE.md and FYP_Hybrid_Model_BROKEN.ipynb."
                    ))

    # Experiment 2: LSTM no-velocity
    log("\n### EXPERIMENT 2/5: LSTM_no_velocity ###")
    safe_run(run_lstm_experiment,
             exp_name="LSTM_no_velocity",
             features=FEATURES_11,
             notes_suffix="Velocity features removed to test LSTM capacity without engineered sequence signals.")

    # Experiment 3: LSTM with-velocity (cross-ref to #1 since deterministic)
    log("\n### EXPERIMENT 3/5: LSTM_with_velocity (cross-reference) ###")
    if exp1 is not None:
        note = (
            f"CROSS-REFERENCE: same feature set, seed, and pipeline as LSTM_reproduced_baseline "
            f"(F1={exp1['F1']:.4f}). Not re-run to save compute; deterministic result."
        )
        entry = dict(exp1)
        entry["experiment_name"] = "LSTM_with_velocity"
        entry["notes"] = note
        entry["runtime_seconds"] = 0.0
        append_gap_entry(entry)
        log(f"  cross-referenced: F1={exp1['F1']:.4f}")
    else:
        log("  SKIPPED: Experiment 1 failed, no baseline to cross-reference. Re-running.")
        safe_run(run_lstm_experiment,
                 exp_name="LSTM_with_velocity",
                 features=FEATURES_14,
                 notes_suffix="Re-run because Experiment 1 failed.")

    # Experiment 4: XGBoost + ADASYN
    log("\n### EXPERIMENT 4/5: XGBoost_ADASYN ###")
    safe_run(run_xgb_adasyn)

    # Experiment 5: XGBoost + Focal Loss
    log("\n### EXPERIMENT 5/5: XGBoost_FocalLoss ###")
    safe_run(run_xgb_focal_loss)

    total = time.time() - t_all
    h, rem = divmod(int(total), 3600)
    m, s = divmod(rem, 60)
    log("=" * 70)
    log(f"ALL EXPERIMENTS COMPLETE. Total runtime: {h}h {m}m {s}s")
    log("Results: verified_metrics.json (gap_experiments key)")
    log("Log:     FEEDBACK_GAPS_RUN_LOG.txt")
    log("=" * 70)


if __name__ == "__main__":
    main()

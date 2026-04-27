"""
Compute PR-AUC for the saved LSTM+RF Hybrid Comparator on the held-out test set
and patch results/verified_metrics.json to add a "PR_AUC" field on the
LSTM_reproduced_baseline entry.

Pipeline (must mirror scripts/save_lstm_rf_model.py exactly):
  - sort test by (cc_num, unix_time)
  - scale 14 features with the SAVED scaler
  - build per-card sliding sequences of length 5 (zero-padded at card start)
  - LSTM forward pass -> probability per row
  - RF input = [LSTM_prob | RAW (unscaled) static features]
  - probs = rf.predict_proba(...)[:, 1]
  - pr_auc = average_precision_score(y_test, probs)

Outputs:
  - prints F1@0.5, ROC-AUC, PR-AUC for sanity
  - updates results/verified_metrics.json -> gap_experiments[name=LSTM_reproduced_baseline].PR_AUC
"""
from __future__ import annotations

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTHONHASHSEED"] = "42"

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "data" / "engineered" / "fraudTest_engineered_with_ids.csv"
SAVE_DIR = ROOT / "models" / "saved" / "02_comparator"
METRICS_PATH = ROOT / "results" / "verified_metrics.json"

SEED = 42
SEQ_LEN = 5
FEATURES = [
    "amt", "city_pop", "hour", "month", "distance_cardholder_merchant",
    "age", "is_weekend", "is_night",
    "velocity_1h", "velocity_24h", "amount_velocity_1h",
    "category_encoded", "gender_encoded", "day_of_week_encoded",
]


def set_all_seeds(s: int = SEED) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_sequences(X: np.ndarray, cc: np.ndarray, seq_len: int) -> np.ndarray:
    """Per-card chronological sequences. X already sorted by (cc_num, unix_time)."""
    n, d = X.shape
    out = np.zeros((n, seq_len, d), dtype=np.float32)
    boundaries = np.r_[0, np.where(np.diff(cc) != 0)[0] + 1, n]
    for k in range(len(boundaries) - 1):
        s, e = boundaries[k], boundaries[k + 1]
        for i in range(s, e):
            start = max(s, i - seq_len + 1)
            seq = X[start:i + 1]
            if len(seq) < seq_len:
                pad = np.zeros((seq_len - len(seq), d), dtype=np.float32)
                seq = np.vstack([pad, seq])
            out[i] = seq
    return out


def main() -> None:
    import keras  # noqa: F401  (registers backend before load)

    print(f"[load] {TEST_CSV.name}")
    df = pd.read_csv(TEST_CSV)
    df = df.sort_values(["cc_num", "unix_time"], kind="mergesort").reset_index(drop=True)
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["is_fraud"].to_numpy(dtype=np.int64)
    cc = df["cc_num"].to_numpy()
    print(f"       rows={len(X):,}  fraud_rate={y.mean():.4f}")

    print(f"[load] {SAVE_DIR.name}/lstm_rf_scaler.joblib")
    scaler = joblib.load(SAVE_DIR / "lstm_rf_scaler.joblib")
    X_s = scaler.transform(X).astype(np.float32)

    print(f"[seq ] sliding windows seq_len={SEQ_LEN}")
    seq = build_sequences(X_s, cc, SEQ_LEN)

    print(f"[load] lstm_rf_keras.keras")
    set_all_seeds(SEED)
    import keras as K
    K.utils.set_random_seed(SEED)
    lstm = K.models.load_model(SAVE_DIR / "lstm_rf_keras.keras")

    print("[lstm] forward pass")
    p = lstm.predict(seq, batch_size=1024, verbose=0).ravel()

    print("[load] lstm_rf_classifier.joblib")
    rf = joblib.load(SAVE_DIR / "lstm_rf_classifier.joblib")

    print("[rf  ] predict_proba on [lstm_prob | RAW static features]")
    X_rf = np.column_stack([p, X])
    proba = rf.predict_proba(X_rf)[:, 1]

    pred05 = (proba >= 0.5).astype(int)
    f1 = float(f1_score(y, pred05))
    prec = float(precision_score(y, pred05, zero_division=0))
    rec = float(recall_score(y, pred05))
    roc = float(roc_auc_score(y, proba))
    pr_auc = float(average_precision_score(y, proba))

    print()
    print("=" * 60)
    print(f"  F1@0.5     = {f1:.4f}   (reference 0.7892)")
    print(f"  Precision  = {prec:.4f}   (reference 0.6770)")
    print(f"  Recall     = {rec:.4f}   (reference 0.9459)")
    print(f"  ROC-AUC    = {roc:.4f}   (reference 0.9981)")
    print(f"  PR-AUC     = {pr_auc:.4f}   <-- NEW")
    print("=" * 60)

    # Patch verified_metrics.json
    print(f"\n[patch] {METRICS_PATH.relative_to(ROOT)}")
    with METRICS_PATH.open(encoding="utf-8") as f:
        d = json.load(f)

    target = None
    for entry in d.get("gap_experiments", []):
        if entry.get("experiment_name") == "LSTM_reproduced_baseline":
            target = entry
            break
    if target is None:
        raise RuntimeError("No LSTM_reproduced_baseline entry found in gap_experiments")

    target["PR_AUC"] = round(pr_auc, 4)
    target.setdefault("notes", "")
    if "PR-AUC computed post-hoc" not in target["notes"]:
        target["notes"] = (
            target["notes"].rstrip() + " "
            "PR-AUC computed post-hoc from saved artifacts via "
            "scripts/compute_lstm_pr_auc.py (deterministic, seed=42)."
        ).strip()

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    print(f"        added PR_AUC = {round(pr_auc, 4)} to LSTM_reproduced_baseline")

    # Also update Table A in app/config.py? — out of scope; user can do that
    # manually after seeing the printed value, or we can wire it next.
    print("\n[done]")


if __name__ == "__main__":
    main()

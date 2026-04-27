"""Train the LSTM + RF Hybrid Comparator and save the trained artifacts to models/saved/.

Reproduces the exact F1=0.7892 pipeline recorded in results/verified_metrics.json
under "LSTM_reproduced_baseline". Seed is fixed so the resulting model is
deterministic against that record.

Outputs:
  models/saved/lstm_rf_keras.keras        — trained LSTM (Keras format)
  models/saved/lstm_rf_classifier.joblib  — trained Random Forest head
  models/saved/lstm_rf_scaler.joblib      — StandardScaler fitted on train

Run:
    python scripts/save_lstm_rf_model.py

Runtime: ~30-60 minutes on CPU.
"""
from __future__ import annotations

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTHONHASHSEED"] = "42"

import random
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---- Paths ----
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_CSV = PROJECT_ROOT / "data" / "engineered" / "fraudTrain_engineered_with_ids.csv"
TEST_CSV  = PROJECT_ROOT / "data" / "engineered" / "fraudTest_engineered_with_ids.csv"
SAVE_DIR  = PROJECT_ROOT / "models" / "saved"

# ---- Hyperparameters (must match LSTM_reproduced_baseline in verified_metrics.json) ----
SEED = 42
SEQ_LEN = 5
FEATURES = [
    "amt", "city_pop", "hour", "month", "distance_cardholder_merchant",
    "age", "is_weekend", "is_night",
    "velocity_1h", "velocity_24h", "amount_velocity_1h",
    "category_encoded", "gender_encoded", "day_of_week_encoded",
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def set_all_seeds(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_sequences(X: np.ndarray, cc: np.ndarray, seq_len: int) -> np.ndarray:
    """Per-card chronological sequences. X must already be sorted by (cc_num, unix_time).
    Returns (n_rows, seq_len, n_features). Pads with zeros for new cards."""
    n, d = X.shape
    out = np.zeros((n, seq_len, d), dtype=np.float32)
    # Find segment boundaries by cc change
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
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing CSV(s):\n  {TRAIN_CSV}\n  {TEST_CSV}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    import keras
    from keras import Sequential, layers, optimizers

    set_all_seeds(SEED)
    keras.utils.set_random_seed(SEED)

    log("Loading CSVs...")
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)

    df_tr = df_tr.sort_values(["cc_num", "unix_time"], kind="mergesort").reset_index(drop=True)
    df_te = df_te.sort_values(["cc_num", "unix_time"], kind="mergesort").reset_index(drop=True)

    X_tr = df_tr[FEATURES].to_numpy(dtype=np.float32)
    X_te = df_te[FEATURES].to_numpy(dtype=np.float32)
    y_tr = df_tr["is_fraud"].to_numpy(dtype=np.int64)
    y_te = df_te["is_fraud"].to_numpy(dtype=np.int64)
    cc_tr = df_tr["cc_num"].to_numpy()
    cc_te = df_te["cc_num"].to_numpy()
    log(f"  train: {X_tr.shape} | fraud rate {y_tr.mean():.4f}")
    log(f"  test:  {X_te.shape} | fraud rate {y_te.mean():.4f}")

    log("Fitting StandardScaler on train...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    log("Building sequences...")
    seq_tr = build_sequences(X_tr_s, cc_tr, SEQ_LEN)
    seq_te = build_sequences(X_te_s, cc_te, SEQ_LEN)
    log(f"  seq_tr: {seq_tr.shape}, seq_te: {seq_te.shape}")

    n0, n1 = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    cw = {0: 1.0, 1: n0 / max(n1, 1)}
    log(f"  class_weight = {cw}")

    log("Compiling LSTM (LSTM64 + Dropout0.3 + Dense32 + Dense16 + Sigmoid)...")
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

    log("Training LSTM (5 epochs, batch=512) — long step...")
    t0 = time.time()
    model.fit(
        seq_tr, y_tr,
        epochs=5, batch_size=512,
        class_weight=cw,
        validation_split=0.1,
        verbose=2,
    )
    log(f"  LSTM training time: {time.time() - t0:.0f}s")

    log("Producing LSTM probabilities for RF input...")
    p_tr = model.predict(seq_tr, batch_size=1024, verbose=0).ravel()
    p_te = model.predict(seq_te, batch_size=1024, verbose=0).ravel()

    log("Fitting Random Forest on [LSTM_prob | static features]...")
    X_rf_tr = np.column_stack([p_tr, X_tr])
    X_rf_te = np.column_stack([p_te, X_te])
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight={0: 1, 1: 150},
        random_state=SEED, n_jobs=-1,
    )
    t0 = time.time()
    rf.fit(X_rf_tr, y_tr)
    log(f"  RF training time: {time.time() - t0:.0f}s")

    log("Evaluating @ threshold 0.5...")
    proba_te = rf.predict_proba(X_rf_te)[:, 1]
    pred_te = (proba_te >= 0.5).astype(int)
    f1 = f1_score(y_te, pred_te)
    prec = precision_score(y_te, pred_te)
    rec = recall_score(y_te, pred_te)
    roc = roc_auc_score(y_te, proba_te)
    log(f"  F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  ROC-AUC={roc:.4f}")
    log(f"  (verified_metrics.json reference: F1=0.7892, P=0.6770, R=0.9459, ROC=0.9981)")

    log("Saving artifacts to models/saved/...")
    keras_path = SAVE_DIR / "lstm_rf_keras.keras"
    rf_path = SAVE_DIR / "lstm_rf_classifier.joblib"
    sc_path = SAVE_DIR / "lstm_rf_scaler.joblib"
    model.save(keras_path)
    joblib.dump(rf, rf_path)
    joblib.dump(scaler, sc_path)
    log(f"  saved: {keras_path}")
    log(f"  saved: {rf_path}")
    log(f"  saved: {sc_path}")

    log("Done.")


if __name__ == "__main__":
    main()

"""Minimal pipeline smoke test — 1000 rows, 1 epoch, verify no crashes."""
import os
os.environ["KERAS_BACKEND"] = "torch"
import sys, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from run_gap_experiments import build_sequences, FEATURES_14, FEATURES_11, SEED, SEQ_LEN

print("=" * 60)
print("SMOKE TEST — 1000 rows, 1 epoch, verify pipeline end-to-end")
print("=" * 60)

df_tr = pd.read_csv(ROOT / "fraudTrain_engineered_with_ids.csv", nrows=10000)
df_te = pd.read_csv(ROOT / "fraudTest_engineered_with_ids.csv", nrows=5000)
# Take a mix of fraud and non-fraud
df_tr = pd.concat([
    df_tr[df_tr["is_fraud"] == 0].sample(900, random_state=SEED),
    df_tr[df_tr["is_fraud"] == 1].sample(min(100, (df_tr["is_fraud"] == 1).sum()), random_state=SEED),
]).sort_values(["cc_num", "unix_time"]).reset_index(drop=True)
df_te = df_te.sort_values(["cc_num", "unix_time"]).reset_index(drop=True)
print(f"train rows: {len(df_tr)} (fraud={df_tr.is_fraud.sum()})")
print(f"test rows:  {len(df_te)} (fraud={df_te.is_fraud.sum()})")

features = FEATURES_14
X_tr = df_tr[features].to_numpy(dtype=np.float32)
X_te = df_te[features].to_numpy(dtype=np.float32)
y_tr = df_tr["is_fraud"].to_numpy(dtype=np.int64)
y_te = df_te["is_fraud"].to_numpy(dtype=np.int64)
cc_tr = df_tr["cc_num"].to_numpy()
cc_te = df_te["cc_num"].to_numpy()

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
X_te_s = scaler.transform(X_te).astype(np.float32)

print("building sequences...")
t0 = time.time()
seq_tr = build_sequences(X_tr_s, cc_tr, SEQ_LEN)
seq_te = build_sequences(X_te_s, cc_te, SEQ_LEN)
print(f"  seq_tr {seq_tr.shape}  seq_te {seq_te.shape}  took {time.time()-t0:.2f}s")

print("compiling & training Keras LSTM for 1 epoch...")
import keras
keras.utils.set_random_seed(SEED)
from keras import Sequential, layers, optimizers
model = Sequential([
    layers.Input(shape=(SEQ_LEN, X_tr.shape[1])),
    layers.LSTM(64),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=optimizers.Adam(0.001), loss="binary_crossentropy")
t0 = time.time()
model.fit(seq_tr, y_tr, epochs=1, batch_size=256, verbose=1)
print(f"  1 epoch on 1k rows took {time.time()-t0:.2f}s")

p_te = model.predict(seq_te, batch_size=512, verbose=0).ravel()
rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1)
rf.fit(np.column_stack([model.predict(seq_tr, batch_size=512, verbose=0).ravel(), X_tr]), y_tr)
proba = rf.predict_proba(np.column_stack([p_te, X_te]))[:, 1]
pred = (proba >= 0.5).astype(int)
print(f"smoke test F1 (meaningless on small sample): {f1_score(y_te, pred):.4f}")
print("OK — full pipeline runs end-to-end without errors.")

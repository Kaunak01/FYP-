"""
Exp 5 fallback — custom focal-loss XGBoost collapsed to F1=0.0000 (predicted all-zero).
Run the pre-planned BalancedBaggingClassifier fallback and append to gap_experiments.
"""
import os, json, time, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from imblearn.ensemble import BalancedBaggingClassifier
from xgboost import XGBClassifier

SEED = 42
FEATURES_14 = [
    "amt","city_pop","hour","month","distance_cardholder_merchant","age",
    "is_weekend","is_night","velocity_1h","velocity_24h","amount_velocity_1h",
    "category_encoded","gender_encoded","day_of_week_encoded",
]
LOG = "FEEDBACK_GAPS_RUN_LOG.txt"

def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("")
log("### EXPERIMENT 5 FALLBACK: XGBoost_FocalLoss -> BalancedBagging ###")

t0 = time.time()
df_tr = pd.read_csv("fraudTrain_engineered_with_ids.csv")
df_te = pd.read_csv("fraudTest_engineered_with_ids.csv")
X_tr = df_tr[FEATURES_14].to_numpy(dtype=np.float32)
X_te = df_te[FEATURES_14].to_numpy(dtype=np.float32)
y_tr = df_tr["is_fraud"].to_numpy(dtype=np.int64)
y_te = df_te["is_fraud"].to_numpy(dtype=np.int64)

log(f"  loaded train={len(y_tr):,} test={len(y_te):,} fraud_rate_train={y_tr.mean():.4%}")

tuned = joblib.load("xgboost_smote_tuned.joblib").get_params()
base_params = {k: tuned[k] for k in [
    "n_estimators","max_depth","learning_rate","subsample",
    "colsample_bytree","min_child_weight","gamma"
] if k in tuned and tuned[k] is not None}
log(f"  reused XGBoost tuned params: {base_params}")

base = XGBClassifier(**base_params, eval_metric="logloss", random_state=SEED, n_jobs=-1, verbosity=0)

bbc = BalancedBaggingClassifier(
    estimator=base,
    n_estimators=10,
    sampling_strategy=1.0,
    replacement=False,
    random_state=SEED,
    n_jobs=1,
)

log("  fitting BalancedBaggingClassifier(XGB tuned, n_estimators=10, ss=1.0)")
bbc.fit(X_tr, y_tr)
log(f"  fit done in {time.time()-t0:.1f}s")

proba = bbc.predict_proba(X_te)[:, 1]

results = []
for thr in (0.5, 0.7):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    entry = {
        "experiment_name": "XGBoost_FocalLoss_fallback_BalancedBagging",
        "gap_addressed": "Gap 2: imbalance handling (focal-loss alternative)",
        "model": "BalancedBaggingClassifier(XGBoost tuned, n_est=10, ss=1.0)",
        "features_used": 14,
        "oversampling": "BalancedBagging (undersampling per bag, balanced 1.0)",
        "threshold": thr,
        "F1": float(f1_score(y_te, pred)),
        "Precision": float(precision_score(y_te, pred, zero_division=0)),
        "Recall": float(recall_score(y_te, pred)),
        "ROC_AUC": float(roc_auc_score(y_te, proba)),
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "runtime_seconds": time.time() - t0,
        "random_state": SEED,
        "notes": "Fallback for custom focal-loss which collapsed to F1=0.0000 (all-negative predictions). BalancedBagging reuses tuned XGB as base estimator but rebalances per bootstrap bag via undersampling majority. Documented as the planned fallback in XGBOOST_BASELINE_CONFIG.json.",
    }
    results.append(entry)
    log(f"  SAVED @ thr={thr} | F1={entry['F1']:.4f} P={entry['Precision']:.4f} R={entry['Recall']:.4f} ROC={entry['ROC_AUC']:.4f}")

with open("verified_metrics.json", "r", encoding="utf-8") as f:
    vm = json.load(f)
vm.setdefault("gap_experiments", []).extend(results)
with open("verified_metrics.json", "w", encoding="utf-8") as f:
    json.dump(vm, f, indent=2, ensure_ascii=False)

log("  appended 2 rows to verified_metrics.json[gap_experiments]")
log(f"FALLBACK COMPLETE in {time.time()-t0:.1f}s")

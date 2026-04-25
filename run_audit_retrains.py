"""Re-runs the long jobs that implement the Codex audit methodology fixes.

Usage:
    python run_audit_retrains.py --true-no-velocity    # P2: true from-scratch no-velocity AE ablation
    python run_audit_retrains.py --pipeline-cv-ae      # P4: AE+XGBoost with SMOTE inside CV
    python run_audit_retrains.py --pipeline-cv-bds     # P4: AE+BDS+XGBoost with SMOTE inside CV
    python run_audit_retrains.py --all                 # all of the above

Each successful run appends a block to verified_metrics.json under the
`audit_reruns` key so the notebook headline metrics are not overwritten.

Artefact source: `models/saved/` only — matches `rerun_verification.py`.
BDS construction: matches `rerun_verification.py` precompute/compute_bds exactly.
Seed: 42 everywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths — single source of truth, matches rerun_verification.py
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
MODELS = ROOT / "models" / "saved"
TRAIN_CSV = ROOT / "fraudTrain_engineered.csv"
TEST_CSV = ROOT / "fraudTest_engineered.csv"
RAW_TRAIN_CSV = ROOT / "fraudTrain.csv"
RAW_TEST_CSV = ROOT / "fraudTest.csv"

SEED = 42
FEATURE_COLS = [
    "amt", "city_pop", "hour", "month", "distance_cardholder_merchant", "age",
    "is_weekend", "is_night", "velocity_1h", "velocity_24h", "amount_velocity_1h",
    "category_encoded", "gender_encoded", "day_of_week_encoded",
]
VELOCITY_COLS = ["velocity_1h", "velocity_24h", "amount_velocity_1h"]
XGB_PARAM_DIST = {
    "n_estimators":     [200, 300, 400],
    "max_depth":        [4, 6, 8, 10],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.1, 0.2],
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_engineered():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    X_train = train[FEATURE_COLS].values
    y_train = train["is_fraud"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["is_fraud"].values
    return train, test, X_train, y_train, X_test, y_test


def metrics_block(name, y_true, y_pred, y_prob, threshold, extra=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    block = {
        "experiment_name": name,
        "threshold": threshold,
        "F1": float(f1_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "random_state": SEED,
    }
    if extra:
        block.update(extra)
    return block


def append_verified(block):
    path = ROOT / "verified_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.setdefault("audit_reruns", []).append(block)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[verified_metrics.json] appended audit_rerun: {block['experiment_name']}")


# ---------------------------------------------------------------------------
# Autoencoder (shared architecture, per-input-dim)
# ---------------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 10), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(10, 5), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 10), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(10, d),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_saved_ae():
    """Load the saved AE + scaler from models/saved/ (same as rerun_verification.py)."""
    import joblib
    scaler = joblib.load(MODELS / "ae_scaler.joblib")
    ae = Autoencoder(d=len(FEATURE_COLS))
    ae.load_state_dict(torch.load(MODELS / "ae_model.pt", map_location="cpu"))
    ae.eval()
    return ae, scaler


def saved_recon_error(ae, scaler, X):
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        r = ae(torch.tensor(Xs)).numpy()
    return np.mean((Xs - r) ** 2, axis=1)


def train_autoencoder(X_normal, epochs=30, patience=3, batch_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = Autoencoder(X_normal.shape[1]).to(device)
    tensor = torch.FloatTensor(X_normal).to(device)
    val_size = int(0.1 * len(tensor))
    train_size = len(tensor) - val_size
    gen = torch.Generator().manual_seed(SEED)
    train_data, val_data = random_split(
        TensorDataset(tensor, tensor), [train_size, val_size], generator=gen
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    opt = torch.optim.Adam(ae.parameters())
    crit = nn.MSELoss()
    best_val, counter, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        ae.train()
        for xb, _ in train_loader:
            pred = ae(xb)
            loss = crit(pred, xb)
            opt.zero_grad(); loss.backward(); opt.step()
        ae.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, _ in val_loader:
                vl += crit(ae(xb), xb).item() * len(xb)
        vl /= val_size
        print(f"  [AE] epoch {epoch+1}/{epochs} val_loss={vl:.6f}")
        if vl < best_val:
            best_val, counter = vl, 0
            best_state = {k: v.clone() for k, v in ae.state_dict().items()}
        else:
            counter += 1
            if counter >= patience:
                print("  [AE] early stop")
                break
    ae.load_state_dict(best_state)
    return ae


def scratch_recon_error(ae, X_scaled):
    device = next(ae.parameters()).device
    ae.eval()
    with torch.no_grad():
        rec = ae(torch.FloatTensor(X_scaled).to(device)).cpu().numpy()
    return np.mean((X_scaled - rec) ** 2, axis=1)


# ---------------------------------------------------------------------------
# BDS feature construction — CANONICAL PATTERN from rerun_verification.py
# (rebuilds per-card stats from the training partition; does not load the
# joblib to avoid the schema mismatch noted in the Codex audit.)
# ---------------------------------------------------------------------------
def build_bds_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Returns (bds_tr, bds_te) where each is a tuple of 4 arrays."""
    # Ensure cc_num is available; fall back to raw CSV if engineered file lacks it.
    for df, raw in ((df_train, RAW_TRAIN_CSV), (df_test, RAW_TEST_CSV)):
        if "cc_num" not in df.columns or df["cc_num"].nunique() <= 1:
            print(f"  attaching cc_num from {raw.name}")
            df["cc_num"] = pd.read_csv(raw, usecols=["cc_num"])["cc_num"].values

    card_amt = df_train.groupby("cc_num")["amt"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "amt_mean", "std": "amt_std", "count": "amt_count"})
    card_amt["amt_std"] = card_amt["amt_std"].fillna(0)

    n_categories = int(max(df_train["category_encoded"].max(), df_test["category_encoded"].max()) + 1)
    card_hour_prob = (df_train.groupby(["cc_num", "hour"]).size()
                      / df_train.groupby("cc_num").size()).unstack(fill_value=0)
    card_cat_prob = (df_train.groupby(["cc_num", "category_encoded"]).size()
                     / df_train.groupby("cc_num").size()).unstack(fill_value=0)
    card_vel = df_train.groupby("cc_num")["velocity_1h"].agg(["mean"]).rename(
        columns={"mean": "vel_mean"})

    gstats = {
        "amt_mean": df_train["amt"].mean(),
        "amt_std":  df_train["amt"].std(),
        "hour_prob": df_train["hour"].value_counts(normalize=True).to_dict(),
        "cat_prob":  df_train["category_encoded"].value_counts(normalize=True).to_dict(),
        "vel_mean":  df_train["velocity_1h"].mean(),
    }

    def precompute(cc, amt, hr, cat, vel):
        d = pd.DataFrame({"cc_num": cc, "amt": amt,
                          "hour": hr.astype(int), "cat": cat.astype(int), "vel": vel})
        d = d.merge(card_amt, left_on="cc_num", right_index=True, how="left")
        d["amt_mean"] = d["amt_mean"].fillna(gstats["amt_mean"])
        d["amt_std"] = d["amt_std"].fillna(gstats["amt_std"])
        d["amt_count"] = d["amt_count"].fillna(0)
        d["unseen"] = d["amt_count"] == 0
        ss = d["amt_std"].where(d["amt_std"] > 0, gstats["amt_std"])
        d["card_amt_z"] = (d["amt"] - d["amt_mean"]).abs() / ss
        d["global_amt_z"] = (d["amt"] - gstats["amt_mean"]).abs() / gstats["amt_std"]

        hs = card_hour_prob.stack(); hs.index.names = ["cc_num", "hour"]
        hl = hs.reset_index(); hl.columns = ["cc_num", "hour", "card_hour_p"]
        d = d.merge(hl, on=["cc_num", "hour"], how="left")
        d["card_hour_p"] = d["card_hour_p"].fillna(0.0)
        d["global_hour_p"] = d["hour"].map(gstats["hour_prob"]).fillna(1 / 24)

        cs = card_cat_prob.stack(); cs.index.names = ["cc_num", "cat"]
        cl = cs.reset_index(); cl.columns = ["cc_num", "cat", "card_cat_p"]
        d = d.merge(cl, on=["cc_num", "cat"], how="left")
        d["card_cat_p"] = d["card_cat_p"].fillna(0.0)
        d["global_cat_p"] = d["cat"].map(gstats["cat_prob"]).fillna(1 / n_categories)

        d = d.merge(card_vel, left_on="cc_num", right_index=True, how="left")
        d["vel_mean"] = d["vel_mean"].fillna(gstats["vel_mean"])
        sv = d["vel_mean"].where(d["vel_mean"] > 0, gstats["vel_mean"])
        d["freq_ratio"] = d["vel"] / sv
        return {k: d[k].values for k in [
            "card_amt_z", "global_amt_z", "amt_count",
            "card_hour_p", "global_hour_p", "card_cat_p", "global_cat_p",
            "freq_ratio", "unseen",
        ]}

    def compute_bds(raw, params):
        at, ac, tt, tc, ft, fc, ct, cc_, mh, sm = params
        mh = int(round(mh))
        ug = (raw["amt_count"] < mh) | raw["unseen"]
        az = np.where(ug, raw["global_amt_z"], raw["card_amt_z"])
        bds_amount = np.clip(np.maximum(az - at, 0), 0, ac)
        hp = np.where(ug, raw["global_hour_p"], raw["card_hour_p"])
        bds_time = np.clip(np.maximum(-np.log(hp + sm) - tt, 0), 0, tc)
        fr = np.maximum(raw["freq_ratio"] - 1.0, 0)
        bds_freq = np.clip(np.maximum(fr - ft, 0), 0, fc)
        cp = np.where(ug, raw["global_cat_p"], raw["card_cat_p"])
        bds_category = np.clip(np.maximum(-np.log(cp + sm) - ct, 0), 0, cc_)
        return bds_amount, bds_time, bds_freq, bds_category

    ga = json.loads((MODELS / "ga_best_params.json").read_text())
    best_params = [ga["params"][k] for k in (
        "amount_threshold", "amount_cap", "time_threshold", "time_cap",
        "freq_threshold", "freq_cap", "cat_threshold", "cat_cap",
        "min_history", "smoothing",
    )]

    raw_tr = precompute(df_train["cc_num"].values, df_train["amt"].values,
                        df_train["hour"].values, df_train["category_encoded"].values,
                        df_train["velocity_1h"].values)
    raw_te = precompute(df_test["cc_num"].values, df_test["amt"].values,
                        df_test["hour"].values, df_test["category_encoded"].values,
                        df_test["velocity_1h"].values)

    return compute_bds(raw_tr, best_params), compute_bds(raw_te, best_params)


# ---------------------------------------------------------------------------
# P2: true from-scratch no-velocity ablation
# ---------------------------------------------------------------------------
def run_true_no_velocity_ablation():
    print("\n" + "=" * 70)
    print("[P2] True from-scratch no-velocity AE + XGBoost ablation")
    print("=" * 70)
    seed_everything()
    t0 = time.time()

    _, _, X_train, y_train, X_test, y_test = load_engineered()

    keep_idx = [i for i, c in enumerate(FEATURE_COLS) if c not in VELOCITY_COLS]
    X_train_nv = X_train[:, keep_idx]
    X_test_nv = X_test[:, keep_idx]
    nv_cols = [c for c in FEATURE_COLS if c not in VELOCITY_COLS]
    print(f"  no-velocity feature set ({len(nv_cols)}): {nv_cols}")

    scaler_nv = StandardScaler()
    X_train_nv_scaled = scaler_nv.fit_transform(X_train_nv)
    X_test_nv_scaled = scaler_nv.transform(X_test_nv)
    X_train_nv_normal = X_train_nv_scaled[y_train == 0]

    print("  training autoencoder from scratch (no-velocity)...")
    ae_nv = train_autoencoder(X_train_nv_normal)

    train_rec_nv = scratch_recon_error(ae_nv, X_train_nv_scaled)
    test_rec_nv = scratch_recon_error(ae_nv, X_test_nv_scaled)

    X_train_hybrid_nv = np.column_stack([X_train_nv, train_rec_nv])
    X_test_hybrid_nv = np.column_stack([X_test_nv, test_rec_nv])

    sm = SMOTE(random_state=SEED)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_train_hybrid_nv, y_train)
    print(f"  SMOTE applied to no-velocity training only: "
          f"normal={(y_tr_sm == 0).sum():,}, fraud={(y_tr_sm == 1).sum():,}")

    # Reuse the locked tuned params from models/saved/ so we isolate the
    # effect of removing velocity rather than re-tuning.
    import joblib
    tuned_path = MODELS / "ae_xgboost_smote_tuned.joblib"
    if tuned_path.exists():
        tuned_model = joblib.load(tuned_path)
        best_params = tuned_model.get_params()
        keep_keys = set(XGB_PARAM_DIST.keys())
        best_params = {k: v for k, v in best_params.items() if k in keep_keys}
        print(f"  reusing tuned params from {tuned_path.name}: {best_params}")
    else:
        best_params = {}
        print(f"  WARNING: {tuned_path} not found — using defaults")

    xgb_nv = XGBClassifier(
        **best_params, eval_metric="logloss", random_state=SEED, n_jobs=-1,
    )
    xgb_nv.fit(X_tr_sm, y_tr_sm)

    y_prob_nv = xgb_nv.predict_proba(X_test_hybrid_nv)[:, 1]
    for thr in (0.5, 0.7):
        y_pred_nv = (y_prob_nv >= thr).astype(int)
        b = metrics_block(
            "AE_XGBoost_true_no_velocity_ablation",
            y_test, y_pred_nv, y_prob_nv, thr,
            extra={
                "gap_addressed": "P2 — true from-scratch no-velocity ablation",
                "features_used": nv_cols + ["recon_error"],
                "oversampling": "SMOTE (post-split, train-only)",
                "notes": (
                    "From-scratch no-velocity AE + XGBoost. "
                    "Scaler, AE, and XGBoost retrained from scratch on the no-velocity feature set. "
                    "Tuned XGB hyperparameters copied from models/saved/ae_xgboost_smote_tuned.joblib "
                    "to isolate the velocity contribution rather than re-tuning."
                ),
            },
        )
        append_verified(b)
        print(f"  thr={thr}: F1={b['F1']:.4f} Prec={b['Precision']:.4f} Rec={b['Recall']:.4f}")

    print(f"  runtime: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# P4: SMOTE-inside-CV Pipeline (AE / BDS variants)
# ---------------------------------------------------------------------------
def _run_pipeline_cv(name, X_train_hybrid, y_train, X_test_hybrid, y_test, features, extra_note):
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=SEED)),
        ("xgb", XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1)),
    ])
    param_dist_pipeline = {f"xgb__{k}": v for k, v in XGB_PARAM_DIST.items()}
    rs = RandomizedSearchCV(
        pipeline, param_distributions=param_dist_pipeline,
        n_iter=30, scoring="f1", cv=3, random_state=SEED, verbose=1, n_jobs=-1,
    )
    print(f"  fitting Pipeline-CV for {name} (30 iter x 3 fold)...")
    rs.fit(X_train_hybrid, y_train)
    print(f"  best params: {rs.best_params_}")
    print(f"  best CV F1 : {rs.best_score_:.4f}")

    y_prob = rs.best_estimator_.predict_proba(X_test_hybrid)[:, 1]
    for thr in (0.5, 0.7):
        y_pred = (y_prob >= thr).astype(int)
        b = metrics_block(
            name, y_test, y_pred, y_prob, thr,
            extra={
                "gap_addressed": "P4 — SMOTE-inside-CV via imblearn Pipeline",
                "features_used": features,
                "oversampling": "SMOTE (inside CV folds, imblearn Pipeline)",
                "best_params": {k: v for k, v in rs.best_params_.items()},
                "best_cv_f1": float(rs.best_score_),
                "notes": extra_note,
            },
        )
        append_verified(b)
        print(f"  thr={thr}: F1={b['F1']:.4f} Prec={b['Precision']:.4f} Rec={b['Recall']:.4f}")


def run_pipeline_cv_ae():
    print("\n" + "=" * 70)
    print("[P4] AE + XGBoost with SMOTE inside CV (imblearn Pipeline)")
    print("=" * 70)
    seed_everything()
    t0 = time.time()

    _, _, X_train, y_train, X_test, y_test = load_engineered()
    ae, scaler = load_saved_ae()
    train_rec = saved_recon_error(ae, scaler, X_train)
    test_rec = saved_recon_error(ae, scaler, X_test)

    X_train_hybrid = np.column_stack([X_train, train_rec])
    X_test_hybrid = np.column_stack([X_test, test_rec])

    _run_pipeline_cv(
        "AE_XGBoost_PipelineCV",
        X_train_hybrid, y_train, X_test_hybrid, y_test,
        features=FEATURE_COLS + ["recon_error"],
        extra_note=(
            "Uses saved AE + scaler from models/saved/ (ae_model.pt, ae_scaler.joblib). "
            "Only the XGBoost tuning methodology changes: SMOTE runs inside each CV fold via "
            "imblearn.pipeline.Pipeline, removing the fold-leakage risk of fitting SMOTE on the "
            "full training matrix before RandomizedSearchCV. Test set untouched by SMOTE."
        ),
    )
    print(f"  runtime: {time.time() - t0:.1f}s")


def run_pipeline_cv_bds(stop_after_construction: bool = False):
    print("\n" + "=" * 70)
    print("[P4] AE + BDS + XGBoost with SMOTE inside CV (imblearn Pipeline)")
    print("=" * 70)
    seed_everything()
    t0 = time.time()

    train_df, test_df, X_train, y_train, X_test, y_test = load_engineered()

    ae, scaler = load_saved_ae()
    train_rec = saved_recon_error(ae, scaler, X_train)
    test_rec = saved_recon_error(ae, scaler, X_test)

    print("  building BDS features (canonical pattern — matches rerun_verification.py)...")
    bds_tr, bds_te = build_bds_features(train_df, test_df)
    print(f"  BDS features built — "
          f"train shape=({len(bds_tr[0])}, 4), test shape=({len(bds_te[0])}, 4)")

    X_train_final = np.column_stack([X_train, train_rec] + list(bds_tr))
    X_test_final = np.column_stack([X_test, test_rec] + list(bds_te))
    print(f"  X_train_final shape = {X_train_final.shape}")
    print(f"  X_test_final  shape = {X_test_final.shape}")

    if stop_after_construction:
        print("\n  stop_after_construction=True — halting before RandomizedSearchCV.")
        print(f"  BDS feature construction completed in {time.time() - t0:.1f}s.")
        return

    _run_pipeline_cv(
        "AE_BDS_XGBoost_PipelineCV",
        X_train_final, y_train, X_test_final, y_test,
        features=FEATURE_COLS + ["recon_error", "bds_amount", "bds_time", "bds_freq", "bds_category"],
        extra_note=(
            "19-feature model. Uses saved AE + scaler + GA params from models/saved/. "
            "BDS features rebuilt on-the-fly using the canonical precompute/compute_bds pattern "
            "from rerun_verification.py (train-only profile statistics; the bds_profiles.joblib "
            "artefact is NOT loaded here to avoid schema mismatch — the pattern deliberately "
            "matches the verification script). "
            "Only the XGBoost tuning methodology changes: SMOTE inside each CV fold via "
            "imblearn.pipeline.Pipeline. Test set untouched by SMOTE."
        ),
    )
    print(f"  runtime: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--true-no-velocity", action="store_true",
                        help="P2: true from-scratch no-velocity AE ablation")
    parser.add_argument("--pipeline-cv-ae", action="store_true",
                        help="P4: AE+XGB with SMOTE inside CV")
    parser.add_argument("--pipeline-cv-bds", action="store_true",
                        help="P4: AE+BDS+XGB with SMOTE inside CV")
    parser.add_argument("--all", action="store_true", help="run all three")
    parser.add_argument("--smoke-test-bds", action="store_true",
                        help="Run --pipeline-cv-bds up to BDS construction then halt "
                             "(no tuning) — useful for verifying the BDS path works.")
    args = parser.parse_args()

    if args.all:
        args.true_no_velocity = args.pipeline_cv_ae = args.pipeline_cv_bds = True

    if not (args.true_no_velocity or args.pipeline_cv_ae or args.pipeline_cv_bds
            or args.smoke_test_bds):
        parser.print_help()
        return

    seed_everything()
    wrote_any = False
    if args.true_no_velocity:
        run_true_no_velocity_ablation()
        wrote_any = True
    if args.pipeline_cv_ae:
        run_pipeline_cv_ae()
        wrote_any = True
    if args.smoke_test_bds:
        run_pipeline_cv_bds(stop_after_construction=True)
    elif args.pipeline_cv_bds:
        run_pipeline_cv_bds(stop_after_construction=False)
        wrote_any = True

    if wrote_any:
        print("\nDone. New blocks appended to verified_metrics.json under `audit_reruns`.")
    else:
        print("\nDone. Smoke test only — verified_metrics.json unchanged.")


if __name__ == "__main__":
    main()

"""Parts 2-4: Robustness Testing, Custom Code Validation, Visualizations"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, math, time, sys, json
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score as sk_f1, precision_score as sk_prec, recall_score as sk_rec, roc_auc_score as sk_auc, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib, warnings
warnings.filterwarnings('ignore')
import functools
print = functools.partial(print, flush=True)

sys.path.insert(0, '.')
from custom_metrics import *
from custom_sampler import stratified_kfold

# ============================================================
# LOAD DATA
# ============================================================
print("="*70)
print("LOADING DATA")
print("="*70)
train_df = pd.read_csv('fraudTrain_engineered.csv')
test_df = pd.read_csv('fraudTest_engineered.csv')
drop_cols = ['is_fraud','unix_time']
feature_cols = [c for c in train_df.columns if c not in drop_cols]
X_train = train_df[feature_cols].values; y_train = train_df['is_fraud'].values
X_test = test_df[feature_cols].values; y_test = test_df['is_fraud'].values
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Load saved model
model = joblib.load('xgboost_best.joblib')
print(f"Loaded xgboost_best.joblib")

# ============================================================
# CUSTOM 1 VALIDATION — Compare custom metrics vs sklearn
# ============================================================
print("\n" + "="*70)
print("CUSTOM 1 — METRICS VALIDATION (custom vs sklearn)")
print("="*70)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

cm_custom = calculate_confusion_matrix(y_test.tolist(), y_pred.tolist())
p_custom = calculate_precision(y_test.tolist(), y_pred.tolist())
r_custom = calculate_recall(y_test.tolist(), y_pred.tolist())
f1_custom = calculate_f1(y_test.tolist(), y_pred.tolist())
spec_custom = calculate_specificity(y_test.tolist(), y_pred.tolist())
mcc_custom = calculate_mcc(y_test.tolist(), y_pred.tolist())

p_sk = sk_prec(y_test, y_pred)
r_sk = sk_rec(y_test, y_pred)
f1_sk = sk_f1(y_test, y_pred)
from sklearn.metrics import matthews_corrcoef
mcc_sk = matthews_corrcoef(y_test, y_pred)

print(f"{'Metric':<20s} {'Custom':>10s} {'sklearn':>10s} {'Match':>6s}")
print("-"*50)
print(f"{'TP':<20s} {cm_custom['TP']:>10d} {confusion_matrix(y_test,y_pred)[1,1]:>10d} {'OK' if cm_custom['TP']==confusion_matrix(y_test,y_pred)[1,1] else 'FAIL':>6s}")
print(f"{'FP':<20s} {cm_custom['FP']:>10d} {confusion_matrix(y_test,y_pred)[0,1]:>10d} {'OK' if cm_custom['FP']==confusion_matrix(y_test,y_pred)[0,1] else 'FAIL':>6s}")
print(f"{'FN':<20s} {cm_custom['FN']:>10d} {confusion_matrix(y_test,y_pred)[1,0]:>10d} {'OK' if cm_custom['FN']==confusion_matrix(y_test,y_pred)[1,0] else 'FAIL':>6s}")
print(f"{'TN':<20s} {cm_custom['TN']:>10d} {confusion_matrix(y_test,y_pred)[0,0]:>10d} {'OK' if cm_custom['TN']==confusion_matrix(y_test,y_pred)[0,0] else 'FAIL':>6s}")
print(f"{'Precision':<20s} {p_custom:>10.4f} {p_sk:>10.4f} {'OK' if abs(p_custom-p_sk)<0.0001 else 'FAIL':>6s}")
print(f"{'Recall':<20s} {r_custom:>10.4f} {r_sk:>10.4f} {'OK' if abs(r_custom-r_sk)<0.0001 else 'FAIL':>6s}")
print(f"{'F1':<20s} {f1_custom:>10.4f} {f1_sk:>10.4f} {'OK' if abs(f1_custom-f1_sk)<0.0001 else 'FAIL':>6s}")
print(f"{'MCC':<20s} {mcc_custom:>10.4f} {mcc_sk:>10.4f} {'OK' if abs(mcc_custom-mcc_sk)<0.0001 else 'FAIL':>6s}")

# ============================================================
# CUSTOM 2 VALIDATION — Stratified KFold
# ============================================================
print("\n" + "="*70)
print("CUSTOM 2 — STRATIFIED KFOLD VALIDATION")
print("="*70)
folds = stratified_kfold(y_train.tolist(), n_splits=5, random_state=42)
for i, (tr, vl) in enumerate(folds):
    fraud_tr = sum(y_train[j] for j in tr)
    fraud_vl = sum(y_train[j] for j in vl)
    print(f"Fold {i+1}: train={len(tr):,} ({100*fraud_tr/len(tr):.3f}% fraud), val={len(vl):,} ({100*fraud_vl/len(vl):.3f}% fraud)")

# ============================================================
# ROBUST 1 — MANUAL 5-FOLD STRATIFIED CV
# ============================================================
print("\n" + "="*70)
print("ROBUST 1 — MANUAL 5-FOLD STRATIFIED CROSS-VALIDATION")
print("="*70)

smote = SMOTE(random_state=42)
best_params = model.get_params()
skip_keys = {'use_label_encoder', 'eval_metric', 'random_state', 'n_jobs'}
fold_results = []

for i, (tr_idx, vl_idx) in enumerate(folds):
    t0 = time.time()
    X_tr = X_train[tr_idx]; y_tr = y_train[tr_idx]
    X_vl = X_train[vl_idx]; y_vl = y_train[vl_idx]

    # SMOTE on train portion only
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    skip_keys = {'use_label_encoder', 'eval_metric', 'random_state', 'n_jobs'}
    xgb = XGBClassifier(**{k:v for k,v in best_params.items() if k not in skip_keys},
                        eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb.fit(X_tr_sm, y_tr_sm)
    yp = xgb.predict(X_vl)

    f1 = sk_f1(y_vl, yp)
    prec = sk_prec(y_vl, yp)
    rec = sk_rec(y_vl, yp)
    fold_results.append((f1, prec, rec))
    print(f"Fold {i+1}: F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f} ({time.time()-t0:.1f}s)")

f1s = [r[0] for r in fold_results]
precs = [r[1] for r in fold_results]
recs = [r[2] for r in fold_results]
print(f"\nMean F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
print(f"Mean Pre: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
print(f"Mean Rec: {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
print(f"Test F1:  0.8646")
print(f"Std < 0.05: {'YES — model is stable' if np.std(f1s) < 0.05 else 'NO — model is unstable'}")

# ============================================================
# ROBUST 2 — CUSTOM THRESHOLD OPTIMIZATION
# ============================================================
print("\n" + "="*70)
print("ROBUST 2 — CUSTOM THRESHOLD OPTIMIZATION")
print("="*70)

thresholds = [t/100 for t in range(5, 96)]
results_thresh = []
for thresh in thresholds:
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_test.tolist(), y_prob.tolist()):
        pred = 1 if yp >= thresh else 0
        if yt==1 and pred==1: tp += 1
        elif yt==0 and pred==1: fp += 1
        elif yt==1 and pred==0: fn += 1
        else: tn += 1
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    results_thresh.append((thresh, f1, prec, rec, tp, fp, fn, tn))

best_t = max(results_thresh, key=lambda x: x[1])
default_t = [r for r in results_thresh if r[0]==0.50][0]

print(f"Optimal threshold: {best_t[0]:.2f}")
print(f"  F1={best_t[1]:.4f}, Precision={best_t[2]:.4f}, Recall={best_t[3]:.4f}")
print(f"  TP={best_t[4]}, FP={best_t[5]}, FN={best_t[6]}, TN={best_t[7]}")
print(f"\nDefault threshold (0.50):")
print(f"  F1={default_t[1]:.4f}, Precision={default_t[2]:.4f}, Recall={default_t[3]:.4f}")
print(f"  TP={default_t[4]}, FP={default_t[5]}, FN={default_t[6]}, TN={default_t[7]}")

# Plot
fig, ax = plt.subplots(figsize=(10,6))
ts = [r[0] for r in results_thresh]
f1s_t = [r[1] for r in results_thresh]
ax.plot(ts, f1s_t, 'b-', linewidth=2)
ax.axvline(best_t[0], color='r', linestyle='--', label=f'Optimal: {best_t[0]:.2f} (F1={best_t[1]:.4f})')
ax.axvline(0.5, color='gray', linestyle=':', label=f'Default: 0.50 (F1={default_t[1]:.4f})')
ax.set_xlabel('Threshold'); ax.set_ylabel('F1 Score')
ax.set_title('Threshold Optimization — XGBoost Best Model')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('threshold_optimization.png', dpi=100); plt.close()
print("Saved: threshold_optimization.png")

# VIZ 1 — Precision/Recall/F1 vs threshold
fig, ax = plt.subplots(figsize=(10,6))
precs_t = [r[2] for r in results_thresh]
recs_t = [r[3] for r in results_thresh]
ax.plot(ts, precs_t, 'g-', label='Precision', linewidth=2)
ax.plot(ts, recs_t, 'r-', label='Recall', linewidth=2)
ax.plot(ts, f1s_t, 'b-', label='F1', linewidth=2)
# Find precision-recall intersection
for j in range(1, len(ts)):
    if (precs_t[j-1] - recs_t[j-1]) * (precs_t[j] - recs_t[j]) <= 0:
        ax.axvline(ts[j], color='purple', linestyle='--', alpha=0.5, label=f'P-R intersection: {ts[j]:.2f}')
        ax.plot(ts[j], precs_t[j], 'ko', markersize=10)
        break
ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
ax.set_title('Precision, Recall, F1 vs Decision Threshold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('precision_recall_threshold.png', dpi=100); plt.close()
print("Saved: precision_recall_threshold.png")

# ============================================================
# ROBUST 3 — McNEMAR'S TEST (from scratch)
# ============================================================
print("\n" + "="*70)
print("ROBUST 3 — McNEMAR'S TEST (from scratch)")
print("="*70)

# Need XGBoost with and without velocity features
velocity_cols = ['velocity_1h','velocity_24h','amount_velocity_1h']
vel_idx = [feature_cols.index(c) for c in velocity_cols]
keep_idx = [i for i in range(len(feature_cols)) if i not in vel_idx]

X_train_novel = X_train[:, keep_idx]; X_test_novel = X_test[:, keep_idx]
X_sm_novel, y_sm_novel = smote.fit_resample(X_train_novel, y_train)
X_sm_full, y_sm_full = smote.fit_resample(X_train, y_train)

xgb_with = XGBClassifier(**{k:v for k,v in best_params.items() if k not in skip_keys},
                          eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_with.fit(X_sm_full, y_sm_full)
pred_with = xgb_with.predict(X_test)

xgb_without = XGBClassifier(**{k:v for k,v in best_params.items() if k not in skip_keys},
                             eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_without.fit(X_sm_novel, y_sm_novel)
pred_without = xgb_without.predict(X_test_novel)

# McNemar counts
n01 = n10 = 0  # n01: without correct, with wrong. n10: with correct, without wrong
for yt, pw, pwo in zip(y_test, pred_with, pred_without):
    correct_with = (pw == yt)
    correct_without = (pwo == yt)
    if correct_without and not correct_with: n01 += 1
    if correct_with and not correct_without: n10 += 1

# McNemar chi-squared with continuity correction
chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10) if (n01 + n10) > 0 else 0

# p-value from chi2 with 1 df using erfc approximation
# chi2 with 1 df: p = erfc(sqrt(chi2/2) / sqrt(1))
p_value = math.erfc(math.sqrt(chi2 / 2))

print(f"McNemar's Test: XGBoost WITH velocity vs WITHOUT velocity")
print(f"  n01 (without correct, with wrong): {n01}")
print(f"  n10 (with correct, without wrong):  {n10}")
print(f"  Chi-squared statistic: {chi2:.4f}")
print(f"  p-value: {p_value:.6f}")
if p_value < 0.05:
    print(f"  RESULT: Velocity features provide a STATISTICALLY SIGNIFICANT improvement (p < 0.05)")
else:
    print(f"  RESULT: Improvement is NOT statistically significant (p >= 0.05)")
print(f"  F1 with velocity: {sk_f1(y_test, pred_with):.4f}")
print(f"  F1 without velocity: {sk_f1(y_test, pred_without):.4f}")

# ============================================================
# ROBUST 4 — LEARNING CURVES
# ============================================================
print("\n" + "="*70)
print("ROBUST 4 — LEARNING CURVES")
print("="*70)

fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
train_f1s = []; test_f1s = []

for frac in fractions:
    n = int(len(y_train) * frac)
    idx = np.random.RandomState(42).choice(len(y_train), n, replace=False)
    Xf = X_train[idx]; yf = y_train[idx]
    Xsm, ysm = smote.fit_resample(Xf, yf)
    xgb_lc = XGBClassifier(**{k:v for k,v in best_params.items() if k not in skip_keys},
                            eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_lc.fit(Xsm, ysm)
    tr_f1 = sk_f1(yf, xgb_lc.predict(Xf))
    te_f1 = sk_f1(y_test, xgb_lc.predict(X_test))
    train_f1s.append(tr_f1); test_f1s.append(te_f1)
    print(f"  {frac*100:5.0f}% ({n:>9,} rows): Train F1={tr_f1:.4f}, Test F1={te_f1:.4f}")

fig, ax = plt.subplots(figsize=(10,6))
sizes = [int(len(y_train)*f) for f in fractions]
ax.plot(sizes, train_f1s, 'b-o', label='Train F1', linewidth=2)
ax.plot(sizes, test_f1s, 'r-s', label='Test F1', linewidth=2)
ax.set_xlabel('Training Set Size'); ax.set_ylabel('F1 Score (Fraud)')
ax.set_title('Learning Curves — XGBoost Best Model')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('learning_curves.png', dpi=100); plt.close()
print("Saved: learning_curves.png")

# ============================================================
# ROBUST 5 — FEATURE CORRELATION
# ============================================================
print("\n" + "="*70)
print("ROBUST 5 — FEATURE CORRELATION")
print("="*70)

corr_df = pd.DataFrame(X_train, columns=feature_cols)
corr_matrix = corr_df.corr()

# Find top 10 most correlated pairs (excluding self-correlation)
pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        pairs.append((feature_cols[i], feature_cols[j], abs(corr_matrix.iloc[i,j])))
pairs.sort(key=lambda x: x[2], reverse=True)

print("Top 10 most correlated feature pairs:")
for f1, f2, c in pairs[:10]:
    flag = " *** HIGH" if c > 0.95 else " ** MODERATE" if c > 0.7 else ""
    print(f"  {f1:>30s} vs {f2:<30s}: {c:.4f}{flag}")

# Check velocity vs BDS-related features
print("\nVelocity feature correlations:")
vel_feats = ['velocity_1h','velocity_24h','amount_velocity_1h']
for vf in vel_feats:
    vi = feature_cols.index(vf)
    for j, fc in enumerate(feature_cols):
        if fc != vf and abs(corr_matrix.iloc[vi,j]) > 0.3:
            print(f"  {vf} vs {fc}: {corr_matrix.iloc[vi,j]:.4f}")

# Heatmap
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, xticklabels=feature_cols, yticklabels=feature_cols)
ax.set_title('Feature Correlation Matrix (14 Features)')
plt.tight_layout(); plt.savefig('feature_correlation.png', dpi=100); plt.close()
print("Saved: feature_correlation.png")

# ============================================================
# VIZ 2 — ERROR ANALYSIS DEEP DIVE
# ============================================================
print("\n" + "="*70)
print("VIZ 2 — ERROR ANALYSIS DEEP DIVE")
print("="*70)

test_analysis = pd.DataFrame(X_test, columns=feature_cols)
test_analysis['is_fraud'] = y_test
test_analysis['predicted'] = pred_with
missed = test_analysis[(test_analysis['is_fraud']==1) & (test_analysis['predicted']==0)]
caught = test_analysis[(test_analysis['is_fraud']==1) & (test_analysis['predicted']==1)]
print(f"Caught: {len(caught)}, Missed: {len(missed)}")

fig, axes = plt.subplots(2, 3, figsize=(18,10))
compare_cols = ['amt','hour','day_of_week_encoded','velocity_1h','velocity_24h','amount_velocity_1h']
for ax, col in zip(axes.flat, compare_cols):
    ax.hist(caught[col], bins=30, alpha=0.6, label='Caught', density=True)
    ax.hist(missed[col], bins=30, alpha=0.6, label='Missed', density=True)
    ax.set_title(col); ax.legend()
plt.suptitle('Error Analysis: Caught vs Missed Fraud Distributions', fontsize=14)
plt.tight_layout(); plt.savefig('error_analysis_distributions.png', dpi=100); plt.close()
print("Saved: error_analysis_distributions.png")

# Identify patterns
print("Missed fraud patterns vs caught:")
for col in compare_cols:
    print(f"  {col:>25s}: missed mean={missed[col].mean():.2f}, caught mean={caught[col].mean():.2f}")

# ============================================================
# VIZ 3 — SHAP DEPENDENCE PLOTS
# ============================================================
print("\n" + "="*70)
print("VIZ 3 — SHAP DEPENDENCE PLOTS")
print("="*70)

import shap
np.random.seed(42)
X_shap = X_test[np.random.choice(len(X_test), 5000, replace=False)]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# Top 5 features
mean_shap = np.abs(shap_values).mean(axis=0)
top5_idx = np.argsort(mean_shap)[::-1][:5]
top5_names = [feature_cols[i] for i in top5_idx]
print(f"Top 5 features for dependence plots: {top5_names}")

fig, axes = plt.subplots(1, 5, figsize=(25,5))
for ax, idx, name in zip(axes, top5_idx, top5_names):
    ax.scatter(X_shap[:, idx], shap_values[:, idx], alpha=0.3, s=5, c=shap_values[:, idx], cmap='RdBu_r')
    ax.set_xlabel(name); ax.set_ylabel('SHAP value')
    ax.set_title(f'{name}'); ax.axhline(0, color='gray', linewidth=0.5)
plt.suptitle('SHAP Dependence Plots — Top 5 Features', fontsize=14)
plt.tight_layout(); plt.savefig('shap_dependence_top5.png', dpi=100); plt.close()
print("Saved: shap_dependence_top5.png")

# ============================================================
# VIZ 4 — RADAR CHART
# ============================================================
print("\n" + "="*70)
print("VIZ 4 — MODEL COMPARISON RADAR CHART")
print("="*70)

from sklearn.metrics import precision_recall_curve, auc as sk_auc_fn

models_data = {
    'XGB CW': {'f1': 0.5215, 'prec': 0.36, 'rec': 0.95, 'roc': 0.9978},
    'XGB SMOTE+T': {'f1': 0.8646, 'prec': 0.93, 'rec': 0.81, 'roc': 0.9972},
    'AE+XGB': {'f1': 0.8705, 'prec': 0.94, 'rec': 0.81, 'roc': 0.9972},
    'AE+BDS+XGB': {'f1': 0.868, 'prec': 0.93, 'rec': 0.81, 'roc': 0.9976},
    'LSTM+RF': {'f1': 0.47, 'prec': 0.32, 'rec': 0.89, 'roc': 0.994},
}
categories = ['F1', 'Precision', 'Recall', 'ROC-AUC']
N = len(categories)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
for name, d in models_data.items():
    values = [d['f1'], d['prec'], d['rec'], d['roc']]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=name)
    ax.fill(angles, values, alpha=0.05)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1.05)
ax.set_title('Model Comparison Radar Chart', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout(); plt.savefig('model_comparison_radar.png', dpi=100, bbox_inches='tight'); plt.close()
print("Saved: model_comparison_radar.png")

# ============================================================
# VIZ 5 — ALL CONFUSION MATRICES
# ============================================================
print("\n" + "="*70)
print("VIZ 5 — ALL CONFUSION MATRICES")
print("="*70)

# We have predictions for with/without velocity from ROBUST 3
cm_data = {
    'XGB CW': np.array([[549997,3577],[111,2034]]),  # from earlier run
    'XGB SMOTE': np.array([[553296,278],[405,1740]]),
    'XGB SMOTE+T': confusion_matrix(y_test, pred_with),
    'AE+XGB Best': np.array([[553460,114],[404,1741]]),
    'AE+XGB NoVel': np.array([[553406,168],[414,1731]]),
    'AE+BDS+XGB': np.array([[553446,128],[402,1743]]),
    'LSTM+RF': np.array([[545104,8470],[201,1944]]),
}

fig, axes = plt.subplots(2, 4, figsize=(24,10))
axes_flat = axes.flat
for (name, cm), ax in zip(cm_data.items(), axes_flat):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal','Fraud'], yticklabels=['Normal','Fraud'])
    ax.set_title(name); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
# Hide extra subplot
axes_flat[7].set_visible(False)
plt.suptitle('All Model Confusion Matrices', fontsize=16)
plt.tight_layout(); plt.savefig('all_confusion_matrices.png', dpi=100); plt.close()
print("Saved: all_confusion_matrices.png")

# ============================================================
# CUSTOM 3 — EARLY STOPPING DEMO
# ============================================================
print("\n" + "="*70)
print("CUSTOM 3 — EARLY STOPPING DEMO")
print("="*70)
from custom_training import EarlyStopping
# Simulate the autoencoder loss history from our actual run
losses = [0.4874, 0.4762, 0.4585, 0.4437, 0.4363, 0.4305, 0.4275, 0.4289, 0.4275, 0.4240, 0.4167, 0.4191, 0.4194, 0.4176]
es = EarlyStopping(patience=3)
print("Simulating autoencoder training with custom EarlyStopping:")
for i, loss in enumerate(losses):
    es.status(i+1, loss)
    if es.step(loss, i+1):
        print(f"  --> STOPPED at epoch {i+1}. Best epoch: {es.best_epoch}, best loss: {es.best_loss:.6f}")
        break

print("\n" + "="*70)
print("ALL PARTS COMPLETE")
print("="*70)

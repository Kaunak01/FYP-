"""
Rerun verification — produces three citeable JSONs:
  verified_metrics.json     — test-set metrics for all saved models (threshold 0.5 AND 0.7)
  ablation_results.json     — feature-group ablations on the full hybrid
  shap_top_features.json    — mean(|SHAP|) ranking for the full hybrid
"""
import json, os, time, warnings, numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score)
warnings.filterwarnings('ignore')

ROOT   = r'C:\Users\User\OneDrive\Desktop\FYP-Fraud-Detection'
MODELS = os.path.join(ROOT, 'models', 'saved')
TRAIN  = os.path.join(ROOT, 'fraudTrain_engineered.csv')
TEST   = os.path.join(ROOT, 'fraudTest_engineered.csv')

FEATURE_COLS = ['amt','city_pop','hour','month','distance_cardholder_merchant',
                'age','is_weekend','is_night','velocity_1h','velocity_24h',
                'amount_velocity_1h','category_encoded','gender_encoded','day_of_week_encoded']
TARGET = 'is_fraud'

print("Loading test & train data...")
df_test  = pd.read_csv(TEST)
df_train = pd.read_csv(TRAIN)
X_test = df_test[FEATURE_COLS].values;  y_test = df_test[TARGET].values
X_train = df_train[FEATURE_COLS].values; y_train = df_train[TARGET].values
print(f"  test  {len(df_test):,} rows | {int(y_test.sum()):,} fraud ({y_test.mean()*100:.2f}%)")
print(f"  train {len(df_train):,} rows | {int(y_train.sum()):,} fraud ({y_train.mean()*100:.2f}%)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Autoencoder (same arch as save_all_models.py)
# ═══════════════════════════════════════════════════════════════════════════════
class Autoencoder(nn.Module):
    def __init__(self, n=14):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n,10), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(10,5), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(5,10), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(10,n))
    def forward(self,x): return self.decoder(self.encoder(x))

def recon_error(X):
    sc = joblib.load(os.path.join(MODELS,'ae_scaler.joblib'))
    Xs = sc.transform(X).astype(np.float32)
    ae = Autoencoder(n=X.shape[1])
    ae.load_state_dict(torch.load(os.path.join(MODELS,'ae_model.pt'), map_location='cpu'))
    ae.eval()
    with torch.no_grad():
        r = ae(torch.tensor(Xs)).numpy()
    return np.mean((Xs - r)**2, axis=1)

print("Computing AE reconstruction errors (train+test)...")
t0=time.time()
re_train = recon_error(X_train)
re_test  = recon_error(X_test)
print(f"  done in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# BDS scores — using the GA-tuned compute_bds from save_all_models.py
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing BDS features (GA-tuned, full logic)...")
t0=time.time()
with open(os.path.join(MODELS,'ga_best_params.json')) as f:
    ga = json.load(f)
best_params = [ga['params'][k] for k in
               ['amount_threshold','amount_cap','time_threshold','time_cap',
                'freq_threshold','freq_cap','cat_threshold','cat_cap',
                'min_history','smoothing']]

# Build per-card stats from TRAIN ONLY (same as save_all_models.py)
df_train['cc_num'] = df_train['cc_num'] if 'cc_num' in df_train.columns else 0
df_test ['cc_num'] = df_test ['cc_num'] if 'cc_num' in df_test .columns else 0
# fraudTrain/Test engineered may not have cc_num — fallback to raw csv join
if 'cc_num' not in df_train.columns or df_train['cc_num'].nunique()<=1:
    raw_tr = pd.read_csv(os.path.join(ROOT,'fraudTrain.csv'), usecols=['cc_num'])
    raw_te = pd.read_csv(os.path.join(ROOT,'fraudTest.csv'),  usecols=['cc_num'])
    df_train['cc_num'] = raw_tr['cc_num'].values
    df_test ['cc_num'] = raw_te['cc_num'].values

card_amt = df_train.groupby('cc_num')['amt'].agg(['mean','std','count']).rename(
    columns={'mean':'amt_mean','std':'amt_std','count':'amt_count'})
card_amt['amt_std'] = card_amt['amt_std'].fillna(0)

n_categories = max(df_train['category_encoded'].max(), df_test['category_encoded'].max()) + 1
card_hour_prob = (df_train.groupby(['cc_num','hour']).size()
                  / df_train.groupby('cc_num').size()).unstack(fill_value=0)
card_cat_prob  = (df_train.groupby(['cc_num','category_encoded']).size()
                  / df_train.groupby('cc_num').size()).unstack(fill_value=0)
card_vel = df_train.groupby('cc_num')['velocity_1h'].agg(['mean']).rename(columns={'mean':'vel_mean'})

gstats = {
    'amt_mean': df_train['amt'].mean(),
    'amt_std' : df_train['amt'].std(),
    'hour_prob': df_train['hour'].value_counts(normalize=True).to_dict(),
    'cat_prob' : df_train['category_encoded'].value_counts(normalize=True).to_dict(),
    'vel_mean' : df_train['velocity_1h'].mean(),
}

def precompute(cc, amt, hr, cat, vel):
    d = pd.DataFrame({'cc_num':cc,'amt':amt,'hour':hr.astype(int),'cat':cat.astype(int),'vel':vel})
    d = d.merge(card_amt, left_on='cc_num', right_index=True, how='left')
    d['amt_mean'] = d['amt_mean'].fillna(gstats['amt_mean'])
    d['amt_std']  = d['amt_std'].fillna(gstats['amt_std'])
    d['amt_count']= d['amt_count'].fillna(0); d['unseen'] = d['amt_count']==0
    ss = d['amt_std'].where(d['amt_std']>0, gstats['amt_std'])
    d['card_amt_z']   = (d['amt']-d['amt_mean']).abs()/ss
    d['global_amt_z'] = (d['amt']-gstats['amt_mean']).abs()/gstats['amt_std']
    hs = card_hour_prob.stack(); hs.index.names=['cc_num','hour']
    hl = hs.reset_index(); hl.columns=['cc_num','hour','card_hour_p']
    d = d.merge(hl, on=['cc_num','hour'], how='left'); d['card_hour_p']=d['card_hour_p'].fillna(0.0)
    d['global_hour_p'] = d['hour'].map(gstats['hour_prob']).fillna(1/24)
    cs = card_cat_prob.stack(); cs.index.names=['cc_num','cat']
    cl = cs.reset_index(); cl.columns=['cc_num','cat','card_cat_p']
    d = d.merge(cl, on=['cc_num','cat'], how='left'); d['card_cat_p']=d['card_cat_p'].fillna(0.0)
    d['global_cat_p'] = d['cat'].map(gstats['cat_prob']).fillna(1/n_categories)
    d = d.merge(card_vel, left_on='cc_num', right_index=True, how='left')
    d['vel_mean'] = d['vel_mean'].fillna(gstats['vel_mean'])
    sv = d['vel_mean'].where(d['vel_mean']>0, gstats['vel_mean'])
    d['freq_ratio'] = d['vel']/sv
    return {k:d[k].values for k in ['card_amt_z','global_amt_z','amt_count',
            'card_hour_p','global_hour_p','card_cat_p','global_cat_p','freq_ratio','unseen']}

def compute_bds(raw, p):
    at,ac,tt,tc,ft,fc,ct,cc_,mh,sm = p; mh=int(round(mh))
    ug = (raw['amt_count']<mh)|raw['unseen']
    az = np.where(ug, raw['global_amt_z'], raw['card_amt_z'])
    a = np.clip(np.maximum(az-at,0),0,ac)
    hp = np.where(ug, raw['global_hour_p'], raw['card_hour_p'])
    tr = -np.log(hp+sm); t = np.clip(np.maximum(tr-tt,0),0,tc)
    fr = np.maximum(raw['freq_ratio']-1.0,0); fsc = np.clip(np.maximum(fr-ft,0),0,fc)
    cp = np.where(ug, raw['global_cat_p'], raw['card_cat_p'])
    cr = -np.log(cp+sm); c = np.clip(np.maximum(cr-ct,0),0,cc_)
    return a,t,fsc,c

raw_tr = precompute(df_train['cc_num'].values, df_train['amt'].values,
                    df_train['hour'].values, df_train['category_encoded'].values,
                    df_train['velocity_1h'].values)
raw_te = precompute(df_test['cc_num'].values, df_test['amt'].values,
                    df_test['hour'].values, df_test['category_encoded'].values,
                    df_test['velocity_1h'].values)
bds_tr = compute_bds(raw_tr, best_params)
bds_te = compute_bds(raw_te, best_params)
print(f"  done in {time.time()-t0:.1f}s")

# Feature matrices
X_test_15 = np.column_stack([X_test, re_test])
X_test_19 = np.column_stack([X_test, re_test] + list(bds_te))
X_train_15 = np.column_stack([X_train, re_train])
X_train_19 = np.column_stack([X_train, re_train] + list(bds_tr))

# ═══════════════════════════════════════════════════════════════════════════════
# 1. VERIFIED METRICS for all 4 saved models @ threshold 0.5 AND 0.7
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Computing verified test metrics for all 4 saved models...")
def metrics(name, y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        'model': name, 'threshold': thr,
        'roc_auc':   float(roc_auc_score(y, p)),
        'pr_auc':    float(average_precision_score(y, p)),
        'f1':        float(f1_score(y, pred, zero_division=0)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall':    float(recall_score(y, pred, zero_division=0)),
        'tp': int(((pred==1)&(y==1)).sum()),
        'fp': int(((pred==1)&(y==0)).sum()),
        'fn': int(((pred==0)&(y==1)).sum()),
        'tn': int(((pred==0)&(y==0)).sum()),
    }

verified = {'test_set_size': int(len(y_test)), 'test_fraud_count': int(y_test.sum()),
            'test_fraud_rate': float(y_test.mean()), 'models': []}

models_probs = {}
for name, file, Xeval in [
    ('XGBoost Baseline (CW)',       'xgboost_baseline_cw.joblib',       X_test),
    ('XGBoost SMOTE+tuned',         'xgboost_smote_tuned.joblib',       X_test),
    ('AE + XGBoost SMOTE+tuned',    'ae_xgboost_smote_tuned.joblib',    X_test_15),
    ('AE + BDS + XGBoost (full)',   'ae_bds_xgboost_smote_tuned.joblib',X_test_19),
]:
    m = joblib.load(os.path.join(MODELS, file))
    p = m.predict_proba(Xeval)[:,1]
    models_probs[name] = p
    for thr in (0.5, 0.7):
        row = metrics(name, y_test, p, thr)
        verified['models'].append(row)
        print(f"  {name:<32} thr={thr}  F1={row['f1']:.4f}  P={row['precision']:.4f}  R={row['recall']:.4f}  ROC={row['roc_auc']:.4f}  PR={row['pr_auc']:.4f}")

with open(os.path.join(ROOT,'verified_metrics.json'),'w') as f:
    json.dump(verified, f, indent=2)
print("  >> saved verified_metrics.json")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ABLATION — retrain full hybrid WITHOUT each feature group at a time
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Running ablation study (retraining full hybrid without each group)...")
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2

# Full model best params come from the saved tuned hybrid
full_model = joblib.load(os.path.join(MODELS,'ae_bds_xgboost_smote_tuned.joblib'))
best_hp = full_model.get_params()
# keep only relevant hyperparameters for retraining
keep_keys = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree',
             'min_child_weight','gamma','reg_alpha','reg_lambda','scale_pos_weight']
hp = {k: best_hp.get(k) for k in keep_keys if best_hp.get(k) is not None}

def ablate(label, X_tr, X_te):
    sm = SMOTE(random_state=42)
    Xs, ys = sm.fit_resample(X_tr, y_train)
    m = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0, **hp)
    m.fit(Xs, ys)
    p = m.predict_proba(X_te)[:,1]
    pred = (p >= 0.5).astype(int)
    return {
        'variant': label,
        'f1_at_0.5': float(f1_score(y_test, pred, zero_division=0)),
        'precision_at_0.5': float(precision_score(y_test, pred, zero_division=0)),
        'recall_at_0.5': float(recall_score(y_test, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, p)),
        'pr_auc':  float(average_precision_score(y_test, p)),
    }, pred

ablations = {}
# Full hybrid (for reference — uses the saved tuned model directly)
p_full  = models_probs['AE + BDS + XGBoost (full)']
pred_full = (p_full >= 0.5).astype(int)
ablations['full_hybrid_saved'] = {
    'variant': 'full hybrid (saved model, not retrained)',
    'f1_at_0.5': float(f1_score(y_test, pred_full, zero_division=0)),
    'precision_at_0.5': float(precision_score(y_test, pred_full, zero_division=0)),
    'recall_at_0.5': float(recall_score(y_test, pred_full, zero_division=0)),
    'roc_auc': float(roc_auc_score(y_test, p_full)),
    'pr_auc':  float(average_precision_score(y_test, p_full)),
}

# WITHOUT velocity features (drop velocity_1h, velocity_24h, amount_velocity_1h — cols 8,9,10)
vel_idx = [FEATURE_COLS.index(c) for c in ['velocity_1h','velocity_24h','amount_velocity_1h']]
keep = [i for i in range(len(FEATURE_COLS)) if i not in vel_idx]
X_tr_noV = np.column_stack([X_train[:,keep], re_train] + list(bds_tr))
X_te_noV = np.column_stack([X_test [:,keep], re_test ] + list(bds_te))
res, pred_noV = ablate('without velocity features', X_tr_noV, X_te_noV)
ablations['without_velocity'] = res

# WITHOUT AE reconstruction error
X_tr_noAE = np.column_stack([X_train] + list(bds_tr))
X_te_noAE = np.column_stack([X_test ] + list(bds_te))
res, pred_noAE = ablate('without AE reconstruction error', X_tr_noAE, X_te_noAE)
ablations['without_ae'] = res

# WITHOUT BDS
X_tr_noB = np.column_stack([X_train, re_train])
X_te_noB = np.column_stack([X_test , re_test ])
res, pred_noB = ablate('without BDS scores', X_tr_noB, X_te_noB)
ablations['without_bds'] = res

# McNemar's test vs full hybrid for each ablation
def mcnemar(y, p_a, p_b):
    a_right = (p_a==y); b_right = (p_b==y)
    b01 = int(((~a_right)&( b_right)).sum())  # a wrong, b right
    b10 = int((( a_right)&(~b_right)).sum())  # a right, b wrong
    if b01+b10 == 0: return {'b01':b01,'b10':b10,'stat':0.0,'p_value':1.0}
    stat = (abs(b01-b10)-1)**2 / (b01+b10)
    p = 1 - chi2.cdf(stat, df=1)
    return {'b01':b01,'b10':b10,'stat':float(stat),'p_value':float(p)}

ablations['mcnemar_full_vs_without_velocity'] = mcnemar(y_test, pred_full, pred_noV)
ablations['mcnemar_full_vs_without_ae']       = mcnemar(y_test, pred_full, pred_noAE)
ablations['mcnemar_full_vs_without_bds']      = mcnemar(y_test, pred_full, pred_noB)

# deltas
base = ablations['full_hybrid_saved']['f1_at_0.5']
for k in ['without_velocity','without_ae','without_bds']:
    ablations[k]['delta_f1_vs_full'] = ablations[k]['f1_at_0.5'] - base

with open(os.path.join(ROOT,'ablation_results.json'),'w') as f:
    json.dump(ablations, f, indent=2)
print("  → saved ablation_results.json")
for k, v in ablations.items():
    if isinstance(v, dict) and 'f1_at_0.5' in v:
        print(f"    {k:<30} F1={v['f1_at_0.5']:.4f}  delta={v.get('delta_f1_vs_full','-'):+}" if isinstance(v.get('delta_f1_vs_full'),(int,float)) else f"    {k:<30} F1={v['f1_at_0.5']:.4f}")
    elif isinstance(v, dict) and 'p_value' in v:
        print(f"    {k:<30} McNemar p={v['p_value']:.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SHAP on full hybrid
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Computing SHAP mean-|value| on full hybrid (sample n=5000)...")
import shap
t0=time.time()
# Stratified sample: 2500 fraud + 2500 normal for stable SHAP
rng = np.random.RandomState(42)
fraud_idx  = np.where(y_test==1)[0]
normal_idx = np.where(y_test==0)[0]
sample_idx = np.concatenate([
    rng.choice(fraud_idx,  min(2500, len(fraud_idx)), replace=False),
    rng.choice(normal_idx, 2500, replace=False)
])
X_shap = X_test_19[sample_idx]
explainer = shap.TreeExplainer(full_model)
shap_values = explainer.shap_values(X_shap)
mean_abs = np.abs(shap_values).mean(axis=0)
feature_names_19 = FEATURE_COLS + ['recon_error','bds_amount','bds_time','bds_freq','bds_category']
ranking = sorted(zip(feature_names_19, mean_abs.tolist()), key=lambda x:-x[1])
shap_out = {
    'model': 'ae_bds_xgboost_smote_tuned.joblib (19 features)',
    'sample_size': int(len(X_shap)),
    'sample_composition': 'stratified: ~2500 fraud + 2500 normal',
    'top_features': [{'rank': i+1, 'feature': n, 'mean_abs_shap': float(v)}
                     for i,(n,v) in enumerate(ranking)],
}
with open(os.path.join(ROOT,'shap_top_features.json'),'w') as f:
    json.dump(shap_out, f, indent=2)
print(f"  done in {time.time()-t0:.1f}s")
print("  >> saved shap_top_features.json")
for i,(n,v) in enumerate(ranking[:10]):
    print(f"    {i+1:2d}. {n:<30} {v:.4f}")

print("\nALL THREE JSONs SAVED in", ROOT)

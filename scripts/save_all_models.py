"""Step 1: Save all models, extract category mapping, compute training stats."""
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, json, time, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import warnings; warnings.filterwarnings('ignore')
import functools
print = functools.partial(print, flush=True)

T0 = time.time()

# ============================================================
# LOAD DATA
# ============================================================
print("="*70)
print("LOADING DATA")
print("="*70)
train_raw = pd.read_csv('fraudTrain.csv', usecols=['cc_num', 'category', 'trans_date_trans_time'])
train_eng = pd.read_csv('fraudTrain_engineered.csv')
test_eng = pd.read_csv('fraudTest_engineered.csv')

drop_cols = ['is_fraud', 'unix_time']
feature_cols = [c for c in train_eng.columns if c not in drop_cols]
X_train = train_eng[feature_cols].values
y_train = train_eng['is_fraud'].values
X_test = test_eng[feature_cols].values
y_test = test_eng['is_fraud'].values
print(f"Train: {X_train.shape}, Test: {X_test.shape}, Features: {feature_cols}")

# ============================================================
# 1. CATEGORY MAPPING
# ============================================================
print("\n" + "="*70)
print("1. EXTRACTING CATEGORY MAPPING")
print("="*70)

cat_raw = train_raw['category'].values
cat_enc = train_eng['category_encoded'].values

# Build mapping: pair raw category names with their encoded values
cat_pairs = pd.DataFrame({'name': cat_raw, 'code': cat_enc}).drop_duplicates().sort_values('code')
name_to_code = dict(zip(cat_pairs['name'], cat_pairs['code']))
code_to_name = dict(zip(cat_pairs['code'], cat_pairs['name']))

# Compute frequency
cat_freq = pd.Series(cat_raw).value_counts()
cat_mapping = {
    'name_to_code': {str(k): int(v) for k, v in name_to_code.items()},
    'code_to_name': {str(k): str(v) for k, v in code_to_name.items()},
    'frequency': {str(k): int(v) for k, v in cat_freq.items()},
    'total_transactions': int(len(cat_raw)),
    'top5_common': list(cat_freq.head(5).index),
    'categories_list': list(cat_pairs['name'].values)
}

with open('category_mapping.json', 'w') as f:
    json.dump(cat_mapping, f, indent=2)
print(f"Saved: category_mapping.json")
print(f"Categories ({len(name_to_code)}):")
for name, code in sorted(name_to_code.items(), key=lambda x: x[1]):
    freq = cat_freq.get(name, 0)
    pct = 100 * freq / len(cat_raw)
    common = "COMMON" if name in cat_mapping['top5_common'] else ""
    print(f"  {code:2d} = {name:<30s} ({freq:>8,} txns, {pct:5.1f}%) {common}")

# ============================================================
# 2. TRAINING STATS
# ============================================================
print("\n" + "="*70)
print("2. COMPUTING TRAINING STATS")
print("="*70)

def compute_stats(arr):
    return {
        'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
        'min': float(np.min(arr)), 'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'p25': float(np.percentile(arr, 25)), 'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)), 'p99': float(np.percentile(arr, 99))
    }

normal_mask = y_train == 0
fraud_mask = y_train == 1

training_stats = {'feature_cols': feature_cols, 'stats': {}}
for i, col in enumerate(feature_cols):
    training_stats['stats'][col] = {
        'all': compute_stats(X_train[:, i]),
        'normal': compute_stats(X_train[normal_mask, i]),
        'fraud': compute_stats(X_train[fraud_mask, i])
    }
    print(f"  {col:<35s} all_mean={X_train[:,i].mean():.4f}, fraud_mean={X_train[fraud_mask,i].mean():.4f}")

with open('training_stats.json', 'w') as f:
    json.dump(training_stats, f, indent=2)
print(f"Saved: training_stats.json")

# ============================================================
# 3. XGBOOST BASELINES (14 features)
# ============================================================
print("\n" + "="*70)
print("3. TRAINING XGBOOST BASELINES")
print("="*70)

n_normal = (y_train == 0).sum()
n_fraud = (y_train == 1).sum()
spw = n_normal / n_fraud

# 3a. Class weights baseline
print("Training xgboost_baseline_cw...")
xgb_cw = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=spw, eval_metric='logloss', random_state=42, n_jobs=-1
)
xgb_cw.fit(X_train, y_train)
f1_cw = f1_score(y_test, xgb_cw.predict(X_test))
joblib.dump(xgb_cw, 'xgboost_baseline_cw.joblib')
print(f"  Saved: xgboost_baseline_cw.joblib (F1={f1_cw:.4f})")

# 3b. SMOTE + Tuned
print("Training xgboost_smote_tuned (RandomizedSearchCV)...")
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)

param_dist = {
    'n_estimators': [200, 300, 400], 'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9], 'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}
rs = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    param_dist, n_iter=30, scoring='f1', cv=3, random_state=42, verbose=0, n_jobs=1
)
rs.fit(X_sm, y_sm)
xgb_tuned = rs.best_estimator_
f1_tuned = f1_score(y_test, xgb_tuned.predict(X_test))
joblib.dump(xgb_tuned, 'xgboost_smote_tuned.joblib')
print(f"  Saved: xgboost_smote_tuned.joblib (F1={f1_tuned:.4f})")
print(f"  Best params: {rs.best_params_}")

# ============================================================
# 4. AUTOENCODER
# ============================================================
print("\n" + "="*70)
print("4. TRAINING AUTOENCODER")
print("="*70)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
X_normal = X_train_sc[y_train == 0]

class Autoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d,10), nn.ReLU(), nn.Dropout(0.2), nn.Linear(10,5), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(5,10), nn.ReLU(), nn.Dropout(0.2), nn.Linear(10,d))
    def forward(self, x): return self.decoder(self.encoder(x))

device = torch.device('cpu')
ae = Autoencoder(X_train_sc.shape[1]).to(device)
nt = torch.FloatTensor(X_normal).to(device)
vs = int(0.1 * len(nt))
tds, vds = random_split(TensorDataset(nt, nt), [len(nt)-vs, vs])
tl = DataLoader(tds, batch_size=512, shuffle=True)
vl = DataLoader(vds, batch_size=512)
opt = torch.optim.Adam(ae.parameters())
crit = nn.MSELoss()
bv, pt, bs = float('inf'), 0, None

for ep in range(30):
    ae.train()
    for xb, _ in tl:
        loss = crit(ae(xb), xb); opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        v = sum(crit(ae(xb), xb).item()*len(xb) for xb, _ in vl) / vs
    print(f"  Epoch {ep+1}/30 — val_loss: {v:.6f}")
    if v < bv:
        bv, pt, bs = v, 0, {k: v.clone() for k, v in ae.state_dict().items()}
    else:
        pt += 1
        if pt >= 3:
            print(f"  Early stopping at epoch {ep+1}")
            break

ae.load_state_dict(bs); ae.eval()
torch.save(ae.state_dict(), 'ae_model.pt')
joblib.dump(scaler, 'ae_scaler.joblib')
print(f"Saved: ae_model.pt, ae_scaler.joblib")

# Compute reconstruction errors for all data
with torch.no_grad():
    train_re = np.mean((X_train_sc - ae(torch.FloatTensor(X_train_sc)).numpy())**2, axis=1)
    test_re = np.mean((X_test_sc - ae(torch.FloatTensor(X_test_sc)).numpy())**2, axis=1)
ratio = train_re[y_train==1].mean() / train_re[y_train==0].mean()
print(f"Recon error ratio: {ratio:.2f}x")

# ============================================================
# 5. AE + XGBOOST (15 features)
# ============================================================
print("\n" + "="*70)
print("5. TRAINING AE + XGBOOST")
print("="*70)

X_train_h = np.column_stack([X_train, train_re])
X_test_h = np.column_stack([X_test, test_re])
X_sm_h, y_sm_h = smote.fit_resample(X_train_h, y_train)

print("Training ae_xgboost_smote_tuned (RandomizedSearchCV)...")
rs2 = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    param_dist, n_iter=30, scoring='f1', cv=3, random_state=42, verbose=0, n_jobs=1
)
rs2.fit(X_sm_h, y_sm_h)
ae_xgb = rs2.best_estimator_
f1_ae = f1_score(y_test, ae_xgb.predict(X_test_h))
joblib.dump(ae_xgb, 'ae_xgboost_smote_tuned.joblib')
print(f"  Saved: ae_xgboost_smote_tuned.joblib (F1={f1_ae:.4f})")
print(f"  Best params: {rs2.best_params_}")

# ============================================================
# 6. BDS PROFILES
# ============================================================
print("\n" + "="*70)
print("6. BUILDING BDS PROFILES")
print("="*70)

train_cc = pd.read_csv('fraudTrain.csv', usecols=['cc_num'])['cc_num'].values
test_cc = pd.read_csv('fraudTest.csv', usecols=['cc_num'])['cc_num'].values

profile_df = pd.DataFrame({
    'cc_num': train_cc, 'amt': train_eng['amt'].values,
    'hour': train_eng['hour'].values.astype(int),
    'category': train_eng['category_encoded'].values.astype(int),
    'velocity_1h': train_eng['velocity_1h'].values
})

card_amt = profile_df.groupby('cc_num')['amt'].agg(['mean','std','count'])
card_amt.columns = ['amt_mean','amt_std','amt_count']
card_amt['amt_std'] = card_amt['amt_std'].fillna(0)

card_hour_counts = profile_df.groupby(['cc_num','hour']).size().unstack(fill_value=0)
for h in range(24):
    if h not in card_hour_counts.columns: card_hour_counts[h] = 0
card_hour_counts = card_hour_counts[sorted(card_hour_counts.columns)]
card_hour_prob = card_hour_counts.div(card_hour_counts.sum(axis=1), axis=0)

n_categories = profile_df['category'].nunique()
card_cat_counts = profile_df.groupby(['cc_num','category']).size().unstack(fill_value=0)
card_cat_prob = card_cat_counts.div(card_cat_counts.sum(axis=1), axis=0)

card_vel = profile_df.groupby('cc_num')['velocity_1h'].agg(['mean','std'])
card_vel.columns = ['vel_mean','vel_std']
card_vel['vel_std'] = card_vel['vel_std'].fillna(0)

global_stats_bds = {
    'amt_mean': float(profile_df['amt'].mean()), 'amt_std': float(profile_df['amt'].std()),
    'hour_prob': profile_df.groupby('hour').size().div(len(profile_df)).to_dict(),
    'cat_prob': profile_df.groupby('category').size().div(len(profile_df)).to_dict(),
    'vel_mean': float(profile_df['velocity_1h'].mean()),
    'n_categories': int(n_categories)
}

profiles = {
    'card_amt': card_amt, 'card_hour_prob': card_hour_prob,
    'card_cat_prob': card_cat_prob, 'card_vel': card_vel,
    'global_stats': global_stats_bds
}
joblib.dump(profiles, 'bds_profiles.joblib')
print(f"Saved: bds_profiles.joblib ({len(card_amt)} card profiles)")

# ============================================================
# 7. GA OPTIMIZATION
# ============================================================
print("\n" + "="*70)
print("7. RUNNING GA")
print("="*70)

PARAM_BOUNDS = [
    (0.0,3.0),(3.0,15.0),(0.0,3.0),(3.0,15.0),
    (0.0,3.0),(3.0,15.0),(0.0,3.0),(3.0,15.0),
    (2.0,20.0),(0.001,0.5)
]

def precompute_bds_raw(cc_nums, amts, hours, cats, velocities):
    df = pd.DataFrame({'cc_num':cc_nums,'amt':amts,'hour':hours.astype(int),'cat':cats.astype(int),'vel':velocities})
    df = df.merge(card_amt[['amt_mean','amt_std','amt_count']], left_on='cc_num', right_index=True, how='left')
    df['amt_mean']=df['amt_mean'].fillna(global_stats_bds['amt_mean'])
    df['amt_std']=df['amt_std'].fillna(global_stats_bds['amt_std'])
    df['amt_count']=df['amt_count'].fillna(0); df['unseen']=df['amt_count']==0
    ss=df['amt_std'].where(df['amt_std']>0, global_stats_bds['amt_std'])
    df['card_amt_z']=(df['amt']-df['amt_mean']).abs()/ss
    df['global_amt_z']=(df['amt']-global_stats_bds['amt_mean']).abs()/global_stats_bds['amt_std']
    hs=card_hour_prob.stack(); hs.index.names=['cc_num','hour']; hl=hs.reset_index(); hl.columns=['cc_num','hour','card_hour_p']
    df=df.merge(hl,on=['cc_num','hour'],how='left'); df['card_hour_p']=df['card_hour_p'].fillna(0.0)
    df['global_hour_p']=df['hour'].map(global_stats_bds['hour_prob']).fillna(1/24)
    cs=card_cat_prob.stack(); cs.index.names=['cc_num','cat']; cl=cs.reset_index(); cl.columns=['cc_num','cat','card_cat_p']
    df=df.merge(cl,on=['cc_num','cat'],how='left'); df['card_cat_p']=df['card_cat_p'].fillna(0.0)
    df['global_cat_p']=df['cat'].map(global_stats_bds['cat_prob']).fillna(1/n_categories)
    df=df.merge(card_vel[['vel_mean']],left_on='cc_num',right_index=True,how='left')
    df['vel_mean']=df['vel_mean'].fillna(global_stats_bds['vel_mean'])
    sv=df['vel_mean'].where(df['vel_mean']>0, global_stats_bds['vel_mean']); df['freq_ratio']=df['vel']/sv
    return {k:df[k].values for k in ['card_amt_z','global_amt_z','card_counts','card_hour_p','global_hour_p','card_cat_p','global_cat_p','freq_ratio','unseen']}

# Fix: rename amt_count to card_counts in precompute
def precompute_fix(cc, amts, hrs, cats, vels):
    df = pd.DataFrame({'cc_num':cc,'amt':amts,'hour':hrs.astype(int),'cat':cats.astype(int),'vel':vels})
    df = df.merge(card_amt[['amt_mean','amt_std','amt_count']], left_on='cc_num', right_index=True, how='left')
    df['amt_mean']=df['amt_mean'].fillna(global_stats_bds['amt_mean'])
    df['amt_std']=df['amt_std'].fillna(global_stats_bds['amt_std'])
    df['amt_count']=df['amt_count'].fillna(0); df['unseen']=df['amt_count']==0
    ss=df['amt_std'].where(df['amt_std']>0, global_stats_bds['amt_std'])
    df['card_amt_z']=(df['amt']-df['amt_mean']).abs()/ss
    df['global_amt_z']=(df['amt']-global_stats_bds['amt_mean']).abs()/global_stats_bds['amt_std']
    hs=card_hour_prob.stack(); hs.index.names=['cc_num','hour']; hl=hs.reset_index(); hl.columns=['cc_num','hour','card_hour_p']
    df=df.merge(hl,on=['cc_num','hour'],how='left'); df['card_hour_p']=df['card_hour_p'].fillna(0.0)
    df['global_hour_p']=df['hour'].map(global_stats_bds['hour_prob']).fillna(1/24)
    cs=card_cat_prob.stack(); cs.index.names=['cc_num','cat']; cl=cs.reset_index(); cl.columns=['cc_num','cat','card_cat_p']
    df=df.merge(cl,on=['cc_num','cat'],how='left'); df['card_cat_p']=df['card_cat_p'].fillna(0.0)
    df['global_cat_p']=df['cat'].map(global_stats_bds['cat_prob']).fillna(1/n_categories)
    df=df.merge(card_vel[['vel_mean']],left_on='cc_num',right_index=True,how='left')
    df['vel_mean']=df['vel_mean'].fillna(global_stats_bds['vel_mean'])
    sv=df['vel_mean'].where(df['vel_mean']>0, global_stats_bds['vel_mean']); df['freq_ratio']=df['vel']/sv
    r = {k:df[k].values for k in ['card_amt_z','global_amt_z','card_hour_p','global_hour_p','card_cat_p','global_cat_p','freq_ratio','unseen']}
    r['card_counts'] = df['amt_count'].values
    return r

def compute_bds(raw, params):
    at,ac,tt,tc,ft,fc,ct,cc_,mh,sm = params; mh=int(round(mh))
    ug = (raw['card_counts']<mh)|raw['unseen']
    az = np.where(ug, raw['global_amt_z'], raw['card_amt_z'])
    a_sc = np.clip(np.maximum(az-at,0),0,ac)
    hp = np.where(ug, raw['global_hour_p'], raw['card_hour_p'])
    t_r = -np.log(hp+sm); t_sc = np.clip(np.maximum(t_r-tt,0),0,tc)
    fr = np.maximum(raw['freq_ratio']-1.0,0); f_sc = np.clip(np.maximum(fr-ft,0),0,fc)
    cp = np.where(ug, raw['global_cat_p'], raw['card_cat_p'])
    c_r = -np.log(cp+sm); c_sc = np.clip(np.maximum(c_r-ct,0),0,cc_)
    return a_sc, t_sc, f_sc, c_sc

print("Precomputing BDS deviations...")
train_bds_raw = precompute_fix(train_cc, train_eng['amt'].values, train_eng['hour'].values,
                                train_eng['category_encoded'].values, train_eng['velocity_1h'].values)
test_bds_raw = precompute_fix(test_cc, test_eng['amt'].values, test_eng['hour'].values,
                               test_eng['category_encoded'].values, test_eng['velocity_1h'].values)

# GA subsample
_, sub_idx = train_test_split(np.arange(len(y_train)), test_size=0.1, stratify=y_train, random_state=42)
X_sub = X_train[sub_idx]; y_sub = y_train[sub_idx]; re_sub = train_re[sub_idx]
sub_raw = {k:v[sub_idx] for k,v in train_bds_raw.items()}
ga_tr, ga_vl = train_test_split(np.arange(len(y_sub)), test_size=0.3, stratify=y_sub, random_state=42)
X_gt,X_gv = X_sub[ga_tr],X_sub[ga_vl]; y_gt,y_gv = y_sub[ga_tr],y_sub[ga_vl]
re_gt,re_gv = re_sub[ga_tr],re_sub[ga_vl]
gtr={k:v[ga_tr] for k,v in sub_raw.items()}; gvr={k:v[ga_vl] for k,v in sub_raw.items()}
ga_spw = (y_gt==0).sum()/max((y_gt==1).sum(),1)

def ga_fitness(ind):
    bt=compute_bds(gtr,ind); bv=compute_bds(gvr,ind)
    Xt=np.column_stack([X_gt,re_gt]+list(bt)); Xv=np.column_stack([X_gv,re_gv]+list(bv))
    m=XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.1,scale_pos_weight=ga_spw,
                    eval_metric='logloss',random_state=42,n_jobs=-1,verbosity=0)
    m.fit(Xt,y_gt); return f1_score(y_gv,m.predict(Xv))

print("Running GA (pop=30, gen=20)...")
rng = np.random.RandomState(42)
population = [[rng.uniform(lo,hi) for lo,hi in PARAM_BOUNDS] for _ in range(30)]
best_ever_f1 = -1; best_ever_ind = None

for gen in range(20):
    t0 = time.time()
    fits = [ga_fitness(ind) for ind in population]
    bi = np.argmax(fits)
    if fits[bi] > best_ever_f1:
        best_ever_f1 = fits[bi]; best_ever_ind = population[bi].copy()
    print(f"  Gen {gen+1:2d}/20 | Best: {max(fits):.4f} | Avg: {np.mean(fits):.4f} | Overall: {best_ever_f1:.4f} | {time.time()-t0:.1f}s")
    si = np.argsort(fits)[::-1]
    new_pop = [population[si[i]].copy() for i in range(2)]
    while len(new_pop) < 30:
        c1=rng.choice(30,3,replace=False); p1=population[c1[np.argmax([fits[i] for i in c1])]].copy()
        c2=rng.choice(30,3,replace=False); p2=population[c2[np.argmax([fits[i] for i in c2])]].copy()
        alpha=rng.uniform(0.2,0.8)
        child=[alpha*a+(1-alpha)*b for a,b in zip(p1,p2)] if rng.random()<0.8 else p1
        for i in range(10):
            if rng.random()<0.2:
                lo,hi=PARAM_BOUNDS[i]; child[i]=np.clip(child[i]+rng.normal(0,(hi-lo)*0.1),lo,hi)
        new_pop.append(child)
    population = new_pop

param_names = ['amount_threshold','amount_cap','time_threshold','time_cap',
               'freq_threshold','freq_cap','cat_threshold','cat_cap','min_history','smoothing']
ga_config = {
    'params': {n: float(v) for n,v in zip(param_names, best_ever_ind)},
    'param_bounds': [[float(lo),float(hi)] for lo,hi in PARAM_BOUNDS],
    'ga_best_f1': float(best_ever_f1)
}
with open('ga_best_params.json', 'w') as f:
    json.dump(ga_config, f, indent=2)
print(f"Saved: ga_best_params.json (best F1={best_ever_f1:.4f})")
print("Optimal params:")
for n, v in zip(param_names, best_ever_ind):
    print(f"  {n:>20s} = {v:.4f}")

# ============================================================
# 8. AE + BDS + XGBOOST (19 features)
# ============================================================
print("\n" + "="*70)
print("8. TRAINING AE + BDS + XGBOOST")
print("="*70)

train_bds_scores = compute_bds(train_bds_raw, best_ever_ind)
test_bds_scores = compute_bds(test_bds_raw, best_ever_ind)

X_train_19 = np.column_stack([X_train, train_re] + list(train_bds_scores))
X_test_19 = np.column_stack([X_test, test_re] + list(test_bds_scores))
print(f"Feature shape: train={X_train_19.shape}, test={X_test_19.shape}")

X_sm_19, y_sm_19 = smote.fit_resample(X_train_19, y_train)
print("Training ae_bds_xgboost_smote_tuned (RandomizedSearchCV)...")
rs3 = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    param_dist, n_iter=30, scoring='f1', cv=3, random_state=42, verbose=0, n_jobs=1
)
rs3.fit(X_sm_19, y_sm_19)
bds_xgb = rs3.best_estimator_
f1_bds = f1_score(y_test, bds_xgb.predict(X_test_19))
joblib.dump(bds_xgb, 'ae_bds_xgboost_smote_tuned.joblib')
print(f"  Saved: ae_bds_xgboost_smote_tuned.joblib (F1={f1_bds:.4f})")
print(f"  Best params: {rs3.best_params_}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("ALL MODELS SAVED — SUMMARY")
print("="*70)
print(f"  1. category_mapping.json         — {len(name_to_code)} categories")
print(f"  2. training_stats.json           — {len(feature_cols)} features x 3 groups x 9 stats")
print(f"  3. xgboost_baseline_cw.joblib    — F1={f1_cw:.4f} (14 features)")
print(f"  4. xgboost_smote_tuned.joblib    — F1={f1_tuned:.4f} (14 features)")
print(f"  5. ae_model.pt                   — Autoencoder weights")
print(f"  6. ae_scaler.joblib              — StandardScaler")
print(f"  7. ae_xgboost_smote_tuned.joblib — F1={f1_ae:.4f} (15 features)")
print(f"  8. bds_profiles.joblib           — {len(card_amt)} card profiles")
print(f"  9. ga_best_params.json           — GA best F1={best_ever_f1:.4f}")
print(f" 10. ae_bds_xgboost_smote_tuned.joblib — F1={f1_bds:.4f} (19 features)")
print(f"\nTotal time: {(time.time()-T0)/60:.1f} minutes")

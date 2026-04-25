# BDS + GA Pipeline — Standalone Script
import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, precision_recall_curve, auc, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import time, warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch version: {torch.__version__}, Device: {device}")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_eng = pd.read_csv('fraudTrain_engineered.csv')
test_eng = pd.read_csv('fraudTest_engineered.csv')
train_cc = pd.read_csv('fraudTrain.csv', usecols=['cc_num'])['cc_num'].values
test_cc = pd.read_csv('fraudTest.csv', usecols=['cc_num'])['cc_num'].values

drop_cols = ['is_fraud', 'unix_time']
feature_cols = [c for c in train_eng.columns if c not in drop_cols]
X_train = train_eng[feature_cols].values
y_train = train_eng['is_fraud'].values
X_test = test_eng[feature_cols].values
y_test = test_eng['is_fraud'].values

print(f"Train: {X_train.shape[0]:,} rows, Test: {X_test.shape[0]:,} rows")
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Unique cards: train={len(np.unique(train_cc)):,}, test={len(np.unique(test_cc)):,}")

# ============================================================
# 2. AUTOENCODER (for reconstruction error)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_normal = X_train_scaled[y_train == 0]

class Autoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d,10), nn.ReLU(), nn.Dropout(0.2), nn.Linear(10,5), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(5,10), nn.ReLU(), nn.Dropout(0.2), nn.Linear(10,d))
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae = Autoencoder(X_train_scaled.shape[1]).to(device)
normal_t = torch.FloatTensor(X_train_normal).to(device)
val_sz = int(0.1 * len(normal_t))
train_ds, val_ds = random_split(TensorDataset(normal_t, normal_t), [len(normal_t)-val_sz, val_sz])
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)
opt = torch.optim.Adam(ae.parameters())
crit = nn.MSELoss()
best_vl, pat, best_st = float('inf'), 0, None

for epoch in range(30):
    ae.train()
    for xb, _ in train_loader:
        loss = crit(ae(xb), xb); opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        vl = sum(crit(ae(xb), xb).item()*len(xb) for xb,_ in val_loader) / val_sz
    print(f"Epoch {epoch+1}/30 - val_loss: {vl:.6f}")
    if vl < best_vl:
        best_vl, pat, best_st = vl, 0, {k:v.clone() for k,v in ae.state_dict().items()}
    else:
        pat += 1
        if pat >= 3: print(f"Early stopping at epoch {epoch+1}"); break

ae.load_state_dict(best_st); ae.eval()
with torch.no_grad():
    train_recon = ae(torch.FloatTensor(X_train_scaled).to(device)).cpu().numpy()
    test_recon = ae(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
train_recon_error = np.mean((X_train_scaled - train_recon)**2, axis=1)
test_recon_error = np.mean((X_test_scaled - test_recon)**2, axis=1)
print(f"Recon error - Normal: {train_recon_error[y_train==0].mean():.4f}, Fraud: {train_recon_error[y_train==1].mean():.4f}, Ratio: {train_recon_error[y_train==1].mean()/train_recon_error[y_train==0].mean():.2f}x")

# ============================================================
# 3. BUILD CARDHOLDER PROFILES
# ============================================================
print("\nBuilding cardholder profiles...")
profile_df = pd.DataFrame({
    'cc_num': train_cc,
    'amt': train_eng['amt'].values,
    'hour': train_eng['hour'].values.astype(int),
    'category': train_eng['category_encoded'].values.astype(int),
    'velocity_1h': train_eng['velocity_1h'].values
})

card_amt = profile_df.groupby('cc_num')['amt'].agg(['mean','std','count'])
card_amt.columns = ['amt_mean','amt_std','amt_count']
card_amt['amt_std'] = card_amt['amt_std'].fillna(0)

card_hour_counts = profile_df.groupby(['cc_num','hour']).size().unstack(fill_value=0)
for h in range(24):
    if h not in card_hour_counts.columns:
        card_hour_counts[h] = 0
card_hour_counts = card_hour_counts[sorted(card_hour_counts.columns)]
card_hour_prob = card_hour_counts.div(card_hour_counts.sum(axis=1), axis=0)

n_categories = profile_df['category'].nunique()
card_cat_counts = profile_df.groupby(['cc_num','category']).size().unstack(fill_value=0)
card_cat_prob = card_cat_counts.div(card_cat_counts.sum(axis=1), axis=0)

card_vel = profile_df.groupby('cc_num')['velocity_1h'].agg(['mean','std'])
card_vel.columns = ['vel_mean','vel_std']
card_vel['vel_std'] = card_vel['vel_std'].fillna(0)

global_amt_mean = profile_df['amt'].mean()
global_amt_std = profile_df['amt'].std()
global_hour_prob = profile_df.groupby('hour').size() / len(profile_df)
global_cat_prob = profile_df.groupby('category').size() / len(profile_df)
global_vel_mean = profile_df['velocity_1h'].mean()

print(f"Profiles for {len(card_amt):,} cards | Global avg amount: ${global_amt_mean:.2f}")

# ============================================================
# 4. PRECOMPUTE RAW DEVIATIONS
# ============================================================
def precompute_raw_deviations(cc_nums, amts, hours, cats, velocities):
    df = pd.DataFrame({'cc_num':cc_nums, 'amt':amts, 'hour':hours.astype(int), 'cat':cats.astype(int), 'vel':velocities})

    # Amount z-scores
    df = df.merge(card_amt[['amt_mean','amt_std','amt_count']], left_on='cc_num', right_index=True, how='left')
    df['amt_mean'] = df['amt_mean'].fillna(global_amt_mean)
    df['amt_std'] = df['amt_std'].fillna(global_amt_std)
    df['amt_count'] = df['amt_count'].fillna(0)
    df['unseen'] = df['amt_count'] == 0
    safe_std = df['amt_std'].where(df['amt_std']>0, global_amt_std)
    df['card_amt_z'] = (df['amt']-df['amt_mean']).abs() / safe_std
    df['global_amt_z'] = (df['amt']-global_amt_mean).abs() / global_amt_std

    # Hour probabilities
    hour_stack = card_hour_prob.stack()
    hour_stack.index.names = ['cc_num','hour']
    hour_lookup = hour_stack.reset_index(); hour_lookup.columns = ['cc_num','hour','card_hour_p']
    df = df.merge(hour_lookup, on=['cc_num','hour'], how='left')
    df['card_hour_p'] = df['card_hour_p'].fillna(0.0)
    df['global_hour_p'] = df['hour'].map(global_hour_prob.to_dict()).fillna(1/24)

    # Category probabilities
    cat_stack = card_cat_prob.stack()
    cat_stack.index.names = ['cc_num','cat']
    cat_lookup = cat_stack.reset_index(); cat_lookup.columns = ['cc_num','cat','card_cat_p']
    df = df.merge(cat_lookup, on=['cc_num','cat'], how='left')
    df['card_cat_p'] = df['card_cat_p'].fillna(0.0)
    df['global_cat_p'] = df['cat'].map(global_cat_prob.to_dict()).fillna(1/n_categories)

    # Frequency ratio
    df = df.merge(card_vel[['vel_mean']], left_on='cc_num', right_index=True, how='left')
    df['vel_mean'] = df['vel_mean'].fillna(global_vel_mean)
    safe_vel = df['vel_mean'].where(df['vel_mean']>0, global_vel_mean)
    df['freq_ratio'] = df['vel'] / safe_vel

    return {k: df[k].values for k in ['card_amt_z','global_amt_z','card_hour_p','global_hour_p','card_cat_p','global_cat_p','freq_ratio','unseen','amt_count']}

# Rename amt_count to card_counts in the result
def precompute_with_rename(cc_nums, amts, hours, cats, velocities):
    r = precompute_raw_deviations(cc_nums, amts, hours, cats, velocities)
    r['card_counts'] = r.pop('amt_count')
    return r

print("Precomputing train deviations...")
t0 = time.time()
train_raw_dev = precompute_with_rename(train_cc, train_eng['amt'].values, train_eng['hour'].values, train_eng['category_encoded'].values, train_eng['velocity_1h'].values)
print(f"  Done in {time.time()-t0:.1f}s")

print("Precomputing test deviations...")
t0 = time.time()
test_raw_dev = precompute_with_rename(test_cc, test_eng['amt'].values, test_eng['hour'].values, test_eng['category_encoded'].values, test_eng['velocity_1h'].values)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Unseen cards in test: {test_raw_dev['unseen'].sum():,} transactions")

# ============================================================
# 5. BDS TRANSFORMATION FUNCTION
# ============================================================
PARAM_BOUNDS = [
    (0.0, 3.0),   (3.0, 15.0),  # amount: threshold, cap
    (0.0, 3.0),   (3.0, 15.0),  # time: threshold, cap
    (0.0, 3.0),   (3.0, 15.0),  # freq: threshold, cap
    (0.0, 3.0),   (3.0, 15.0),  # cat: threshold, cap
    (2.0, 20.0),                 # min_history
    (0.001, 0.5),                # smoothing
]
N_PARAMS = len(PARAM_BOUNDS)

def compute_bds_scores(raw_dev, params):
    amt_thresh,amt_cap,time_thresh,time_cap,freq_thresh,freq_cap,cat_thresh,cat_cap,min_hist,smoothing = params
    min_hist = int(round(min_hist))
    use_global = (raw_dev['card_counts'] < min_hist) | raw_dev['unseen']

    amt_z = np.where(use_global, raw_dev['global_amt_z'], raw_dev['card_amt_z'])
    amount_score = np.clip(np.maximum(amt_z - amt_thresh, 0), 0, amt_cap)

    hour_p = np.where(use_global, raw_dev['global_hour_p'], raw_dev['card_hour_p'])
    time_raw = -np.log(hour_p + smoothing)
    time_score = np.clip(np.maximum(time_raw - time_thresh, 0), 0, time_cap)

    freq_raw = np.maximum(raw_dev['freq_ratio'] - 1.0, 0)
    freq_score = np.clip(np.maximum(freq_raw - freq_thresh, 0), 0, freq_cap)

    cat_p = np.where(use_global, raw_dev['global_cat_p'], raw_dev['card_cat_p'])
    cat_raw = -np.log(cat_p + smoothing)
    cat_score = np.clip(np.maximum(cat_raw - cat_thresh, 0), 0, cat_cap)

    return amount_score, time_score, freq_score, cat_score

default_params = [(lo+hi)/2 for lo,hi in PARAM_BOUNDS]
a,t,f,c = compute_bds_scores(train_raw_dev, default_params)
print(f"Default BDS non-zero %: amount={100*(a>0).mean():.1f}%, time={100*(t>0).mean():.1f}%, freq={100*(f>0).mean():.1f}%, cat={100*(c>0).mean():.1f}%")

# ============================================================
# 6. GENETIC ALGORITHM — FROM SCRATCH
# ============================================================
# GA subsample: 10% stratified
np.random.seed(42)
_, sub_idx = train_test_split(np.arange(len(y_train)), test_size=0.1, stratify=y_train, random_state=42)
X_sub = X_train[sub_idx]; y_sub = y_train[sub_idx]; recon_sub = train_recon_error[sub_idx]
sub_raw_dev = {k: v[sub_idx] for k,v in train_raw_dev.items()}

ga_train_idx, ga_val_idx = train_test_split(np.arange(len(y_sub)), test_size=0.3, stratify=y_sub, random_state=42)
X_ga_train, X_ga_val = X_sub[ga_train_idx], X_sub[ga_val_idx]
y_ga_train, y_ga_val = y_sub[ga_train_idx], y_sub[ga_val_idx]
recon_ga_train, recon_ga_val = recon_sub[ga_train_idx], recon_sub[ga_val_idx]
ga_train_raw = {k: v[ga_train_idx] for k,v in sub_raw_dev.items()}
ga_val_raw = {k: v[ga_val_idx] for k,v in sub_raw_dev.items()}
ga_scale_pos = (y_ga_train==0).sum() / max((y_ga_train==1).sum(), 1)

print(f"\nGA subsample: {len(y_sub):,} ({y_sub.sum()} frauds) | GA train: {len(y_ga_train):,} | GA val: {len(y_ga_val):,}")

def create_individual():
    return [np.random.uniform(lo,hi) for lo,hi in PARAM_BOUNDS]

def fitness(individual):
    bds_tr = compute_bds_scores(ga_train_raw, individual)
    bds_vl = compute_bds_scores(ga_val_raw, individual)
    X_tr = np.column_stack([X_ga_train, recon_ga_train] + list(bds_tr))
    X_vl = np.column_stack([X_ga_val, recon_ga_val] + list(bds_vl))
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=ga_scale_pos, eval_metric='logloss',
                          random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_ga_train)
    return f1_score(y_ga_val, model.predict(X_vl))

def tournament_select(population, fitnesses, k=3):
    cands = np.random.choice(len(population), size=k, replace=False)
    return population[cands[np.argmax([fitnesses[i] for i in cands])]].copy()

def arithmetic_crossover(p1, p2):
    alpha = np.random.uniform(0.2, 0.8)
    return [alpha*a + (1-alpha)*b for a,b in zip(p1,p2)]

def gaussian_mutation(ind, rate=0.2):
    m = ind.copy()
    for i in range(N_PARAMS):
        if np.random.random() < rate:
            lo,hi = PARAM_BOUNDS[i]
            m[i] = np.clip(m[i] + np.random.normal(0, (hi-lo)*0.1), lo, hi)
    return m

def run_ga(pop_size=30, generations=20, elitism=2):
    print(f"GA: pop={pop_size}, gen={generations}, elitism={elitism}")
    print("="*60)
    population = [create_individual() for _ in range(pop_size)]
    best_hist, avg_hist = [], []
    best_ever, best_ind = -1, None

    for gen in range(generations):
        t0 = time.time()
        fits = [fitness(ind) for ind in population]
        gb, ga = max(fits), np.mean(fits)
        best_hist.append(gb); avg_hist.append(ga)
        bi = np.argmax(fits)
        if gb > best_ever:
            best_ever, best_ind = gb, population[bi].copy()
        print(f"Gen {gen+1:2d}/{generations} | Best: {gb:.4f} | Avg: {ga:.4f} | Overall: {best_ever:.4f} | {time.time()-t0:.1f}s")

        si = np.argsort(fits)[::-1]
        new_pop = [population[si[i]].copy() for i in range(elitism)]
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fits)
            p2 = tournament_select(population, fits)
            child = arithmetic_crossover(p1,p2) if np.random.random()<0.8 else p1.copy()
            new_pop.append(gaussian_mutation(child))
        population = new_pop

    print("="*60)
    print(f"GA Complete! Best F1: {best_ever:.4f}")
    return best_ind, best_ever, best_hist, avg_hist

print("\nStarting GA...\n")
ga_start = time.time()
best_params, best_f1, best_history, avg_history = run_ga(30, 20, 2)
ga_time = time.time() - ga_start
print(f"\nGA time: {ga_time/60:.1f} minutes")

param_names = ['amount_threshold','amount_cap','time_threshold','time_cap',
               'freq_threshold','freq_cap','cat_threshold','cat_cap','min_history','smoothing']
print("\nOptimal BDS Parameters:")
for n,v,(lo,hi) in zip(param_names, best_params, PARAM_BOUNDS):
    print(f"  {n:>20s} = {v:.4f}  [{lo}, {hi}]")

# GA convergence plot
plt.figure(figsize=(10,5))
plt.plot(range(1,len(best_history)+1), best_history, 'b-o', label='Best F1', linewidth=2)
plt.plot(range(1,len(avg_history)+1), avg_history, 'r--s', label='Avg F1', linewidth=1.5)
plt.fill_between(range(1,len(best_history)+1), avg_history, best_history, alpha=0.15)
plt.xlabel('Generation'); plt.ylabel('F1 Score')
plt.title('GA Convergence — BDS Parameter Optimisation')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('bds_fig_1.png', dpi=100, bbox_inches='tight'); plt.close()

# ============================================================
# 7. APPLY BEST PARAMS TO FULL DATA
# ============================================================
print("\nApplying GA-optimised BDS to full dataset...")
train_bds = compute_bds_scores(train_raw_dev, best_params)
test_bds = compute_bds_scores(test_raw_dev, best_params)
bds_names = ['bds_amount','bds_time','bds_freq','bds_category']

X_train_final = np.column_stack([X_train, train_recon_error] + list(train_bds))
X_test_final = np.column_stack([X_test, test_recon_error] + list(test_bds))
final_feature_cols = feature_cols + ['recon_error'] + bds_names

print(f"Final features ({len(final_feature_cols)}): {final_feature_cols}")
print("\nBDS Score Stats (test):")
for name, scores in zip(bds_names, test_bds):
    fm, nm = scores[y_test==1].mean(), scores[y_test==0].mean()
    print(f"  {name:>15s} - Normal: {nm:.4f}, Fraud: {fm:.4f}, Ratio: {fm/max(nm,0.0001):.2f}x")

# ============================================================
# 8. TRAIN FINAL XGBOOST MODELS
# ============================================================
n_normal = (y_train==0).sum(); n_fraud = (y_train==1).sum()
spw = n_normal / n_fraud

# Config 1: Class weights
print("\n--- Config 1: Class Weights ---")
xgb_cw = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=spw, eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_cw.fit(X_train_final, y_train)
y_pred_cw = xgb_cw.predict(X_test_final)
y_prob_cw = xgb_cw.predict_proba(X_test_final)[:,1]
f1_cw = f1_score(y_test, y_pred_cw)
print(classification_report(y_test, y_pred_cw, target_names=['Normal','Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_cw):.4f} | F1: {f1_cw:.4f}")

# Config 2: SMOTE
print("\n--- Config 2: SMOTE ---")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train)
print(f"After SMOTE: {(y_train_smote==0).sum():,} normal, {(y_train_smote==1).sum():,} fraud")
xgb_sm = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_sm.fit(X_train_smote, y_train_smote)
y_pred_sm = xgb_sm.predict(X_test_final)
y_prob_sm = xgb_sm.predict_proba(X_test_final)[:,1]
f1_sm = f1_score(y_test, y_pred_sm)
print(classification_report(y_test, y_pred_sm, target_names=['Normal','Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_sm):.4f} | F1: {f1_sm:.4f}")

# Config 3: SMOTE + Tuned
print("\n--- Config 3: SMOTE + RandomizedSearchCV ---")
param_dist = {
    'n_estimators': [200,300,400], 'max_depth': [4,6,8,10],
    'learning_rate': [0.01,0.05,0.1], 'subsample': [0.7,0.8,0.9],
    'colsample_bytree': [0.7,0.8,0.9], 'min_child_weight': [1,3,5], 'gamma': [0,0.1,0.2]
}
rs = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    param_dist, n_iter=30, scoring='f1', cv=3, random_state=42, verbose=1, n_jobs=-1
)
rs.fit(X_train_smote, y_train_smote)
xgb_tuned = rs.best_estimator_
y_pred_tuned = xgb_tuned.predict(X_test_final)
y_prob_tuned = xgb_tuned.predict_proba(X_test_final)[:,1]
f1_tuned = f1_score(y_test, y_pred_tuned)
print(f"\nBest params: {rs.best_params_}")
print(classification_report(y_test, y_pred_tuned, target_names=['Normal','Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_tuned):.4f} | F1: {f1_tuned:.4f}")

# ============================================================
# 9. RESULTS COMPARISON
# ============================================================
results = pd.DataFrame({
    'Model': ['LSTM+RF (Hybrid1)','AE+XGBoost (Hybrid2)',
              'AE+BDS(GA)+XGB (CW)','AE+BDS(GA)+XGB (SMOTE)','AE+BDS(GA)+XGB (SMOTE+Tuned)'],
    'F1': [0.47, 0.87, f1_cw, f1_sm, f1_tuned],
    'ROC-AUC': [0.994, 0.997, roc_auc_score(y_test,y_prob_cw), roc_auc_score(y_test,y_prob_sm), roc_auc_score(y_test,y_prob_tuned)]
}).sort_values('F1', ascending=False)
print("\n" + results.to_string(index=False))

# Confusion matrices
fig, axes = plt.subplots(1,3,figsize=(18,5))
for ax,(name,yp) in zip(axes, [('CW',y_pred_cw),('SMOTE',y_pred_sm),('SMOTE+Tuned',y_pred_tuned)]):
    cm = confusion_matrix(y_test, yp)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal','Fraud'], yticklabels=['Normal','Fraud'])
    ax.set_title(f'AE+BDS(GA)+XGB ({name})'); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout(); plt.savefig('bds_fig_2.png', dpi=100, bbox_inches='tight'); plt.close()

# ROC + PR curves
fig, axes = plt.subplots(1,2,figsize=(16,6))
for name,yp in [('CW',y_prob_cw),('SMOTE',y_prob_sm),('SMOTE+Tuned',y_prob_tuned)]:
    fpr,tpr,_ = roc_curve(y_test,yp); axes[0].plot(fpr,tpr,label=f'{name} (AUC={roc_auc_score(y_test,yp):.4f})')
    pr,rc,_ = precision_recall_curve(y_test,yp); axes[1].plot(rc,pr,label=f'{name} (PR-AUC={auc(rc,pr):.4f})')
axes[0].plot([0,1],[0,1],'k--',alpha=0.5); axes[0].set_title('ROC'); axes[0].legend()
axes[1].set_title('Precision-Recall'); axes[1].legend()
plt.tight_layout(); plt.savefig('bds_fig_3.png', dpi=100, bbox_inches='tight'); plt.close()

# ============================================================
# 10. ERROR ANALYSIS
# ============================================================
ta = pd.DataFrame({'amt':test_eng['amt'].values, 'hour':test_eng['hour'].values,
                    'is_fraud':y_test, 'predicted':y_pred_tuned, 'recon_error':test_recon_error,
                    'bds_amount':test_bds[0], 'bds_time':test_bds[1], 'bds_freq':test_bds[2], 'bds_category':test_bds[3]})
missed = ta[(ta['is_fraud']==1)&(ta['predicted']==0)]
caught = ta[(ta['is_fraud']==1)&(ta['predicted']==1)]
fp = ta[(ta['is_fraud']==0)&(ta['predicted']==1)]
print(f"\nTotal frauds: {(y_test==1).sum()} | Caught: {len(caught)} | Missed: {len(missed)} | FP: {len(fp)}")
print(f"Missed - avg: ${missed['amt'].mean():.2f}, median: ${missed['amt'].median():.2f}, %<$50: {100*(missed['amt']<50).mean():.1f}%")
print(f"Caught - avg: ${caught['amt'].mean():.2f}, median: ${caught['amt'].median():.2f}")
print(f"Missed BDS - amt:{missed['bds_amount'].mean():.3f} time:{missed['bds_time'].mean():.3f} freq:{missed['bds_freq'].mean():.3f} cat:{missed['bds_category'].mean():.3f}")
print(f"Caught BDS - amt:{caught['bds_amount'].mean():.3f} time:{caught['bds_time'].mean():.3f} freq:{caught['bds_freq'].mean():.3f} cat:{caught['bds_category'].mean():.3f}")

fig,axes = plt.subplots(2,2,figsize=(14,10))
for ax,name in zip(axes.flat, bds_names):
    ax.hist(caught[name],bins=50,alpha=0.7,label='Caught',density=True)
    ax.hist(missed[name],bins=50,alpha=0.7,label='Missed',density=True)
    ax.set_title(f'{name}: Caught vs Missed'); ax.legend()
plt.suptitle('BDS Scores - Caught vs Missed Frauds',fontsize=14)
plt.tight_layout(); plt.savefig('bds_fig_4.png', dpi=100, bbox_inches='tight'); plt.close()

# ============================================================
# 11. SHAP ANALYSIS
# ============================================================
import shap
np.random.seed(42)
X_shap = X_test_final[np.random.choice(len(X_test_final), 5000, replace=False)]
explainer = shap.TreeExplainer(xgb_tuned)
shap_values = explainer.shap_values(X_shap)
print(f"\nSHAP computed: {shap_values.shape}")

plt.figure(figsize=(10,8))
shap.summary_plot(shap_values, X_shap, feature_names=final_feature_cols, plot_type='bar', show=False)
plt.title('SHAP Feature Importance'); plt.tight_layout()
plt.savefig('bds_fig_5.png', dpi=100, bbox_inches='tight'); plt.close()

plt.figure(figsize=(10,10))
shap.summary_plot(shap_values, X_shap, feature_names=final_feature_cols, show=False)
plt.title('SHAP Beeswarm'); plt.tight_layout()
plt.savefig('bds_fig_6.png', dpi=100, bbox_inches='tight'); plt.close()

mean_shap = np.abs(shap_values).mean(axis=0)
importance = pd.DataFrame({'Feature':final_feature_cols, 'Mean |SHAP|':mean_shap}).sort_values('Mean |SHAP|',ascending=False)
print("\nFeature Importance:")
print(importance.to_string(index=False))

contribs = ['velocity_1h','velocity_24h','amount_velocity_1h','recon_error','bds_amount','bds_time','bds_freq','bds_category']
print("\n--- Personal Contributions ---")
for _,row in importance.iterrows():
    if row['Feature'] in contribs:
        rank = importance['Feature'].tolist().index(row['Feature'])+1
        print(f"  #{rank:2d}: {row['Feature']:>25s} ({row['Mean |SHAP|']:.4f})")

# ============================================================
# 12. FINAL SUMMARY
# ============================================================
print("\n" + "="*65)
print("FINAL RESULTS SUMMARY")
print("="*65)
print(f"\n  {'Model':<45s} {'F1':>6s}")
print(f"  {'-'*45} {'-'*6}")
print(f"  {'LSTM + RF (Hybrid 1)':<45s} {'0.47':>6s}")
print(f"  {'AE + XGBoost (Hybrid 2)':<45s} {'0.87':>6s}")
print(f"  {'AE + BDS(GA) + XGBoost (CW)':<45s} {f1_cw:>6.4f}")
print(f"  {'AE + BDS(GA) + XGBoost (SMOTE)':<45s} {f1_sm:>6.4f}")
print(f"  {'AE + BDS(GA) + XGBoost (SMOTE+Tuned)':<45s} {f1_tuned:>6.4f}")
print(f"\nGA: pop=30, gen={len(best_history)}, best F1={best_f1:.4f}, time={ga_time/60:.1f}min")
print(f"\nOptimal BDS params:")
for n,v in zip(param_names, best_params):
    print(f"  {n:>20s} = {v:.4f}")
print(f"\nSHAP Top 5:")
for i,(_,row) in enumerate(importance.head().iterrows()):
    tag = " <-- BDS" if row['Feature'].startswith('bds_') else " <-- Velocity" if 'velocity' in row['Feature'] else " <-- AE" if row['Feature']=='recon_error' else ""
    print(f"  #{i+1}: {row['Feature']} ({row['Mean |SHAP|']:.4f}){tag}")
print("="*65)

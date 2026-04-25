"""CUSTOM 4 — GA Analysis: fitness distributions, parameter convergence, multi-seed stability"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, time, math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import warnings; warnings.filterwarnings('ignore')
import functools
print = functools.partial(print, flush=True)

# Load data
train_eng = pd.read_csv('fraudTrain_engineered.csv')
test_eng = pd.read_csv('fraudTest_engineered.csv')
train_cc = pd.read_csv('fraudTrain.csv', usecols=['cc_num'])['cc_num'].values
drop_cols = ['is_fraud','unix_time']
feature_cols = [c for c in train_eng.columns if c not in drop_cols]
X_train = train_eng[feature_cols].values; y_train = train_eng['is_fraud'].values

# Scale + autoencoder (quick retrain)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_train)
X_normal = X_sc[y_train == 0]

class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,10), nn.ReLU(), nn.Dropout(0.2), nn.Linear(10,5), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(5,10), nn.ReLU(), nn.Dropout(0.2), nn.Linear(10,d))
    def forward(self, x): return self.dec(self.enc(x))

print("Training autoencoder...")
ae = AE(X_sc.shape[1])
nt = torch.FloatTensor(X_normal)
vs = int(0.1*len(nt))
tds, vds = random_split(TensorDataset(nt,nt), [len(nt)-vs, vs])
tl = DataLoader(tds, batch_size=512, shuffle=True)
vl = DataLoader(vds, batch_size=512)
opt = torch.optim.Adam(ae.parameters())
crit = nn.MSELoss()
bv, pt, bs = float('inf'), 0, None
for ep in range(30):
    ae.train()
    for xb,_ in tl: l=crit(ae(xb),xb); opt.zero_grad(); l.backward(); opt.step()
    ae.eval()
    with torch.no_grad(): v=sum(crit(ae(xb),xb).item()*len(xb) for xb,_ in vl)/vs
    if v<bv: bv,pt,bs=v,0,{k:v.clone() for k,v in ae.state_dict().items()}
    else:
        pt+=1
        if pt>=3: break
ae.load_state_dict(bs); ae.eval()
with torch.no_grad(): tr_re = np.mean((X_sc - ae(torch.FloatTensor(X_sc)).numpy())**2, axis=1)
print("AE done.")

# Build profiles (simplified — same as BDS notebook)
profile_df = pd.DataFrame({'cc_num':train_cc, 'amt':train_eng['amt'].values, 'hour':train_eng['hour'].values.astype(int),
                            'category':train_eng['category_encoded'].values.astype(int), 'velocity_1h':train_eng['velocity_1h'].values})
card_amt = profile_df.groupby('cc_num')['amt'].agg(['mean','std','count'])
card_amt.columns = ['amt_mean','amt_std','amt_count']; card_amt['amt_std']=card_amt['amt_std'].fillna(0)
card_hour_counts = profile_df.groupby(['cc_num','hour']).size().unstack(fill_value=0)
for h in range(24):
    if h not in card_hour_counts.columns: card_hour_counts[h]=0
card_hour_counts = card_hour_counts[sorted(card_hour_counts.columns)]
card_hour_prob = card_hour_counts.div(card_hour_counts.sum(axis=1), axis=0)
n_categories = profile_df['category'].nunique()
card_cat_counts = profile_df.groupby(['cc_num','category']).size().unstack(fill_value=0)
card_cat_prob = card_cat_counts.div(card_cat_counts.sum(axis=1), axis=0)
card_vel = profile_df.groupby('cc_num')['velocity_1h'].agg(['mean','std'])
card_vel.columns = ['vel_mean','vel_std']; card_vel['vel_std']=card_vel['vel_std'].fillna(0)
ga_m = profile_df['amt'].mean(); ga_s = profile_df['amt'].std()
gh = profile_df.groupby('hour').size()/len(profile_df)
gc = profile_df.groupby('category').size()/len(profile_df)
gv = profile_df['velocity_1h'].mean()

def precompute(cc, amts, hrs, cats, vels):
    df = pd.DataFrame({'cc_num':cc,'amt':amts,'hour':hrs.astype(int),'cat':cats.astype(int),'vel':vels})
    df = df.merge(card_amt[['amt_mean','amt_std','amt_count']], left_on='cc_num', right_index=True, how='left')
    df['amt_mean']=df['amt_mean'].fillna(ga_m); df['amt_std']=df['amt_std'].fillna(ga_s); df['amt_count']=df['amt_count'].fillna(0)
    df['unseen']=df['amt_count']==0
    ss=df['amt_std'].where(df['amt_std']>0, ga_s)
    df['card_amt_z']=(df['amt']-df['amt_mean']).abs()/ss; df['global_amt_z']=(df['amt']-ga_m).abs()/ga_s
    hs=card_hour_prob.stack(); hs.index.names=['cc_num','hour']; hl=hs.reset_index(); hl.columns=['cc_num','hour','card_hour_p']
    df=df.merge(hl,on=['cc_num','hour'],how='left'); df['card_hour_p']=df['card_hour_p'].fillna(0.0)
    df['global_hour_p']=df['hour'].map(gh.to_dict()).fillna(1/24)
    cs=card_cat_prob.stack(); cs.index.names=['cc_num','cat']; cl=cs.reset_index(); cl.columns=['cc_num','cat','card_cat_p']
    df=df.merge(cl,on=['cc_num','cat'],how='left'); df['card_cat_p']=df['card_cat_p'].fillna(0.0)
    df['global_cat_p']=df['cat'].map(gc.to_dict()).fillna(1/n_categories)
    df=df.merge(card_vel[['vel_mean']],left_on='cc_num',right_index=True,how='left')
    df['vel_mean']=df['vel_mean'].fillna(gv); sv=df['vel_mean'].where(df['vel_mean']>0,gv); df['freq_ratio']=df['vel']/sv
    return {k:df[k].values for k in ['card_amt_z','global_amt_z','card_hour_p','global_hour_p','card_cat_p','global_cat_p','freq_ratio','unseen','amt_count']}

def rename(r): r['card_counts']=r.pop('amt_count'); return r

print("Precomputing...")
train_raw = rename(precompute(train_cc, train_eng['amt'].values, train_eng['hour'].values, train_eng['category_encoded'].values, train_eng['velocity_1h'].values))

PARAM_BOUNDS = [(0.0,3.0),(3.0,15.0),(0.0,3.0),(3.0,15.0),(0.0,3.0),(3.0,15.0),(0.0,3.0),(3.0,15.0),(2.0,20.0),(0.001,0.5)]
N_PARAMS = 10

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

# GA subsample
_, sub_idx = train_test_split(np.arange(len(y_train)), test_size=0.1, stratify=y_train, random_state=42)
X_sub=X_train[sub_idx]; y_sub=y_train[sub_idx]; re_sub=tr_re[sub_idx]
sub_raw={k:v[sub_idx] for k,v in train_raw.items()}
ga_tr, ga_vl = train_test_split(np.arange(len(y_sub)), test_size=0.3, stratify=y_sub, random_state=42)
X_gt,X_gv=X_sub[ga_tr],X_sub[ga_vl]; y_gt,y_gv=y_sub[ga_tr],y_sub[ga_vl]
re_gt,re_gv=re_sub[ga_tr],re_sub[ga_vl]
gtr={k:v[ga_tr] for k,v in sub_raw.items()}; gvr={k:v[ga_vl] for k,v in sub_raw.items()}
spw=(y_gt==0).sum()/max((y_gt==1).sum(),1)

def fitness(ind):
    bt=compute_bds(gtr,ind); bv=compute_bds(gvr,ind)
    Xt=np.column_stack([X_gt,re_gt]+list(bt)); Xv=np.column_stack([X_gv,re_gv]+list(bv))
    m=XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.1,scale_pos_weight=spw,eval_metric='logloss',random_state=42,n_jobs=-1,verbosity=0)
    m.fit(Xt,y_gt); return f1_score(y_gv,m.predict(Xv))

def run_ga_full(seed, pop_size=30, generations=20, elitism=2):
    """Run GA and return full history including all fitness values per generation."""
    rng = np.random.RandomState(seed)
    population = [[rng.uniform(lo,hi) for lo,hi in PARAM_BOUNDS] for _ in range(pop_size)]
    all_fits = []; best_params_per_gen = []; best_ever = -1; best_ind = None

    for gen in range(generations):
        fits = [fitness(ind) for ind in population]
        all_fits.append(fits)
        bi = np.argmax(fits)
        best_params_per_gen.append(population[bi].copy())
        if fits[bi] > best_ever: best_ever = fits[bi]; best_ind = population[bi].copy()

        si = np.argsort(fits)[::-1]
        new_pop = [population[si[i]].copy() for i in range(elitism)]
        while len(new_pop) < pop_size:
            c1 = rng.choice(len(population), 3, replace=False); p1 = population[c1[np.argmax([fits[i] for i in c1])]].copy()
            c2 = rng.choice(len(population), 3, replace=False); p2 = population[c2[np.argmax([fits[i] for i in c2])]].copy()
            alpha = rng.uniform(0.2,0.8)
            child = [alpha*a+(1-alpha)*b for a,b in zip(p1,p2)] if rng.random()<0.8 else p1
            for i in range(N_PARAMS):
                if rng.random()<0.2:
                    lo,hi=PARAM_BOUNDS[i]; child[i]=np.clip(child[i]+rng.normal(0,(hi-lo)*0.1),lo,hi)
            new_pop.append(child)
        population = new_pop

    return best_ind, best_ever, all_fits, best_params_per_gen

# Run GA with 3 different seeds (5 is too slow, ~3min each)
print("\n" + "="*70)
print("CUSTOM 4 — GA MULTI-SEED ANALYSIS")
print("="*70)

param_names = ['amt_thresh','amt_cap','time_thresh','time_cap','freq_thresh','freq_cap','cat_thresh','cat_cap','min_hist','smoothing']
all_results = []

for seed in [42, 123, 7]:
    print(f"\nRunning GA with seed={seed}...")
    t0 = time.time()
    best_ind, best_f1, all_fits, best_per_gen = run_ga_full(seed)
    elapsed = time.time() - t0
    all_results.append((seed, best_ind, best_f1, all_fits, best_per_gen))
    print(f"  Seed {seed}: Best F1={best_f1:.4f}, Time={elapsed:.1f}s")
    for n, v in zip(param_names, best_ind):
        print(f"    {n:>15s} = {v:.4f}")

# Print comparison table
print(f"\n{'Param':<16s}", end="")
for seed,_,_,_,_ in all_results:
    print(f" Seed={seed:>4d}", end="")
print()
print("-"*60)
for j, name in enumerate(param_names):
    print(f"{name:<16s}", end="")
    for _,bi,_,_,_ in all_results:
        print(f" {bi[j]:>9.4f}", end="")
    print()
print(f"{'Best F1':<16s}", end="")
for _,_,bf,_,_ in all_results:
    print(f" {bf:>9.4f}", end="")
print()

# Plot fitness distributions: gen 1 vs gen 20
r0 = all_results[0]  # seed=42
fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].hist(r0[3][0], bins=15, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title(f'Fitness Distribution — Generation 1\nMean={np.mean(r0[3][0]):.4f}, Std={np.std(r0[3][0]):.4f}')
axes[0].set_xlabel('F1 Score'); axes[0].set_ylabel('Count')
axes[1].hist(r0[3][-1], bins=15, alpha=0.7, color='green', edgecolor='black')
axes[1].set_title(f'Fitness Distribution — Generation 20\nMean={np.mean(r0[3][-1]):.4f}, Std={np.std(r0[3][-1]):.4f}')
axes[1].set_xlabel('F1 Score'); axes[1].set_ylabel('Count')
plt.suptitle('GA Fitness Distribution: Gen 1 vs Gen 20', fontsize=14)
plt.tight_layout(); plt.savefig('ga_fitness_distribution.png', dpi=100); plt.close()
print("Saved: ga_fitness_distribution.png")

# Plot parameter convergence
fig, axes = plt.subplots(2, 5, figsize=(25,8))
for j, (ax, name) in enumerate(zip(axes.flat, param_names)):
    vals = [r0[4][g][j] for g in range(20)]
    ax.plot(range(1,21), vals, 'b-o', markersize=4)
    ax.set_title(name); ax.set_xlabel('Generation'); ax.grid(True, alpha=0.3)
    ax.axhline(r0[1][j], color='r', linestyle='--', alpha=0.5)
plt.suptitle('GA Parameter Convergence (Seed=42)', fontsize=14)
plt.tight_layout(); plt.savefig('ga_parameter_convergence.png', dpi=100); plt.close()
print("Saved: ga_parameter_convergence.png")

print("\n" + "="*70)
print("CUSTOM 4 COMPLETE")
print("="*70)

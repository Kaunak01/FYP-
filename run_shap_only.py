"""Stage 3 only: SHAP on full hybrid."""
import json, os, time, warnings, numpy as np, pandas as pd, joblib, torch, torch.nn as nn
warnings.filterwarnings('ignore')
ROOT = r'C:\Users\User\OneDrive\Desktop\FYP-Fraud-Detection'
MODELS = os.path.join(ROOT, 'models', 'saved')
TEST   = os.path.join(ROOT, 'fraudTest_engineered.csv')
FEATURE_COLS = ['amt','city_pop','hour','month','distance_cardholder_merchant',
                'age','is_weekend','is_night','velocity_1h','velocity_24h',
                'amount_velocity_1h','category_encoded','gender_encoded','day_of_week_encoded']

print("Loading test data & full hybrid model...")
df = pd.read_csv(TEST)
X = df[FEATURE_COLS].values
y = df['is_fraud'].values

class AE(nn.Module):
    def __init__(self,n=14):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(n,10),nn.ReLU(),nn.Dropout(0.2),nn.Linear(10,5),nn.ReLU())
        self.decoder=nn.Sequential(nn.Linear(5,10),nn.ReLU(),nn.Dropout(0.2),nn.Linear(10,n))
    def forward(self,x): return self.decoder(self.encoder(x))

sc = joblib.load(os.path.join(MODELS,'ae_scaler.joblib'))
ae = AE(); ae.load_state_dict(torch.load(os.path.join(MODELS,'ae_model.pt'),map_location='cpu')); ae.eval()
with torch.no_grad():
    re_test = np.mean((sc.transform(X).astype(np.float32) - ae(torch.tensor(sc.transform(X).astype(np.float32))).numpy())**2, axis=1)

# BDS — reuse saved profiles via same logic as save_all_models.py
with open(os.path.join(MODELS,'ga_best_params.json')) as f: ga=json.load(f)
best_params = [ga['params'][k] for k in ['amount_threshold','amount_cap','time_threshold','time_cap',
                                          'freq_threshold','freq_cap','cat_threshold','cat_cap',
                                          'min_history','smoothing']]

raw_tr_df = pd.read_csv(os.path.join(ROOT,'fraudTrain.csv'), usecols=['cc_num'])
raw_te_df = pd.read_csv(os.path.join(ROOT,'fraudTest.csv'),  usecols=['cc_num'])
df_train = pd.read_csv(os.path.join(ROOT,'fraudTrain_engineered.csv'))
df_train['cc_num']=raw_tr_df['cc_num'].values
df['cc_num']=raw_te_df['cc_num'].values

card_amt = df_train.groupby('cc_num')['amt'].agg(['mean','std','count']).rename(columns={'mean':'amt_mean','std':'amt_std','count':'amt_count'})
card_amt['amt_std']=card_amt['amt_std'].fillna(0)
n_cat = max(df_train['category_encoded'].max(), df['category_encoded'].max())+1
card_hour_prob = (df_train.groupby(['cc_num','hour']).size()/df_train.groupby('cc_num').size()).unstack(fill_value=0)
card_cat_prob  = (df_train.groupby(['cc_num','category_encoded']).size()/df_train.groupby('cc_num').size()).unstack(fill_value=0)
card_vel = df_train.groupby('cc_num')['velocity_1h'].agg(['mean']).rename(columns={'mean':'vel_mean'})
gstats = {'amt_mean':df_train['amt'].mean(),'amt_std':df_train['amt'].std(),
          'hour_prob':df_train['hour'].value_counts(normalize=True).to_dict(),
          'cat_prob':df_train['category_encoded'].value_counts(normalize=True).to_dict(),
          'vel_mean':df_train['velocity_1h'].mean()}

def precompute(cc,amt,hr,cat,vel):
    d=pd.DataFrame({'cc_num':cc,'amt':amt,'hour':hr.astype(int),'cat':cat.astype(int),'vel':vel})
    d=d.merge(card_amt,left_on='cc_num',right_index=True,how='left')
    d['amt_mean']=d['amt_mean'].fillna(gstats['amt_mean']); d['amt_std']=d['amt_std'].fillna(gstats['amt_std'])
    d['amt_count']=d['amt_count'].fillna(0); d['unseen']=d['amt_count']==0
    ss=d['amt_std'].where(d['amt_std']>0, gstats['amt_std'])
    d['card_amt_z']=(d['amt']-d['amt_mean']).abs()/ss
    d['global_amt_z']=(d['amt']-gstats['amt_mean']).abs()/gstats['amt_std']
    hs=card_hour_prob.stack(); hs.index.names=['cc_num','hour']
    hl=hs.reset_index(); hl.columns=['cc_num','hour','card_hour_p']
    d=d.merge(hl,on=['cc_num','hour'],how='left'); d['card_hour_p']=d['card_hour_p'].fillna(0.0)
    d['global_hour_p']=d['hour'].map(gstats['hour_prob']).fillna(1/24)
    cs=card_cat_prob.stack(); cs.index.names=['cc_num','cat']
    cl=cs.reset_index(); cl.columns=['cc_num','cat','card_cat_p']
    d=d.merge(cl,on=['cc_num','cat'],how='left'); d['card_cat_p']=d['card_cat_p'].fillna(0.0)
    d['global_cat_p']=d['cat'].map(gstats['cat_prob']).fillna(1/n_cat)
    d=d.merge(card_vel,left_on='cc_num',right_index=True,how='left')
    d['vel_mean']=d['vel_mean'].fillna(gstats['vel_mean'])
    sv=d['vel_mean'].where(d['vel_mean']>0, gstats['vel_mean']); d['freq_ratio']=d['vel']/sv
    return {k:d[k].values for k in ['card_amt_z','global_amt_z','amt_count','card_hour_p','global_hour_p','card_cat_p','global_cat_p','freq_ratio','unseen']}

def compute_bds(raw,p):
    at,ac,tt,tc,ft,fc,ct,cc_,mh,sm=p; mh=int(round(mh))
    ug=(raw['amt_count']<mh)|raw['unseen']
    az=np.where(ug,raw['global_amt_z'],raw['card_amt_z']); a=np.clip(np.maximum(az-at,0),0,ac)
    hp=np.where(ug,raw['global_hour_p'],raw['card_hour_p']); tr=-np.log(hp+sm); t=np.clip(np.maximum(tr-tt,0),0,tc)
    fr=np.maximum(raw['freq_ratio']-1.0,0); fsc=np.clip(np.maximum(fr-ft,0),0,fc)
    cp=np.where(ug,raw['global_cat_p'],raw['card_cat_p']); cr=-np.log(cp+sm); c=np.clip(np.maximum(cr-ct,0),0,cc_)
    return a,t,fsc,c

raw_te=precompute(df['cc_num'].values, df['amt'].values, df['hour'].values, df['category_encoded'].values, df['velocity_1h'].values)
bds_te=compute_bds(raw_te, best_params)
X_test_19 = np.column_stack([X, re_test] + list(bds_te))

import shap
model = joblib.load(os.path.join(MODELS,'ae_bds_xgboost_smote_tuned.joblib'))

rng=np.random.RandomState(42)
fraud_idx=np.where(y==1)[0]; normal_idx=np.where(y==0)[0]
sample=np.concatenate([rng.choice(fraud_idx,min(2500,len(fraud_idx)),replace=False),
                      rng.choice(normal_idx,2500,replace=False)])
X_shap=X_test_19[sample]
print(f"SHAP on {len(X_shap)} samples...")
t0=time.time()
sv=shap.TreeExplainer(model).shap_values(X_shap)
print(f"  done in {time.time()-t0:.1f}s")

mean_abs=np.abs(sv).mean(axis=0)
names=FEATURE_COLS+['recon_error','bds_amount','bds_time','bds_freq','bds_category']
rank=sorted(zip(names,mean_abs.tolist()), key=lambda x:-x[1])
out={'model':'ae_bds_xgboost_smote_tuned.joblib (19 features)',
     'sample_size':int(len(X_shap)),
     'sample_composition':'stratified: ~2500 fraud + 2500 normal',
     'top_features':[{'rank':i+1,'feature':n,'mean_abs_shap':float(v)} for i,(n,v) in enumerate(rank)]}
with open(os.path.join(ROOT,'shap_top_features.json'),'w') as f:
    json.dump(out,f,indent=2)
print(">> saved shap_top_features.json")
for i,(n,v) in enumerate(rank):
    print(f"  {i+1:2d}. {n:<30} {v:.4f}")

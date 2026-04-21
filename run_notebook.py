# Auto-generated from FYP_Autoencoder_XGBoost.ipynb
import matplotlib; matplotlib.use("Agg")

# # Hybrid 2: Autoencoder + XGBoost for Credit Card Fraud Detection
# 
# **Approach:** Train an autoencoder on **normal transactions only** to learn the pattern of legitimate spending. The reconstruction error becomes a new feature — fraudulent transactions should have higher reconstruction error because they deviate from normal patterns. This error feature is then combined with the original engineered features and fed into XGBoost.
# 
# **Pipeline:**
# 1. Load engineered datasets
# 2. Train autoencoder on normal transactions only
# 3. Compute reconstruction error for all transactions
# 4. Merge reconstruction error as a new feature
# 5. Train XGBoost with class weights (baseline)
# 6. Train XGBoost with SMOTE
# 7. Train XGBoost with SMOTE + hyperparameter tuning
# 8. Evaluation and comparison

# === Cell 1 ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, precision_recall_curve, auc, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ## 1. Load Engineered Datasets

# === Cell 3 ===
# Load engineered datasets
train_df = pd.read_csv('fraudTrain_engineered.csv')
test_df = pd.read_csv('fraudTest_engineered.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nFraud rate (train): {train_df['is_fraud'].mean():.4f} ({train_df['is_fraud'].sum()} frauds)")
print(f"Fraud rate (test): {test_df['is_fraud'].mean():.4f} ({test_df['is_fraud'].sum()} frauds)")

# Define features — drop unix_time and target
drop_cols = ['is_fraud', 'unix_time']
feature_cols = [c for c in train_df.columns if c not in drop_cols]
print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

X_train = train_df[feature_cols].values
y_train = train_df['is_fraud'].values
X_test = test_df[feature_cols].values
y_test = test_df['is_fraud'].values

# === Cell 4 ===
# Scale features for autoencoder
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split training data: normal transactions only for autoencoder training
X_train_normal = X_train_scaled[y_train == 0]
print(f"Normal transactions for AE training: {X_train_normal.shape[0]:,}")
print(f"Fraud transactions (excluded from AE training): {(y_train == 1).sum():,}")

# ## 2. Build and Train Autoencoder
# 
# Simple symmetric encoder-decoder architecture. The bottleneck forces the model to learn a compressed representation of normal transactions. Fraudulent transactions, being different from normal patterns, will have higher reconstruction error.

# === Cell 6 ===
# Autoencoder architecture (PyTorch)
input_dim = X_train_scaled.shape[1]

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, 5),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

autoencoder = Autoencoder(input_dim).to(device)
print(autoencoder)
total_params = sum(p.numel() for p in autoencoder.parameters())
print(f"\nTotal parameters: {total_params}")

# === Cell 7 ===
# Train on NORMAL transactions only
from torch.utils.data import random_split

normal_tensor = torch.FloatTensor(X_train_normal).to(device)
val_size = int(0.1 * len(normal_tensor))
train_size = len(normal_tensor) - val_size
train_data, val_data = random_split(TensorDataset(normal_tensor, normal_tensor), [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
val_loader = DataLoader(val_data, batch_size=512)

optimizer = torch.optim.Adam(autoencoder.parameters())
criterion = nn.MSELoss()

# Training with early stopping
best_val_loss = float('inf')
patience, patience_counter = 3, 0
best_state = None
history = {'loss': [], 'val_loss': []}

for epoch in range(30):
    # Train
    autoencoder.train()
    train_loss = 0
    for xb, _ in train_loader:
        pred = autoencoder(xb)
        loss = criterion(pred, xb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= train_size

    # Validate
    autoencoder.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, _ in val_loader:
            pred = autoencoder(xb)
            val_loss += criterion(pred, xb).item() * len(xb)
    val_loss /= val_size

    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    print(f"Epoch {epoch+1}/30 — train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = autoencoder.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

autoencoder.load_state_dict(best_state)
print(f"\nBest validation loss: {best_val_loss:.6f}")

# === Cell 8 ===
# Plot training history
plt.figure(figsize=(8, 4))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.savefig("fig_1.png", dpi=100, bbox_inches="tight")
plt.close()

print(f"Final training loss: {history['loss'][-1]:.6f}")
print(f"Final validation loss: {history['val_loss'][-1]:.6f}")

# ## 3. Compute Reconstruction Error

# === Cell 10 ===
# Reconstruction error = mean squared error per sample
autoencoder.eval()
with torch.no_grad():
    train_reconstructed = autoencoder(torch.FloatTensor(X_train_scaled).to(device)).cpu().numpy()
    test_reconstructed = autoencoder(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()

train_recon_error = np.mean((X_train_scaled - train_reconstructed) ** 2, axis=1)
test_recon_error = np.mean((X_test_scaled - test_reconstructed) ** 2, axis=1)

print(f"Reconstruction error stats (TRAIN):")
print(f"  Normal  — mean: {train_recon_error[y_train == 0].mean():.4f}, std: {train_recon_error[y_train == 0].std():.4f}")
print(f"  Fraud   — mean: {train_recon_error[y_train == 1].mean():.4f}, std: {train_recon_error[y_train == 1].std():.4f}")
print(f"  Ratio (fraud/normal): {train_recon_error[y_train == 1].mean() / train_recon_error[y_train == 0].mean():.2f}x")

# === Cell 11 ===
# Visualise reconstruction error distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(train_recon_error[y_train == 0], bins=100, alpha=0.7, label='Normal', density=True)
axes[0].hist(train_recon_error[y_train == 1], bins=100, alpha=0.7, label='Fraud', density=True)
axes[0].set_title('Reconstruction Error Distribution')
axes[0].set_xlabel('Reconstruction Error (MSE)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_xlim(0, np.percentile(train_recon_error, 99))

# Box plot
error_df = pd.DataFrame({
    'Reconstruction Error': train_recon_error,
    'Class': ['Normal' if y == 0 else 'Fraud' for y in y_train]
})
sns.boxplot(data=error_df, x='Class', y='Reconstruction Error', ax=axes[1])
axes[1].set_title('Reconstruction Error by Class')
axes[1].set_ylim(0, np.percentile(train_recon_error, 99))

plt.tight_layout()
plt.savefig("fig_2.png", dpi=100, bbox_inches="tight")
plt.close()

# ## 4. Merge Reconstruction Error as Feature
# 
# Add the autoencoder's reconstruction error as a new feature alongside the original engineered features for XGBoost.

# === Cell 13 ===
# Add reconstruction error as a new feature
X_train_hybrid = np.column_stack([X_train, train_recon_error])
X_test_hybrid = np.column_stack([X_test, test_recon_error])

hybrid_feature_cols = feature_cols + ['recon_error']
print(f"Hybrid feature set ({len(hybrid_feature_cols)} features): {hybrid_feature_cols}")
print(f"X_train_hybrid shape: {X_train_hybrid.shape}")
print(f"X_test_hybrid shape: {X_test_hybrid.shape}")

# ## 5. XGBoost — Configuration 1: Class Weights Only (Baseline)

# === Cell 15 ===
# Calculate class weight ratio
n_normal = (y_train == 0).sum()
n_fraud = (y_train == 1).sum()
scale_pos_weight = n_normal / n_fraud
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Config 1: Class weights only
xgb_v1 = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_v1.fit(X_train_hybrid, y_train)
y_pred_v1 = xgb_v1.predict(X_test_hybrid)
y_prob_v1 = xgb_v1.predict_proba(X_test_hybrid)[:, 1]

print("\n=== Config 1: AE + XGBoost (Class Weights) ===")
print(classification_report(y_test, y_pred_v1, target_names=['Normal', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_v1):.4f}")
f1_v1 = f1_score(y_test, y_pred_v1)
print(f"Fraud F1: {f1_v1:.4f}")

# ## 6. XGBoost — Configuration 2: SMOTE

# === Cell 17 ===
# Config 2: SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_hybrid, y_train)

print(f"Before SMOTE: Normal={n_normal:,}, Fraud={n_fraud:,}")
print(f"After SMOTE:  Normal={(y_train_smote == 0).sum():,}, Fraud={(y_train_smote == 1).sum():,}")

xgb_v2 = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_v2.fit(X_train_smote, y_train_smote)
y_pred_v2 = xgb_v2.predict(X_test_hybrid)
y_prob_v2 = xgb_v2.predict_proba(X_test_hybrid)[:, 1]

print("\n=== Config 2: AE + XGBoost (SMOTE) ===")
print(classification_report(y_test, y_pred_v2, target_names=['Normal', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_v2):.4f}")
f1_v2 = f1_score(y_test, y_pred_v2)
print(f"Fraud F1: {f1_v2:.4f}")

# ## 7. XGBoost — Configuration 3: SMOTE + Hyperparameter Tuning

# === Cell 19 ===
# Config 3: SMOTE + RandomizedSearchCV
param_dist = {
    'n_estimators': [200, 300, 400],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

xgb_search = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

random_search = RandomizedSearchCV(
    xgb_search,
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1',
    cv=3,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

print("Running RandomizedSearchCV (30 iterations, 3-fold CV)...")
random_search.fit(X_train_smote, y_train_smote)

print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best CV F1: {random_search.best_score_:.4f}")

# === Cell 20 ===
# Evaluate best tuned model
xgb_v3 = random_search.best_estimator_
y_pred_v3 = xgb_v3.predict(X_test_hybrid)
y_prob_v3 = xgb_v3.predict_proba(X_test_hybrid)[:, 1]

print("=== Config 3: AE + XGBoost (SMOTE + Tuned) ===")
print(classification_report(y_test, y_pred_v3, target_names=['Normal', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_v3):.4f}")
f1_v3 = f1_score(y_test, y_pred_v3)
print(f"Fraud F1: {f1_v3:.4f}")

# ## 8. Results Comparison

# === Cell 22 ===
# Comparison table
results = pd.DataFrame({
    'Model': [
        'AE + XGBoost (Class Weights)',
        'AE + XGBoost (SMOTE)',
        'AE + XGBoost (SMOTE + Tuned)',
        'LSTM + RF (best, from Hybrid 1)',
    ],
    'F1 (Fraud)': [f1_v1, f1_v2, f1_v3, 0.47],
    'ROC-AUC': [
        roc_auc_score(y_test, y_prob_v1),
        roc_auc_score(y_test, y_prob_v2),
        roc_auc_score(y_test, y_prob_v3),
        0.9939
    ]
})
results = results.sort_values('F1 (Fraud)', ascending=False)
print(results.to_string(index=False))

# === Cell 23 ===
# Confusion matrices for all 3 configs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
configs = [
    ('Class Weights', y_pred_v1),
    ('SMOTE', y_pred_v2),
    ('SMOTE + Tuned', y_pred_v3)
]

for ax, (name, y_pred) in zip(axes, configs):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    ax.set_title(f'AE + XGBoost ({name})')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig("fig_3.png", dpi=100, bbox_inches="tight")
plt.close()

# === Cell 24 ===
# ROC curves
plt.figure(figsize=(8, 6))
for name, y_prob in [('Class Weights', y_prob_v1), ('SMOTE', y_prob_v2), ('SMOTE + Tuned', y_prob_v3)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.4f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — Autoencoder + XGBoost Hybrid')
plt.legend()
plt.tight_layout()
plt.savefig("fig_4.png", dpi=100, bbox_inches="tight")
plt.close()

# === Cell 25 ===
# Precision-Recall curves
plt.figure(figsize=(8, 6))
for name, y_prob in [('Class Weights', y_prob_v1), ('SMOTE', y_prob_v2), ('SMOTE + Tuned', y_prob_v3)]:
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (PR-AUC={pr_auc:.4f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves — Autoencoder + XGBoost Hybrid')
plt.legend()
plt.tight_layout()
plt.savefig("fig_5.png", dpi=100, bbox_inches="tight")
plt.close()

# ## 9. Error Analysis

# === Cell 27 ===
# Error analysis on best model (SMOTE + Tuned)
best_pred = y_pred_v3
best_prob = y_prob_v3

# Missed frauds (false negatives) and false positives
test_analysis = test_df[feature_cols].copy()
test_analysis['is_fraud'] = y_test
test_analysis['predicted'] = best_pred
test_analysis['fraud_prob'] = best_prob
test_analysis['recon_error'] = test_recon_error

missed_frauds = test_analysis[(test_analysis['is_fraud'] == 1) & (test_analysis['predicted'] == 0)]
caught_frauds = test_analysis[(test_analysis['is_fraud'] == 1) & (test_analysis['predicted'] == 1)]
false_positives = test_analysis[(test_analysis['is_fraud'] == 0) & (test_analysis['predicted'] == 1)]

print(f"Total frauds in test: {(y_test == 1).sum()}")
print(f"Caught frauds: {len(caught_frauds)}")
print(f"Missed frauds: {len(missed_frauds)}")
print(f"False positives: {len(false_positives)}")

print(f"\n--- Missed Frauds ---")
print(f"Average amount: ${missed_frauds['amt'].mean():.2f}")
print(f"Median amount: ${missed_frauds['amt'].median():.2f}")
print(f"% under $50: {(missed_frauds['amt'] < 50).mean() * 100:.1f}%")

print(f"\n--- Caught Frauds ---")
print(f"Average amount: ${caught_frauds['amt'].mean():.2f}")
print(f"Median amount: ${caught_frauds['amt'].median():.2f}")

print(f"\n--- Reconstruction Error ---")
print(f"Missed frauds avg recon error: {missed_frauds['recon_error'].mean():.4f}")
print(f"Caught frauds avg recon error: {caught_frauds['recon_error'].mean():.4f}")

# === Cell 28 ===
# Visualise error analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Amount distribution: caught vs missed
axes[0].hist(caught_frauds['amt'], bins=50, alpha=0.7, label='Caught', density=True)
axes[0].hist(missed_frauds['amt'], bins=50, alpha=0.7, label='Missed', density=True)
axes[0].set_title('Transaction Amount: Caught vs Missed Frauds')
axes[0].set_xlabel('Amount ($)')
axes[0].legend()

# Hour distribution: caught vs missed
axes[1].hist(caught_frauds['hour'], bins=24, alpha=0.7, label='Caught', density=True)
axes[1].hist(missed_frauds['hour'], bins=24, alpha=0.7, label='Missed', density=True)
axes[1].set_title('Hour of Day: Caught vs Missed Frauds')
axes[1].set_xlabel('Hour')
axes[1].legend()

# Recon error: caught vs missed
axes[2].hist(caught_frauds['recon_error'], bins=50, alpha=0.7, label='Caught', density=True)
axes[2].hist(missed_frauds['recon_error'], bins=50, alpha=0.7, label='Missed', density=True)
axes[2].set_title('Reconstruction Error: Caught vs Missed Frauds')
axes[2].set_xlabel('Reconstruction Error')
axes[2].legend()

plt.tight_layout()
plt.savefig("fig_6.png", dpi=100, bbox_inches="tight")
plt.close()

# ## 10. Ablation Study — Velocity Features
# 
# Retrain XGBoost (SMOTE + tuned) **without** velocity features to prove they improve performance.

# === Cell 30 ===
# Remove velocity features
velocity_cols = ['velocity_1h', 'velocity_24h', 'amount_velocity_1h']
ablation_feature_cols = [c for c in feature_cols if c not in velocity_cols]
print(f"Features WITH velocity ({len(feature_cols)}): {feature_cols}")
print(f"Features WITHOUT velocity ({len(ablation_feature_cols)}): {ablation_feature_cols}")

# Prepare ablation data (no velocity, but keep recon_error)
velocity_indices = [feature_cols.index(c) for c in velocity_cols]
keep_indices = [i for i in range(len(feature_cols)) if i not in velocity_indices]

X_train_ablation = np.column_stack([X_train[:, keep_indices], train_recon_error])
X_test_ablation = np.column_stack([X_test[:, keep_indices], test_recon_error])

print(f"\nAblation X_train shape: {X_train_ablation.shape}")
print(f"Ablation X_test shape: {X_test_ablation.shape}")

# SMOTE on ablation data
X_train_abl_smote, y_train_abl_smote = smote.fit_resample(X_train_ablation, y_train)

# === Cell 31 ===
# Train with same best hyperparameters but without velocity features
best_params = random_search.best_params_

xgb_ablation = XGBClassifier(
    **best_params,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_ablation.fit(X_train_abl_smote, y_train_abl_smote)
y_pred_ablation = xgb_ablation.predict(X_test_ablation)
y_prob_ablation = xgb_ablation.predict_proba(X_test_ablation)[:, 1]

f1_ablation = f1_score(y_test, y_pred_ablation)

print("=== Ablation: AE + XGBoost WITHOUT Velocity Features ===")
print(classification_report(y_test, y_pred_ablation, target_names=['Normal', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_ablation):.4f}")
print(f"Fraud F1: {f1_ablation:.4f}")

print(f"\n{'='*50}")
print(f"ABLATION RESULT:")
print(f"  WITH velocity features:    F1 = {f1_v3:.4f}")
print(f"  WITHOUT velocity features: F1 = {f1_ablation:.4f}")
print(f"  Difference:                F1 = {f1_v3 - f1_ablation:+.4f}")
print(f"{'='*50}")

# === Cell 32 ===
# Ablation confusion matrix comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_with = confusion_matrix(y_test, y_pred_v3)
cm_without = confusion_matrix(y_test, y_pred_ablation)

sns.heatmap(cm_with, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
axes[0].set_title(f'WITH Velocity Features (F1={f1_v3:.4f})')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

sns.heatmap(cm_without, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
axes[1].set_title(f'WITHOUT Velocity Features (F1={f1_ablation:.4f})')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.suptitle('Ablation Study: Impact of Velocity Features', fontsize=14)
plt.tight_layout()
plt.savefig("fig_7.png", dpi=100, bbox_inches="tight")
plt.close()

# ## 11. SHAP Analysis
# 
# SHAP (SHapley Additive exPlanations) on the best model to show which features drive fraud predictions.

# === Cell 34 ===
import shap

# Use a sample for SHAP (full dataset is too large)
np.random.seed(42)
shap_sample_size = 5000
sample_idx = np.random.choice(len(X_test_hybrid), shap_sample_size, replace=False)
X_shap_sample = X_test_hybrid[sample_idx]

# TreeExplainer is fast for XGBoost
explainer = shap.TreeExplainer(xgb_v3)
shap_values = explainer.shap_values(X_shap_sample)

print(f"SHAP values computed for {shap_sample_size} samples")
print(f"SHAP values shape: {shap_values.shape}")

# === Cell 35 ===
# SHAP summary plot (bar — feature importance ranking)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap_sample, feature_names=hybrid_feature_cols, plot_type='bar', show=False)
plt.title('SHAP Feature Importance — Autoencoder + XGBoost')
plt.tight_layout()
plt.savefig("fig_8.png", dpi=100, bbox_inches="tight")
plt.close()

# === Cell 36 ===
# SHAP beeswarm plot (detailed feature impact)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap_sample, feature_names=hybrid_feature_cols, show=False)
plt.title('SHAP Feature Impact — Autoencoder + XGBoost')
plt.tight_layout()
plt.savefig("fig_9.png", dpi=100, bbox_inches="tight")
plt.close()

# === Cell 37 ===
# Top features by mean absolute SHAP value
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'Feature': hybrid_feature_cols,
    'Mean |SHAP|': mean_abs_shap
}).sort_values('Mean |SHAP|', ascending=False)

print("Feature Importance Ranking (by mean |SHAP|):")
print(feature_importance.to_string(index=False))

# Highlight velocity features
velocity_features = ['velocity_1h', 'velocity_24h', 'amount_velocity_1h']
velocity_ranks = feature_importance[feature_importance['Feature'].isin(velocity_features)]
print(f"\nVelocity features ranking:")
for _, row in velocity_ranks.iterrows():
    rank = feature_importance['Feature'].tolist().index(row['Feature']) + 1
    print(f"  #{rank}: {row['Feature']} (mean |SHAP| = {row['Mean |SHAP|']:.4f})")

# Reconstruction error rank
recon_rank = feature_importance['Feature'].tolist().index('recon_error') + 1
recon_shap = feature_importance[feature_importance['Feature'] == 'recon_error']['Mean |SHAP|'].values[0]
print(f"  #{recon_rank}: recon_error (mean |SHAP| = {recon_shap:.4f})")

# ## 12. Final Summary

# === Cell 39 ===
print("="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"\nAutoencoder + XGBoost Hybrid:")
print(f"  Config 1 (Class Weights):  F1 = {f1_v1:.4f}")
print(f"  Config 2 (SMOTE):          F1 = {f1_v2:.4f}")
print(f"  Config 3 (SMOTE + Tuned):  F1 = {f1_v3:.4f}")
print(f"\nComparison with Hybrid 1:")
print(f"  LSTM + RF (best):          F1 = 0.4747")
print(f"  AE + XGBoost (best):       F1 = {f1_v3:.4f}")
print(f"  Improvement:               F1 = {f1_v3 - 0.4747:+.4f}")
print(f"\nAblation Study:")
print(f"  WITH velocity features:    F1 = {f1_v3:.4f}")
print(f"  WITHOUT velocity features: F1 = {f1_ablation:.4f}")
print(f"  Velocity contribution:     F1 = {f1_v3 - f1_ablation:+.4f}")
print(f"\nSHAP: Top 5 features by importance:")
for i, (_, row) in enumerate(feature_importance.head().iterrows()):
    print(f"  #{i+1}: {row['Feature']} ({row['Mean |SHAP|']:.4f})")
print("="*60)

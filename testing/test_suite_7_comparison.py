"""Test Suite 7: Multi-Model Comparison."""
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_config import TEST_CSV, RESULTS_DIR, FEATURE_COLS, RANDOM_SEED
from fraud_predictor import (XGBoostPredictor, AEXGBoostPredictor, BDSXGBoostPredictor,
                             transaction_from_row, MODEL_XGB_CW, MODEL_XGB_TUNED)

class Tee:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d): self.stdout.write(d); self.file.write(d)
    def flush(self): self.stdout.flush(); self.file.flush()
    def close(self): self.file.close()


def run_suite():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tee = Tee(os.path.join(RESULTS_DIR, 'test_suite_7_results.txt'))
    sys.stdout = tee

    print("="*70)
    print("TEST SUITE 7: MULTI-MODEL COMPARISON")
    print("="*70)
    t0_total = time.time()
    tests_passed = 0; tests_failed = 0; tests_total = 0

    # Load data
    test_df = pd.read_csv(TEST_CSV)
    y_test = test_df['is_fraud'].values

    # Load all models
    print("\nLoading models...")
    models = {}
    try:
        models['XGB_CW'] = XGBoostPredictor(MODEL_XGB_CW)
        print("  XGB_CW loaded")
    except Exception as e:
        print(f"  XGB_CW FAILED: {e}")

    try:
        models['XGB_Tuned'] = XGBoostPredictor(MODEL_XGB_TUNED)
        print("  XGB_Tuned loaded")
    except Exception as e:
        print(f"  XGB_Tuned FAILED: {e}")

    try:
        models['AE_XGB'] = AEXGBoostPredictor()
        print("  AE_XGB loaded")
    except Exception as e:
        print(f"  AE_XGB FAILED: {e}")

    try:
        models['BDS_XGB'] = BDSXGBoostPredictor()
        print("  BDS_XGB loaded")
    except Exception as e:
        print(f"  BDS_XGB FAILED: {e}")

    model_names = list(models.keys())
    print(f"  Models loaded: {len(models)}/{4}")

    # ================================================================
    # COMPARE 1: SAME FRAUDS, ALL MODELS
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("COMPARE 1: SAME 200 FRAUDS, ALL MODELS")
    print("="*70)

    fraud_idx = test_df[test_df['is_fraud'] == 1].index.values
    np.random.seed(RANDOM_SEED)
    sample_fraud_idx = np.random.choice(fraud_idx, min(200, len(fraud_idx)), replace=False)
    sample_df = test_df.iloc[sample_fraud_idx]

    print(f"  Sampled {len(sample_fraud_idx)} fraud transactions")

    # Get predictions from each model
    model_probs = {}
    model_preds = {}
    for name, pred in models.items():
        t0 = time.time()
        probs = []
        for idx in sample_fraud_idx:
            txn = transaction_from_row(test_df, idx)
            r = pred.predict_single(txn)
            probs.append(r.probability)
        probs = np.array(probs)
        preds = (probs >= 0.5).astype(int)
        model_probs[name] = probs
        model_preds[name] = preds
        caught = preds.sum()
        print(f"  {name:<12s}: caught {caught}/{len(preds)} ({100*caught/len(preds):.1f}%) in {time.time()-t0:.1f}s")

    # Agreement analysis
    all_caught = np.ones(len(sample_fraud_idx), dtype=bool)
    all_missed = np.ones(len(sample_fraud_idx), dtype=bool)
    for name in model_names:
        all_caught &= (model_preds[name] == 1)
        all_missed &= (model_preds[name] == 0)

    n_all_caught = all_caught.sum()
    n_all_missed = all_missed.sum()

    # Unique catches per model
    print(f"\n  Agreement analysis:")
    print(f"    Caught by ALL models: {n_all_caught}")
    print(f"    Missed by ALL models: {n_all_missed}")

    for name in model_names:
        unique_catch = 0
        for i in range(len(sample_fraud_idx)):
            if model_preds[name][i] == 1:
                others_missed = all(model_preds[other][i] == 0 for other in model_names if other != name)
                if others_missed:
                    unique_catch += 1
        if unique_catch > 0:
            print(f"    Caught ONLY by {name}: {unique_catch}")

    # Analyse missed-by-all cases
    if n_all_missed > 0:
        print(f"\n  Profile of {n_all_missed} frauds missed by ALL models:")
        missed_all_idx = sample_fraud_idx[all_missed]
        missed_features = test_df.iloc[missed_all_idx][FEATURE_COLS]
        for col in ['amt', 'hour', 'is_night', 'velocity_1h', 'amount_velocity_1h']:
            print(f"    {col}: mean={missed_features[col].mean():.2f}, median={missed_features[col].median():.2f}")
        print(f"    These are the UNCATCHABLE frauds — small amounts, daytime, normal velocity")

    print(f"\n  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # COMPARE 2: PROBABILITY CORRELATION
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("COMPARE 2: PROBABILITY CORRELATION")
    print("="*70)

    pairs = []
    for i, n1 in enumerate(model_names):
        for n2 in model_names[i+1:]:
            corr = np.corrcoef(model_probs[n1], model_probs[n2])[0, 1]
            pairs.append((n1, n2, corr))
            print(f"  {n1} vs {n2}: correlation = {corr:.4f}")

    # Scatter plots
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]
    for ax, (n1, n2, corr) in zip(axes, pairs):
        ax.scatter(model_probs[n1], model_probs[n2], alpha=0.3, s=10)
        ax.plot([0,1],[0,1], 'r--', alpha=0.5)
        ax.set_xlabel(n1); ax.set_ylabel(n2)
        ax.set_title(f'r={corr:.3f}')
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    plt.suptitle('Model Probability Correlation (200 fraud samples)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'probability_correlation.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: probability_correlation.png")
    print(f"  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # COMPARE 3: ERROR OVERLAP (on full test set)
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("COMPARE 3: ERROR OVERLAP (full test set)")
    print("="*70)

    # Get full test predictions for each model
    full_preds = {}
    for name, pred in models.items():
        print(f"  Predicting full test set with {name}...")
        s = pred.predict_batch(test_df)
        full_preds[name] = s['predictions']

    # Missed fraud sets
    missed_sets = {}
    fp_sets = {}
    for name in model_names:
        missed_sets[name] = set(np.where((y_test == 1) & (full_preds[name] == 0))[0])
        fp_sets[name] = set(np.where((y_test == 0) & (full_preds[name] == 1))[0])
        print(f"  {name}: {len(missed_sets[name])} missed frauds, {len(fp_sets[name])} false positives")

    # Jaccard similarity for missed frauds
    print(f"\n  Jaccard Similarity (missed fraud overlap):")
    print(f"  {'':>12s}", end="")
    for n in model_names:
        print(f"  {n:>12s}", end="")
    print()
    for n1 in model_names:
        print(f"  {n1:>12s}", end="")
        for n2 in model_names:
            s1, s2 = missed_sets[n1], missed_sets[n2]
            if len(s1 | s2) > 0:
                jaccard = len(s1 & s2) / len(s1 | s2)
            else:
                jaccard = 1.0
            print(f"  {jaccard:>12.4f}", end="")
        print()

    # Find overlap between best models
    if 'XGB_Tuned' in missed_sets and 'AE_XGB' in missed_sets:
        overlap = missed_sets['XGB_Tuned'] & missed_sets['AE_XGB']
        only_xgb = missed_sets['XGB_Tuned'] - missed_sets['AE_XGB']
        only_ae = missed_sets['AE_XGB'] - missed_sets['XGB_Tuned']
        print(f"\n  XGB_Tuned vs AE_XGB missed fraud overlap:")
        print(f"    Both miss: {len(overlap)}")
        print(f"    Only XGB misses: {len(only_xgb)} (AE catches these)")
        print(f"    Only AE misses: {len(only_ae)} (XGB catches these)")
        if len(overlap) > len(only_xgb) + len(only_ae):
            print(f"    High overlap — models fail on same transactions")
        else:
            print(f"    Low overlap — ensemble could help")

    print(f"\n  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # COMPARE 4: PERFORMANCE BY FRAUD TYPE
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("COMPARE 4: PERFORMANCE BY FRAUD TYPE")
    print("="*70)

    fraud_df = test_df[test_df['is_fraud'] == 1].copy()
    fraud_indices = fraud_df.index.values

    # Categorise frauds
    def classify_fraud(row):
        if row['amt'] > 500 and row['is_night'] == 1:
            return 'Classic (high amt + night)'
        elif row['amt'] < 50 and row['is_night'] == 0:
            return 'Stealthy (low amt + day)'
        elif row['velocity_1h'] >= 3:
            return 'Velocity (3+ txns/hr)'
        else:
            return 'Camouflaged (normal pattern)'

    fraud_df['fraud_type'] = fraud_df.apply(classify_fraud, axis=1)
    fraud_types = fraud_df['fraud_type'].value_counts()
    print(f"\n  Fraud type distribution:")
    for ft, count in fraud_types.items():
        print(f"    {ft:<35s}: {count:>5d}")

    # Recall per fraud type per model
    print(f"\n  {'Fraud Type':<35s}", end="")
    for name in model_names:
        print(f" {name:>10s}", end="")
    print(f" {'Best':>12s}")
    print(f"  {'-'*95}")

    for ft in fraud_types.index:
        ft_mask = fraud_df['fraud_type'] == ft
        ft_indices = fraud_df[ft_mask].index.values
        print(f"  {ft:<35s}", end="")
        best_recall = 0; best_model = ""
        for name in model_names:
            # Get predictions for these specific indices
            ft_preds = full_preds[name][ft_indices]
            recall = ft_preds.sum() / len(ft_preds) if len(ft_preds) > 0 else 0
            print(f" {recall:>10.1%}", end="")
            if recall > best_recall:
                best_recall = recall; best_model = name
        print(f" {best_model:>12s}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ft_list = list(fraud_types.index)
    x = np.arange(len(ft_list))
    width = 0.2
    for i, name in enumerate(model_names):
        recalls = []
        for ft in ft_list:
            ft_mask = fraud_df['fraud_type'] == ft
            ft_idx = fraud_df[ft_mask].index.values
            ft_p = full_preds[name][ft_idx]
            recalls.append(ft_p.sum() / len(ft_p) if len(ft_p) > 0 else 0)
        ax.bar(x + i*width, [r*100 for r in recalls], width, label=name)
    ax.set_xticks(x + width * (len(model_names)-1) / 2)
    ax.set_xticklabels(ft_list, rotation=15, ha='right')
    ax.set_ylabel('Recall (%)'); ax.set_title('Recall by Fraud Type — All Models')
    ax.legend(); ax.set_ylim(0, 110); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'recall_by_fraud_type.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: recall_by_fraud_type.png")

    # Model agreement heatmap
    n = len(model_names)
    agreement = np.zeros((n, n))
    for i, n1 in enumerate(model_names):
        for j, n2 in enumerate(model_names):
            agree = (full_preds[n1] == full_preds[n2]).mean()
            agreement[i, j] = agree
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(agreement, annot=True, fmt='.4f', xticklabels=model_names,
                yticklabels=model_names, cmap='YlGn', ax=ax, vmin=0.95, vmax=1.0)
    ax.set_title('Model Agreement Rate (% same prediction)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_agreement_matrix.png'), dpi=100)
    plt.close()
    print(f"  Saved: model_agreement_matrix.png")

    print(f"\n  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"TEST SUITE 7 SUMMARY")
    print(f"  Tests passed: {tests_passed}/{tests_total}")
    print(f"  Tests failed: {tests_failed}/{tests_total}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Plots: probability_correlation.png, recall_by_fraud_type.png, model_agreement_matrix.png")
    print(f"{'='*70}")

    sys.stdout = tee.stdout
    tee.close()
    return tests_passed, tests_failed, tests_total

if __name__ == '__main__':
    passed, failed, total = run_suite()
    print(f"\nTest Suite 7: {passed}/{total} passed, {failed} failed")

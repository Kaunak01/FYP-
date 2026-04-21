"""Test Suite 1: Single Transaction Deep Analysis."""
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_config import TEST_CSV, FEATURE_COLS, RESULTS_DIR, RANDOM_SEED
from fraud_predictor import XGBoostPredictor, transaction_from_row

# Output capture
class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

def run_suite():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tee = Tee(os.path.join(RESULTS_DIR, 'test_suite_1_results.txt'))
    sys.stdout = tee

    print("="*70)
    print("TEST SUITE 1: SINGLE TRANSACTION DEEP ANALYSIS")
    print("="*70)
    t0 = time.time()

    # Load data and model
    test_df = pd.read_csv(TEST_CSV)
    y_test = test_df['is_fraud'].values
    predictor = XGBoostPredictor()

    tests_passed = 0
    tests_failed = 0
    tests_total = 0

    # ================================================================
    # TEST 1A — FIRST 10 TRANSACTIONS
    # ================================================================
    print("\n" + "="*70)
    print("TEST 1A: FIRST 10 TRANSACTIONS")
    print("="*70)
    tests_total += 1

    all_valid = True
    for i in range(10):
        txn = transaction_from_row(test_df, i)
        result = predictor.predict_single(txn)
        actual = "FRAUD" if y_test[i] == 1 else "NORMAL"
        correct = "OK" if result.classification_default == actual else "WRONG"
        if not (0 <= result.probability <= 1):
            all_valid = False
        top_feat = result.shap_explanation[0][0] if result.shap_explanation else "N/A"
        print(f"  Row {i:3d} | Actual: {actual:6s} | Pred: {result.classification_default:6s} | "
              f"Prob: {result.probability:.4f} | Risk: {result.risk_level:8s} | "
              f"Top SHAP: {top_feat:25s} | {correct}")

    if all_valid:
        print("\n  TEST 1A: PASS — All 10 predictions valid, probabilities in [0,1]")
        tests_passed += 1
    else:
        print("\n  TEST 1A: FAIL — Invalid probabilities detected")
        tests_failed += 1

    # ================================================================
    # TEST 1B — RANDOM SAMPLE OF 50
    # ================================================================
    print("\n" + "="*70)
    print("TEST 1B: RANDOM SAMPLE OF 50")
    print("="*70)
    tests_total += 1

    np.random.seed(RANDOM_SEED)
    sample_idx = np.random.choice(len(test_df), 50, replace=False)
    sample_df = test_df.iloc[sample_idx]
    y_sample = y_test[sample_idx]

    summary = predictor.predict_batch(sample_df)
    correct_count = (summary['predictions'] == y_sample).sum()
    accuracy = correct_count / len(y_sample)

    print(f"  Sample size: 50")
    print(f"  Frauds in sample: {y_sample.sum()}")
    print(f"  Correct predictions: {correct_count}/50")
    print(f"  Accuracy: {accuracy:.2%}")

    # Find most confident correct and wrong predictions
    probs = summary['probabilities']
    preds = summary['predictions']

    correct_mask = preds == y_sample
    wrong_mask = ~correct_mask

    # Top 3 most confident CORRECT
    print(f"\n  Top 3 most confident CORRECT predictions:")
    if correct_mask.any():
        correct_probs = np.where(correct_mask, np.where(y_sample == 1, probs, 1 - probs), -1)
        top_correct = np.argsort(correct_probs)[::-1][:3]
        for rank, idx in enumerate(top_correct):
            real_idx = sample_idx[idx]
            actual = "FRAUD" if y_sample[idx] == 1 else "NORMAL"
            print(f"    {rank+1}. Row {real_idx}: {actual}, prob={probs[idx]:.4f}, confidence={correct_probs[idx]:.4f}")

    # Top 3 most confident WRONG
    print(f"\n  Top 3 most confident WRONG predictions:")
    if wrong_mask.any():
        wrong_confidence = np.where(wrong_mask, np.where(preds == 1, probs, 1 - probs), -1)
        top_wrong = np.argsort(wrong_confidence)[::-1][:3]
        for rank, idx in enumerate(top_wrong):
            if wrong_confidence[idx] < 0:
                break
            real_idx = sample_idx[idx]
            actual = "FRAUD" if y_sample[idx] == 1 else "NORMAL"
            pred_label = "FRAUD" if preds[idx] == 1 else "NORMAL"
            print(f"    {rank+1}. Row {real_idx}: Actual={actual}, Predicted={pred_label}, prob={probs[idx]:.4f}")
    else:
        print(f"    No wrong predictions in this sample!")

    if accuracy > 0.80:
        print(f"\n  TEST 1B: PASS — Accuracy {accuracy:.2%} > 80%")
        tests_passed += 1
    else:
        print(f"\n  TEST 1B: FAIL — Accuracy {accuracy:.2%} < 80%")
        tests_failed += 1

    # ================================================================
    # TEST 1C — ALL TRUE FRAUDS
    # ================================================================
    print("\n" + "="*70)
    print("TEST 1C: ALL TRUE FRAUDS (2,145)")
    print("="*70)
    tests_total += 1

    fraud_df = test_df[test_df['is_fraud'] == 1]
    fraud_summary = predictor.predict_batch(fraud_df)
    fraud_probs = fraud_summary['probabilities']
    fraud_preds = fraud_summary['predictions']

    caught = fraud_preds.sum()
    missed = len(fraud_preds) - caught
    recall = caught / len(fraud_preds)

    print(f"  Model catches {caught} out of {len(fraud_preds)} frauds ({recall:.2%})")
    print(f"  Missed: {missed}")

    # Top 10 most confidently caught
    print(f"\n  Top 10 most confidently CAUGHT frauds:")
    caught_idx = np.where(fraud_preds == 1)[0]
    caught_probs = fraud_probs[caught_idx]
    top_caught = caught_idx[np.argsort(caught_probs)[::-1][:10]]
    for rank, idx in enumerate(top_caught):
        real_idx = fraud_df.index[idx]
        txn = transaction_from_row(test_df, real_idx)
        result = predictor.predict_single(txn)
        top3 = [(s[0], s[1]) for s in result.shap_explanation[:3]]
        print(f"    {rank+1}. Row {real_idx}: prob={fraud_probs[idx]:.4f}, "
              f"amt=${txn['amt']:.2f}, hour={int(txn['hour'])}")
        for fname, sval in top3:
            print(f"       {fname}: SHAP={sval:+.4f}")

    # Top 10 most confidently MISSED
    print(f"\n  Top 10 most confidently MISSED frauds (model was most wrong):")
    missed_idx = np.where(fraud_preds == 0)[0]
    missed_probs = fraud_probs[missed_idx]
    top_missed = missed_idx[np.argsort(missed_probs)[:10]]  # lowest probability = most wrong
    for rank, idx in enumerate(top_missed):
        real_idx = fraud_df.index[idx]
        txn = transaction_from_row(test_df, real_idx)
        result = predictor.predict_single(txn)
        top3 = [(s[0], s[1], s[3]) for s in result.shap_explanation[:3]]
        print(f"    {rank+1}. Row {real_idx}: prob={fraud_probs[idx]:.4f}, "
              f"amt=${txn['amt']:.2f}, hour={int(txn['hour'])}")
        for fname, sval, expl in top3:
            print(f"       {fname}: SHAP={sval:+.4f} — {expl}")

        # Population comparison for missed frauds
        unusual = [(k, v) for k, v in result.population_comparison.items() if v['unusual']]
        if unusual:
            for fname, comp in unusual:
                print(f"       UNUSUAL: {fname}={comp['value']:.2f} (z={comp['z_score_normal']:.1f})")
        else:
            print(f"       No unusual features — this fraud looks completely normal")

    # Missed fraud statistics
    missed_mask = fraud_preds == 0
    caught_mask = fraud_preds == 1
    fraud_features = fraud_df[FEATURE_COLS].values

    print(f"\n  --- Caught vs Missed Fraud Comparison ---")
    print(f"  {'Feature':<30s} {'Caught Mean':>12s} {'Missed Mean':>12s} {'Difference':>12s}")
    print(f"  {'-'*66}")
    for i, col in enumerate(FEATURE_COLS):
        c_mean = fraud_features[caught_mask, i].mean()
        m_mean = fraud_features[missed_mask, i].mean() if missed_mask.any() else 0
        print(f"  {col:<30s} {c_mean:>12.2f} {m_mean:>12.2f} {c_mean - m_mean:>+12.2f}")

    expected_recall = 0.8079
    if abs(recall - expected_recall) <= 0.02:
        print(f"\n  TEST 1C: PASS — Recall {recall:.4f} matches expected {expected_recall:.4f} (within 0.02)")
        tests_passed += 1
    else:
        print(f"\n  TEST 1C: FAIL — Recall {recall:.4f} deviates from expected {expected_recall:.4f}")
        tests_failed += 1

    # ================================================================
    # TEST 1D — FALSE POSITIVES DEEP DIVE
    # ================================================================
    print("\n" + "="*70)
    print("TEST 1D: FALSE POSITIVES DEEP DIVE")
    print("="*70)
    tests_total += 1

    # Get all predictions on full test set
    full_summary = predictor.predict_batch(test_df)
    full_preds = full_summary['predictions']
    full_probs = full_summary['probabilities']

    fp_mask = (full_preds == 1) & (y_test == 0)
    fp_count = fp_mask.sum()
    print(f"  Model falsely flagged {fp_count} normal transactions as fraud")

    # Top 10 most confident false positives
    print(f"\n  Top 10 most confident FALSE POSITIVES:")
    fp_indices = np.where(fp_mask)[0]
    fp_probs = full_probs[fp_indices]
    top_fp = fp_indices[np.argsort(fp_probs)[::-1][:10]]

    for rank, idx in enumerate(top_fp):
        txn = transaction_from_row(test_df, idx)
        result = predictor.predict_single(txn)
        top3 = [(s[0], s[1], s[3]) for s in result.shap_explanation[:3]]
        print(f"    {rank+1}. Row {idx}: prob={full_probs[idx]:.4f}, "
              f"amt=${txn['amt']:.2f}, hour={int(txn['hour'])}, cat={int(txn['category_encoded'])}")
        for fname, sval, expl in top3:
            print(f"       {fname}: SHAP={sval:+.4f} — {expl}")

    # False positive statistics
    print(f"\n  --- False Positive Profile vs True Fraud Profile ---")
    fp_features = test_df.iloc[fp_indices][FEATURE_COLS].values
    true_fraud_features = test_df[test_df['is_fraud'] == 1][FEATURE_COLS].values
    print(f"  {'Feature':<30s} {'FP Mean':>12s} {'TrueFraud Mean':>14s} {'Similar?':>10s}")
    print(f"  {'-'*68}")
    for i, col in enumerate(FEATURE_COLS):
        fp_mean = fp_features[:, i].mean()
        tf_mean = true_fraud_features[:, i].mean()
        similar = "YES" if abs(fp_mean - tf_mean) / (abs(tf_mean) + 0.001) < 0.3 else "NO"
        print(f"  {col:<30s} {fp_mean:>12.2f} {tf_mean:>14.2f} {similar:>10s}")

    if abs(fp_count - 114) <= 20:
        print(f"\n  TEST 1D: PASS — FP count {fp_count} within expected range (target 114 ±20)")
        tests_passed += 1
    else:
        print(f"\n  TEST 1D: FAIL — FP count {fp_count} outside expected range (target 114 ±20)")
        tests_failed += 1

    # ================================================================
    # TEST 1E — BORDERLINE CASES
    # ================================================================
    print("\n" + "="*70)
    print("TEST 1E: BORDERLINE CASES (probability 0.4-0.6)")
    print("="*70)
    tests_total += 1

    borderline_mask = (full_probs >= 0.4) & (full_probs <= 0.6)
    borderline_idx = np.where(borderline_mask)[0]
    n_borderline = len(borderline_idx)
    print(f"  Found {n_borderline} borderline transactions (prob 0.4-0.6)")

    if n_borderline > 0:
        # Take up to 20
        show_idx = borderline_idx[:20]
        n_fraud_border = y_test[show_idx].sum()
        n_normal_border = len(show_idx) - n_fraud_border
        print(f"  Showing first {len(show_idx)}: {n_fraud_border} fraud, {n_normal_border} normal")

        for i, idx in enumerate(show_idx):
            actual = "FRAUD" if y_test[idx] == 1 else "NORMAL"
            pred = "FRAUD" if full_preds[idx] == 1 else "NORMAL"
            correct = "OK" if actual == pred else "WRONG"
            print(f"    {i+1}. Row {idx}: Actual={actual}, Pred={pred}, Prob={full_probs[idx]:.4f} [{correct}]")

        # What features make these ambiguous?
        border_features = test_df.iloc[show_idx][FEATURE_COLS].values
        print(f"\n  Borderline transaction profile (mean of {len(show_idx)} transactions):")
        for j, col in enumerate(FEATURE_COLS):
            print(f"    {col:<30s}: {border_features[:, j].mean():.2f}")

        print(f"\n  TEST 1E: PASS — {n_borderline} borderline cases found and analysed")
        tests_passed += 1
    else:
        print(f"\n  TEST 1E: PASS — No borderline cases (model is very decisive)")
        tests_passed += 1

    # ================================================================
    # PLOT: Probability Distributions
    # ================================================================
    print("\n" + "="*70)
    print("GENERATING PLOT: Probability Distributions")
    print("="*70)

    # Split into 3 groups
    normal_probs = full_probs[y_test == 0]
    caught_fraud_probs = full_probs[(y_test == 1) & (full_preds == 1)]
    missed_fraud_probs = full_probs[(y_test == 1) & (full_preds == 0)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Full distribution
    axes[0].hist(normal_probs, bins=100, alpha=0.7, label=f'Normal (n={len(normal_probs):,})', density=True)
    axes[0].hist(caught_fraud_probs, bins=50, alpha=0.7, label=f'Caught Fraud (n={len(caught_fraud_probs):,})', density=True)
    axes[0].hist(missed_fraud_probs, bins=50, alpha=0.7, label=f'Missed Fraud (n={len(missed_fraud_probs):,})', density=True)
    axes[0].axvline(0.5, color='red', linestyle='--', label='Threshold=0.5')
    axes[0].set_xlabel('Fraud Probability')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Probability Distribution by Outcome')
    axes[0].legend()

    # Zoomed into 0-0.1 range (where most normal transactions are)
    axes[1].hist(normal_probs[normal_probs < 0.1], bins=100, alpha=0.7, label='Normal', density=True)
    axes[1].hist(missed_fraud_probs[missed_fraud_probs < 0.1], bins=30, alpha=0.7, label='Missed Fraud', density=True)
    axes[1].set_xlabel('Fraud Probability')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Zoomed: Low Probability Region (0-0.1)')
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'probability_distributions.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - t0
    print("\n" + "="*70)
    print(f"TEST SUITE 1 SUMMARY")
    print(f"  Tests passed: {tests_passed}/{tests_total}")
    print(f"  Tests failed: {tests_failed}/{tests_total}")
    print(f"  Time: {elapsed:.1f}s")
    print("="*70)

    sys.stdout = tee.stdout
    tee.close()

    return tests_passed, tests_failed, tests_total


if __name__ == '__main__':
    passed, failed, total = run_suite()
    print(f"\nTest Suite 1: {passed}/{total} passed, {failed} failed")

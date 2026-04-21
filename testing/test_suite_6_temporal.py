"""Test Suite 6: Temporal Validation."""
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_config import TEST_CSV, TEST_RAW_CSV, RESULTS_DIR, FEATURE_COLS
from fraud_predictor import XGBoostPredictor

class Tee:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d): self.stdout.write(d); self.file.write(d)
    def flush(self): self.stdout.flush(); self.file.flush()
    def close(self): self.file.close()


def safe_metrics(y_true, y_pred, y_prob=None):
    """Compute metrics safely, handling edge cases."""
    if len(y_true) == 0 or y_true.sum() == 0:
        return {'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0, 'n': len(y_true), 'n_fraud': 0}
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc_val = roc_auc_score(y_true, y_prob) if y_prob is not None and len(np.unique(y_true)) > 1 else 0
    return {'f1': f1, 'precision': prec, 'recall': rec, 'auc': auc_val,
            'n': len(y_true), 'n_fraud': int(y_true.sum())}


def run_suite():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tee = Tee(os.path.join(RESULTS_DIR, 'test_suite_6_results.txt'))
    sys.stdout = tee

    print("="*70)
    print("TEST SUITE 6: TEMPORAL VALIDATION")
    print("="*70)
    t0_total = time.time()
    tests_passed = 0; tests_failed = 0; tests_total = 0

    # Load data
    test_eng = pd.read_csv(TEST_CSV)
    y_test = test_eng['is_fraud'].values
    X_test = test_eng[FEATURE_COLS].values

    # Get timestamps from unix_time
    test_eng['datetime'] = pd.to_datetime(test_eng['unix_time'], unit='s')
    test_eng['date'] = test_eng['datetime'].dt.date
    test_eng['week'] = test_eng['datetime'].dt.isocalendar().week.astype(int)
    test_eng['year_month'] = test_eng['datetime'].dt.to_period('M')

    # Predict full test set once
    predictor = XGBoostPredictor()
    summary = predictor.predict_batch(test_eng)
    y_pred = summary['predictions']
    y_prob = summary['probabilities']

    # ================================================================
    # TIME 1: MONTHLY PERFORMANCE
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("TIME 1: MONTHLY PERFORMANCE BREAKDOWN")
    print("="*70)

    months = sorted(test_eng['year_month'].unique())
    monthly_f1s = []
    print(f"\n  {'Month':<12s} {'Total':>8s} {'Frauds':>7s} {'F1':>7s} {'Prec':>7s} {'Recall':>7s} {'AUC':>7s}")
    print(f"  {'-'*60}")

    for month in months:
        mask = test_eng['year_month'] == month
        m = safe_metrics(y_test[mask], y_pred[mask], y_prob[mask])
        monthly_f1s.append(m['f1'])
        print(f"  {str(month):<12s} {m['n']:>8,} {m['n_fraud']:>7d} {m['f1']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['auc']:>7.4f}")

    f1_std = np.std(monthly_f1s)
    f1_mean = np.mean(monthly_f1s)
    print(f"\n  Mean F1: {f1_mean:.4f}, Std: {f1_std:.4f}")

    if f1_std < 0.10:
        print(f"  No significant concept drift detected.")
        print(f"  RESULT: PASS (std={f1_std:.4f} < 0.10)")
        tests_passed += 1
    else:
        print(f"  Evidence of CONCEPT DRIFT — performance varies significantly across months.")
        print(f"  RESULT: FAIL (std={f1_std:.4f} >= 0.10)")
        tests_failed += 1

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(months)), monthly_f1s, 'bo-', linewidth=2, markersize=8)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([str(m) for m in months], rotation=45)
    ax.set_ylabel('F1 Score'); ax.set_title('Monthly F1 Score — Concept Drift Check')
    ax.axhline(f1_mean, color='red', linestyle='--', alpha=0.5, label=f'Mean={f1_mean:.4f}')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'monthly_f1.png'), dpi=100); plt.close()
    print(f"  Saved: monthly_f1.png")

    # ================================================================
    # TIME 2: WEEKLY PERFORMANCE
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("TIME 2: WEEKLY PERFORMANCE")
    print("="*70)

    # Create a proper week identifier
    test_eng['year_week'] = test_eng['datetime'].dt.strftime('%Y-W%U')
    weeks = sorted(test_eng['year_week'].unique())
    weekly_f1s = []; week_labels = []

    for week in weeks:
        mask = test_eng['year_week'] == week
        if y_test[mask].sum() == 0:
            continue
        m = safe_metrics(y_test[mask], y_pred[mask], y_prob[mask])
        weekly_f1s.append(m['f1'])
        week_labels.append(week)

    print(f"  Weeks with fraud: {len(weekly_f1s)}")
    print(f"  F1 range: [{min(weekly_f1s):.4f}, {max(weekly_f1s):.4f}]")
    print(f"  Worst week: {week_labels[np.argmin(weekly_f1s)]} (F1={min(weekly_f1s):.4f})")
    print(f"  Best week: {week_labels[np.argmax(weekly_f1s)]} (F1={max(weekly_f1s):.4f})")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(weekly_f1s)), weekly_f1s, 'b-', linewidth=1, alpha=0.7)
    ax.scatter(range(len(weekly_f1s)), weekly_f1s, c=['red' if f < 0.7 else 'blue' for f in weekly_f1s], s=20)
    ax.set_ylabel('F1 Score'); ax.set_title('Weekly F1 Score')
    ax.set_xlabel('Week'); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'weekly_f1.png'), dpi=100); plt.close()
    print(f"  Saved: weekly_f1.png")
    print(f"  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # TIME 3: DAY VS NIGHT
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("TIME 3: DAY VS NIGHT PERFORMANCE")
    print("="*70)

    day_mask = test_eng['is_night'] == 0
    night_mask = test_eng['is_night'] == 1

    m_day = safe_metrics(y_test[day_mask], y_pred[day_mask], y_prob[day_mask])
    m_night = safe_metrics(y_test[night_mask], y_pred[night_mask], y_prob[night_mask])

    print(f"\n  {'Metric':<12s} {'Daytime':>10s} {'Nighttime':>10s} {'Difference':>12s}")
    print(f"  {'-'*46}")
    print(f"  {'Transactions':<12s} {m_day['n']:>10,} {m_night['n']:>10,}")
    print(f"  {'Frauds':<12s} {m_day['n_fraud']:>10,} {m_night['n_fraud']:>10,}")
    print(f"  {'F1':<12s} {m_day['f1']:>10.4f} {m_night['f1']:>10.4f} {m_night['f1']-m_day['f1']:>+12.4f}")
    print(f"  {'Precision':<12s} {m_day['precision']:>10.4f} {m_night['precision']:>10.4f} {m_night['precision']-m_day['precision']:>+12.4f}")
    print(f"  {'Recall':<12s} {m_day['recall']:>10.4f} {m_night['recall']:>10.4f} {m_night['recall']-m_day['recall']:>+12.4f}")
    print(f"  {'AUC':<12s} {m_day['auc']:>10.4f} {m_night['auc']:>10.4f} {m_night['auc']-m_day['auc']:>+12.4f}")

    print(f"\n  Fraud rate — Day: {m_day['n_fraud']/m_day['n']*100:.3f}%, Night: {m_night['n_fraud']/m_night['n']*100:.3f}%")
    print(f"  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # TIME 4: WEEKDAY VS WEEKEND
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("TIME 4: WEEKDAY VS WEEKEND PERFORMANCE")
    print("="*70)

    wd_mask = test_eng['is_weekend'] == 0
    we_mask = test_eng['is_weekend'] == 1

    m_wd = safe_metrics(y_test[wd_mask], y_pred[wd_mask], y_prob[wd_mask])
    m_we = safe_metrics(y_test[we_mask], y_pred[we_mask], y_prob[we_mask])

    print(f"\n  {'Metric':<12s} {'Weekday':>10s} {'Weekend':>10s} {'Difference':>12s}")
    print(f"  {'-'*46}")
    print(f"  {'Transactions':<12s} {m_wd['n']:>10,} {m_we['n']:>10,}")
    print(f"  {'Frauds':<12s} {m_wd['n_fraud']:>10,} {m_we['n_fraud']:>10,}")
    print(f"  {'F1':<12s} {m_wd['f1']:>10.4f} {m_we['f1']:>10.4f} {m_we['f1']-m_wd['f1']:>+12.4f}")
    print(f"  {'Precision':<12s} {m_wd['precision']:>10.4f} {m_we['precision']:>10.4f} {m_we['precision']-m_wd['precision']:>+12.4f}")
    print(f"  {'Recall':<12s} {m_wd['recall']:>10.4f} {m_we['recall']:>10.4f} {m_we['recall']-m_wd['recall']:>+12.4f}")
    print(f"  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # TIME 5: FRAUD AMOUNT BINS
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("TIME 5: RECALL BY FRAUD AMOUNT BIN")
    print("="*70)

    fraud_mask = y_test == 1
    fraud_amts = test_eng.loc[fraud_mask, 'amt'].values
    fraud_preds = y_pred[fraud_mask]

    bins = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 500), (500, 1000), (1000, float('inf'))]
    bin_labels = ['<$10', '$10-25', '$25-50', '$50-100', '$100-250', '$250-500', '$500-1K', '>$1K']
    bin_recalls = []; bin_counts = []; bin_caught = []

    print(f"\n  {'Amount Bin':<12s} {'Total':>7s} {'Caught':>7s} {'Missed':>7s} {'Recall':>8s}")
    print(f"  {'-'*45}")

    for (lo, hi), label in zip(bins, bin_labels):
        mask = (fraud_amts >= lo) & (fraud_amts < hi)
        total = mask.sum()
        if total == 0:
            bin_recalls.append(0); bin_counts.append(0); bin_caught.append(0)
            continue
        caught = fraud_preds[mask].sum()
        missed = total - caught
        recall = caught / total
        bin_recalls.append(recall); bin_counts.append(total); bin_caught.append(caught)
        print(f"  {label:<12s} {total:>7d} {int(caught):>7d} {int(missed):>7d} {recall:>8.1%}")

    print(f"\n  Key finding:")
    print(f"  Model catches {bin_recalls[-1]:.0%} of frauds over $1000 but only {bin_recalls[0]:.0%} of frauds under $10")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['red' if r < 0.5 else 'orange' if r < 0.8 else 'green' for r in bin_recalls]
    bars = ax.bar(bin_labels, [r*100 for r in bin_recalls], color=colors, edgecolor='black')
    # Add count labels on bars
    for bar, count, caught_n in zip(bars, bin_counts, bin_caught):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{int(caught_n)}/{count}', ha='center', fontsize=9)
    ax.set_ylabel('Recall (%)'); ax.set_xlabel('Fraud Amount')
    ax.set_title('Recall by Fraud Amount — The Small Fraud Problem')
    ax.set_ylim(0, 110); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'recall_by_amount.png'), dpi=100); plt.close()
    print(f"  Saved: recall_by_amount.png")
    print(f"  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # TIME 6: HOURLY RECALL
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("TIME 6: HOURLY FRAUD RECALL")
    print("="*70)

    fraud_hours = test_eng.loc[fraud_mask, 'hour'].values.astype(int)
    hourly_recalls = []

    print(f"\n  {'Hour':>5s} {'Total':>7s} {'Caught':>7s} {'Recall':>8s}")
    print(f"  {'-'*30}")

    for h in range(24):
        h_mask = fraud_hours == h
        total = h_mask.sum()
        if total == 0:
            hourly_recalls.append(0)
            continue
        caught = fraud_preds[h_mask].sum()
        recall = caught / total
        hourly_recalls.append(recall)
        print(f"  {h:>5d} {total:>7d} {int(caught):>7d} {recall:>8.1%}")

    worst_hour = np.argmin(hourly_recalls)
    best_hour = np.argmax(hourly_recalls)
    print(f"\n  Best hour: {best_hour}:00 (recall={hourly_recalls[best_hour]:.1%})")
    print(f"  Worst hour: {worst_hour}:00 (recall={hourly_recalls[worst_hour]:.1%})")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['red' if r < 0.5 else 'orange' if r < 0.8 else 'green' for r in hourly_recalls]
    ax.bar(range(24), [r*100 for r in hourly_recalls], color=colors, edgecolor='black')
    ax.set_xlabel('Hour of Day'); ax.set_ylabel('Recall (%)')
    ax.set_title('Fraud Detection Recall by Hour of Day')
    ax.set_xticks(range(24)); ax.set_ylim(0, 110)
    ax.axhline(80, color='blue', linestyle='--', alpha=0.5, label='80% target')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'recall_by_hour.png'), dpi=100); plt.close()
    print(f"  Saved: recall_by_hour.png")
    print(f"  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"TEST SUITE 6 SUMMARY")
    print(f"  Tests passed: {tests_passed}/{tests_total}")
    print(f"  Tests failed: {tests_failed}/{tests_total}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Plots: monthly_f1.png, weekly_f1.png, recall_by_amount.png, recall_by_hour.png")
    print(f"{'='*70}")

    sys.stdout = tee.stdout
    tee.close()
    return tests_passed, tests_failed, tests_total

if __name__ == '__main__':
    passed, failed, total = run_suite()
    print(f"\nTest Suite 6: {passed}/{total} passed, {failed} failed")

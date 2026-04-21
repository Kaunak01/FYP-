"""Test Suite 3: Decision Boundary Analysis."""
import sys, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_config import RESULTS_DIR, FEATURE_COLS, TRAINING_STATS, get_median_transaction
from fraud_predictor import XGBoostPredictor

class Tee:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d): self.stdout.write(d); self.file.write(d)
    def flush(self): self.stdout.flush(); self.file.flush()
    def close(self): self.file.close()


def run_suite():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tee = Tee(os.path.join(RESULTS_DIR, 'test_suite_3_results.txt'))
    sys.stdout = tee

    print("="*70)
    print("TEST SUITE 3: DECISION BOUNDARY ANALYSIS")
    print("="*70)
    t0 = time.time()

    predictor = XGBoostPredictor()
    median_txn = get_median_transaction()

    # Predict the median transaction first
    from fraud_predictor import PredictionResult
    raw = np.array([median_txn[f] for f in FEATURE_COLS], dtype=np.float64)
    median_prob = predictor.model.predict_proba(raw.reshape(1, -1))[0, 1]
    print(f"\nMedian transaction probability: {median_prob:.6f}")
    print(f"Median transaction features:")
    for f in FEATURE_COLS:
        print(f"  {f:<35s} = {median_txn[f]:.4f}")

    # Sweep each feature
    boundary_results = []
    fig_all, axes_all = plt.subplots(4, 4, figsize=(24, 20))
    axes_flat = axes_all.flat

    tests_total = 1
    tests_passed = 0
    any_crash = False

    for feat_idx, feat_name in enumerate(FEATURE_COLS):
        stats = TRAINING_STATS['stats'][feat_name]['all']
        feat_min = stats['min']
        feat_max = stats['max']

        # For binary features, just test 0 and 1
        if feat_name in ('is_weekend', 'is_night', 'gender_encoded'):
            sweep_values = [0.0, 1.0]
        elif feat_name in ('category_encoded',):
            sweep_values = list(range(14))
        elif feat_name in ('day_of_week_encoded',):
            sweep_values = list(range(7))
        elif feat_name == 'hour':
            sweep_values = list(range(24))
        elif feat_name == 'month':
            sweep_values = list(range(1, 13))
        else:
            # Continuous: 100 steps from min to max
            sweep_values = np.linspace(feat_min, feat_max, 100).tolist()

        probabilities = []
        for val in sweep_values:
            txn = median_txn.copy()
            txn[feat_name] = float(val)
            features = np.array([txn[f] for f in FEATURE_COLS], dtype=np.float64)
            try:
                prob = predictor.model.predict_proba(features.reshape(1, -1))[0, 1]
                probabilities.append(prob)
            except Exception as e:
                probabilities.append(np.nan)
                any_crash = True

        # Find boundary crossing (where prob crosses 0.5)
        boundary_val = None
        boundary_z = None
        for i in range(1, len(probabilities)):
            if probabilities[i-1] < 0.5 <= probabilities[i] or probabilities[i-1] >= 0.5 > probabilities[i]:
                # Linear interpolation
                p0, p1 = probabilities[i-1], probabilities[i]
                v0, v1 = sweep_values[i-1], sweep_values[i]
                if p1 != p0:
                    boundary_val = v0 + (0.5 - p0) * (v1 - v0) / (p1 - p0)
                else:
                    boundary_val = (v0 + v1) / 2
                # Z-score of boundary
                mean = stats['mean']
                std = stats['std']
                boundary_z = (boundary_val - mean) / std if std > 0 else 0
                break

        prob_range = max(probabilities) - min(probabilities)
        boundary_results.append({
            'feature': feat_name,
            'boundary_val': boundary_val,
            'boundary_z': boundary_z,
            'prob_min': min(probabilities),
            'prob_max': max(probabilities),
            'prob_range': prob_range,
        })

        print(f"\n  {feat_name}:")
        print(f"    Sweep range: [{feat_min:.2f}, {feat_max:.2f}]")
        print(f"    Prob range: [{min(probabilities):.4f}, {max(probabilities):.4f}] (delta={prob_range:.4f})")
        if boundary_val is not None:
            print(f"    BOUNDARY at {feat_name}={boundary_val:.4f} (z={boundary_z:.2f} std from mean)")
        else:
            print(f"    No boundary crossing (never crosses 0.5)")

        # Individual plot
        ax = axes_flat[feat_idx]
        ax.plot(sweep_values, probabilities, 'b-', linewidth=2)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.7)
        if boundary_val is not None:
            ax.axvline(boundary_val, color='green', linestyle=':', alpha=0.7)
            ax.set_title(f'{feat_name}\nBoundary={boundary_val:.2f}', fontsize=10)
        else:
            ax.set_title(f'{feat_name}\nNo boundary', fontsize=10)
        ax.set_xlabel(feat_name, fontsize=8)
        ax.set_ylabel('Prob', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Save individual plot
        fig_ind, ax_ind = plt.subplots(figsize=(8, 5))
        ax_ind.plot(sweep_values, probabilities, 'b-', linewidth=2)
        ax_ind.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Threshold=0.5')
        if boundary_val is not None:
            ax_ind.axvline(boundary_val, color='green', linestyle=':', linewidth=2,
                          label=f'Boundary={boundary_val:.2f}')
        ax_ind.set_xlabel(feat_name)
        ax_ind.set_ylabel('Fraud Probability')
        ax_ind.set_title(f'Decision Boundary for {feat_name}')
        ax_ind.legend()
        ax_ind.set_ylim(-0.05, 1.05)
        ax_ind.grid(True, alpha=0.3)
        safe_name = feat_name.replace('/', '_')
        plt.tight_layout()
        fig_ind.savefig(os.path.join(RESULTS_DIR, f'boundary_{safe_name}.png'), dpi=100)
        plt.close(fig_ind)

    # Hide unused subplots (14 features, 16 slots)
    for i in range(len(FEATURE_COLS), 16):
        axes_flat[i].set_visible(False)

    plt.suptitle('Decision Boundaries — All 14 Features (Median Transaction Base)', fontsize=14)
    plt.tight_layout()
    fig_all.savefig(os.path.join(RESULTS_DIR, 'all_boundaries.png'), dpi=100, bbox_inches='tight')
    plt.close(fig_all)

    # Summary table
    print("\n" + "="*70)
    print("DECISION BOUNDARY SUMMARY")
    print("="*70)
    print(f"\n{'Feature':<35s} {'Boundary':>10s} {'Z-Score':>8s} {'ProbRange':>10s} {'Interpretation'}")
    print("-" * 100)
    features_with_boundary = 0
    for r in boundary_results:
        if r['boundary_val'] is not None:
            features_with_boundary += 1
            # Interpretation
            if r['feature'] == 'amt':
                interp = f"Transactions over ${r['boundary_val']:.0f} flagged (all else avg)"
            elif r['feature'] == 'amount_velocity_1h':
                interp = f"Spending over ${r['boundary_val']:.0f}/hr flagged"
            elif r['feature'] == 'velocity_1h':
                interp = f"More than {r['boundary_val']:.1f} txns/hr flagged"
            elif r['feature'] == 'hour':
                interp = f"Transactions after hour {r['boundary_val']:.0f} more likely fraud"
            elif r['feature'] == 'is_night':
                interp = f"Nighttime transactions flagged"
            else:
                interp = f"Values above {r['boundary_val']:.2f} increase fraud risk"
            print(f"{r['feature']:<35s} {r['boundary_val']:>10.2f} {r['boundary_z']:>8.2f} "
                  f"{r['prob_range']:>10.4f} {interp}")
        else:
            print(f"{r['feature']:<35s} {'N/A':>10s} {'N/A':>8s} {r['prob_range']:>10.4f} "
                  f"Cannot flip prediction alone")

    print(f"\nFeatures that can individually flip prediction: {features_with_boundary}/{len(FEATURE_COLS)}")

    # Feature influence ranking by probability range
    print(f"\nFeature influence ranking (by probability range when varied alone):")
    sorted_results = sorted(boundary_results, key=lambda x: x['prob_range'], reverse=True)
    for i, r in enumerate(sorted_results):
        print(f"  {i+1:2d}. {r['feature']:<35s} range={r['prob_range']:.4f}")

    # Pass/fail
    if not any_crash:
        print(f"\n  TEST SUITE 3: PASS - All {len(FEATURE_COLS)} feature sweeps completed without errors")
        tests_passed = 1
    else:
        print(f"\n  TEST SUITE 3: FAIL - Some sweeps crashed")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"TEST SUITE 3 SUMMARY")
    print(f"  Tests passed: {tests_passed}/{tests_total}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Plots saved: all_boundaries.png + 14 individual boundary_*.png")
    print(f"{'='*70}")

    sys.stdout = tee.stdout
    tee.close()
    return tests_passed, tests_total - tests_passed, tests_total


if __name__ == '__main__':
    passed, failed, total = run_suite()
    print(f"\nTest Suite 3: {passed}/{total} passed, {failed} failed")

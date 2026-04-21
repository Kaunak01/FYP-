"""Drift Detector: monitors model performance degradation using PSI (Population Stability Index)."""
import json
import math
import logging
import numpy as np
from app.config import STATS_FILES, FEATURE_COLS

logger = logging.getLogger(__name__)

# Load baseline stats
with open(STATS_FILES['training_stats']) as f:
    _BASELINE_STATS = json.load(f)


class DriftDetector:
    """Monitors model and data drift using PSI (Population Stability Index).

    PSI < 0.1: No drift
    0.1 <= PSI < 0.25: Moderate drift (warning)
    PSI >= 0.25: Significant drift (retrain needed)

    This is the banking industry standard for monitoring credit and fraud models.
    """

    def __init__(self, baseline_metrics=None):
        self.baseline_metrics = baseline_metrics or {
            'f1': 0.8646, 'precision': 0.9297, 'recall': 0.8079, 'roc_auc': 0.9972
        }
        self.baseline_stats = _BASELINE_STATS

    def compute_psi(self, expected, actual, n_bins=10):
        """Population Stability Index — coded from scratch.

        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

        Args:
            expected: array of values from baseline (training) distribution
            actual: array of values from current (production) distribution
            n_bins: number of bins for histogram comparison

        Returns:
            float: PSI value
        """
        # Create bins from expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        # Remove duplicate breakpoints
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            return 0.0

        # Compute bin percentages
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        # Replace zeros with small value to avoid log(0)
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

        # PSI formula
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def interpret_psi(self, psi_value):
        """Interpret PSI value."""
        if psi_value < 0.1:
            return 'GREEN', 'No significant drift detected'
        elif psi_value < 0.25:
            return 'YELLOW', 'Moderate drift detected - monitor closely'
        else:
            return 'RED', 'Significant drift - model retraining recommended'

    def check_prediction_drift(self, recent_f1, recent_precision=None, recent_recall=None):
        """Compare recent performance metrics to baseline.

        Returns:
            dict with drift status and details
        """
        f1_drop = self.baseline_metrics['f1'] - recent_f1

        if f1_drop > 0.10:
            status = 'RED'
            message = f'CRITICAL: F1 dropped by {f1_drop:.4f} (baseline={self.baseline_metrics["f1"]:.4f}, current={recent_f1:.4f}). Model needs retraining.'
        elif f1_drop > 0.05:
            status = 'YELLOW'
            message = f'WARNING: F1 dropped by {f1_drop:.4f} (baseline={self.baseline_metrics["f1"]:.4f}, current={recent_f1:.4f}). Monitor closely.'
        elif f1_drop > 0:
            status = 'GREEN'
            message = f'Minor F1 change of {f1_drop:.4f}. Within acceptable range.'
        else:
            status = 'GREEN'
            message = f'F1 improved by {abs(f1_drop):.4f}. No degradation detected.'

        return {
            'status': status,
            'message': message,
            'baseline_f1': self.baseline_metrics['f1'],
            'current_f1': recent_f1,
            'f1_change': -f1_drop,
        }

    def check_feature_drift(self, recent_data, feature_names=None):
        """Check if feature distributions have shifted from training baseline.

        Args:
            recent_data: numpy array or dict of {feature_name: array}
            feature_names: list of feature names (if recent_data is numpy array)

        Returns:
            dict with per-feature PSI values and overall status
        """
        results = {}
        overall_status = 'GREEN'

        if isinstance(recent_data, dict):
            features_to_check = recent_data
        else:
            features_to_check = {}
            names = feature_names or FEATURE_COLS
            for i, fname in enumerate(names):
                if i < recent_data.shape[1]:
                    features_to_check[fname] = recent_data[:, i]

        for fname, actual_values in features_to_check.items():
            if fname not in self.baseline_stats['stats']:
                continue

            stats = self.baseline_stats['stats'][fname]['all']

            # Reconstruct approximate baseline distribution from stored stats
            # Use mean and std to generate a representative baseline
            np.random.seed(42)
            baseline_approx = np.random.normal(stats['mean'], stats['std'], size=len(actual_values))
            # Clip to observed range
            baseline_approx = np.clip(baseline_approx, stats['min'], stats['max'])

            psi = self.compute_psi(baseline_approx, actual_values)
            psi_status, psi_msg = self.interpret_psi(psi)

            results[fname] = {
                'psi': psi,
                'status': psi_status,
                'message': psi_msg,
                'baseline_mean': stats['mean'],
                'current_mean': float(np.mean(actual_values)),
                'mean_shift': float(np.mean(actual_values) - stats['mean']),
            }

            # Update overall status to worst
            if psi_status == 'RED':
                overall_status = 'RED'
            elif psi_status == 'YELLOW' and overall_status == 'GREEN':
                overall_status = 'YELLOW'

        return {
            'overall_status': overall_status,
            'features': results,
            'drifted_features': [f for f, r in results.items() if r['status'] != 'GREEN'],
        }

    def check_prediction_distribution(self, recent_probabilities, baseline_fraud_rate=0.0039):
        """Check if the distribution of predicted probabilities has shifted.

        Args:
            recent_probabilities: array of predicted probabilities
            baseline_fraud_rate: expected fraud rate in test set

        Returns:
            dict with distribution check results
        """
        flag_rate = float(np.mean(recent_probabilities >= 0.5))
        avg_prob = float(np.mean(recent_probabilities))

        issues = []
        status = 'GREEN'

        if flag_rate > 0.05:  # More than 5% flagged
            issues.append(f'Flagging {flag_rate:.1%} of transactions (expected ~{baseline_fraud_rate:.1%}). Model may be over-sensitive.')
            status = 'YELLOW'

        if flag_rate == 0 and len(recent_probabilities) > 100:
            issues.append('No transactions flagged. Model may be broken or data format has changed.')
            status = 'RED'

        if avg_prob > 0.1:
            issues.append(f'Average probability {avg_prob:.4f} is unusually high (expected ~0.002).')
            status = 'YELLOW'

        if not issues:
            issues.append(f'Distribution looks normal. Flag rate: {flag_rate:.3%}, avg probability: {avg_prob:.4f}')

        return {
            'status': status,
            'flag_rate': flag_rate,
            'avg_probability': avg_prob,
            'issues': issues,
        }

    def generate_report(self, prediction_drift=None, feature_drift=None, distribution_check=None):
        """Generate a complete drift monitoring report.

        Returns:
            dict with overall status and all check results
        """
        checks = {}
        overall = 'GREEN'

        if prediction_drift:
            checks['prediction_drift'] = prediction_drift
            if prediction_drift['status'] == 'RED':
                overall = 'RED'
            elif prediction_drift['status'] == 'YELLOW' and overall != 'RED':
                overall = 'YELLOW'

        if feature_drift:
            checks['feature_drift'] = feature_drift
            if feature_drift['overall_status'] == 'RED':
                overall = 'RED'
            elif feature_drift['overall_status'] == 'YELLOW' and overall != 'RED':
                overall = 'YELLOW'

        if distribution_check:
            checks['distribution'] = distribution_check
            if distribution_check['status'] == 'RED':
                overall = 'RED'
            elif distribution_check['status'] == 'YELLOW' and overall != 'RED':
                overall = 'YELLOW'

        recommendations = []
        if overall == 'RED':
            recommendations.append('Immediate model retraining required')
            recommendations.append('Investigate data pipeline for format changes')
        elif overall == 'YELLOW':
            recommendations.append('Schedule model evaluation within 1 week')
            recommendations.append('Monitor drifted features daily')

        return {
            'overall_status': overall,
            'checks': checks,
            'recommendations': recommendations,
            'summary': f'Drift status: {overall}' + (f' - {len(recommendations)} action(s) recommended' if recommendations else ''),
        }


# ---- Quick test ----
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pandas as pd

    detector = DriftDetector()

    # Test 1: PSI computation
    print("="*60)
    print("TEST 1: PSI Computation")
    print("="*60)
    np.random.seed(42)
    baseline = np.random.normal(100, 20, 10000)
    no_drift = np.random.normal(100, 20, 10000)
    mild_drift = np.random.normal(110, 20, 10000)
    severe_drift = np.random.normal(150, 30, 10000)

    psi_none = detector.compute_psi(baseline, no_drift)
    psi_mild = detector.compute_psi(baseline, mild_drift)
    psi_severe = detector.compute_psi(baseline, severe_drift)

    status_none, msg_none = detector.interpret_psi(psi_none)
    status_mild, msg_mild = detector.interpret_psi(psi_mild)
    status_severe, msg_severe = detector.interpret_psi(psi_severe)

    print(f"  No drift:     PSI={psi_none:.4f} [{status_none}] {msg_none}")
    print(f"  Mild drift:   PSI={psi_mild:.4f} [{status_mild}] {msg_mild}")
    print(f"  Severe drift: PSI={psi_severe:.4f} [{status_severe}] {msg_severe}")

    # Test 2: Prediction drift
    print("\n" + "="*60)
    print("TEST 2: Prediction Drift Detection")
    print("="*60)
    for test_f1, desc in [(0.86, "Normal (slight drop)"), (0.80, "Warning"), (0.72, "Critical")]:
        result = detector.check_prediction_drift(test_f1)
        print(f"  F1={test_f1}: [{result['status']}] {result['message']}")

    # Test 3: Feature drift on real data
    print("\n" + "="*60)
    print("TEST 3: Feature Drift (train vs test — should show no drift)")
    print("="*60)
    test_df = pd.read_csv('fraudTest_engineered.csv')
    drop_cols = ['is_fraud', 'unix_time']
    feature_cols = [c for c in test_df.columns if c not in drop_cols]
    X_test = test_df[feature_cols].values

    feature_result = detector.check_feature_drift(X_test, feature_cols)
    print(f"  Overall: [{feature_result['overall_status']}]")
    print(f"  Drifted features: {feature_result['drifted_features']}")
    for fname, r in feature_result['features'].items():
        if r['status'] != 'GREEN':
            print(f"    {fname}: PSI={r['psi']:.4f} [{r['status']}] shift={r['mean_shift']:.2f}")

    # Test 4: Simulate drift by adding $100 to all amounts
    print("\n" + "="*60)
    print("TEST 4: Simulated Drift (+$100 to all amounts)")
    print("="*60)
    X_drifted = X_test.copy()
    amt_idx = feature_cols.index('amt')
    X_drifted[:, amt_idx] += 100  # Add $100 to every transaction

    drift_result = detector.check_feature_drift(X_drifted, feature_cols)
    print(f"  Overall: [{drift_result['overall_status']}]")
    print(f"  Drifted features: {drift_result['drifted_features']}")
    amt_drift = drift_result['features'].get('amt', {})
    print(f"  amt PSI: {amt_drift.get('psi', 0):.4f} [{amt_drift.get('status', '?')}]")
    print(f"  amt shift: baseline_mean=${amt_drift.get('baseline_mean', 0):.2f} -> current_mean=${amt_drift.get('current_mean', 0):.2f}")

    # Test 5: Prediction distribution check
    print("\n" + "="*60)
    print("TEST 5: Prediction Distribution Check")
    print("="*60)
    normal_probs = np.random.exponential(0.001, 10000)  # Mostly very low
    normal_probs = np.clip(normal_probs, 0, 1)
    dist_ok = detector.check_prediction_distribution(normal_probs)
    print(f"  Normal distribution: [{dist_ok['status']}] {dist_ok['issues'][0]}")

    broken_probs = np.random.uniform(0.3, 0.8, 10000)  # Way too high
    dist_bad = detector.check_prediction_distribution(broken_probs)
    print(f"  Broken distribution: [{dist_bad['status']}] {dist_bad['issues'][0]}")

    # Test 6: Full report
    print("\n" + "="*60)
    print("TEST 6: Full Drift Report")
    print("="*60)
    report = detector.generate_report(
        prediction_drift=detector.check_prediction_drift(0.85),
        feature_drift=drift_result,
        distribution_check=dist_ok,
    )
    print(f"  Overall: [{report['overall_status']}]")
    print(f"  Summary: {report['summary']}")
    for rec in report['recommendations']:
        print(f"  -> {rec}")

    print("\n  ALL DRIFT DETECTOR TESTS COMPLETE")

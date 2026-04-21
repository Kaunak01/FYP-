"""Postprocessor: transforms raw model output into human-readable explanations."""
import json
import logging
from app.config import STATS_FILES, FEATURE_COLS

logger = logging.getLogger(__name__)

# Load training stats once
with open(STATS_FILES['training_stats']) as f:
    _STATS = json.load(f)

with open(STATS_FILES['category_mapping']) as f:
    _CAT = json.load(f)
    _CODE_TO_NAME = {int(k): v for k, v in _CAT['code_to_name'].items()}

# Human-readable feature names
_FEATURE_LABELS = {
    'amt': 'Transaction amount',
    'city_pop': 'City population',
    'hour': 'Hour of day',
    'month': 'Month',
    'distance_cardholder_merchant': 'Distance to merchant',
    'age': 'Cardholder age',
    'is_weekend': 'Weekend transaction',
    'is_night': 'Nighttime transaction',
    'velocity_1h': 'Transactions in last hour',
    'velocity_24h': 'Transactions in last 24h',
    'amount_velocity_1h': 'Amount spent in last hour',
    'category_encoded': 'Merchant category',
    'gender_encoded': 'Cardholder gender',
    'day_of_week_encoded': 'Day of week',
    'recon_error': 'Anomaly score (autoencoder)',
    'bds_amount': 'Spending deviation score',
    'bds_time': 'Time deviation score',
    'bds_freq': 'Frequency deviation score',
    'bds_category': 'Category deviation score',
}

# Action recommendations by risk level
_RECOMMENDATIONS = {
    'CRITICAL': {
        'action': 'BLOCK IMMEDIATELY',
        'detail': 'Block transaction and contact cardholder immediately.',
        'color': '#e74c3c',
    },
    'HIGH': {
        'action': 'HOLD FOR REVIEW',
        'detail': 'Hold for manual review. Do not process until analyst confirms.',
        'color': '#e67e22',
    },
    'MEDIUM': {
        'action': 'PROCESS WITH MONITORING',
        'detail': 'Process but flag for end-of-day review by analyst.',
        'color': '#f39c12',
    },
    'LOW': {
        'action': 'PROCESS NORMALLY',
        'detail': 'No issues detected. Process normally.',
        'color': '#2ecc71',
    },
}


class Postprocessor:
    """Generates human-readable prediction explanations."""

    def format_prediction(self, features, probability, risk_level, classification,
                          shap_values=None, shap_feature_names=None,
                          rule_result=None, metadata=None):
        """Generate a complete human-readable prediction report.

        Returns:
            dict with: risk_assessment, key_factors, recommendation, summary, shap_chart_data
        """
        report = {}

        # ---- 1. Risk Assessment ----
        pct = probability * 100
        report['risk_assessment'] = {
            'probability': probability,
            'probability_pct': f'{pct:.1f}%',
            'risk_level': risk_level,
            'classification': classification,
            'headline': self._headline(risk_level, probability, classification),
            'color': _RECOMMENDATIONS.get(risk_level, _RECOMMENDATIONS['LOW'])['color'],
        }

        # ---- 2. Key Factors (from SHAP) ----
        toward_fraud = []
        away_fraud = []

        if shap_values is not None and shap_feature_names is not None:
            sorted_idx = sorted(range(len(shap_values)),
                                key=lambda i: abs(shap_values[i]), reverse=True)

            for idx in sorted_idx[:8]:  # Top 8 factors
                fname = shap_feature_names[idx]
                sval = shap_values[idx]
                fval = features.get(fname, 0) if isinstance(features, dict) else features[idx]

                explanation = self._explain_feature(fname, fval, sval)
                entry = {
                    'feature': fname,
                    'label': _FEATURE_LABELS.get(fname, fname),
                    'shap_value': float(sval),
                    'feature_value': float(fval),
                    'direction': 'toward_fraud' if sval > 0 else 'away_from_fraud',
                    'explanation': explanation,
                }

                if sval > 0:
                    toward_fraud.append(entry)
                else:
                    away_fraud.append(entry)

        report['toward_fraud'] = toward_fraud
        report['away_from_fraud'] = away_fraud

        # ---- 3. Recommendation ----
        rec = _RECOMMENDATIONS.get(risk_level, _RECOMMENDATIONS['LOW'])
        report['recommendation'] = {
            'action': rec['action'],
            'detail': rec['detail'],
            'color': rec['color'],
        }

        # ---- 4. Rule triggers ----
        report['rules'] = []
        if rule_result and rule_result.any_triggered:
            for name, level, reason in rule_result.triggered_rules:
                report['rules'].append({
                    'name': name,
                    'level': level,
                    'reason': reason,
                })

        # ---- 5. Population comparison ----
        report['population_comparison'] = []
        feat_dict = features if isinstance(features, dict) else {FEATURE_COLS[i]: features[i] for i in range(len(FEATURE_COLS))}

        for fname in FEATURE_COLS:
            if fname not in _STATS['stats']:
                continue
            val = feat_dict.get(fname, 0)
            normal_mean = _STATS['stats'][fname]['normal']['mean']
            normal_std = _STATS['stats'][fname]['normal']['std']
            fraud_mean = _STATS['stats'][fname]['fraud']['mean']

            z_normal = (val - normal_mean) / normal_std if normal_std > 0 else 0
            unusual = 'VERY UNUSUAL' if abs(z_normal) > 3 else 'UNUSUAL' if abs(z_normal) > 2 else ''

            report['population_comparison'].append({
                'feature': fname,
                'label': _FEATURE_LABELS.get(fname, fname),
                'value': float(val),
                'normal_mean': float(normal_mean),
                'fraud_mean': float(fraud_mean),
                'z_score': float(z_normal),
                'flag': unusual,
            })

        # ---- 6. SHAP chart data (for Chart.js) ----
        report['shap_chart_data'] = None
        if shap_values is not None and shap_feature_names is not None:
            sorted_idx = sorted(range(len(shap_values)),
                                key=lambda i: abs(shap_values[i]), reverse=True)[:10]
            report['shap_chart_data'] = {
                'labels': [_FEATURE_LABELS.get(shap_feature_names[i], shap_feature_names[i]) for i in sorted_idx],
                'values': [float(shap_values[i]) for i in sorted_idx],
                'colors': ['#e74c3c' if shap_values[i] > 0 else '#2ecc71' for i in sorted_idx],
                'feature_values': [float(features.get(shap_feature_names[i], 0) if isinstance(features, dict) else features[i]) for i in sorted_idx],
            }

        # ---- 7. Plain text summary ----
        report['summary'] = self._build_summary(report, feat_dict, metadata)

        return report

    def _headline(self, risk_level, probability, classification):
        pct = probability * 100
        if classification == 'FRAUD':
            return f'FRAUD DETECTED - {pct:.1f}% fraud probability'
        elif classification == 'REVIEW':
            return f'FLAGGED FOR REVIEW - Rules detected suspicious patterns'
        elif classification == 'MONITOR':
            return f'MONITORING - Minor risk indicators detected'
        else:
            return f'TRANSACTION APPROVED - {pct:.1f}% fraud probability'

    def _explain_feature(self, fname, fval, sval):
        """Generate a plain English explanation for one feature."""
        direction = 'increases' if sval > 0 else 'decreases'

        if fname == 'amt':
            if sval > 0:
                return f'${fval:.2f} is a high transaction amount, which {direction} fraud risk'
            return f'${fval:.2f} is a normal transaction amount'

        if fname == 'is_night':
            if fval == 1:
                return f'Transaction at night (10PM-6AM) - nighttime transactions are 14x more likely to be fraudulent'
            return f'Daytime transaction - lower fraud risk during business hours'

        if fname == 'amount_velocity_1h':
            if sval > 0:
                return f'${fval:.2f} spent in the last hour is unusually high, suggesting potential card compromise'
            return f'${fval:.2f} spent in the last hour is within normal range'

        if fname == 'velocity_1h':
            if sval > 0:
                return f'{fval:.0f} transactions in the last hour is above average frequency'
            return f'{fval:.0f} transaction(s) in the last hour is a normal pace'

        if fname == 'velocity_24h':
            if sval > 0:
                return f'{fval:.0f} transactions in 24 hours indicates elevated activity'
            return f'{fval:.0f} transactions in 24 hours is typical activity'

        if fname == 'category_encoded':
            cat_name = _CODE_TO_NAME.get(int(fval), f'code {int(fval)}')
            if sval > 0:
                return f'Merchant category "{cat_name}" has historically higher fraud rates'
            return f'Merchant category "{cat_name}" is a common low-risk category'

        if fname == 'hour':
            if sval > 0:
                return f'Transactions at {int(fval)}:00 are associated with higher fraud risk'
            return f'Transactions at {int(fval)}:00 are common and low-risk'

        if fname == 'age':
            return f'Cardholder age {fval:.0f} {direction} fraud risk for this transaction profile'

        if fname == 'distance_cardholder_merchant':
            if sval > 0:
                return f'{fval:.1f}km from merchant is an unusual distance for this cardholder'
            return f'{fval:.1f}km from merchant is within typical range'

        if fname == 'city_pop':
            return f'City population {fval:,.0f} {direction} the fraud risk assessment'

        if fname == 'recon_error':
            if sval > 0:
                return f'Anomaly score {fval:.4f} indicates this transaction deviates from normal patterns'
            return f'Anomaly score {fval:.4f} indicates normal transaction pattern'

        # Generic
        label = _FEATURE_LABELS.get(fname, fname)
        return f'{label} = {fval:.2f} {direction} fraud risk'

    def _build_summary(self, report, features, metadata=None):
        """Build a plain text summary."""
        lines = []
        ra = report['risk_assessment']
        lines.append(f"--- {ra['headline']} ---")
        lines.append(f"Probability: {ra['probability_pct']} | Risk: {ra['risk_level']} | Decision: {ra['classification']}")
        lines.append("")

        if report['toward_fraud']:
            lines.append("Why this was flagged:")
            for f in report['toward_fraud'][:3]:
                lines.append(f"  * {f['explanation']}")

        if report['away_from_fraud']:
            lines.append("Factors arguing AGAINST fraud:")
            for f in report['away_from_fraud'][:3]:
                lines.append(f"  * {f['explanation']}")

        if report['rules']:
            lines.append("Rules triggered:")
            for r in report['rules']:
                lines.append(f"  ! [{r['level']}] {r['reason']}")

        rec = report['recommendation']
        lines.append(f"\nRECOMMENDED ACTION: {rec['action']}")
        lines.append(rec['detail'])

        return '\n'.join(lines)


# ---- Quick test ----
if __name__ == '__main__':
    import sys, os, numpy as np
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    pp = Postprocessor()

    # Test 1: CRITICAL fraud
    print("="*60)
    print("TEST 1: CRITICAL FRAUD ($800, 3am)")
    print("="*60)
    features1 = {
        'amt': 800, 'city_pop': 5000, 'hour': 3, 'month': 3,
        'distance_cardholder_merchant': 45, 'age': 25, 'is_weekend': 1,
        'is_night': 1, 'velocity_1h': 3, 'velocity_24h': 8,
        'amount_velocity_1h': 2400, 'category_encoded': 4, 'gender_encoded': 1,
        'day_of_week_encoded': 5,
    }
    shap_vals = np.array([1.8, -0.1, 0.5, -0.2, 0.3, 0.4, 0.8, 1.5, 0.6, -0.3, 3.2, 1.1, -0.1, -0.5])
    report1 = pp.format_prediction(
        features1, probability=0.993, risk_level='CRITICAL', classification='FRAUD',
        shap_values=shap_vals, shap_feature_names=FEATURE_COLS
    )
    print(report1['summary'])

    # Test 2: LOW normal
    print("\n" + "="*60)
    print("TEST 2: LOW RISK ($15, noon)")
    print("="*60)
    features2 = {
        'amt': 15, 'city_pop': 50000, 'hour': 12, 'month': 6,
        'distance_cardholder_merchant': 5, 'age': 45, 'is_weekend': 0,
        'is_night': 0, 'velocity_1h': 1, 'velocity_24h': 3,
        'amount_velocity_1h': 45, 'category_encoded': 4, 'gender_encoded': 0,
        'day_of_week_encoded': 2,
    }
    shap_vals2 = np.array([-1.8, -0.1, 0.5, -0.2, -0.3, -0.1, -0.3, -4.1, -0.2, -1.5, -2.4, 1.1, -0.1, -0.5])
    report2 = pp.format_prediction(
        features2, probability=0.001, risk_level='LOW', classification='NORMAL',
        shap_values=shap_vals2, shap_feature_names=FEATURE_COLS
    )
    print(report2['summary'])

    # Test 3: REVIEW (model missed, rules caught)
    print("\n" + "="*60)
    print("TEST 3: REVIEW (model NORMAL but rules HIGH)")
    print("="*60)
    from app.pipeline.rule_engine import RuleEngine, RuleResult
    engine = RuleEngine()
    features3 = {
        'amt': 100, 'city_pop': 5000, 'hour': 22, 'month': 3,
        'distance_cardholder_merchant': 10, 'age': 30, 'is_weekend': 0,
        'is_night': 1, 'velocity_1h': 8, 'velocity_24h': 15,
        'amount_velocity_1h': 800, 'category_encoded': 9, 'gender_encoded': 1,
        'day_of_week_encoded': 4,
    }
    rule_result = engine.evaluate(features3)
    report3 = pp.format_prediction(
        features3, probability=0.15, risk_level='HIGH', classification='REVIEW',
        shap_values=shap_vals, shap_feature_names=FEATURE_COLS,
        rule_result=rule_result
    )
    print(report3['summary'])

    # Verify chart data structure
    print("\n" + "="*60)
    print("TEST 4: Chart data structure check")
    print("="*60)
    cd = report1['shap_chart_data']
    print(f"  Labels: {cd['labels'][:5]}")
    print(f"  Values: {[f'{v:.2f}' for v in cd['values'][:5]]}")
    print(f"  Colors: {cd['colors'][:5]}")
    print(f"  Population comparison entries: {len(report1['population_comparison'])}")
    unusual = [p for p in report1['population_comparison'] if p['flag']]
    print(f"  Unusual features: {[(p['feature'], p['flag']) for p in unusual]}")

    print("\n  ALL POSTPROCESSOR TESTS COMPLETE")

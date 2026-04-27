"""REST API endpoints for fraud detection system."""
import time
import logging
from functools import wraps
from collections import defaultdict
from flask import Blueprint, request, jsonify
from app.config import (
    DEFAULT_THRESHOLD, RISK_LEVELS,
    MODEL_CODE_TO_INTERNAL, MODEL_INTERNAL_TO_DISPLAY,
    MODEL_CATEGORIES, MODEL_CATEGORY_LABELS, MODEL_DESCRIPTIONS,
    SIMULATION_DATASETS, FEATURE_DISPLAY_NAMES, FEATURE_COLS,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Rate limiting (in-memory)
_request_counts = defaultdict(list)
_RATE_LIMIT = 100  # per minute


def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr
        now = time.time()
        _request_counts[ip] = [t for t in _request_counts[ip] if now - t < 60]
        if len(_request_counts[ip]) >= _RATE_LIMIT:
            return jsonify({'error': 'Rate limit exceeded. Max 100 requests per minute.'}), 429
        _request_counts[ip].append(now)
        return f(*args, **kwargs)
    return decorated


def get_risk_level(prob):
    for level, (lo, hi) in RISK_LEVELS.items():
        if lo <= prob < hi:
            return level
    return 'CRITICAL'


def _generate_shap_summary(features, prediction):
    """Generate plain English explanation of top SHAP contributors."""
    top = [f for f in features if f['contribution'] > 0.02][:3]
    reasons = []
    for f in top:
        n, v = f['name'], f['value']
        if n == 'is_night' and v >= 0.5:
            reasons.append('it occurred at night')
        elif n == 'amount_velocity_1h':
            reasons.append(f"the card spent ${v:.0f} in the last hour")
        elif n == 'amt':
            reasons.append(f"the transaction amount (${v:.2f}) is unusually high")
        elif n == 'velocity_1h' and v > 1:
            reasons.append(f"there were {int(v)} recent transactions in one hour")
        elif n == 'velocity_24h':
            reasons.append(f"there were {int(v)} transactions in the last 24 hours")
        elif n == 'hour':
            reasons.append(f"the transaction time ({int(v)}:00) is unusual")
        elif n == 'recon_error':
            reasons.append(f"the transaction pattern is anomalous (AE score: {v:.2f})")
        elif n == 'bds_amount':
            reasons.append("the amount deviates from this cardholder's typical spending")
        elif n == 'bds_time':
            reasons.append("the transaction time is unusual for this cardholder")
        elif n == 'bds_freq':
            reasons.append("the transaction frequency is unusually high")
        elif n == 'bds_category':
            reasons.append("this merchant category is uncommon for this cardholder")
        elif n == 'category_encoded':
            reasons.append("the merchant category is commonly associated with fraud")
        elif n == 'distance_cardholder_merchant':
            reasons.append(f"the merchant is {v:.0f}km from the cardholder's location")
        else:
            reasons.append(f"{f['display_name'].lower()} contributed to the score")
    if not reasons:
        return f"Flagged with {prediction*100:.1f}% fraud probability based on a combination of subtle patterns."
    elif len(reasons) == 1:
        return f"This transaction was flagged primarily because {reasons[0]}."
    else:
        return f"This transaction was flagged primarily because {', '.join(reasons[:-1])}, and {reasons[-1]}."


def init_api(app, model_manager, preprocessor, rule_engine, postprocessor, drift_detector, db):
    """Initialize API routes with dependencies."""

    @api_bp.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy' if model_manager.active_model else 'degraded',
            'model_loaded': model_manager.active_model is not None,
            'active_model': model_manager.active_model_name,
            'database_connected': db is not None,
        })

    @api_bp.route('/predict', methods=['POST'])
    @rate_limit
    def predict():
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        try:
            # Preprocess
            velocity_override = None
            if all(k in data for k in ['velocity_1h', 'velocity_24h', 'amount_velocity_1h']):
                velocity_override = {k: data[k] for k in ['velocity_1h', 'velocity_24h', 'amount_velocity_1h']}

            features, metadata = preprocessor.process(data, velocity_override=velocity_override)

            # Build prior-feature sequence for LSTM+RF (no-op for other models)
            prior = None
            if model_manager.active_model_name == 'LSTM+RF' and metadata.get('card_number'):
                prior = db.get_card_recent_features(metadata['card_number'], limit=4)

            # Predict (with SHAP for single transactions)
            result = model_manager.predict_with_shap(features, prior_features=prior)
            probability = result['probability']
            risk_level = get_risk_level(probability)

            # Rule engine
            rule_result = rule_engine.evaluate(features)
            classification, combined_prob, decision_reason = rule_engine.combine_decision(
                probability, rule_result, DEFAULT_THRESHOLD
            )

            # Postprocess
            report = postprocessor.format_prediction(
                features, probability, risk_level, classification,
                shap_values=result.get('shap_values'),
                shap_feature_names=result.get('shap_feature_names'),
                rule_result=rule_result, metadata=metadata
            )

            # Store in database
            txn_record = {
                'transaction_id': metadata['transaction_id'],
                'card_number': metadata['card_number'],
                'timestamp': metadata['timestamp'],
                'amount': features['amt'],
                'category': metadata['category_name'],
                'probability': probability,
                'risk_level': risk_level,
                'classification': classification,
                'rule_triggers': ','.join(r[0] for r in rule_result.triggered_rules) if rule_result.any_triggered else None,
                'processing_time_ms': result['processing_time_ms'],
                'velocity_source': metadata['velocity_source'],
            }
            txn_record.update(features)
            db.store_transaction(txn_record)

            # Store card history
            db.add_card_transaction(
                metadata['card_number'], metadata['timestamp'],
                features['amt'], int(features['category_encoded'])
            )

            # Store alert if flagged
            if classification in ('FRAUD', 'REVIEW'):
                db.store_alert({
                    'transaction_id': metadata['transaction_id'],
                    'probability': probability,
                    'risk_level': risk_level,
                    'classification': classification,
                    'amount': features['amt'],
                    'category': metadata['category_name'],
                    'rule_triggers': txn_record['rule_triggers'],
                    'explanation': report['summary'],
                })

            return jsonify({
                'transaction_id': metadata['transaction_id'],
                'probability': probability,
                'risk_level': risk_level,
                'classification': classification,
                'combined_probability': combined_prob,
                'processing_time_ms': result['processing_time_ms'],
                'velocity_source': metadata['velocity_source'],
                'report': report,
            })

        except Exception as e:
            logger.exception("Prediction error")
            return jsonify({'error': str(e)}), 500

    @api_bp.route('/predict/batch', methods=['POST'])
    @rate_limit
    def predict_batch():
        data = request.get_json()
        if not data or 'transactions' not in data:
            return jsonify({'error': 'Expected JSON with "transactions" array'}), 400

        results = []
        total_fraud = 0
        total_review = 0
        total_normal = 0
        total_amount_at_risk = 0

        for txn in data['transactions']:
            try:
                velocity_override = None
                if all(k in txn for k in ['velocity_1h', 'velocity_24h', 'amount_velocity_1h']):
                    velocity_override = {k: txn[k] for k in ['velocity_1h', 'velocity_24h', 'amount_velocity_1h']}

                features, metadata = preprocessor.process(txn, velocity_override=velocity_override)
                pred = model_manager.predict(features)
                probability = pred['probability']
                risk_level = get_risk_level(probability)
                rule_result = rule_engine.evaluate(features)
                classification, _, _ = rule_engine.combine_decision(probability, rule_result)

                if classification == 'FRAUD':
                    total_fraud += 1
                    total_amount_at_risk += features['amt']
                elif classification == 'REVIEW':
                    total_review += 1
                    total_amount_at_risk += features['amt']
                else:
                    total_normal += 1

                results.append({
                    'transaction_id': metadata['transaction_id'],
                    'probability': probability,
                    'risk_level': risk_level,
                    'classification': classification,
                    'amount': features['amt'],
                })
            except Exception as e:
                results.append({'transaction_id': txn.get('transaction_id', '?'), 'error': str(e)})

        return jsonify({
            'total': len(results),
            'fraud_count': total_fraud,
            'review_count': total_review,
            'normal_count': total_normal,
            'total_amount_at_risk': total_amount_at_risk,
            'results': results,
        })

    @api_bp.route('/model/info', methods=['GET'])
    def model_info():
        info = model_manager.get_model_info()
        info['active_display'] = MODEL_INTERNAL_TO_DISPLAY.get(info.get('active_model'), info.get('active_model'))
        return jsonify(info)

    @api_bp.route('/model/switch', methods=['POST'])
    def switch_model():
        data = request.get_json()
        # Accept either code name (xgboost_baseline) or internal name (XGBoost (Class Weights))
        name = data.get('model_name') or data.get('model')
        internal_name = MODEL_CODE_TO_INTERNAL.get(name, name)  # map code → internal, fallback to raw
        if model_manager.set_active(internal_name):
            display = MODEL_INTERNAL_TO_DISPLAY.get(model_manager.active_model_name, model_manager.active_model_name)
            return jsonify({'status': 'ok', 'active_model': model_manager.active_model_name, 'display_name': display})
        return jsonify({'error': f'Model "{name}" not found'}), 404

    @api_bp.route('/model/list', methods=['GET'])
    def list_models():
        """Return all available models with display names and availability."""
        import os
        from app.config import MODEL_FILES
        models = []
        for code, internal in MODEL_CODE_TO_INTERNAL.items():
            display = MODEL_INTERNAL_TO_DISPLAY.get(internal, internal)
            available = internal in model_manager.models
            category = MODEL_CATEGORIES.get(internal, 'supplementary')
            models.append({
                'code': code, 'internal': internal, 'display': display,
                'available': available,
                'active': internal == model_manager.active_model_name,
                'category': category,
                'category_label': MODEL_CATEGORY_LABELS.get(category, category),
                'description': MODEL_DESCRIPTIONS.get(internal, ''),
            })
        datasets = []
        for key, info in SIMULATION_DATASETS.items():
            exists = info['file'] is None or os.path.exists(info['file'])
            datasets.append({'key': key, 'label': info['label'], 'rows': info['rows'], 'available': exists})
        return jsonify({'models': models, 'datasets': datasets,
                        'active_model': model_manager.active_model_name,
                        'active_display': MODEL_INTERNAL_TO_DISPLAY.get(model_manager.active_model_name, model_manager.active_model_name)})

    @api_bp.route('/model/performance', methods=['GET'])
    def model_performance():
        stats = db.get_stats()
        drift_report = drift_detector.generate_report()
        return jsonify({
            'stats': stats,
            'drift': drift_report,
            'active_model': model_manager.active_model_name,
            'baseline_metrics': drift_detector.baseline_metrics,
        })

    @api_bp.route('/feedback', methods=['POST'])
    @rate_limit
    def feedback():
        data = request.get_json()
        txn_id = data.get('transaction_id')
        actual = data.get('actual_label')
        notes = data.get('analyst_notes', '')
        if not txn_id or not actual:
            return jsonify({'error': 'transaction_id and actual_label required'}), 400
        db.store_feedback(txn_id, actual, notes)
        return jsonify({'status': 'ok', 'message': f'Feedback recorded for {txn_id}'})

    @api_bp.route('/explain', methods=['POST'])
    @rate_limit
    def explain():
        """Compute SHAP explanation for a single transaction's features."""
        data = request.get_json() or {}
        features = data.get('features')
        if not features or not isinstance(features, dict):
            return jsonify({'error': 'features dict required'}), 400
        try:
            result = model_manager.predict_with_shap(features)
            if result.get('error'):
                return jsonify({'error': result['error']}), 500
            shap_vals = result.get('shap_values')
            feature_names = result.get('shap_feature_names', [])
            model_features = result.get('features')  # full numpy array used by model
            if shap_vals is None:
                return jsonify({'error': 'SHAP computation failed — values are None'}), 500

            # Base value — handle scalar or array (XGBoost returns scalar for binary)
            model_name = model_manager.active_model_name
            explainer = model_manager.explainers.get(model_name)
            base_val = 0.0
            if explainer is not None:
                ev = explainer.expected_value
                if hasattr(ev, '__len__'):
                    base_val = float(ev[-1])
                else:
                    base_val = float(ev)

            feature_data = []
            for i, name in enumerate(feature_names):
                val = float(model_features[i]) if model_features is not None and i < len(model_features) else 0.0
                feature_data.append({
                    'name': name,
                    'display_name': FEATURE_DISPLAY_NAMES.get(name, name.replace('_', ' ').title()),
                    'value': round(val, 4),
                    'contribution': round(float(shap_vals[i]), 6),
                })
            feature_data.sort(key=lambda x: abs(x['contribution']), reverse=True)

            return jsonify({
                'success': True,
                'prediction': round(result['probability'], 4),
                'base_value': round(base_val, 4),
                'model': model_name,
                'features': feature_data,
                'summary': _generate_shap_summary(feature_data, result['probability']),
            })
        except Exception as e:
            logger.exception("Explain error")
            return jsonify({'error': str(e)}), 500

    @api_bp.route('/cardholder/<card_id>/history', methods=['GET'])
    def cardholder_history(card_id):
        """Return full transaction history for a card from current simulation data."""
        result = db.get_cardholder_history(card_id)
        if result is None:
            return jsonify({'error': f'No transactions found for card {card_id}'}), 404
        return jsonify(result)

    @api_bp.route('/stats', methods=['GET'])
    def stats():
        return jsonify(db.get_stats())

    @api_bp.route('/recent', methods=['GET'])
    def recent_transactions():
        limit = int(request.args.get('limit', 50))
        return jsonify({'transactions': db.get_recent_transactions(limit=limit)})

    @api_bp.route('/analyse', methods=['POST'])
    def analyse_dataset():
        """Rich analysis endpoint — returns detailed breakdown of an uploaded dataset."""
        data = request.get_json()
        if not data or 'transactions' not in data:
            return jsonify({'error': 'Expected JSON with "transactions" array'}), 400

        transactions = data['transactions']
        has_labels = data.get('has_labels', False)
        results = []
        probabilities = []
        classifications = []
        amounts = []
        hours = []
        categories = []
        actuals = []

        # Per-card running prior-feature buffers — used to build LSTM sequences
        # within the batch when LSTM+RF is the active model.
        lstm_active = (model_manager.active_model_name == 'LSTM+RF')
        card_priors = {}

        for txn in transactions:
            try:
                velocity_override = None
                if all(k in txn for k in ['velocity_1h', 'velocity_24h', 'amount_velocity_1h']):
                    velocity_override = {k: txn[k] for k in ['velocity_1h', 'velocity_24h', 'amount_velocity_1h']}

                features, metadata = preprocessor.process(txn, velocity_override=velocity_override)

                prior = None
                if lstm_active:
                    card_id = metadata.get('card_number') or txn.get('cc_num') or '__nocard__'
                    prior = card_priors.get(card_id, [])
                    feat_for_seq = {c: float(features.get(c, 0.0)) for c in FEATURE_COLS}

                pred = model_manager.predict(features, prior_features=prior) if lstm_active else model_manager.predict(features)
                if lstm_active:
                    buf = card_priors.setdefault(card_id, [])
                    buf.append(feat_for_seq)
                    if len(buf) > 4:
                        buf.pop(0)
                probability = pred['probability']
                risk_level = get_risk_level(probability)
                rule_result = rule_engine.evaluate(features)
                classification, combined_prob, reason = rule_engine.combine_decision(probability, rule_result)

                probabilities.append(probability)
                classifications.append(classification)
                amounts.append(features['amt'])
                hours.append(int(features['hour']))
                categories.append(metadata.get('category_name', 'unknown'))

                actual = txn.get('_actual', txn.get('is_fraud'))
                if actual is not None:
                    actuals.append(int(actual))

                results.append({
                    'transaction_id': metadata.get('transaction_id', f'ROW-{len(results)+1}'),
                    'probability': probability,
                    'risk_level': risk_level,
                    'classification': classification,
                    'amount': features['amt'],
                    'hour': int(features['hour']),
                    'category': metadata.get('category_name', 'unknown'),
                    'is_night': int(features['is_night']),
                    'velocity_1h': features['velocity_1h'],
                    'velocity_24h': features['velocity_24h'],
                    'amount_velocity_1h': features['amount_velocity_1h'],
                    'rule_triggers': [r[0] for r in rule_result.triggered_rules] if rule_result.any_triggered else [],
                    'actual': actual,
                    # Full feature dict for SHAP explanation on click
                    'features': {col: float(features[col]) for col in FEATURE_COLS},
                })
            except Exception as e:
                results.append({'transaction_id': txn.get('transaction_id', '?'), 'error': str(e),
                                'probability': 0, 'classification': 'NORMAL', 'risk_level': 'LOW',
                                'amount': 0, 'hour': 0, 'category': 'unknown',
                                'velocity_1h': 0, 'velocity_24h': 0, 'amount_velocity_1h': 0, 'is_night': 0,
                                'rule_triggers': []})
                probabilities.append(0)
                classifications.append('NORMAL')
                amounts.append(0)
                hours.append(0)
                categories.append('unknown')

        import numpy as np

        # Counts
        counts = {'FRAUD': 0, 'REVIEW': 0, 'MONITOR': 0, 'NORMAL': 0}
        for c in classifications:
            counts[c] = counts.get(c, 0) + 1

        # Amount stats
        fraud_amounts = [r['amount'] for r in results if r['classification'] == 'FRAUD']
        normal_amounts = [r['amount'] for r in results if r['classification'] == 'NORMAL']

        # Hour breakdown
        hour_counts = {h: 0 for h in range(24)}
        hour_fraud = {h: 0 for h in range(24)}
        for r in results:
            h = r.get('hour', 0)
            hour_counts[h] = hour_counts.get(h, 0) + 1
            if r['classification'] in ('FRAUD', 'REVIEW'):
                hour_fraud[h] = hour_fraud.get(h, 0) + 1

        # Category breakdown
        cat_counts = {}
        cat_fraud = {}
        for r in results:
            cat = r.get('category', 'unknown')
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            if r['classification'] in ('FRAUD', 'REVIEW'):
                cat_fraud[cat] = cat_fraud.get(cat, 0) + 1

        # Probability distribution (bins)
        prob_bins = [0]*10
        for p in probabilities:
            b = min(int(p * 10), 9)
            prob_bins[b] += 1

        # Risk summary
        amount_at_risk = sum(r['amount'] for r in results if r['classification'] in ('FRAUD', 'REVIEW'))
        amount_safe = sum(r['amount'] for r in results if r['classification'] == 'NORMAL')

        # Performance metrics (if labels available)
        performance = None
        if len(actuals) == len(results) and len(actuals) > 0:
            # F1/precision/recall are computed against FRAUD-only (matches training evaluation
            # where a single probability threshold decides positive/negative). REVIEW is a
            # human-review tier and intentionally excludes many borderline transactions that
            # would otherwise inflate false positives.
            tp = sum(1 for a, c in zip(actuals, classifications) if a == 1 and c == 'FRAUD')
            fp = sum(1 for a, c in zip(actuals, classifications) if a == 0 and c == 'FRAUD')
            fn = sum(1 for a, c in zip(actuals, classifications) if a == 1 and c != 'FRAUD')
            tn = sum(1 for a, c in zip(actuals, classifications) if a == 0 and c != 'FRAUD')
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            performance = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
                'accuracy': round((tp + tn) / len(actuals), 4),
                'total_fraud': sum(actuals), 'total_normal': len(actuals) - sum(actuals),
                'missed_frauds': [r for r in results if r.get('actual') == 1 and r['classification'] != 'FRAUD'][:20],
                'false_positives': [r for r in results if r.get('actual') == 0 and r['classification'] == 'FRAUD'][:20],
            }

        # Top flagged transactions
        sorted_results = sorted(results, key=lambda x: x['probability'], reverse=True)
        top_flagged = sorted_results[:10]

        return jsonify({
            'total': len(results),
            'counts': counts,
            'amount_at_risk': round(amount_at_risk, 2),
            'amount_safe': round(amount_safe, 2),
            'avg_probability': round(float(np.mean(probabilities)), 6) if probabilities else 0,
            'max_probability': round(float(max(probabilities)), 6) if probabilities else 0,
            'prob_distribution': prob_bins,
            'hour_counts': hour_counts,
            'hour_fraud': hour_fraud,
            'category_counts': cat_counts,
            'category_fraud': cat_fraud,
            'amount_stats': {
                'flagged_avg': round(float(np.mean(fraud_amounts)), 2) if fraud_amounts else 0,
                'flagged_max': round(float(max(fraud_amounts)), 2) if fraud_amounts else 0,
                'normal_avg': round(float(np.mean(normal_amounts)), 2) if normal_amounts else 0,
            },
            'performance': performance,
            'top_flagged': top_flagged,
            'results': results,
        })

    app.register_blueprint(api_bp)

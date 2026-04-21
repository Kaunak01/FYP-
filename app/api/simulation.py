"""Simulation endpoint: processes transactions via Server-Sent Events."""
import csv
import json
import time
import os
import random
import logging
import collections
from flask import Blueprint, Response, request, jsonify, stream_with_context
from app.config import DEFAULT_THRESHOLD, RISK_LEVELS, FEATURE_COLS, SIMULATION_DATASETS, MODEL_INTERNAL_TO_DISPLAY

logger = logging.getLogger(__name__)

sim_bp = Blueprint('simulation', __name__, url_prefix='/api/simulation')

# Thread-safe queue for injected attack transactions (deque is thread-safe for append/pop)
_inject_queue = collections.deque()
# Sequence counter shared across stream (updated by the running generator)
_sim_seq = [0]  # list so it's mutable from inner functions

# Load demo transactions (JSON — original 102 rows)
_demo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo_transactions.json')
_demo_transactions = []
if os.path.exists(_demo_path):
    with open(_demo_path) as f:
        _demo_transactions = json.load(f)
    logger.info("Loaded %d demo transactions", len(_demo_transactions))

# Category code → name (for CSV rows)
_cat_map = {}
_cat_map_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'models', 'stats', 'category_mapping.json')
if os.path.exists(_cat_map_path):
    with open(_cat_map_path) as f:
        _raw = json.load(f)
        _cat_map = {int(k): v for k, v in _raw.get('code_to_name', {}).items()}


def get_risk_level(prob):
    for level, (lo, hi) in RISK_LEVELS.items():
        if lo <= prob < hi:
            return level
    return 'CRITICAL'


def _stream_demo(model_manager, preprocessor, rule_engine, db, delay, max_txns, compare=False):
    """Stream original 102 demo transactions (JSON format)."""
    demo_count = 0
    for i, txn in enumerate(_demo_transactions[:max_txns]):
        # Drain inject queue — yield injected attack events first
        while _inject_queue:
            try:
                inj = _inject_queue.popleft()
                demo_count += 1
                inj['sequence'] = demo_count
                inj['total'] = max_txns
                yield inj
            except IndexError:
                break
        demo_count += 1
        t0 = time.perf_counter()
        velocity_override = {
            'velocity_1h': txn['velocity_1h'],
            'velocity_24h': txn['velocity_24h'],
            'amount_velocity_1h': txn['amount_velocity_1h'],
        }
        features, metadata = preprocessor.process(txn, velocity_override=velocity_override)
        result = model_manager.predict(features)
        probability = result['probability']
        risk_level = get_risk_level(probability)
        rule_result = rule_engine.evaluate(features)
        classification, combined_prob, decision_reason = rule_engine.combine_decision(
            probability, rule_result, DEFAULT_THRESHOLD
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        txn_record = {
            'transaction_id': txn['transaction_id'], 'card_number': txn['card_number'],
            'timestamp': txn['timestamp'], 'amount': txn['amount'],
            'category': txn['merchant_category'], 'probability': probability,
            'risk_level': risk_level, 'classification': classification,
            'rule_triggers': ','.join(r[0] for r in rule_result.triggered_rules) if rule_result.any_triggered else None,
            'processing_time_ms': elapsed_ms, 'velocity_source': 'override',
        }
        txn_record.update(features)
        db.store_transaction(txn_record)
        db.add_card_transaction(txn['card_number'], txn['timestamp'], txn['amount'], int(features['category_encoded']))

        is_alert = classification in ('FRAUD', 'REVIEW')
        if is_alert:
            db.store_alert({
                'transaction_id': txn['transaction_id'], 'probability': probability,
                'risk_level': risk_level, 'classification': classification,
                'amount': txn['amount'], 'category': txn['merchant_category'],
                'rule_triggers': txn_record['rule_triggers'], 'explanation': decision_reason,
            })

        yield {
            'sequence': demo_count, 'total': max_txns,
            'transaction_id': txn['transaction_id'], 'card_number': txn['card_number'],
            'amount': txn['amount'],
            'timestamp': txn['timestamp'], 'category': txn['merchant_category'],
            'probability': round(probability, 6), 'risk_level': risk_level,
            'classification': classification, 'actual_is_fraud': txn['actual_is_fraud'],
            'correct': (classification in ('FRAUD', 'REVIEW') and txn['actual_is_fraud'] == 1) or
                       (classification == 'NORMAL' and txn['actual_is_fraud'] == 0),
            'is_alert': is_alert,
            'rule_triggers': [r[0] for r in rule_result.triggered_rules] if rule_result.any_triggered else [],
            'processing_time_ms': round(elapsed_ms, 2),
            'narration': txn.get('narration'), 'act': txn.get('act'),
            'hour': int(features.get('hour', 0)),
            'features': {k: float(features[k]) for k in FEATURE_COLS if k in features},
            'compare_predictions': model_manager.predict_all(features) if compare else None,
        }

        if delay > 0:
            time.sleep(delay)


def _stream_csv(filepath, model_manager, preprocessor, rule_engine, db, delay, max_txns, compare=False):
    """Stream transactions from a CSV file row by row."""
    # Count total rows first (for progress bar)
    total = 0
    with open(filepath, newline='', encoding='utf-8') as f:
        total = sum(1 for _ in f) - 1  # minus header
    if max_txns and max_txns < total:
        total = max_txns

    count = 0
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Drain inject queue — yield injected attack events before next row
            while _inject_queue:
                try:
                    inj = _inject_queue.popleft()
                    count += 1
                    inj['sequence'] = count
                    inj['total'] = total
                    yield inj
                except IndexError:
                    break

            if max_txns and count >= max_txns:
                break

            t0 = time.perf_counter()
            try:
                # Build features directly from CSV (already engineered)
                features = {col: float(row[col]) for col in FEATURE_COLS}

                # Build synthetic metadata
                cat_code = int(float(row.get('category_encoded', 0)))
                cat_name = _cat_map.get(cat_code, f'category_{cat_code}')
                txn_id = f'SIM-{i+1:06d}'
                # Deterministic fake card number from row index
                card_num = f'CARD-{(i % 500) + 1000:04d}'
                timestamp = str(row.get('unix_time', ''))
                actual_fraud = int(float(row.get('is_fraud', 0)))

                result = model_manager.predict(features)
                probability = result['probability']
                risk_level = get_risk_level(probability)
                rule_result = rule_engine.evaluate(features)
                classification, combined_prob, decision_reason = rule_engine.combine_decision(
                    probability, rule_result, DEFAULT_THRESHOLD
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

                txn_record = {
                    'transaction_id': txn_id, 'card_number': card_num,
                    'timestamp': timestamp, 'amount': features['amt'],
                    'category': cat_name, 'probability': probability,
                    'risk_level': risk_level, 'classification': classification,
                    'rule_triggers': ','.join(r[0] for r in rule_result.triggered_rules) if rule_result.any_triggered else None,
                    'processing_time_ms': elapsed_ms, 'velocity_source': 'precomputed',
                }
                txn_record.update(features)
                db.store_transaction(txn_record)
                db.add_card_transaction(card_num, timestamp, features['amt'], cat_code)

                is_alert = classification in ('FRAUD', 'REVIEW')
                if is_alert:
                    db.store_alert({
                        'transaction_id': txn_id, 'probability': probability,
                        'risk_level': risk_level, 'classification': classification,
                        'amount': features['amt'], 'category': cat_name,
                        'rule_triggers': txn_record['rule_triggers'], 'explanation': decision_reason,
                    })

                count += 1
                yield {
                    'sequence': count, 'total': total,
                    'transaction_id': txn_id, 'card_number': card_num,
                    'amount': features['amt'],
                    'timestamp': timestamp, 'category': cat_name,
                    'probability': round(probability, 6), 'risk_level': risk_level,
                    'classification': classification, 'actual_is_fraud': actual_fraud,
                    'correct': (classification in ('FRAUD', 'REVIEW') and actual_fraud == 1) or
                               (classification == 'NORMAL' and actual_fraud == 0),
                    'is_alert': is_alert,
                    'rule_triggers': [r[0] for r in rule_result.triggered_rules] if rule_result.any_triggered else [],
                    'processing_time_ms': round(elapsed_ms, 2),
                    'narration': None, 'act': 'simulation',
                    'hour': int(features.get('hour', 0)),
                    'features': dict(features),
                    'compare_predictions': model_manager.predict_all(features) if compare else None,
                }

                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                logger.warning("CSV row %d error: %s", i, e)
                continue


def init_simulation(app, model_manager, preprocessor, rule_engine, postprocessor, db):
    """Initialize simulation routes."""

    @sim_bp.route('/start', methods=['POST'])
    def start_simulation():
        """Start processing transactions via SSE."""
        data = request.get_json() or {}
        delay = float(data.get('delay', 0.5))
        dataset_key = data.get('dataset', 'demo')
        compare = bool(data.get('compare', False))

        # Resolve dataset
        dataset_info = SIMULATION_DATASETS.get(dataset_key, SIMULATION_DATASETS['demo'])
        csv_path = dataset_info.get('file')
        use_csv = csv_path is not None

        if use_csv and not os.path.exists(csv_path):
            return jsonify({'error': f'Dataset file not found: {csv_path}'}), 404

        max_txns = int(data.get('max', dataset_info['rows']))

        # Reset database and inject queue for fresh simulation
        db.reset()
        _inject_queue.clear()

        active_display = MODEL_INTERNAL_TO_DISPLAY.get(model_manager.active_model_name, model_manager.active_model_name)

        def generate():
            # Send start event with metadata
            start_meta = {
                'event': 'start',
                'dataset': dataset_key,
                'dataset_label': dataset_info['label'],
                'total': max_txns,
                'model': model_manager.active_model_name,
                'model_display': active_display,
            }
            yield f"event: meta\ndata: {json.dumps(start_meta)}\n\n"

            if use_csv:
                streamer = _stream_csv(csv_path, model_manager, preprocessor, rule_engine, db, delay, max_txns, compare=compare)
            else:
                streamer = _stream_demo(model_manager, preprocessor, rule_engine, db, delay, max_txns, compare=compare)

            for event_data in streamer:
                yield f"data: {json.dumps(event_data)}\n\n"

            # Final summary
            stats = db.get_stats()
            stats['model_display'] = active_display
            yield f"event: complete\ndata: {json.dumps(stats)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

    @sim_bp.route('/status', methods=['GET'])
    def simulation_status():
        return jsonify(db.get_stats())

    @sim_bp.route('/datasets', methods=['GET'])
    def list_datasets():
        """Return available simulation datasets with availability check."""
        result = []
        for key, info in SIMULATION_DATASETS.items():
            exists = info['file'] is None or os.path.exists(info['file'])
            result.append({'key': key, 'label': info['label'], 'rows': info['rows'], 'available': exists})
        return jsonify({'datasets': result})

    @sim_bp.route('/transactions', methods=['GET'])
    def demo_transaction_list():
        return jsonify({
            'total': len(_demo_transactions),
            'acts': {act: sum(1 for t in _demo_transactions if t['act'] == act)
                     for act in set(t['act'] for t in _demo_transactions)},
        })

    @sim_bp.route('/inject_attack', methods=['POST'])
    def inject_attack():
        """Inject a synthetic fraud attack burst into the running simulation."""
        data = request.get_json() or {}
        count = min(int(data.get('count', 3)), 6)  # max 6 injected at once

        # Synthetic attack features — designed to score high fraud probability
        # Based on SHAP top features: is_night > amt > amount_velocity_1h > hour
        ATTACK_SCENARIOS = [
            # Rapid ATM-style withdrawals late at night
            {'amt': 850.00,  'is_night': 1, 'hour': 2,  'velocity_1h': 6, 'amount_velocity_1h': 4500.0, 'category_encoded': 1,  'cat_name': 'gas_transport'},
            {'amt': 1200.00, 'is_night': 1, 'hour': 2,  'velocity_1h': 7, 'amount_velocity_1h': 5200.0, 'category_encoded': 8,  'cat_name': 'shopping_net'},
            {'amt': 975.50,  'is_night': 1, 'hour': 23, 'velocity_1h': 5, 'amount_velocity_1h': 3900.0, 'category_encoded': 11, 'cat_name': 'misc_net'},
            # High-value electronics/travel purchases with velocity spike
            {'amt': 1850.00, 'is_night': 1, 'hour': 1,  'velocity_1h': 8, 'amount_velocity_1h': 7200.0, 'category_encoded': 10, 'cat_name': 'travel'},
            {'amt': 620.00,  'is_night': 0, 'hour': 14, 'velocity_1h': 9, 'amount_velocity_1h': 6100.0, 'category_encoded': 3,  'cat_name': 'grocery_pos'},
            {'amt': 1475.00, 'is_night': 1, 'hour': 3,  'velocity_1h': 7, 'amount_velocity_1h': 8300.0, 'category_encoded': 8,  'cat_name': 'shopping_pos'},
        ]

        injected = []
        attack_card = f'CARD-ATK{random.randint(1000, 9999)}'

        for j in range(count):
            scenario = ATTACK_SCENARIOS[j % len(ATTACK_SCENARIOS)]
            amt = scenario['amt'] * random.uniform(0.85, 1.15)  # ±15% noise

            features = {
                'amt':                          round(amt, 2),
                'city_pop':                     random.randint(50000, 200000),
                'hour':                         scenario['hour'],
                'month':                        random.randint(1, 12),
                'distance_cardholder_merchant': round(random.uniform(30, 120), 1),
                'age':                          random.randint(25, 45),
                'is_weekend':                   0,
                'is_night':                     scenario['is_night'],
                'velocity_1h':                  scenario['velocity_1h'],
                'velocity_24h':                 scenario['velocity_1h'] + random.randint(5, 12),
                'amount_velocity_1h':           round(scenario['amount_velocity_1h'] * random.uniform(0.9, 1.1), 2),
                'category_encoded':             float(scenario['category_encoded']),
                'gender_encoded':               float(random.randint(0, 1)),
                'day_of_week_encoded':          float(random.randint(0, 6)),
            }

            try:
                result = model_manager.predict(features)
                probability = result['probability']
                risk_level = get_risk_level(probability)
                rule_result = rule_engine.evaluate(features)
                classification, combined_prob, decision_reason = rule_engine.combine_decision(
                    probability, rule_result, DEFAULT_THRESHOLD
                )
            except Exception as e:
                logger.warning("inject_attack prediction error: %s", e)
                probability = 0.92
                risk_level = 'CRITICAL'
                classification = 'FRAUD'
                rule_result = None
                decision_reason = 'Injected attack'

            txn_id = f'ATK-{random.randint(100000, 999999)}'

            txn_record = {
                'transaction_id': txn_id, 'card_number': attack_card,
                'timestamp': '', 'amount': features['amt'],
                'category': scenario['cat_name'], 'probability': probability,
                'risk_level': risk_level, 'classification': classification,
                'rule_triggers': ','.join(r[0] for r in rule_result.triggered_rules) if rule_result and rule_result.any_triggered else 'INJECTED_ATTACK',
                'processing_time_ms': 0.0, 'velocity_source': 'injected',
            }
            txn_record.update(features)
            try:
                db.store_transaction(txn_record)
                db.store_alert({
                    'transaction_id': txn_id, 'probability': probability,
                    'risk_level': risk_level, 'classification': classification,
                    'amount': features['amt'], 'category': scenario['cat_name'],
                    'rule_triggers': txn_record['rule_triggers'], 'explanation': decision_reason,
                })
            except Exception:
                pass

            event = {
                'sequence': -1,  # will be patched by generator when draining queue
                'total': -1,
                'transaction_id': txn_id,
                'card_number': attack_card,
                'amount': features['amt'],
                'timestamp': '',
                'category': scenario['cat_name'],
                'probability': round(probability, 6),
                'risk_level': risk_level,
                'classification': classification,
                'actual_is_fraud': 1,
                'correct': classification in ('FRAUD', 'REVIEW'),
                'is_alert': True,
                'injected': True,  # flag for frontend special rendering
                'rule_triggers': [r[0] for r in rule_result.triggered_rules] if rule_result and rule_result.any_triggered else ['INJECTED_ATTACK'],
                'processing_time_ms': 0.0,
                'narration': None, 'act': 'injected_attack',
                'hour': int(features['hour']),
                'features': {k: float(features[k]) for k in FEATURE_COLS if k in features},
                'compare_predictions': None,
            }
            _inject_queue.append(event)
            injected.append(txn_id)

        return jsonify({
            'ok': True,
            'injected': len(injected),
            'transaction_ids': injected,
            'card': attack_card,
        })

    app.register_blueprint(sim_bp)

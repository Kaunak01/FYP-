import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import logging
logging.disable(logging.CRITICAL)

from app.main import create_app
app, db, mm = create_app()

with app.test_client() as client:
    # Test 1: Health
    r = client.get('/api/health')
    print('=== /api/health ===')
    print('Status:', r.status_code)
    h = r.get_json()
    print('Model loaded:', h.get('model_loaded'))
    print('DB connected:', h.get('db_connected'))

    # Test 2: Predict legit
    legit = {
        'amount': 25.50,
        'timestamp': '2020-06-22 14:30:00',
        'merchant_category': 'grocery_pos',
        'card_number': 'CC_TEST_001',
        'merchant_lat': 40.7128, 'merchant_long': -74.0060,
        'card_lat': 40.7580, 'card_long': -73.9855,
        'gender': 'M', 'age': 35, 'city_pop': 500000
    }
    r = client.post('/api/predict', json=legit)
    print('\n=== /api/predict (legit $25.50 grocery afternoon) ===')
    print('Status:', r.status_code)
    d = r.get_json()
    if r.status_code == 200:
        print('Probability:', d.get('probability'))
        print('Triage:', d.get('triage_band'))
        print('Classification:', d.get('classification'))
        has_shap = 'explanation' in d or 'shap_explanation' in d
        print('Has SHAP:', has_shap)
    else:
        print('Error:', d)

    # Test 3: Predict suspicious
    fraud = {
        'amount': 950.00,
        'timestamp': '2020-06-22 02:30:00',
        'merchant_category': 'shopping_net',
        'card_number': 'CC_TEST_002',
        'merchant_lat': 51.5074, 'merchant_long': -0.1278,
        'card_lat': 40.7580, 'card_long': -73.9855,
        'gender': 'F', 'age': 28, 'city_pop': 100000
    }
    r = client.post('/api/predict', json=fraud)
    print('\n=== /api/predict (suspicious $950 shopping_net 2:30am) ===')
    print('Status:', r.status_code)
    d = r.get_json()
    if r.status_code == 200:
        print('Probability:', d.get('probability'))
        print('Triage:', d.get('triage_band'))
        print('Classification:', d.get('classification'))
        print('Rules:', d.get('rule_triggers', []))
    else:
        print('Error:', d)

    # Test 4: Model switching
    print('\n=== Model Switching ===')
    for model in ['XGBoost (SMOTE+Tuned)', 'LSTM+RF', 'AE+BDS+XGBoost']:
        r = client.post('/api/model/switch', json={'model_name': model})
        body = r.get_json() or r.data.decode('utf-8')[:100]
        print(f'  Switch to {model}: {r.status_code} - {body}')

    # Test 5: All dashboard pages
    print('\n=== Dashboard Pages ===')
    for page in ['/', '/analyse', '/monitor', '/predict', '/performance', '/settings']:
        r = client.get(page)
        print(f'  GET {page}: {r.status_code}')

    # Test 6: Model info
    r = client.get('/api/model/info')
    print('\n=== /api/model-info ===')
    print('Status:', r.status_code)
    info = r.get_json()
    if info:
        print('Active model:', info.get('name'))
        print('Features:', info.get('n_features'))
    else:
        print('Response:', r.data.decode('utf-8')[:200])

    # Test 7: Alerts
    r = client.get('/api/alerts')
    print('\n=== /api/alerts ===')
    print('Status:', r.status_code)
    alerts = r.get_json()
    if alerts:
        print('Alert count:', len(alerts) if isinstance(alerts, list) else 'N/A')
    else:
        print('Response:', r.data.decode('utf-8')[:200])

    print('\n=== ALL TESTS COMPLETE ===')

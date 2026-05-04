"""Dashboard page routes."""
import os
import json
import csv
import io
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, send_file, Response

dashboard_bp = Blueprint('dashboard', __name__)

# Internal model name → entry name in results/verified_metrics.json
_VERIFIED_METRICS_KEY = {
    'XGBoost (Class Weights)': 'XGBoost Baseline (CW)',
    'XGBoost (SMOTE+Tuned)':   'XGBoost SMOTE+tuned',
    'AE+XGBoost':              'AE + XGBoost SMOTE+tuned',
    'AE+BDS+XGBoost':          'AE + BDS + XGBoost (full)',
}


def _load_active_model_metrics(active_model_name):
    """Return {f1, precision, recall} for the active model at threshold=0.5,
    or None if not found. LSTM+RF lives in `gap_experiments` and uses
    different key casing."""
    here = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.normpath(os.path.join(here, '..', '..', 'results', 'verified_metrics.json'))
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        data = json.load(f)

    if active_model_name == 'LSTM+RF':
        for e in data.get('gap_experiments', []):
            if e.get('experiment_name') == 'LSTM_reproduced_baseline':
                return {'f1': e['F1'], 'precision': e['Precision'], 'recall': e['Recall']}
        return None

    target = _VERIFIED_METRICS_KEY.get(active_model_name)
    if not target:
        return None
    for m in data.get('models', []):
        if m.get('model') == target and m.get('threshold') == 0.5:
            return {'f1': m['f1'], 'precision': m['precision'], 'recall': m['recall']}
    return None


def init_dashboard(app, model_manager, db):

    @dashboard_bp.route('/')
    def index():
        stats = db.get_stats()
        model_info = model_manager.get_model_info()
        active_metrics = _load_active_model_metrics(model_info.get('active_model'))
        return render_template('dashboard.html',
                               stats=stats,
                               model_info=model_info,
                               active_metrics=active_metrics)

    @dashboard_bp.route('/analyse')
    def analyse_page():
        return render_template('analyse.html')

    @dashboard_bp.route('/monitor')
    def monitor_page():
        return render_template('monitor.html')

    @dashboard_bp.route('/predict')
    def predict_page():
        return render_template('predict.html')

    @dashboard_bp.route('/performance')
    def performance_page():
        return render_template('model_performance.html', model_info=model_manager.get_model_info())

    @dashboard_bp.route('/settings')
    def settings_page():
        return render_template('settings.html', model_info=model_manager.get_model_info())

    @dashboard_bp.route('/api/sample/<sample_type>')
    def download_sample(sample_type):
        """Generate sample CSV files for testing."""
        from app.config import DATA_DIR
        test_path = os.path.join(DATA_DIR, 'fraudTest_engineered.csv')
        test_df = pd.read_csv(test_path)

        np.random.seed(42)
        if sample_type == 'mixed':
            normals = test_df[test_df['is_fraud'] == 0].sample(45, random_state=42)
            frauds = test_df[test_df['is_fraud'] == 1].sample(5, random_state=42)
            sample = pd.concat([normals, frauds]).sample(frac=1, random_state=42)
            filename = 'sample_mixed.csv'
        elif sample_type == 'fraud':
            normals = test_df[test_df['is_fraud'] == 0].sample(15, random_state=42)
            frauds = test_df[test_df['is_fraud'] == 1].sample(15, random_state=42)
            sample = pd.concat([normals, frauds]).sample(frac=1, random_state=42)
            filename = 'sample_fraud_heavy.csv'
        else:
            return "Unknown sample type", 404

        output = io.StringIO()
        sample.to_csv(output, index=False)
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )

    app.register_blueprint(dashboard_bp)

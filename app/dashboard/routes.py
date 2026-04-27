"""Dashboard page routes."""
import os
import json
import csv
import io
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, send_file, Response

dashboard_bp = Blueprint('dashboard', __name__)


def init_dashboard(app, model_manager, db):

    @dashboard_bp.route('/')
    def index():
        stats = db.get_stats()
        model_info = model_manager.get_model_info()
        return render_template('dashboard.html', stats=stats, model_info=model_info)

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

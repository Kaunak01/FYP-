"""Model Manager: loads, switches, and manages all saved models."""
import os
import json
import time
import logging
import numpy as np
import joblib
import torch
import torch.nn as nn
import shap
from app.config import MODEL_FILES, STATS_FILES, FEATURE_COLS

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d,10), nn.ReLU(), nn.Dropout(0.2), nn.Linear(10,5), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(5,10), nn.ReLU(), nn.Dropout(0.2), nn.Linear(10,d))
    def forward(self, x): return self.decoder(self.encoder(x))


class ModelManager:
    """Manages all available fraud detection models."""

    def __init__(self):
        self.models = {}
        self.active_model = None
        self.active_model_name = None
        self.ae = None
        self.scaler = None
        self.bds_profiles = None
        self.ga_params = None
        self.explainers = {}
        self._load_all()

    def _load_all(self):
        """Load all available models with graceful fallback."""
        # Load XGBoost models
        for name, key in [('XGBoost (Class Weights)', 'xgb_cw'),
                          ('XGBoost (SMOTE+Tuned)', 'xgb_tuned'),
                          ('AE+XGBoost', 'ae_xgb'),
                          ('AE+BDS+XGBoost', 'bds_xgb')]:
            path = MODEL_FILES.get(key)
            if path and os.path.exists(path):
                try:
                    self.models[name] = joblib.load(path)
                    logger.info("Loaded %s from %s", name, path)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", name, e)

        # Load autoencoder + scaler
        ae_path = MODEL_FILES.get('ae_model')
        scaler_path = MODEL_FILES.get('ae_scaler')
        if ae_path and os.path.exists(ae_path) and scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                self.ae = Autoencoder(len(FEATURE_COLS))
                self.ae.load_state_dict(torch.load(ae_path, weights_only=True))
                self.ae.eval()
                logger.info("Loaded autoencoder and scaler")
            except Exception as e:
                logger.warning("Failed to load autoencoder: %s", e)

        # Load BDS profiles + GA params
        profiles_path = MODEL_FILES.get('bds_profiles')
        ga_path = MODEL_FILES.get('ga_params')
        if profiles_path and os.path.exists(profiles_path):
            try:
                self.bds_profiles = joblib.load(profiles_path)
                logger.info("Loaded BDS profiles")
            except Exception as e:
                logger.warning("Failed to load BDS profiles: %s", e)
        if ga_path and os.path.exists(ga_path):
            try:
                with open(ga_path) as f:
                    self.ga_params = json.load(f)
                logger.info("Loaded GA params")
            except Exception as e:
                logger.warning("Failed to load GA params: %s", e)

        # Set active model (prefer best hybrid model)
        for preferred in ['AE+BDS+XGBoost', 'AE+XGBoost', 'XGBoost (SMOTE+Tuned)', 'XGBoost (Class Weights)']:
            if preferred in self.models:
                self.set_active(preferred)
                break

        if not self.models:
            logger.error("NO MODELS LOADED — system offline")

    def set_active(self, model_name):
        """Switch the active prediction model."""
        if model_name in self.models:
            self.active_model = self.models[model_name]
            self.active_model_name = model_name
            # Create SHAP explainer if not cached
            if model_name not in self.explainers:
                self.explainers[model_name] = shap.TreeExplainer(self.active_model)
            logger.info("Active model set to: %s", model_name)
            return True
        logger.warning("Model '%s' not found", model_name)
        return False

    def get_model_info(self):
        """Return info about all loaded models and active model."""
        info = {
            'active_model': self.active_model_name,
            'available_models': list(self.models.keys()),
            'autoencoder_loaded': self.ae is not None,
            'bds_profiles_loaded': self.bds_profiles is not None,
            'feature_count': self._get_feature_count(),
        }
        if self.active_model:
            params = self.active_model.get_params()
            info['active_params'] = {
                'n_estimators': params.get('n_estimators'),
                'max_depth': params.get('max_depth'),
                'learning_rate': params.get('learning_rate'),
            }
        return info

    def _get_feature_count(self):
        """Number of features the active model expects."""
        if self.active_model_name == 'AE+XGBoost':
            return 15
        elif self.active_model_name == 'AE+BDS+XGBoost':
            return 19
        return 14

    def compute_recon_error(self, features_array):
        """Compute autoencoder reconstruction error."""
        if self.ae is None or self.scaler is None:
            return 0.0
        scaled = self.scaler.transform(features_array.reshape(1, -1))
        with torch.no_grad():
            recon = self.ae(torch.FloatTensor(scaled)).numpy()
        return float(np.mean((scaled - recon) ** 2))

    def compute_bds_scores(self, features_dict):
        """Compute 4 BDS scores using global stats fallback."""
        if self.bds_profiles is None or self.ga_params is None:
            return 0.0, 0.0, 0.0, 0.0

        gs = self.bds_profiles['global_stats']
        params = list(self.ga_params['params'].values())
        at, ac, tt, tc, ft, fc, ct, cc_, mh, sm = params

        amt = features_dict.get('amt', 0)
        hour = int(features_dict.get('hour', 12))
        cat = int(features_dict.get('category_encoded', 0))
        vel = features_dict.get('velocity_1h', 1)

        # Amount deviation
        amt_z = abs(amt - gs['amt_mean']) / gs['amt_std'] if gs['amt_std'] > 0 else 0
        amount_score = min(max(amt_z - at, 0), ac)

        # Time deviation
        hour_prob = gs['hour_prob'].get(str(hour), 1/24)
        import math
        time_raw = -math.log(hour_prob + sm)
        time_score = min(max(time_raw - tt, 0), tc)

        # Frequency deviation
        freq_raw = max(vel / gs['vel_mean'] - 1.0, 0) if gs['vel_mean'] > 0 else 0
        freq_score = min(max(freq_raw - ft, 0), fc)

        # Category deviation
        cat_prob = gs['cat_prob'].get(str(cat), 1 / gs['n_categories'])
        cat_raw = -math.log(cat_prob + sm)
        cat_score = min(max(cat_raw - ct, 0), cc_)

        return amount_score, time_score, freq_score, cat_score

    def predict_all(self, features_dict):
        """Run the same transaction through all 4 loaded models. Used for compare mode."""
        base_arr = np.array([features_dict[f] for f in FEATURE_COLS], dtype=np.float64)
        recon_err = self.compute_recon_error(base_arr) if self.ae is not None else 0.0
        ae_arr = np.append(base_arr, recon_err)
        bds = self.compute_bds_scores(features_dict)
        bds_arr = np.concatenate([ae_arr, list(bds)])

        feature_map = {
            'XGBoost (Class Weights)': base_arr,
            'XGBoost (SMOTE+Tuned)':   base_arr,
            'AE+XGBoost':              ae_arr,
            'AE+BDS+XGBoost':          bds_arr,
        }
        code_map = {
            'xgboost_baseline': 'XGBoost (Class Weights)',
            'xgboost_smote':    'XGBoost (SMOTE+Tuned)',
            'ae_xgboost':       'AE+XGBoost',
            'ae_bds_xgboost':   'AE+BDS+XGBoost',
        }

        results = {}
        for code, internal in code_map.items():
            model = self.models.get(internal)
            if model is None:
                continue
            X = feature_map[internal].reshape(1, -1)
            prob = float(model.predict_proba(X)[0, 1])
            if prob >= 0.5:
                decision = 'FRAUD'
            elif prob >= 0.35:
                decision = 'REVIEW'
            else:
                decision = 'NORMAL'
            results[code] = {'probability': round(prob, 4), 'decision': decision}
        return results

    def predict(self, features_dict):
        """Make a prediction using the active model.

        Args:
            features_dict: dict with 14 base features

        Returns:
            dict with probability, shap_values, shap_feature_names, processing_time_ms
        """
        if self.active_model is None:
            return {'error': 'No model loaded', 'probability': 0.0}

        t0 = time.perf_counter()

        # Build base feature array
        base_features = np.array([features_dict[f] for f in FEATURE_COLS], dtype=np.float64)
        feature_names = FEATURE_COLS.copy()

        # Add recon_error if AE model
        if self.active_model_name in ('AE+XGBoost', 'AE+BDS+XGBoost'):
            recon_error = self.compute_recon_error(base_features)
            model_features = np.append(base_features, recon_error)
            feature_names = feature_names + ['recon_error']
        else:
            model_features = base_features

        # Add BDS scores if BDS model
        if self.active_model_name == 'AE+BDS+XGBoost':
            bds = self.compute_bds_scores(features_dict)
            model_features = np.concatenate([model_features, list(bds)])
            feature_names = feature_names + ['bds_amount', 'bds_time', 'bds_freq', 'bds_category']

        # Predict
        X = model_features.reshape(1, -1)
        probability = float(self.active_model.predict_proba(X)[0, 1])

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            'probability': probability,
            'shap_values': None,
            'shap_feature_names': feature_names,
            'features': model_features,
            'processing_time_ms': elapsed_ms,
        }

    def predict_with_shap(self, features_dict):
        """Predict with SHAP explanation (slower, for single transaction analysis)."""
        result = self.predict(features_dict)
        if result.get('error'):
            return result

        base_features = np.array([features_dict[f] for f in FEATURE_COLS], dtype=np.float64)
        feature_names = FEATURE_COLS.copy()
        if self.active_model_name in ('AE+XGBoost', 'AE+BDS+XGBoost'):
            recon_error = self.compute_recon_error(base_features)
            model_features = np.append(base_features, recon_error)
            feature_names = feature_names + ['recon_error']
        else:
            model_features = base_features
        if self.active_model_name == 'AE+BDS+XGBoost':
            bds = self.compute_bds_scores(features_dict)
            model_features = np.concatenate([model_features, list(bds)])
            feature_names = feature_names + ['bds_amount', 'bds_time', 'bds_freq', 'bds_category']

        X = model_features.reshape(1, -1)
        if self.active_model_name not in self.explainers:
            self.explainers[self.active_model_name] = shap.TreeExplainer(self.active_model)
        explainer = self.explainers[self.active_model_name]
        try:
            result['shap_values'] = explainer.shap_values(X)[0]
            result['shap_feature_names'] = feature_names
        except Exception as e:
            logger.warning("SHAP failed: %s", e)

        return result

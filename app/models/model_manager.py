"""Model Manager: loads, switches, and manages all saved models."""
import os
# Use PyTorch as the Keras 3 backend so we don't require TensorFlow.
os.environ.setdefault('KERAS_BACKEND', 'torch')

import json
import time
import logging
import numpy as np
import joblib
import torch
import torch.nn as nn
import shap
from app.config import MODEL_FILES, STATS_FILES, FEATURE_COLS, LSTM_SEQ_LEN

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
        # LSTM + RF hybrid (comparator). Keras backbone produces a fraud probability
        # from a sequence of 5 prior transactions; RF takes [lstm_prob, ...14 scaled features].
        self.lstm_keras = None
        self.lstm_rf_scaler = None
        self.lstm_rf_clf = None
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

        # Load LSTM + RF hybrid (comparator)
        keras_path  = MODEL_FILES.get('lstm_keras')
        rf_sc_path  = MODEL_FILES.get('lstm_rf_scaler')
        rf_clf_path = MODEL_FILES.get('lstm_rf_clf')
        if all(p and os.path.exists(p) for p in (keras_path, rf_sc_path, rf_clf_path)):
            try:
                import keras as _keras
                self.lstm_keras    = _keras.models.load_model(keras_path, compile=False)
                self.lstm_rf_scaler = joblib.load(rf_sc_path)
                self.lstm_rf_clf   = joblib.load(rf_clf_path)
                self.models['LSTM+RF'] = self.lstm_rf_clf
                logger.info("Loaded LSTM+RF hybrid (Keras + scaler + RF)")
            except Exception as e:
                logger.warning("Failed to load LSTM+RF hybrid: %s", e)

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
            # Create SHAP explainer if not cached. TreeExplainer works for both
            # XGBoost variants and the LSTM+RF's RandomForest head; for the latter
            # the LSTM-derived feature shows up as one of 15 features.
            if model_name not in self.explainers:
                try:
                    self.explainers[model_name] = shap.TreeExplainer(self.active_model)
                except Exception as e:
                    logger.warning("SHAP explainer init failed for %s: %s", model_name, e)
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
        elif self.active_model_name == 'LSTM+RF':
            return 15  # 1 LSTM probability + 14 scaled static features
        return 14

    def compute_lstm_prob(self, features_dict, prior_features=None):
        """Build a (SEQ_LEN, 14) sequence and return the LSTM fraud probability.

        prior_features: optional list of feature dicts (oldest → newest) for the
        same card; the current row is appended last. If fewer than SEQ_LEN-1
        priors are supplied, the head of the sequence is zero-padded — this
        is the cold-start case and matches how the model behaves on cards with
        very short history.
        """
        if self.lstm_keras is None or self.lstm_rf_scaler is None:
            return 0.0
        rows = list(prior_features or [])[-(LSTM_SEQ_LEN - 1):]
        rows.append(features_dict)
        # Pad at the start with zeros if we have fewer than SEQ_LEN rows.
        pad = LSTM_SEQ_LEN - len(rows)
        seq = np.zeros((LSTM_SEQ_LEN, len(FEATURE_COLS)), dtype=np.float32)
        for i, row in enumerate(rows):
            arr = np.array([row.get(c, 0.0) for c in FEATURE_COLS], dtype=np.float32).reshape(1, -1)
            seq[pad + i] = self.lstm_rf_scaler.transform(arr)[0]
        prob = float(self.lstm_keras.predict(seq.reshape(1, LSTM_SEQ_LEN, len(FEATURE_COLS)), verbose=0)[0, 0])
        return prob

    def _build_lstm_rf_input(self, features_dict, prior_features=None):
        """Return (15-vec, feature_names) for the LSTM+RF classifier."""
        lstm_prob = self.compute_lstm_prob(features_dict, prior_features)
        static = np.array([features_dict.get(c, 0.0) for c in FEATURE_COLS], dtype=np.float64).reshape(1, -1)
        scaled = self.lstm_rf_scaler.transform(static)[0]
        vec = np.concatenate([[lstm_prob], scaled])
        names = ['lstm_sequence_prob'] + list(FEATURE_COLS)
        return vec, names

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

    def predict_all(self, features_dict, prior_features=None):
        """Run the same transaction through all loaded models. Used for compare mode."""
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
            'lstm_rf_hybrid':   'LSTM+RF',
        }

        results = {}
        for code, internal in code_map.items():
            model = self.models.get(internal)
            if model is None:
                continue
            if internal == 'LSTM+RF':
                vec, _ = self._build_lstm_rf_input(features_dict, prior_features)
                X = vec.reshape(1, -1)
            else:
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

    def predict(self, features_dict, prior_features=None):
        """Make a prediction using the active model.

        Args:
            features_dict: dict with 14 base features
            prior_features: optional list of prior feature dicts (for LSTM+RF sequence)

        Returns:
            dict with probability, shap_values, shap_feature_names, processing_time_ms
        """
        if self.active_model is None:
            return {'error': 'No model loaded', 'probability': 0.0}

        t0 = time.perf_counter()

        if self.active_model_name == 'LSTM+RF':
            model_features, feature_names = self._build_lstm_rf_input(features_dict, prior_features)
            X = model_features.reshape(1, -1)
            probability = float(self.active_model.predict_proba(X)[0, 1])
            return {
                'probability': probability,
                'shap_values': None,
                'shap_feature_names': feature_names,
                'features': model_features,
                'processing_time_ms': (time.perf_counter() - t0) * 1000,
            }

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

    def predict_with_shap(self, features_dict, prior_features=None):
        """Predict with SHAP explanation (slower, for single transaction analysis)."""
        result = self.predict(features_dict, prior_features=prior_features)
        if result.get('error'):
            return result

        if self.active_model_name == 'LSTM+RF':
            model_features = result['features']
            feature_names = result['shap_feature_names']
        else:
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
            try:
                self.explainers[self.active_model_name] = shap.TreeExplainer(self.active_model)
            except Exception as e:
                logger.warning("SHAP explainer init failed: %s", e)
                self.explainers[self.active_model_name] = None
        explainer = self.explainers.get(self.active_model_name)
        try:
            if explainer is not None:
                sv = explainer.shap_values(X)
                # SHAP shape varies by backend:
                #   XGBoost (binary):     ndarray (1, n_features)
                #   RandomForest (newer): ndarray (1, n_features, 2) — last axis = classes
                #   RandomForest (older): list[2] of ndarray (1, n_features)
                if isinstance(sv, list) and len(sv) == 2:
                    sv_arr = np.asarray(sv[1])[0]
                else:
                    sv_arr = np.asarray(sv)[0]
                if sv_arr.ndim == 2 and sv_arr.shape[-1] == 2:
                    sv_arr = sv_arr[:, 1]
                result['shap_values'] = sv_arr
                result['shap_feature_names'] = feature_names
                result['features'] = model_features
        except Exception as e:
            logger.warning("SHAP failed: %s", e)

        return result

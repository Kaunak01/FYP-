"""Core fraud prediction engine with class hierarchy."""
import numpy as np
import pandas as pd
import time
import joblib
import json
import torch
import torch.nn as nn
import shap
from test_config import (
    FEATURE_COLS, FEATURE_DESCRIPTIONS, FEATURE_VALID_RANGES,
    DEFAULT_THRESHOLD, OPTIMAL_THRESHOLD, TRAINING_STATS,
    CATEGORY_CODE_TO_NAME, get_risk_level, generate_explanation,
    MODEL_XGB_CW, MODEL_XGB_TUNED, MODEL_AE_PT, MODEL_AE_SCALER,
    MODEL_AE_XGB, MODEL_BDS_PROFILES, MODEL_GA_PARAMS, MODEL_BDS_XGB,
)


class PredictionResult:
    """Stores prediction output with full explanation."""

    def __init__(self, probability, feature_values, feature_names,
                 shap_values=None, processing_time_ms=0.0):
        self.probability = probability
        self.classification_default = "FRAUD" if probability >= DEFAULT_THRESHOLD else "NORMAL"
        self.classification_optimal = "FRAUD" if probability >= OPTIMAL_THRESHOLD else "NORMAL"
        self.risk_level = get_risk_level(probability)
        self.feature_values = feature_values
        self.feature_names = feature_names
        self.processing_time_ms = processing_time_ms

        # SHAP explanation
        self.shap_explanation = []
        if shap_values is not None:
            sorted_idx = np.argsort(np.abs(shap_values))[::-1]
            for idx in sorted_idx:
                fname = feature_names[idx]
                sval = shap_values[idx]
                fval = feature_values[idx]
                explanation = generate_explanation(fname, sval, fval)
                self.shap_explanation.append((fname, sval, fval, explanation))

        # Population comparison
        self.population_comparison = {}
        for i, fname in enumerate(feature_names):
            if fname in TRAINING_STATS['stats']:
                stats = TRAINING_STATS['stats'][fname]
                val = feature_values[i]
                mean_all = stats['all']['mean']
                std_all = stats['all']['std']
                mean_normal = stats['normal']['mean']
                std_normal = stats['normal']['std']
                mean_fraud = stats['fraud']['mean']

                z_all = (val - mean_all) / std_all if std_all > 0 else 0
                z_normal = (val - mean_normal) / std_normal if std_normal > 0 else 0

                unusual = "VERY UNUSUAL" if abs(z_normal) > 3 else "UNUSUAL" if abs(z_normal) > 2 else ""
                self.population_comparison[fname] = {
                    'value': val, 'pop_mean': mean_all, 'pop_std': std_all,
                    'normal_mean': mean_normal, 'fraud_mean': mean_fraud,
                    'z_score_all': z_all, 'z_score_normal': z_normal,
                    'unusual': unusual
                }

    def print_report(self, top_shap=5):
        """Print formatted prediction report."""
        print(f"\n{'='*65}")
        print(f"FRAUD PREDICTION REPORT")
        print(f"{'='*65}")
        print(f"  Probability:     {self.probability:.4f} ({self.probability*100:.1f}%)")
        print(f"  Classification:  {self.classification_default} (threshold={DEFAULT_THRESHOLD})")
        print(f"  Optimal class.:  {self.classification_optimal} (threshold={OPTIMAL_THRESHOLD})")
        print(f"  Risk level:      {self.risk_level}")
        print(f"  Processing time: {self.processing_time_ms:.2f} ms")

        if self.shap_explanation:
            print(f"\n  Top {top_shap} SHAP Drivers:")
            for i, (fname, sval, fval, explanation) in enumerate(self.shap_explanation[:top_shap]):
                direction = "TOWARD FRAUD" if sval > 0 else "AWAY FROM FRAUD"
                print(f"    {i+1}. {fname}: SHAP={sval:+.4f} ({direction})")
                print(f"       {explanation}")

        if self.population_comparison:
            print(f"\n  Population Comparison (unusual features):")
            unusual_feats = [(k, v) for k, v in self.population_comparison.items() if v['unusual']]
            if unusual_feats:
                for fname, comp in unusual_feats:
                    print(f"    {fname}: value={comp['value']:.2f}, normal_mean={comp['normal_mean']:.2f}, "
                          f"z={comp['z_score_normal']:.1f} std [{comp['unusual']}]")
            else:
                print(f"    No features are statistically unusual for this transaction.")
        print(f"{'='*65}")


class FraudPredictor:
    """Base class for fraud prediction. Children override _preprocess()."""

    def __init__(self, model_path, feature_names=None):
        self.model = joblib.load(model_path)
        self.feature_names = feature_names or FEATURE_COLS.copy()
        self.explainer = shap.TreeExplainer(self.model)
        self._model_path = model_path

    def _validate_transaction(self, transaction_dict):
        """Validate a transaction dict has all required features."""
        missing = [f for f in FEATURE_COLS if f not in transaction_dict]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        warnings_list = []
        for feat, val in transaction_dict.items():
            if feat not in FEATURE_COLS:
                continue
            if not isinstance(val, (int, float, np.integer, np.floating)):
                raise TypeError(f"Feature '{feat}' must be numeric, got {type(val).__name__}: {val}")
            if feat in FEATURE_VALID_RANGES:
                lo, hi = FEATURE_VALID_RANGES[feat]
                if lo is not None and val < lo:
                    warnings_list.append(f"WARNING: {feat}={val} is below minimum {lo}")
                if hi is not None and val > hi:
                    warnings_list.append(f"WARNING: {feat}={val} is above maximum {hi}")

        return warnings_list

    def _preprocess(self, feature_array):
        """Override in children for model-specific preprocessing.
        Takes raw 14-feature array, returns model-ready array."""
        return feature_array

    def predict_single(self, transaction_dict):
        """Predict fraud probability for a single transaction."""
        t0 = time.perf_counter()

        warnings_list = self._validate_transaction(transaction_dict)
        for w in warnings_list:
            print(f"  {w}")

        # Build feature array in correct order
        raw_features = np.array([transaction_dict[f] for f in FEATURE_COLS], dtype=np.float64)

        # Preprocess (child classes add recon_error, BDS scores, etc.)
        model_features = self._preprocess(raw_features)

        # Predict
        X = model_features.reshape(1, -1)
        probability = self.model.predict_proba(X)[0, 1]

        # SHAP
        shap_vals = self.explainer.shap_values(X)[0]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return PredictionResult(
            probability=probability,
            feature_values=model_features,
            feature_names=self.feature_names,
            shap_values=shap_vals,
            processing_time_ms=elapsed_ms
        )

    def predict_batch(self, dataframe):
        """Predict on a DataFrame. Returns list of PredictionResult and summary."""
        t0 = time.perf_counter()
        results = []

        raw_features = dataframe[FEATURE_COLS].values
        model_features = np.array([self._preprocess(row) for row in raw_features])

        probs = self.model.predict_proba(model_features)[:, 1]
        preds = (probs >= DEFAULT_THRESHOLD).astype(int)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        summary = {
            'total': len(probs),
            'fraud_count': int(preds.sum()),
            'normal_count': int((preds == 0).sum()),
            'avg_probability': float(probs.mean()),
            'max_probability': float(probs.max()),
            'processing_time_ms': elapsed_ms,
            'predictions': preds,
            'probabilities': probs,
        }

        return summary

    def explain_prediction(self, transaction_dict):
        """Get full SHAP explanation for a transaction."""
        result = self.predict_single(transaction_dict)
        return result

    def compare_to_population(self, transaction_dict):
        """Compare a transaction's features to population statistics."""
        result = self.predict_single(transaction_dict)
        print(f"\n{'Feature':<35s} {'Value':>10s} {'NormalMean':>10s} {'FraudMean':>10s} {'Z-Score':>8s} {'Flag':>12s}")
        print("-" * 90)
        for fname in FEATURE_COLS:
            if fname in result.population_comparison:
                c = result.population_comparison[fname]
                print(f"{fname:<35s} {c['value']:>10.2f} {c['normal_mean']:>10.2f} {c['fraud_mean']:>10.2f} "
                      f"{c['z_score_normal']:>8.1f} {c['unusual']:>12s}")
        return result


class XGBoostPredictor(FraudPredictor):
    """XGBoost with 14 raw features. No preprocessing needed."""

    def __init__(self, model_path=MODEL_XGB_TUNED):
        super().__init__(model_path, feature_names=FEATURE_COLS.copy())

    def _preprocess(self, feature_array):
        return feature_array


class AEXGBoostPredictor(FraudPredictor):
    """Autoencoder + XGBoost. Adds reconstruction error as 15th feature."""

    def __init__(self, model_path=MODEL_AE_XGB, ae_path=MODEL_AE_PT, scaler_path=MODEL_AE_SCALER):
        # Load AE and scaler before super().__init__
        self.scaler = joblib.load(scaler_path)
        self.ae = self._load_autoencoder(ae_path)
        feature_names = FEATURE_COLS.copy() + ['recon_error']
        super().__init__(model_path, feature_names=feature_names)

    def _load_autoencoder(self, ae_path):
        class Autoencoder(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(d,10), nn.ReLU(), nn.Dropout(0.2), nn.Linear(10,5), nn.ReLU())
                self.decoder = nn.Sequential(nn.Linear(5,10), nn.ReLU(), nn.Dropout(0.2), nn.Linear(10,d))
            def forward(self, x): return self.decoder(self.encoder(x))
        ae = Autoencoder(14)
        ae.load_state_dict(torch.load(ae_path, weights_only=True))
        ae.eval()
        return ae

    def _preprocess(self, feature_array):
        """Add reconstruction error as 15th feature."""
        scaled = self.scaler.transform(feature_array.reshape(1, -1))
        with torch.no_grad():
            recon = self.ae(torch.FloatTensor(scaled)).numpy()
        recon_error = float(np.mean((scaled - recon) ** 2))
        return np.append(feature_array, recon_error)


class BDSXGBoostPredictor(FraudPredictor):
    """Autoencoder + BDS + XGBoost. Adds recon_error + 4 BDS scores = 19 features."""

    def __init__(self, model_path=MODEL_BDS_XGB, ae_path=MODEL_AE_PT,
                 scaler_path=MODEL_AE_SCALER, profiles_path=MODEL_BDS_PROFILES,
                 ga_params_path=MODEL_GA_PARAMS):
        self.scaler = joblib.load(scaler_path)
        self.ae = AEXGBoostPredictor._load_autoencoder(self, ae_path)
        self.profiles = joblib.load(profiles_path)
        with open(ga_params_path) as f:
            ga = json.load(f)
        self.bds_params = list(ga['params'].values())

        feature_names = FEATURE_COLS.copy() + ['recon_error', 'bds_amount', 'bds_time', 'bds_freq', 'bds_category']
        super().__init__(model_path, feature_names=feature_names)

    def _compute_bds_single(self, feature_array):
        """Compute 4 BDS scores for a single transaction using global stats fallback."""
        # Since we don't have cc_num for synthetic transactions, use global stats
        gs = self.profiles['global_stats']
        params = self.bds_params
        at, ac, tt, tc, ft, fc, ct, cc_, mh, sm = params

        amt = feature_array[FEATURE_COLS.index('amt')]
        hour = int(feature_array[FEATURE_COLS.index('hour')])
        cat = int(feature_array[FEATURE_COLS.index('category_encoded')])
        vel = feature_array[FEATURE_COLS.index('velocity_1h')]

        # Amount deviation (using global stats)
        amt_z = abs(amt - gs['amt_mean']) / gs['amt_std'] if gs['amt_std'] > 0 else 0
        amount_score = min(max(amt_z - at, 0), ac)

        # Time deviation
        hour_prob = gs['hour_prob'].get(str(hour), 1/24)
        time_raw = -np.log(hour_prob + sm)
        time_score = min(max(time_raw - tt, 0), tc)

        # Frequency deviation
        freq_raw = max(vel / gs['vel_mean'] - 1.0, 0) if gs['vel_mean'] > 0 else 0
        freq_score = min(max(freq_raw - ft, 0), fc)

        # Category deviation
        cat_prob = gs['cat_prob'].get(str(cat), 1 / gs['n_categories'])
        cat_raw = -np.log(cat_prob + sm)
        cat_score = min(max(cat_raw - ct, 0), cc_)

        return amount_score, time_score, freq_score, cat_score

    def _preprocess(self, feature_array):
        """Add recon_error + 4 BDS scores = 19 features."""
        # Reconstruction error
        scaled = self.scaler.transform(feature_array.reshape(1, -1))
        with torch.no_grad():
            recon = self.ae(torch.FloatTensor(scaled)).numpy()
        recon_error = float(np.mean((scaled - recon) ** 2))

        # BDS scores
        bds = self._compute_bds_single(feature_array)

        return np.concatenate([feature_array, [recon_error], list(bds)])


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================
def load_all_predictors():
    """Load all available predictors."""
    return {
        'XGBoost (CW)': XGBoostPredictor(MODEL_XGB_CW),
        'XGBoost (Tuned)': XGBoostPredictor(MODEL_XGB_TUNED),
        'AE+XGBoost': AEXGBoostPredictor(),
        'AE+BDS+XGBoost': BDSXGBoostPredictor(),
    }

def transaction_from_row(df, row_idx):
    """Convert a DataFrame row to a transaction dict."""
    return {col: float(df.iloc[row_idx][col]) for col in FEATURE_COLS}


if __name__ == '__main__':
    import pandas as pd
    from test_config import TEST_CSV

    print("Testing FraudPredictor on 3 transactions...\n")

    # Load test data
    test_df = pd.read_csv(TEST_CSV)

    # Test 1: XGBoostPredictor on a normal transaction (row 0)
    print("=" * 70)
    print("TEST 1: XGBoostPredictor — Row 0 (should be NORMAL)")
    print("=" * 70)
    xgb_pred = XGBoostPredictor()
    txn1 = transaction_from_row(test_df, 0)
    result1 = xgb_pred.predict_single(txn1)
    result1.print_report()

    # Test 2: AEXGBoostPredictor on a fraud transaction
    print("\n" + "=" * 70)
    print("TEST 2: AEXGBoostPredictor — First fraud in test set")
    print("=" * 70)
    fraud_idx = test_df[test_df['is_fraud'] == 1].index[0]
    ae_pred = AEXGBoostPredictor()
    txn2 = transaction_from_row(test_df, fraud_idx)
    result2 = ae_pred.predict_single(txn2)
    result2.print_report()
    print(f"  Actual label: FRAUD")

    # Test 3: BDSXGBoostPredictor on a synthetic suspicious transaction
    print("\n" + "=" * 70)
    print("TEST 3: BDSXGBoostPredictor — Synthetic suspicious transaction")
    print("=" * 70)
    from test_config import get_median_transaction, CATEGORY_NAME_TO_CODE
    txn3 = get_median_transaction()
    txn3['amt'] = 800.0
    txn3['hour'] = 3.0
    txn3['is_night'] = 1.0
    txn3['velocity_1h'] = 5.0
    txn3['amount_velocity_1h'] = 2500.0
    bds_pred = BDSXGBoostPredictor()
    result3 = bds_pred.predict_single(txn3)
    result3.print_report()

    print("\n" + "=" * 70)
    print("ALL 3 TESTS COMPLETE — NO CRASHES")
    print("=" * 70)

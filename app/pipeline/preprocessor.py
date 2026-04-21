"""Preprocessor: transforms raw bank transaction input into model-ready features."""
import math
import json
import logging
from datetime import datetime
from app.config import FEATURE_COLS, STATS_FILES, GENDER_MAP, DAY_MAP

logger = logging.getLogger(__name__)


class Preprocessor:
    """Takes raw transaction input (bank format), applies all transformations,
    returns feature vector in exact column order the model expects."""

    def __init__(self, db=None):
        self.db = db  # Database instance for velocity computation

        # Load stats
        with open(STATS_FILES['training_stats']) as f:
            self.training_stats = json.load(f)

        with open(STATS_FILES['category_mapping']) as f:
            cat_data = json.load(f)
            self.name_to_code = cat_data['name_to_code']
            self.code_to_name = {int(k): v for k, v in cat_data['code_to_name'].items()}

        with open(STATS_FILES['category_aliases']) as f:
            self.category_aliases = json.load(f)

        # Compute defaults from training stats
        self.defaults = {}
        for feat in FEATURE_COLS:
            if feat in self.training_stats['stats']:
                self.defaults[feat] = self.training_stats['stats'][feat]['all']['median']
        self.gender_default = self.training_stats['stats']['gender_encoded']['all']['mean']

        # Category frequency for default
        self.default_category = max(cat_data['frequency'], key=cat_data['frequency'].get)
        self.default_category_code = self.name_to_code[self.default_category]

        # 99th percentile caps for input sanitization
        self.caps = {}
        for feat in FEATURE_COLS:
            if feat in self.training_stats['stats']:
                self.caps[feat] = self.training_stats['stats'][feat]['all']['p99']

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Haversine distance in km between two points."""
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    def _encode_category(self, raw_category):
        """Two-step category encoding: bank term → sparkov name → integer code."""
        if raw_category is None:
            logger.warning("Missing category — defaulting to '%s'", self.default_category)
            return self.default_category_code, self.default_category

        raw = str(raw_category).strip()

        # Step 1: exact match to Sparkov category name
        if raw in self.name_to_code:
            return self.name_to_code[raw], raw

        # Step 2: lowercase alias lookup
        raw_lower = raw.lower().strip()
        if raw_lower in self.name_to_code:
            return self.name_to_code[raw_lower], raw_lower

        if raw_lower in self.category_aliases:
            sparkov_name = self.category_aliases[raw_lower]
            return self.name_to_code[sparkov_name], sparkov_name

        # Step 3: no match — default
        logger.warning("Unknown category '%s' — defaulting to '%s'. "
                       "Consider adding this to category_aliases.json", raw, self.default_category)
        return self.default_category_code, self.default_category

    def _encode_gender(self, raw_gender):
        """Encode gender with fallback to training mean."""
        if raw_gender is None or str(raw_gender).strip() == '':
            logger.warning("Missing gender — using training mean (%.2f)", self.gender_default)
            return self.gender_default

        g = str(raw_gender).strip().upper()
        if g in GENDER_MAP:
            return GENDER_MAP[g]

        logger.warning("Unknown gender '%s' — using training mean (%.2f)", raw_gender, self.gender_default)
        return self.gender_default

    def _sanitize_numeric(self, value, field_name, min_val=None, max_val=None):
        """Sanitize a numeric input: type check, range check, cap at p99."""
        if value is None:
            default = self.defaults.get(field_name, 0)
            logger.warning("Missing %s — using default %.2f", field_name, default)
            return default

        try:
            val = float(value)
        except (ValueError, TypeError):
            default = self.defaults.get(field_name, 0)
            logger.warning("Invalid %s='%s' — using default %.2f", field_name, value, default)
            return default

        if min_val is not None and val < min_val:
            logger.warning("%s=%.2f below minimum %.2f — clamping", field_name, val, min_val)
            val = min_val

        if max_val is not None and val > max_val:
            logger.warning("%s=%.2f above maximum %.2f — clamping", field_name, val, max_val)
            val = max_val

        # Cap at 99th percentile if extreme
        if field_name in self.caps and val > self.caps[field_name] * 2:
            logger.warning("%s=%.2f exceeds 2x p99 (%.2f) — capping", field_name, val, self.caps[field_name])
            val = self.caps[field_name] * 2

        return val

    def _parse_timestamp(self, timestamp_str, unix_time=None):
        """Parse timestamp and extract temporal features.
        Falls back to unix_time (epoch seconds) if timestamp_str is missing/invalid.
        """
        if timestamp_str is None and unix_time is not None:
            try:
                from datetime import timezone
                dt = datetime.fromtimestamp(float(unix_time))
                timestamp_str = dt.isoformat()
            except Exception:
                pass

        if timestamp_str is None:
            now = datetime.now()
            logger.warning("Missing timestamp — using current time")
            timestamp_str = now.isoformat()

        if isinstance(timestamp_str, datetime):
            dt = timestamp_str
        else:
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M:%S']:
                try:
                    dt = datetime.strptime(str(timestamp_str).strip(), fmt)
                    break
                except ValueError:
                    continue
            else:
                # Last-resort: try unix_time if normal parsing fails
                if unix_time is not None:
                    try:
                        dt = datetime.fromtimestamp(float(unix_time))
                    except Exception:
                        dt = datetime.now()
                else:
                    logger.warning("Cannot parse timestamp '%s' — using current time", timestamp_str)
                    dt = datetime.now()

        hour = dt.hour
        month = dt.month
        day_of_week = dt.weekday()  # Monday=0, Sunday=6
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if (hour >= 22 or hour < 6) else 0

        return dt, hour, month, day_of_week, is_weekend, is_night

    def process(self, raw_input, velocity_override=None):
        """Transform raw bank input into model-ready feature vector.

        Args:
            raw_input: dict with bank transaction fields
            velocity_override: dict with pre-computed velocity values
                              {'velocity_1h': x, 'velocity_24h': y, 'amount_velocity_1h': z}
                              If None, compute from card_history in database.

        Returns:
            dict with all 14 features in correct order, plus metadata
        """
        warnings = []
        original_input = dict(raw_input)

        # ---- Amount ----
        amt = self._sanitize_numeric(raw_input.get('amount', raw_input.get('amt')), 'amt', min_val=0)

        # ---- City population ----
        city_pop = self._sanitize_numeric(raw_input.get('city_population', raw_input.get('city_pop')),
                                          'city_pop', min_val=0)

        # ---- Timestamp → hour, month, day_of_week, is_weekend, is_night ----
        dt, hour, month, day_of_week, is_weekend, is_night = self._parse_timestamp(
            raw_input.get('timestamp', raw_input.get('trans_date_trans_time')),
            unix_time=raw_input.get('unix_time'),
        )
        # If pre-engineered temporal features are present in the input (e.g. from an
        # engineered CSV), use them directly — they are more accurate than re-deriving
        # from a potentially missing/wrong timestamp.
        if raw_input.get('hour') is not None:
            try:
                hour = int(float(raw_input['hour']))
                is_night = 1 if (hour >= 22 or hour < 6) else 0
            except (ValueError, TypeError):
                pass
        if raw_input.get('month') is not None:
            try:
                month = int(float(raw_input['month']))
            except (ValueError, TypeError):
                pass
        if raw_input.get('is_night') is not None:
            try:
                is_night = int(float(raw_input['is_night']))
            except (ValueError, TypeError):
                pass
        if raw_input.get('is_weekend') is not None:
            try:
                is_weekend = int(float(raw_input['is_weekend']))
            except (ValueError, TypeError):
                pass
        if raw_input.get('day_of_week_encoded') is not None:
            try:
                day_of_week = int(float(raw_input['day_of_week_encoded']))
            except (ValueError, TypeError):
                pass

        # ---- Distance ----
        if 'distance_cardholder_merchant' in raw_input:
            distance = self._sanitize_numeric(raw_input['distance_cardholder_merchant'],
                                              'distance_cardholder_merchant', min_val=0)
        elif all(k in raw_input for k in ['cardholder_lat', 'cardholder_long', 'merchant_lat', 'merchant_long']):
            try:
                distance = self._haversine(
                    float(raw_input['cardholder_lat']), float(raw_input['cardholder_long']),
                    float(raw_input['merchant_lat']), float(raw_input['merchant_long'])
                )
            except (ValueError, TypeError):
                distance = self.defaults.get('distance_cardholder_merchant', 76.0)
                logger.warning("Invalid lat/long — using default distance %.1f", distance)
        else:
            distance = self.defaults.get('distance_cardholder_merchant', 76.0)

        # ---- Age ----
        age = self._sanitize_numeric(raw_input.get('cardholder_age', raw_input.get('age')),
                                     'age', min_val=1, max_val=120)

        # ---- Gender ----
        gender_encoded = self._encode_gender(
            raw_input.get('cardholder_gender', raw_input.get('gender', raw_input.get('gender_encoded')))
        )

        # ---- Category ----
        raw_cat = raw_input.get('merchant_category', raw_input.get('category',
                                raw_input.get('category_encoded')))
        # If it's already a number, use directly
        try:
            cat_code = int(float(raw_cat))
            if 0 <= cat_code <= 13:
                category_encoded = cat_code
                category_name = self.code_to_name.get(cat_code, f'code_{cat_code}')
            else:
                category_encoded, category_name = self._encode_category(raw_cat)
        except (ValueError, TypeError):
            category_encoded, category_name = self._encode_category(raw_cat)

        # ---- Velocity features ----
        velocity_source = 'unknown'
        card_number = raw_input.get('card_number', raw_input.get('cc_num', 'UNKNOWN'))

        if velocity_override and all(k in velocity_override for k in ['velocity_1h', 'velocity_24h', 'amount_velocity_1h']):
            velocity_1h = float(velocity_override['velocity_1h'])
            velocity_24h = float(velocity_override['velocity_24h'])
            amount_velocity_1h = float(velocity_override['amount_velocity_1h'])
            velocity_source = 'override'
        elif self.db is not None:
            # Compute from card history
            v1h, v24h, amt_v1h = self.db.get_card_velocity(card_number, dt)
            velocity_1h = float(v1h)
            velocity_24h = float(v24h)
            amount_velocity_1h = float(amt_v1h) + amt  # Add current transaction amount
            velocity_source = 'computed'
        else:
            # No database, no override — use defaults (first transaction)
            velocity_1h = 1.0
            velocity_24h = 1.0
            amount_velocity_1h = amt
            velocity_source = 'default'

        # ---- Build feature dict ----
        features = {
            'amt': amt,
            'city_pop': city_pop,
            'hour': float(hour),
            'month': float(month),
            'distance_cardholder_merchant': distance,
            'age': age,
            'is_weekend': float(is_weekend),
            'is_night': float(is_night),
            'velocity_1h': velocity_1h,
            'velocity_24h': velocity_24h,
            'amount_velocity_1h': amount_velocity_1h,
            'category_encoded': float(category_encoded),
            'gender_encoded': float(gender_encoded),
            'day_of_week_encoded': float(day_of_week),
        }

        metadata = {
            'transaction_id': raw_input.get('transaction_id', f'TXN-{id(raw_input)}'),
            'card_number': str(card_number),
            'timestamp': dt.isoformat(),
            'category_name': category_name,
            'velocity_source': velocity_source,
            'original_input': original_input,
        }

        return features, metadata


# ---- Quick test ----
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    pp = Preprocessor(db=None)

    # Test 1: Full bank input
    print("="*60)
    print("TEST 1: Full bank transaction input")
    print("="*60)
    txn1 = {
        "transaction_id": "TXN-001",
        "card_number": "4521-XXXX-XXXX-1234",
        "amount": 523.50,
        "merchant_category": "electronics",
        "merchant_lat": 40.7128,
        "merchant_long": -74.0060,
        "cardholder_lat": 40.7580,
        "cardholder_long": -73.9855,
        "timestamp": "2024-03-15 23:45:00",
        "cardholder_age": 34,
        "cardholder_gender": "M",
        "city_population": 8336817
    }
    features, meta = pp.process(txn1)
    print(f"  Input: ${txn1['amount']} at {txn1['merchant_category']}, {txn1['timestamp']}")
    print(f"  Category: '{txn1['merchant_category']}' -> {meta['category_name']} (code={features['category_encoded']:.0f})")
    print(f"  Distance: {features['distance_cardholder_merchant']:.2f} km")
    print(f"  is_night: {features['is_night']:.0f}, is_weekend: {features['is_weekend']:.0f}")
    print(f"  Velocity source: {meta['velocity_source']}")
    print(f"  Features: {list(features.values())}")

    # Test 2: Minimal input (missing fields)
    print("\n" + "="*60)
    print("TEST 2: Minimal input (missing fields)")
    print("="*60)
    txn2 = {
        "amount": 15.00,
        "merchant_category": "coffee shop",
        "timestamp": "2024-03-16 08:30:00",
    }
    features2, meta2 = pp.process(txn2)
    print(f"  Input: ${txn2['amount']} at '{txn2['merchant_category']}'")
    print(f"  Category: '{txn2['merchant_category']}' -> {meta2['category_name']} (code={features2['category_encoded']:.0f})")
    print(f"  Age: {features2['age']:.0f} (default)")
    print(f"  Gender: {features2['gender_encoded']:.2f} (default mean)")
    print(f"  City pop: {features2['city_pop']:.0f} (default)")
    print(f"  Velocity source: {meta2['velocity_source']}")

    # Test 3: Edge cases (bad inputs)
    print("\n" + "="*60)
    print("TEST 3: Bad inputs (sanitization)")
    print("="*60)
    txn3 = {
        "amount": -50,
        "merchant_category": "crypto exchange",
        "timestamp": "not-a-date",
        "cardholder_age": 200,
        "cardholder_gender": "X",
        "city_population": "unknown",
    }
    features3, meta3 = pp.process(txn3)
    print(f"  Input: amt={txn3['amount']}, cat='{txn3['merchant_category']}', age={txn3['cardholder_age']}")
    print(f"  Sanitized amt: {features3['amt']:.2f} (clamped from -50)")
    print(f"  Category: '{txn3['merchant_category']}' -> {meta3['category_name']} (defaulted)")
    print(f"  Age: {features3['age']:.0f} (capped from 200)")
    print(f"  Gender: {features3['gender_encoded']:.2f} (unknown -> default)")

    # Verify all 14 features present
    print(f"\n  All features present: {all(f in features3 for f in FEATURE_COLS)}")
    print(f"  Feature count: {len(features3)}")
    print("\n  ALL 3 TESTS COMPLETE")

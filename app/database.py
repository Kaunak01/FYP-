"""SQLite database manager for fraud detection system."""
import sqlite3
import os
import logging
from datetime import datetime, timedelta
from app.config import DB_PATH

logger = logging.getLogger(__name__)


class Database:
    """Manages all database operations for the fraud detection system."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_tables()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _create_tables(self):
        conn = self._connect()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE,
                    card_number TEXT,
                    timestamp TEXT,
                    amount REAL,
                    category TEXT,
                    amt REAL, city_pop REAL, hour INTEGER, month INTEGER,
                    distance_cardholder_merchant REAL, age REAL,
                    is_weekend INTEGER, is_night INTEGER,
                    velocity_1h REAL, velocity_24h REAL, amount_velocity_1h REAL,
                    category_encoded INTEGER, gender_encoded INTEGER,
                    day_of_week_encoded INTEGER,
                    probability REAL,
                    risk_level TEXT,
                    classification TEXT,
                    rule_triggers TEXT,
                    processing_time_ms REAL,
                    velocity_source TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT,
                    probability REAL,
                    risk_level TEXT,
                    classification TEXT,
                    amount REAL,
                    category TEXT,
                    rule_triggers TEXT,
                    explanation TEXT,
                    status TEXT DEFAULT 'pending',
                    analyst_notes TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    reviewed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT,
                    actual_label TEXT,
                    analyst_notes TEXT,
                    submitted_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS card_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_number TEXT,
                    timestamp TEXT,
                    amount REAL,
                    category_encoded INTEGER,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    model_name TEXT,
                    f1 REAL, precision_score REAL, recall REAL, roc_auc REAL,
                    transactions_processed INTEGER,
                    alerts_raised INTEGER,
                    false_positives INTEGER,
                    drift_status TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_card_history_card ON card_history(card_number);
                CREATE INDEX IF NOT EXISTS idx_card_history_ts ON card_history(timestamp);
                CREATE INDEX IF NOT EXISTS idx_transactions_card ON transactions(card_number);
                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
            """)
            conn.commit()
            logger.info("Database tables created/verified")
        finally:
            conn.close()

    # ---- Card History ----

    def add_card_transaction(self, card_number, timestamp, amount, category_encoded):
        """Store a transaction in card history for velocity computation."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO card_history (card_number, timestamp, amount, category_encoded) VALUES (?, ?, ?, ?)",
                (str(card_number), str(timestamp), float(amount), int(category_encoded))
            )
            conn.commit()
        finally:
            conn.close()

    def get_card_recent_features(self, card_number, limit=4):
        """Return up to `limit` most recent transactions' 14 feature columns
        for a card, ordered oldest → newest. Used to build LSTM sequences."""
        conn = self._connect()
        try:
            cols = ('amt, city_pop, hour, month, distance_cardholder_merchant, age, '
                    'is_weekend, is_night, velocity_1h, velocity_24h, amount_velocity_1h, '
                    'category_encoded, gender_encoded, day_of_week_encoded')
            rows = conn.execute(
                f"SELECT {cols} FROM transactions WHERE card_number = ? "
                f"ORDER BY id DESC LIMIT ?",
                (str(card_number), int(limit))
            ).fetchall()
            return [dict(r) for r in reversed(rows)]
        finally:
            conn.close()

    def get_card_velocity(self, card_number, current_timestamp):
        """Compute velocity features from card history.
        Returns (velocity_1h, velocity_24h, amount_velocity_1h)."""
        conn = self._connect()
        try:
            ts = datetime.fromisoformat(str(current_timestamp)) if isinstance(current_timestamp, str) else current_timestamp
            # Use space-separated format to match stored timestamps
            ts_1h = (ts - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            ts_24h = (ts - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')

            # Count transactions in last 1 hour (including current)
            row_1h = conn.execute(
                "SELECT COUNT(*) as cnt, COALESCE(SUM(amount), 0) as total "
                "FROM card_history WHERE card_number = ? AND timestamp >= ? AND timestamp <= ?",
                (str(card_number), ts_1h, ts_str)
            ).fetchone()

            # Count transactions in last 24 hours
            row_24h = conn.execute(
                "SELECT COUNT(*) as cnt FROM card_history "
                "WHERE card_number = ? AND timestamp >= ? AND timestamp <= ?",
                (str(card_number), ts_24h, ts_str)
            ).fetchone()

            # +1 for the current transaction (not yet stored)
            velocity_1h = (row_1h['cnt'] if row_1h else 0) + 1
            velocity_24h = (row_24h['cnt'] if row_24h else 0) + 1
            amount_velocity_1h = (row_1h['total'] if row_1h else 0)  # current amount added by caller

            return velocity_1h, velocity_24h, amount_velocity_1h
        finally:
            conn.close()

    # ---- Transactions ----

    def store_transaction(self, txn_data):
        """Store a processed transaction."""
        conn = self._connect()
        try:
            cols = ['transaction_id', 'card_number', 'timestamp', 'amount', 'category',
                    'amt', 'city_pop', 'hour', 'month', 'distance_cardholder_merchant',
                    'age', 'is_weekend', 'is_night', 'velocity_1h', 'velocity_24h',
                    'amount_velocity_1h', 'category_encoded', 'gender_encoded',
                    'day_of_week_encoded', 'probability', 'risk_level', 'classification',
                    'rule_triggers', 'processing_time_ms', 'velocity_source']
            values = [txn_data.get(c) for c in cols]
            placeholders = ','.join(['?'] * len(cols))
            col_names = ','.join(cols)
            conn.execute(f"INSERT OR REPLACE INTO transactions ({col_names}) VALUES ({placeholders})", values)
            conn.commit()
        finally:
            conn.close()

    def get_recent_transactions(self, limit=100):
        """Get most recent transactions."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM transactions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ---- Alerts ----

    def store_alert(self, alert_data):
        """Store an alert."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO alerts (transaction_id, probability, risk_level, classification, "
                "amount, category, rule_triggers, explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (alert_data.get('transaction_id'), alert_data.get('probability'),
                 alert_data.get('risk_level'), alert_data.get('classification'),
                 alert_data.get('amount'), alert_data.get('category'),
                 alert_data.get('rule_triggers'), alert_data.get('explanation'))
            )
            conn.commit()
        finally:
            conn.close()

    def get_alerts(self, status=None, limit=100):
        """Get alerts, optionally filtered by status."""
        conn = self._connect()
        try:
            if status:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_alert_status(self, alert_id, status, notes=None):
        """Update alert status (pending → confirmed/false_alarm/dismissed)."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE alerts SET status = ?, analyst_notes = ?, reviewed_at = datetime('now') WHERE id = ?",
                (status, notes, alert_id)
            )
            conn.commit()
        finally:
            conn.close()

    # ---- Feedback ----

    def store_feedback(self, transaction_id, actual_label, notes=None):
        """Store analyst feedback."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO feedback (transaction_id, actual_label, analyst_notes) VALUES (?, ?, ?)",
                (transaction_id, actual_label, notes)
            )
            conn.commit()
        finally:
            conn.close()

    # ---- Stats ----

    def get_stats(self):
        """Get summary statistics."""
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            confirmed = conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'confirmed'").fetchone()[0]
            false_alarms = conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'false_alarm'").fetchone()[0]
            pending = conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'pending'").fetchone()[0]

            money_saved = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM alerts WHERE status = 'confirmed'"
            ).fetchone()[0]
            money_at_risk = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM transactions "
                "WHERE classification = 'NORMAL' AND probability > 0.3"
            ).fetchone()[0]

            return {
                'total_processed': total,
                'alerts_raised': alerts,
                'confirmed_frauds': confirmed,
                'false_alarms': false_alarms,
                'pending_review': pending,
                'money_saved': money_saved,
                'money_at_risk': money_at_risk,
            }
        finally:
            conn.close()

    def get_cardholder_history(self, card_id):
        """Get full transaction history for a card with velocity & attack analysis."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT transaction_id, timestamp, amount, category, probability, classification,
                   velocity_1h, velocity_24h, amount_velocity_1h, hour, is_night,
                   amt, city_pop, month, distance_cardholder_merchant, age, is_weekend,
                   category_encoded, gender_encoded, day_of_week_encoded
                   FROM transactions WHERE card_number = ?
                   ORDER BY rowid DESC""",
                (str(card_id),)
            ).fetchall()

            if not rows:
                return None

            txns = [dict(r) for r in rows]
            total = len(txns)
            total_spent = sum(float(t.get('amount') or 0) for t in txns)
            avg_amount = total_spent / total if total > 0 else 0
            fraud_count = sum(1 for t in txns if t.get('classification') in ('FRAUD', 'REVIEW'))
            fraud_rate = fraud_count / total * 100 if total > 0 else 0
            timestamps = [t['timestamp'] for t in txns if t.get('timestamp')]
            first_seen = min(timestamps) if timestamps else None
            last_seen  = max(timestamps) if timestamps else None

            # Velocity analysis
            normal_txns = [t for t in txns if t.get('classification') == 'NORMAL']
            fraud_txns  = [t for t in txns if t.get('classification') in ('FRAUD', 'REVIEW')]

            normal_vel_vals    = [float(t.get('velocity_1h') or 1) for t in normal_txns]
            normal_daily_vel   = round(sum(normal_vel_vals) / len(normal_vel_vals), 1) if normal_vel_vals else 1.0
            normal_hourly_spend = (sum(float(t.get('amount') or 0) for t in normal_txns) / len(normal_txns)) if normal_txns else 0
            attack_velocity    = max((float(t.get('velocity_1h') or 0) for t in fraud_txns), default=0) if fraud_txns else 0
            attack_hourly_spend= max((float(t.get('amount_velocity_1h') or 0) for t in fraud_txns), default=0) if fraud_txns else 0
            vel_spike  = round(attack_velocity / max(normal_daily_vel, 0.1), 1)
            spend_spike= round(attack_hourly_spend / max(normal_hourly_spend, 1), 1)

            # Category breakdown
            cat_counts = {}
            for t in txns:
                cat = t.get('category') or 'unknown'
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

            # Time distribution
            time_dist = {'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0}
            for t in txns:
                h = int(t.get('hour') or 0)
                if   6 <= h < 12: time_dist['morning']   += 1
                elif 12 <= h < 18: time_dist['afternoon'] += 1
                elif 18 <= h < 22: time_dist['evening']   += 1
                else:              time_dist['night']      += 1

            # Attack detection: first consecutive fraud run from top (most-recent-first)
            attack_detected = fraud_count > 0
            attack_start_index = None
            attack_duration_minutes = None
            if fraud_txns:
                run_end = 0
                for i, t in enumerate(txns):
                    if t.get('classification') in ('FRAUD', 'REVIEW'):
                        run_end = i
                    else:
                        break
                attack_start_index = run_end
                try:
                    from datetime import datetime as _dt
                    fmt = '%Y-%m-%d %H:%M:%S'
                    t1 = _dt.strptime(str(txns[0]['timestamp'])[:19].replace('T', ' '), fmt)
                    t2 = _dt.strptime(str(txns[run_end]['timestamp'])[:19].replace('T', ' '), fmt)
                    attack_duration_minutes = max(0, int(abs((t1 - t2).total_seconds() / 60)))
                except Exception:
                    attack_duration_minutes = None

            # Feature columns for SHAP pass-through
            _FCOLS = ['amt','city_pop','age','hour','month','distance_cardholder_merchant',
                      'category_encoded','gender_encoded','day_of_week_encoded','is_weekend',
                      'is_night','velocity_1h','velocity_24h','amount_velocity_1h']

            transaction_list = []
            for i, t in enumerate(txns):
                features = {col: float(t[col]) for col in _FCOLS if t.get(col) is not None}
                transaction_list.append({
                    'index': total - i,
                    'timestamp': t.get('timestamp'),
                    'amount': round(float(t.get('amount') or 0), 2),
                    'category': t.get('category') or 'unknown',
                    'probability': round(float(t.get('probability') or 0), 4),
                    'decision': t.get('classification') or 'NORMAL',
                    'velocity_1h': int(float(t.get('velocity_1h') or 0)),
                    'amount_velocity_1h': round(float(t.get('amount_velocity_1h') or 0), 2),
                    'is_fraud': 1 if t.get('classification') in ('FRAUD', 'REVIEW') else 0,
                    'features': features,
                })

            return {
                'card_id': f"****{str(card_id)[-4:]}",
                'summary': {
                    'total_transactions': total,
                    'total_spent': round(total_spent, 2),
                    'avg_amount': round(avg_amount, 2),
                    'fraud_count': fraud_count,
                    'fraud_rate': round(fraud_rate, 1),
                    'first_seen': str(first_seen) if first_seen else None,
                    'last_seen':  str(last_seen)  if last_seen  else None,
                },
                'velocity_analysis': {
                    'normal_daily_velocity':  normal_daily_vel,
                    'normal_hourly_spend':    round(normal_hourly_spend, 2),
                    'attack_velocity':        float(attack_velocity),
                    'attack_hourly_spend':    round(float(attack_hourly_spend), 2),
                    'velocity_spike_factor':  vel_spike,
                    'spend_spike_factor':     spend_spike,
                },
                'category_breakdown': cat_counts,
                'time_distribution': time_dist,
                'transactions': transaction_list,
                'attack_detected': attack_detected,
                'attack_start_index': attack_start_index,
                'attack_duration_minutes': attack_duration_minutes,
            }
        finally:
            conn.close()

    def reset(self):
        """Clear all tables. Used before a new simulation."""
        conn = self._connect()
        try:
            for table in ['transactions', 'alerts', 'feedback', 'card_history', 'model_metrics']:
                conn.execute(f"DELETE FROM {table}")
            conn.commit()
            logger.info("Database reset complete")
        finally:
            conn.close()

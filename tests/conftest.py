"""Shared fixtures for the FYP pytest suite.

This file is loaded automatically by pytest. It provides:
- A session-scoped Flask app + ModelManager fixture (loads the Proposed
  AE+BDS+XGBoost model once per session — heavy cold-start ~5–10 s).
- A function-scoped, isolated Database fixture (fresh SQLite file per test).
- Sample known-legit and known-fraud transaction dicts.
- A session-scoped MonkeyPatch helper so the real fraud_detection.db file
  is never touched by the suite.
"""
from __future__ import annotations

import os
import pytest
from _pytest.monkeypatch import MonkeyPatch


# ---------------------------------------------------------------------------
# Session-scoped monkeypatch helper (pytest's built-in monkeypatch is
# function-scoped only)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ---------------------------------------------------------------------------
# Function-scoped fresh Database (new SQLite file per test, no real-DB pollution)
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_db(tmp_path):
    """Brand-new SQLite Database backed by a temp file. One per test."""
    from app.database import Database
    db_file = tmp_path / "fyp_test.db"
    return Database(db_path=str(db_file))


# ---------------------------------------------------------------------------
# Session-scoped Flask app + db + model_manager (Proposed model active by default)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def flask_stack(monkeypatch_session, tmp_path_factory):
    """Build the real Flask app via create_app() with an isolated DB path
    so the suite never writes to the production fraud_detection.db.
    Returns (app, db, model_manager). Active model is whatever
    ModelManager prefers (first available of: AE+BDS+XGBoost, AE+XGBoost,
    XGBoost SMOTE+Tuned, XGBoost CW)."""
    test_db_dir = tmp_path_factory.mktemp("flask_session_db")
    test_db_path = str(test_db_dir / "session.db")

    # Patch DB_PATH on both the config module and the database module
    # before create_app() runs.
    import app.config as _cfg
    monkeypatch_session.setattr(_cfg, "DB_PATH", test_db_path, raising=True)
    import app.database as _dbmod
    monkeypatch_session.setattr(_dbmod, "DB_PATH", test_db_path, raising=True)

    from app.main import create_app
    flask_app, db, model_manager = create_app()
    flask_app.config["TESTING"] = True
    return flask_app, db, model_manager


@pytest.fixture(scope="session")
def model_manager(flask_stack):
    """Convenience accessor for the session ModelManager."""
    _, _, mm = flask_stack
    return mm


@pytest.fixture(scope="session")
def flask_client(flask_stack):
    """Flask test client backed by the session app."""
    flask_app, _, _ = flask_stack
    return flask_app.test_client()


# ---------------------------------------------------------------------------
# Sample transactions (raw API input shape — the keys /api/predict expects)
# ---------------------------------------------------------------------------
@pytest.fixture
def known_legit_txn():
    """A small daytime grocery-store-like transaction that should score low.
    Uses the raw input shape accepted by /api/predict, including the three
    velocity features so the API uses them as overrides (no DB lookup)."""
    return {
        "transaction_id": "TEST-LEGIT-001",
        "card_number": "CARD-0001",
        "merchant_category": "grocery_pos",
        "amt": 12.50,
        "city_pop": 45000,
        "age": 42,
        "hour": 13,
        "month": 6,
        "distance_cardholder_merchant": 3.2,
        "gender_encoded": 0,
        "day_of_week_encoded": 2,
        "is_weekend": 0,
        "is_night": 0,
        "velocity_1h": 1,
        "velocity_24h": 2,
        "amount_velocity_1h": 12.50,
    }


@pytest.fixture
def known_fraud_txn():
    """A canonical fraud-pattern transaction: high amount, night-time,
    long distance, velocity burst, online-shopping category. Same input
    shape as known_legit_txn."""
    return {
        "transaction_id": "TEST-FRAUD-001",
        "card_number": "CARD-9999",
        "merchant_category": "shopping_net",
        "amt": 1450.00,
        "city_pop": 250000,
        "age": 27,
        "hour": 3,
        "month": 11,
        "distance_cardholder_merchant": 280.0,
        "gender_encoded": 1,
        "day_of_week_encoded": 5,
        "is_weekend": 1,
        "is_night": 1,
        "velocity_1h": 6,
        "velocity_24h": 9,
        "amount_velocity_1h": 5400.00,
    }

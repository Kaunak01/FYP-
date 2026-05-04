"""Velocity feature tests against the production INFERENCE-PATH function.

These tests target `app.database.Database.get_card_velocity()` — the SQL-based
implementation used at request time by the Flask `/api/predict` endpoint.
The training-path velocity formula (pandas `groupby('cc_num').rolling('1H')`)
is exercised directly in test_train_serve_consistency.py.

Note on semantics: `get_card_velocity` adds +1 to the count for the current
(not-yet-stored) transaction. The amount-velocity SUM is over stored history
ONLY; the production caller (Preprocessor.process) is responsible for adding
the current transaction's own amount on top — so the SUM returned here is the
'history-only' figure.
"""
from __future__ import annotations

from datetime import datetime, timedelta


CARD = "TESTCARD-0001"


def _store(db, ts, amt, cat=4):
    """Insert a single past transaction into card_history."""
    db.add_card_transaction(CARD, ts.strftime("%Y-%m-%d %H:%M:%S"), amt, cat)


def test_velocity_1h_known_input(fresh_db):
    """5 transactions for one card span 90 minutes. The current transaction
    is at t0; the four stored ones are at t0-{55,45,30,10}min.

    Expected velocity_1h for the current transaction = 5
        (4 history rows inside the 1-hour window + 1 for current).
    The row at t0-55 is inside (1 h = 60 min), so all 4 are counted."""
    now = datetime(2026, 5, 1, 12, 0, 0)
    for offset_min in (55, 45, 30, 10):
        _store(fresh_db, now - timedelta(minutes=offset_min), amt=20.0)

    v1h, _v24h, _amt = fresh_db.get_card_velocity(CARD, now)
    assert v1h == 5, f"expected v1h=5, got {v1h}"


def test_amount_velocity_1h_known_input(fresh_db):
    """4 stored transactions in the last hour with amounts [40, 60, 25, 75].
    The SQL implementation returns the SUM of stored amounts in the 1h window
    (the current transaction's amount is added by the caller, not here).

    Expected stored-only sum = 200.0.
    """
    now = datetime(2026, 5, 1, 12, 0, 0)
    amounts = [40.0, 60.0, 25.0, 75.0]
    offsets = [55, 45, 30, 10]
    for offset_min, amt in zip(offsets, amounts):
        _store(fresh_db, now - timedelta(minutes=offset_min), amt=amt)

    _v1h, _v24h, amount_velocity_1h = fresh_db.get_card_velocity(CARD, now)
    assert amount_velocity_1h == sum(amounts), (
        f"expected stored 1h sum={sum(amounts)}, got {amount_velocity_1h}"
    )


def test_velocity_24h_known_input(fresh_db):
    """5 transactions spread across 30 hours: at t0-{29h, 23h, 10h, 1h}
    plus the current one at t0. The 29 h history row is OUTSIDE the
    24 h window and must NOT be counted.

    Expected velocity_24h for the current transaction = 4
        (3 history rows inside 24 h + 1 for current).
    """
    now = datetime(2026, 5, 1, 12, 0, 0)
    # Outside 24 h window
    _store(fresh_db, now - timedelta(hours=29), amt=15.0)
    # Inside 24 h window
    _store(fresh_db, now - timedelta(hours=23), amt=15.0)
    _store(fresh_db, now - timedelta(hours=10), amt=15.0)
    _store(fresh_db, now - timedelta(hours=1), amt=15.0)

    _v1h, v24h, _amt = fresh_db.get_card_velocity(CARD, now)
    assert v24h == 4, f"expected v24h=4 (3 stored within 24h + 1 current), got {v24h}"

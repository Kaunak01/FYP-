"""Triage band tests against the production code (post-refactor).

The production code now uses the dissertation §4.7 3-tier scheme:

    probability < 0.30           → "NONE"     (no triage action)
    0.30 <= probability < 0.50   → "MONITOR"
    0.50 <= probability < 0.70   → "REVIEW"
    probability >= 0.70          → "FRAUD"

Boundaries belong to the higher band (exactly 0.30 → MONITOR, exactly 0.50 →
REVIEW, exactly 0.70 → FRAUD). The function under test is
`app.api.routes.get_triage_band`. The API response key is `triage_band`.
"""
from app.api.routes import get_triage_band


def test_triage_none_band():
    """probability=0.15 → 'NONE' (no triage action, below all thresholds)."""
    assert get_triage_band(0.15) == "NONE"


def test_triage_monitor_band():
    """probability=0.40 → 'MONITOR' (between 0.30 and 0.50)."""
    assert get_triage_band(0.40) == "MONITOR"


def test_triage_review_band():
    """probability=0.60 → 'REVIEW' (between 0.50 and 0.70)."""
    assert get_triage_band(0.60) == "REVIEW"


def test_triage_fraud_band():
    """probability=0.85 → 'FRAUD' (at or above 0.70)."""
    assert get_triage_band(0.85) == "FRAUD"

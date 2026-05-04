"""Flask route tests via test_client() — no real server is started.

Post-refactor the `/api/predict` response carries both:
  - `triage_band`     : NONE / MONITOR / REVIEW / FRAUD (pure-probability,
                        from get_triage_band, matches dissertation §4.7)
  - `classification`  : NORMAL / MONITOR / REVIEW / FRAUD (rule-combined
                        decision, from RuleEngine.combine_decision)

These two CAN differ when a rule fires (e.g. p=0.20 → triage_band='NONE',
but rules HIGH → classification='REVIEW'). The tests below assert the
post-refactor key names and the expected values for legit / fraud samples.
"""
from __future__ import annotations


def test_score_endpoint_returns_valid_json(flask_client, known_legit_txn):
    """POST /api/predict with a known-legit transaction. The response must
    be HTTP 200 and JSON containing the post-refactor keys
    (probability, triage_band, classification)."""
    resp = flask_client.post("/api/predict", json=known_legit_txn)
    assert resp.status_code == 200, (
        f"expected 200, got {resp.status_code}; body={resp.get_data(as_text=True)[:300]}"
    )
    body = resp.get_json()
    assert body is not None, "response body is not JSON"
    for key in ("probability", "triage_band", "classification"):
        assert key in body, f"key {key!r} missing from response: {list(body.keys())}"

    # Probability must be a real probability
    p = body["probability"]
    assert isinstance(p, (int, float)), f"probability is {type(p).__name__}, not numeric"
    assert 0.0 <= p <= 1.0, f"probability out of range: {p}"
    # triage_band must be one of the four documented values
    assert body["triage_band"] in ("NONE", "MONITOR", "REVIEW", "FRAUD"), (
        f"unexpected triage_band: {body['triage_band']!r}"
    )


def test_score_endpoint_fraud_sample(flask_client, known_fraud_txn):
    """POST a canonical-fraud-pattern transaction. The Proposed model
    should score it above 0.5; both `triage_band` and `classification`
    should reflect REVIEW or FRAUD. Hard-fails if the model misses,
    per the user's standing decision."""
    resp = flask_client.post("/api/predict", json=known_fraud_txn)
    assert resp.status_code == 200, (
        f"expected 200, got {resp.status_code}; body={resp.get_data(as_text=True)[:300]}"
    )
    body = resp.get_json()
    assert body is not None
    p = body["probability"]
    assert p > 0.5, (
        f"canonical fraud-pattern transaction scored only p={p:.4f}; "
        "expected >0.5 from the active fraud-detection model"
    )
    assert body["triage_band"] in ("REVIEW", "FRAUD"), (
        f"triage_band {body['triage_band']!r} is not REVIEW or FRAUD"
    )
    assert body["classification"] in ("REVIEW", "FRAUD"), (
        f"classification {body['classification']!r} is not REVIEW or FRAUD"
    )


def test_dashboard_route_renders(flask_client):
    """GET / (the dashboard / homepage). Response must be 200 and the HTML
    body must contain 'FraudLens' (the product name appears in the layout
    template). Smoke test that the dashboard renders end-to-end with the
    real ModelManager + Database wired in."""
    resp = flask_client.get("/")
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}"
    body = resp.get_data(as_text=True)
    assert "FraudLens" in body, "dashboard HTML does not contain 'FraudLens'"

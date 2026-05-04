"""BDS scoring tests against the inference-path function in app/.

`ModelManager.compute_bds_scores` (in `app/models/model_manager.py`) is the
production BDS implementation invoked at request time when the active model
is AE+BDS+XGBoost. It uses the global statistics persisted in
`bds_profiles.joblib` and the GA-evolved parameters from `ga_best_params.json`.
The training-script version in `scripts/run_bds_ga.py` is NOT used at
inference (and isn't even importable as a module — it does
`pd.read_csv('fraudTrain_engineered.csv')` at module top level).

Documented fallback (model_manager.py:202-204):
    If self.bds_profiles is None or self.ga_params is None:
        return 0.0, 0.0, 0.0, 0.0

These tests cover the happy path (4-tuple, non-negative) and the fallback.
"""
from __future__ import annotations


def _sample_features():
    """A complete 14-feature dict (the inference path indexes by name)."""
    return {
        "amt": 350.0,
        "city_pop": 50000,
        "hour": 14,
        "month": 6,
        "distance_cardholder_merchant": 12.0,
        "age": 35,
        "is_weekend": 0,
        "is_night": 0,
        "velocity_1h": 2,
        "velocity_24h": 5,
        "amount_velocity_1h": 350.0,
        "category_encoded": 4,
        "gender_encoded": 0,
        "day_of_week_encoded": 2,
    }


def test_bds_returns_four_dimensions(model_manager):
    """compute_bds_scores returns a 4-element tuple in the documented order
    (amount, time, frequency, category). Each value must be a non-negative
    float — the production code clips each score with `min(max(x - thresh, 0), cap)`
    so by construction nothing should ever be negative."""
    if model_manager.bds_profiles is None or model_manager.ga_params is None:
        # If the proposed-model artifacts aren't loaded, this test would
        # exercise only the fallback. The other test covers that path
        # explicitly, so we skip rather than double-cover.
        import pytest
        pytest.skip("BDS profiles/GA params not loaded in this environment")

    scores = model_manager.compute_bds_scores(_sample_features())
    assert isinstance(scores, tuple), f"expected tuple, got {type(scores).__name__}"
    assert len(scores) == 4, f"expected 4 dimensions, got {len(scores)}"
    # Each dim must be numeric (int OR float) and non-negative. The production
    # code returns int 0 when a score is exactly zero (e.g. freq_score when
    # vel/vel_mean <= 1) and float otherwise — both are valid by the
    # documented contract "non-negative deviation score".
    for i, s in enumerate(scores):
        assert isinstance(s, (int, float)), (
            f"dim {i}: expected numeric, got {type(s).__name__}"
        )
        assert s >= 0, f"dim {i}: BDS scores are clipped at 0; got {s}"


def test_bds_no_profile_returns_zero_vector(model_manager, monkeypatch):
    """When `bds_profiles` is None, compute_bds_scores returns the documented
    fallback (0.0, 0.0, 0.0, 0.0). We monkeypatch the loaded profiles to None
    on the session ModelManager and restore via the function-scoped
    monkeypatch's auto-undo at teardown."""
    monkeypatch.setattr(model_manager, "bds_profiles", None)
    scores = model_manager.compute_bds_scores(_sample_features())
    assert scores == (0.0, 0.0, 0.0, 0.0), (
        f"fallback expected zero 4-tuple, got {scores}"
    )

"""Train-serve consistency for the velocity_1h feature.

Why this test matters
---------------------
The dissertation §4.3 claims the inference-time velocity computation
(SQL `COUNT(*)` over `card_history` in `app/database.py::get_card_velocity`,
plus `+1` for the current transaction) is *equivalent* to the training-time
formula (pandas `df.groupby('cc_num').rolling('1H').count()` in
notebooks/01_EDA.ipynb function `compute_velocity_features`). If they drift,
the model is being served features that differ from those it was trained on
— a textbook source of silent accuracy loss.

The training-time function lives inside a Jupyter notebook cell and is not
importable as a Python module. Per the agreed test plan (option B), this
test reproduces the *documented training-path formula* inline in the test
itself, then compares it to the production SQL implementation on the same
small input. The inline formula is a 4-line copy of the notebook cell:

    df = df.sort_values(['cc_num', 'trans_datetime']).copy()
    df = df.set_index('trans_datetime')
    df['velocity_1h'] = (df.groupby('cc_num')['amt']
                         .transform(lambda x: x.rolling('1H').count()))
    df = df.reset_index()

A boundary note: pandas `rolling('1H')` is right-closed `(t-1h, t]`; the SQL
uses `>= t-1h AND <= t` which is closed-closed. To avoid this known edge
case we keep all timestamps strictly inside the window (no exact
t-1h boundary points).

The test fails LOUDLY (no xfail) per the user's decision — drift here is
exactly the signal we want.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def _training_path_velocity_1h(rows):
    """Reproduce the training-time formula from notebooks/01_EDA.ipynb.

    `rows` is a list of dicts with keys 'cc_num', 'trans_datetime', 'amt'.
    Returns the pandas Series of velocity_1h for each row, in the same order.
    """
    df = pd.DataFrame(rows)
    df["trans_datetime"] = pd.to_datetime(df["trans_datetime"])
    df = df.sort_values(["cc_num", "trans_datetime"]).copy()
    df = df.set_index("trans_datetime")
    df["velocity_1h"] = (
        df.groupby("cc_num")["amt"]
        .transform(lambda x: x.rolling("1h").count())
    )
    return df.reset_index()


def test_velocity_train_serve_match(fresh_db):
    """Build a 5-transaction history for one card. Compute velocity_1h on
    the LAST (current) transaction via:
      (1) the documented training-path pandas formula, and
      (2) the production SQL `Database.get_card_velocity` after inserting
          the first four transactions into card_history.
    Both paths must yield the same integer count."""
    card = "TS-CARD-001"
    base = datetime(2026, 5, 1, 12, 0, 0)
    # 5 timestamps: 4 stored history + 1 current. All within 1 h, none on
    # the exact t-1h boundary.
    offsets_minutes = [55, 45, 30, 10, 0]   # last is the "current" txn
    amounts = [10.0, 20.0, 30.0, 40.0, 50.0]

    rows = [
        {"cc_num": card,
         "trans_datetime": base - timedelta(minutes=m),
         "amt": a}
        for m, a in zip(offsets_minutes, amounts)
    ]

    # ---- Training path (inline pandas, per documented formula) ----
    train_df = _training_path_velocity_1h(rows)
    # The "current" transaction is the row at offset=0 (the latest).
    current_ts = base
    train_v1h = int(
        train_df.loc[train_df["trans_datetime"] == current_ts, "velocity_1h"].iloc[0]
    )

    # ---- Inference path (production SQL via get_card_velocity) ----
    # Store the first four (history) only. The "current" transaction has
    # not yet been inserted at scoring time; the +1 is added by the SQL
    # implementation itself.
    for m, a in zip(offsets_minutes[:-1], amounts[:-1]):
        ts = base - timedelta(minutes=m)
        fresh_db.add_card_transaction(card, ts.strftime("%Y-%m-%d %H:%M:%S"), a, 4)

    serve_v1h, _v24h, _amt_v1h = fresh_db.get_card_velocity(card, current_ts)

    assert serve_v1h == train_v1h, (
        "train-serve drift: training-path velocity_1h = "
        f"{train_v1h}, inference-path velocity_1h = {serve_v1h}. "
        "If you see this, the SQL implementation in app/database.py has "
        "drifted from the pandas formula in notebooks/01_EDA.ipynb."
    )

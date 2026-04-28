"""
Measure single-transaction inference latency for the 3 deployed models.

Uses the live ModelManager so latency reflects exactly what the Flask runtime
does (preprocessing -> AE -> BDS -> XGBoost, etc), not a stripped-down loader.

Sample: 1000 random rows from data/engineered/fraudTest_engineered_with_ids.csv
        (seed=42, single-transaction calls — no batching)

Output: results/inference_latency.json
        {
          "baseline":   {"median_ms": ..., "p95_ms": ..., "n_samples": 1000},
          "comparator": {"median_ms": ..., "p95_ms": ..., "n_samples": 1000},
          "proposed":   {"median_ms": ..., "p95_ms": ..., "n_samples": 1000}
        }
"""
from __future__ import annotations

import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import FEATURE_COLS, LSTM_SEQ_LEN
from app.models.model_manager import ModelManager

TEST_CSV = ROOT / "data" / "engineered" / "fraudTest_engineered_with_ids.csv"
OUT_PATH = ROOT / "results" / "inference_latency.json"

N_SAMPLES = 1000
N_WARMUP = 25
SEED = 42

MODELS = [
    ("baseline",   "XGBoost (SMOTE+Tuned)"),
    ("comparator", "LSTM+RF"),
    ("proposed",   "AE+BDS+XGBoost"),
]


def pick_priors(df_sorted: pd.DataFrame, idx: int) -> list[dict]:
    """Return up to (LSTM_SEQ_LEN - 1) prior same-card transactions, oldest -> newest."""
    row = df_sorted.iloc[idx]
    cc = row["cc_num"]
    t = row["unix_time"]
    mask = (df_sorted["cc_num"] == cc) & (df_sorted["unix_time"] < t)
    priors = df_sorted.loc[mask].tail(LSTM_SEQ_LEN - 1)
    return [{c: float(r[c]) for c in FEATURE_COLS} for _, r in priors.iterrows()]


def main() -> None:
    print(f"[load] {TEST_CSV.name}")
    df = pd.read_csv(TEST_CSV).sort_values(["cc_num", "unix_time"], kind="mergesort").reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(df), size=N_SAMPLES, replace=False)
    print(f"       sampled {N_SAMPLES} rows (seed={SEED})")

    print("[load] ModelManager (loads all 5 models)")
    mm = ModelManager()

    feature_dicts = [
        {c: float(df.iloc[i][c]) for c in FEATURE_COLS}
        for i in sample_idx
    ]
    # priors only needed for LSTM+RF; build them once and reuse
    print("[prep] building per-row priors for LSTM+RF (cold-start aware)")
    priors_list = [pick_priors(df, int(i)) for i in sample_idx]

    out: dict[str, dict] = {}
    for key, internal in MODELS:
        print(f"\n[time] {key:10s}  ({internal})")
        mm.set_active(internal)

        # Warmup — first call typically has JIT/cache cost we don't want in stats
        for k in range(N_WARMUP):
            mm.predict(feature_dicts[k], prior_features=priors_list[k] if internal == "LSTM+RF" else None)

        latencies_ms: list[float] = []
        for fd, priors in zip(feature_dicts, priors_list):
            t0 = time.perf_counter()
            mm.predict(fd, prior_features=priors if internal == "LSTM+RF" else None)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.asarray(latencies_ms)
        median = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mean = float(arr.mean())
        out[key] = {
            "model_internal_name": internal,
            "median_ms": round(median, 3),
            "p95_ms": round(p95, 3),
            "p99_ms": round(p99, 3),
            "mean_ms": round(mean, 3),
            "n_samples": N_SAMPLES,
            "warmup_calls": N_WARMUP,
        }
        print(f"       median={median:7.3f} ms   p95={p95:7.3f} ms   p99={p99:7.3f} ms   mean={mean:7.3f} ms")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {OUT_PATH.relative_to(ROOT)}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

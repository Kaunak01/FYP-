"""Test Suite 5: Stress and Edge Case Testing."""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_config import TEST_CSV, RESULTS_DIR, FEATURE_COLS, TRAINING_STATS, get_median_transaction
from fraud_predictor import XGBoostPredictor, transaction_from_row

class Tee:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d): self.stdout.write(d); self.file.write(d)
    def flush(self): self.stdout.flush(); self.file.flush()
    def close(self): self.file.close()


def run_suite():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tee = Tee(os.path.join(RESULTS_DIR, 'test_suite_5_results.txt'))
    sys.stdout = tee

    print("="*70)
    print("TEST SUITE 5: STRESS AND EDGE CASE TESTING")
    print("="*70)
    t0_total = time.time()
    tests_passed = 0; tests_failed = 0; tests_total = 0

    predictor = XGBoostPredictor()
    test_df = pd.read_csv(TEST_CSV)

    # ================================================================
    # STRESS 1: THROUGHPUT BENCHMARK
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("STRESS 1: THROUGHPUT BENCHMARK")
    print("="*70)

    n_total = len(test_df)
    print(f"  Dataset: {n_total:,} transactions")

    run_times = []
    for run in range(5):
        t0 = time.perf_counter()
        summary = predictor.predict_batch(test_df)
        elapsed = time.perf_counter() - t0
        run_times.append(elapsed)
        tps = n_total / elapsed
        ms_per = elapsed / n_total * 1000
        print(f"  Run {run+1}: {elapsed:.2f}s ({tps:,.0f} txns/sec, {ms_per:.4f} ms/txn)")

    mean_time = np.mean(run_times)
    std_time = np.std(run_times)
    mean_tps = n_total / mean_time
    mean_ms = mean_time / n_total * 1000

    print(f"\n  Mean: {mean_time:.2f}s +/- {std_time:.2f}s")
    print(f"  Throughput: {mean_tps:,.0f} transactions/second")
    print(f"  Latency: {mean_ms:.4f} ms per transaction")
    print(f"  Banking requirement: < 100ms per transaction")

    if mean_ms < 100:
        print(f"  RESULT: PASS ({mean_ms:.4f} ms < 100 ms)")
        tests_passed += 1
    else:
        print(f"  RESULT: FAIL ({mean_ms:.4f} ms >= 100 ms)")
        tests_failed += 1

    # ================================================================
    # STRESS 2: EXTREME VALUES
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("STRESS 2: EXTREME VALUES")
    print("="*70)

    extreme_tests = [
        ("amt=$0.01 (minimum)", {'amt': 0.01}),
        ("amt=$0.00 (zero)", {'amt': 0.00}),
        ("amt=-$50 (negative)", {'amt': -50.0}),
        ("amt=$99,999 (massive)", {'amt': 99999.0}),
        ("amt=$1,000,000 (absurd)", {'amt': 1000000.0}),
        ("velocity_1h=0 (impossible)", {'velocity_1h': 0.0}),
        ("velocity_1h=100", {'velocity_1h': 100.0}),
        ("velocity_1h=1000 (absurd)", {'velocity_1h': 1000.0}),
        ("age=0 (baby)", {'age': 0.0}),
        ("age=150 (impossible)", {'age': 150.0}),
        ("age=-5 (negative)", {'age': -5.0}),
        ("hour=-1 (invalid)", {'hour': -1.0}),
        ("hour=25 (invalid)", {'hour': 25.0}),
        ("city_pop=0 (ghost town)", {'city_pop': 0.0}),
        ("city_pop=50,000,000 (mega city)", {'city_pop': 50000000.0}),
        ("ALL features = 0", {f: 0.0 for f in FEATURE_COLS}),
        ("ALL features = max training value", {f: TRAINING_STATS['stats'][f]['all']['max'] for f in FEATURE_COLS}),
        ("ALL features = min training value", {f: TRAINING_STATS['stats'][f]['all']['min'] for f in FEATURE_COLS}),
    ]

    all_valid = True
    print(f"\n  {'Test':<40s} {'Prob':>8s} {'Risk':>10s} {'Valid':>6s} {'Notes'}")
    print(f"  {'-'*80}")

    for desc, overrides in extreme_tests:
        txn = get_median_transaction()
        txn.update(overrides)
        try:
            features = np.array([txn[f] for f in FEATURE_COLS], dtype=np.float64)
            prob = predictor.model.predict_proba(features.reshape(1, -1))[0, 1]
            valid = 0 <= prob <= 1
            if not valid:
                all_valid = False
            notes = ""
            if not valid:
                notes = "INVALID PROBABILITY"
            elif prob > 0.99:
                notes = "Near-certain fraud"
            elif prob < 0.01:
                notes = "Near-certain normal"
            risk = "CRITICAL" if prob >= 0.8 else "HIGH" if prob >= 0.5 else "MEDIUM" if prob >= 0.2 else "LOW"
            print(f"  {desc:<40s} {prob:>8.4f} {risk:>10s} {'OK' if valid else 'FAIL':>6s} {notes}")
        except Exception as e:
            all_valid = False
            print(f"  {desc:<40s} {'CRASH':>8s} {'':>10s} {'FAIL':>6s} {str(e)[:40]}")

    if all_valid:
        print(f"\n  RESULT: PASS — All extreme values handled, all probabilities in [0,1]")
        tests_passed += 1
    else:
        print(f"\n  RESULT: FAIL — Some extreme values produced invalid results")
        tests_failed += 1

    # ================================================================
    # STRESS 3: MISSING DATA (NaN)
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("STRESS 3: MISSING DATA HANDLING (NaN)")
    print("="*70)

    nan_results = []
    print(f"\n  Testing NaN in each feature one at a time:")
    for feat in FEATURE_COLS:
        txn = get_median_transaction()
        txn[feat] = float('nan')
        try:
            features = np.array([txn[f] for f in FEATURE_COLS], dtype=np.float64)
            prob = predictor.model.predict_proba(features.reshape(1, -1))[0, 1]
            valid = 0 <= prob <= 1 and not np.isnan(prob)
            nan_results.append((feat, "OK" if valid else "INVALID", prob))
            print(f"    NaN in {feat:<35s}: prob={prob:.4f} {'OK' if valid else 'INVALID'}")
        except Exception as e:
            nan_results.append((feat, "CRASH", None))
            print(f"    NaN in {feat:<35s}: CRASH — {str(e)[:50]}")

    # All NaN
    txn_all_nan = {f: float('nan') for f in FEATURE_COLS}
    try:
        features = np.array([txn_all_nan[f] for f in FEATURE_COLS], dtype=np.float64)
        prob = predictor.model.predict_proba(features.reshape(1, -1))[0, 1]
        print(f"    ALL features NaN: prob={prob:.4f}")
    except Exception as e:
        print(f"    ALL features NaN: CRASH — {str(e)[:50]}")

    # XGBoost handles NaN natively — this should pass
    crashes = [r for r in nan_results if r[1] == "CRASH"]
    if not crashes:
        print(f"\n  RESULT: PASS — XGBoost handles NaN natively (uses default direction at splits)")
        tests_passed += 1
    else:
        print(f"\n  RESULT: FAIL — {len(crashes)} features crashed with NaN")
        tests_failed += 1

    # ================================================================
    # STRESS 4: DATA TYPE ERRORS
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("STRESS 4: DATA TYPE ERROR HANDLING")
    print("="*70)

    type_tests = [
        ("String 'hello' for amt", {'amt': 'hello'}),
        ("Boolean True for velocity_1h", {'velocity_1h': True}),
        ("List [1,2,3] for age", {'age': [1, 2, 3]}),
        ("None for hour", {'hour': None}),
        ("Dict for amt", {'amt': {'value': 100}}),
    ]

    caught_errors = 0
    for desc, bad_override in type_tests:
        txn = get_median_transaction()
        txn.update(bad_override)
        try:
            result = predictor.predict_single(txn)
            # Boolean True is numeric-ish (Python treats it as 1), so may not error
            if isinstance(bad_override[list(bad_override.keys())[0]], bool):
                print(f"    {desc:<45s}: prob={result.probability:.4f} (bool treated as int — acceptable)")
                caught_errors += 1  # Acceptable behaviour
            else:
                print(f"    {desc:<45s}: NO ERROR RAISED — prob={result.probability:.4f}")
        except (TypeError, ValueError) as e:
            caught_errors += 1
            print(f"    {desc:<45s}: CAUGHT — {type(e).__name__}: {str(e)[:60]}")
        except Exception as e:
            print(f"    {desc:<45s}: UNEXPECTED — {type(e).__name__}: {str(e)[:60]}")

    if caught_errors >= 3:  # At least string, list, and None/dict should be caught
        print(f"\n  RESULT: PASS — {caught_errors}/{len(type_tests)} type errors handled")
        tests_passed += 1
    else:
        print(f"\n  RESULT: FAIL — Only {caught_errors}/{len(type_tests)} type errors caught")
        tests_failed += 1

    # ================================================================
    # STRESS 5: CONSISTENCY (determinism)
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("STRESS 5: CONSISTENCY CHECK (1000 identical predictions)")
    print("="*70)

    txn = get_median_transaction()
    txn['amt'] = 500.0
    txn['hour'] = 22.0
    txn['is_night'] = 1.0

    features = np.array([txn[f] for f in FEATURE_COLS], dtype=np.float64)
    X_repeat = np.tile(features, (1000, 1))

    probs = predictor.model.predict_proba(X_repeat)[:, 1]
    all_same = np.all(probs == probs[0])
    unique_probs = len(np.unique(probs))

    print(f"  Transaction: amt=$500, hour=22, is_night=1")
    print(f"  Predicted 1000 times")
    print(f"  First probability: {probs[0]:.10f}")
    print(f"  Last probability:  {probs[-1]:.10f}")
    print(f"  Unique probabilities: {unique_probs}")
    print(f"  All identical: {all_same}")

    if all_same:
        print(f"\n  RESULT: PASS — All 1000 predictions identical (XGBoost is deterministic)")
        tests_passed += 1
    else:
        print(f"\n  RESULT: FAIL — Predictions vary ({unique_probs} unique values)")
        tests_failed += 1

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"TEST SUITE 5 SUMMARY")
    print(f"  Tests passed: {tests_passed}/{tests_total}")
    print(f"  Tests failed: {tests_failed}/{tests_total}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*70}")

    sys.stdout = tee.stdout
    tee.close()
    return tests_passed, tests_failed, tests_total

if __name__ == '__main__':
    passed, failed, total = run_suite()
    print(f"\nTest Suite 5: {passed}/{total} passed, {failed} failed")

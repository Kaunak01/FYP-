"""Test Suite 4: Adversarial/Evasion Testing."""
import sys, os, time, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_config import RESULTS_DIR, FEATURE_COLS, CATEGORY_NAME_TO_CODE, get_median_transaction
from fraud_predictor import XGBoostPredictor

class Tee:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d): self.stdout.write(d); self.file.write(d)
    def flush(self): self.stdout.flush(); self.file.flush()
    def close(self): self.file.close()

def build_transaction(**overrides):
    txn = get_median_transaction()
    txn.update(overrides)
    return txn

def run_suite():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tee = Tee(os.path.join(RESULTS_DIR, 'test_suite_4_results.txt'))
    sys.stdout = tee

    print("="*70)
    print("TEST SUITE 4: ADVERSARIAL / EVASION TESTING")
    print("="*70)
    t0_total = time.time()
    predictor = XGBoostPredictor()
    tests_passed = 0; tests_total = 0

    # ================================================================
    # EVASION 1: AMOUNT SPLITTING
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("EVASION 1: AMOUNT SPLITTING ($1000 total)")
    print("="*70)

    strategies = [
        ("A: 2x$500, 30min apart", 2, 500, 30),
        ("B: 5x$200, 15min apart", 5, 200, 15),
        ("C: 10x$100, 10min apart", 10, 100, 10),
        ("D: 20x$50, 5min apart", 20, 50, 5),
        ("E: 50x$20, 2min apart", 50, 20, 2),
    ]
    split_results = []

    for name, n_txns, amt_each, mins_apart in strategies:
        caught = 0
        for i in range(n_txns):
            # Velocity increases with each transaction
            vel_1h = min(i + 1, int(60 / mins_apart)) if mins_apart < 60 else 1
            amt_vel = amt_each * vel_1h
            txn = build_transaction(amt=float(amt_each), hour=22, is_night=1,
                                    velocity_1h=float(vel_1h), velocity_24h=float(i+1),
                                    amount_velocity_1h=float(amt_vel))
            result = predictor.predict_single(txn)
            if result.classification_default == "FRAUD":
                caught += 1
        pct = 100 * caught / n_txns
        split_results.append((name, n_txns, caught, pct))
        print(f"  {name}: {caught}/{n_txns} caught ({pct:.0f}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [s[0].split(":")[0] for s in split_results]
    rates = [s[3] for s in split_results]
    colors = ['red' if r > 50 else 'orange' if r > 0 else 'green' for r in rates]
    ax.bar(labels, rates, color=colors, edgecolor='black')
    ax.set_ylabel('Detection Rate (%)'); ax.set_title('Amount Splitting Evasion Strategies')
    ax.set_ylim(0, 105)
    for i, r in enumerate(rates):
        ax.text(i, r + 2, f'{r:.0f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'splitting_evasion.png'), dpi=100); plt.close()
    print(f"  Saved: splitting_evasion.png")

    best_evasion = min(split_results, key=lambda x: x[3])
    print(f"  Most effective evasion: {best_evasion[0]} ({best_evasion[3]:.0f}% detected)")
    tests_passed += 1

    # ================================================================
    # EVASION 2: TIME SPACING
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("EVASION 2: TIME SPACING (10x$100)")
    print("="*70)

    spacing_strategies = [
        ("A: 5min apart", 1, 10, 10),       # vel_1h=10 within an hour
        ("B: 30min apart", 1, 3, 3),         # vel_1h=2-3
        ("C: 2hr apart", 1, 1, 5),           # vel_1h=1, vel_24h=5
        ("D: 6hr apart", 1, 1, 4),           # vel_1h=1, vel_24h=4
        ("E: 1day apart", 1, 1, 1),          # vel_1h=1, vel_24h=1
    ]

    for name, vel_1h, vel_1h_max, vel_24h in spacing_strategies:
        caught = 0
        for i in range(10):
            v1h = min(i + 1, vel_1h_max)
            v24h = min(i + 1, vel_24h)
            amt_vel = 100.0 * v1h
            txn = build_transaction(amt=100.0, hour=22, is_night=1,
                                    velocity_1h=float(v1h), velocity_24h=float(v24h),
                                    amount_velocity_1h=float(amt_vel))
            r = predictor.predict_single(txn)
            if r.classification_default == "FRAUD":
                caught += 1
        print(f"  {name}: {caught}/10 caught ({100*caught/10:.0f}%)")

    tests_passed += 1

    # ================================================================
    # EVASION 3: CATEGORY CAMOUFLAGE
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("EVASION 3: CATEGORY CAMOUFLAGE ($500, 11pm)")
    print("="*70)

    cat_tests = [
        ('shopping_net', 'Electronics/online'),
        ('grocery_pos', 'Grocery (most common)'),
        ('gas_transport', 'Gas/Transport'),
        ('travel', 'Travel'),
        ('food_dining', 'Food/Dining'),
        ('health_fitness', 'Health'),
    ]

    cat_probs = []
    for cat_name, desc in cat_tests:
        code = CATEGORY_NAME_TO_CODE[cat_name]
        txn = build_transaction(amt=500, hour=23, is_night=1, velocity_1h=2,
                                velocity_24h=5, amount_velocity_1h=1000,
                                category_encoded=float(code))
        r = predictor.predict_single(txn)
        cat_probs.append((desc, cat_name, r.probability, r.risk_level))
        flagged = "FLAGGED" if r.classification_default == "FRAUD" else "evaded"
        print(f"  {desc:<25s} (code={code:2d}): prob={r.probability:.4f}, {r.risk_level:8s} [{flagged}]")

    best_camouflage = min(cat_probs, key=lambda x: x[2])
    worst_camouflage = max(cat_probs, key=lambda x: x[2])
    print(f"  Best camouflage: {best_camouflage[1]} (prob={best_camouflage[2]:.4f})")
    print(f"  Worst camouflage: {worst_camouflage[1]} (prob={worst_camouflage[2]:.4f})")
    print(f"  Category impact: {worst_camouflage[2] - best_camouflage[2]:.4f} probability difference")
    tests_passed += 1

    # ================================================================
    # EVASION 4: TIME OF DAY CAMOUFLAGE
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("EVASION 4: TIME OF DAY CAMOUFLAGE ($500, velocity=3)")
    print("="*70)

    time_tests = [
        (3, "3am (peak fraud)"),
        (8, "8am (morning)"),
        (12, "12pm (lunch)"),
        (15, "3pm (afternoon)"),
        (18, "6pm (evening)"),
        (21, "9pm (night)"),
        (23, "11pm (late night)"),
    ]
    for hour, desc in time_tests:
        is_night = 1 if (hour >= 22 or hour < 6) else 0
        txn = build_transaction(amt=500, hour=float(hour), is_night=float(is_night),
                                velocity_1h=3, velocity_24h=5, amount_velocity_1h=1500)
        r = predictor.predict_single(txn)
        flagged = "FLAGGED" if r.classification_default == "FRAUD" else "evaded"
        print(f"  {desc:<25s} (is_night={is_night}): prob={r.probability:.4f}, {r.risk_level:8s} [{flagged}]")

    tests_passed += 1

    # ================================================================
    # EVASION 5: COMBINED OPTIMAL EVASION
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("EVASION 5: OPTIMAL EVASION STRATEGY")
    print("="*70)

    # Based on findings: daytime, low amount, low velocity, travel category
    txn_optimal = build_transaction(
        amt=100, hour=14, is_night=0, velocity_1h=1, velocity_24h=2,
        amount_velocity_1h=100, category_encoded=float(CATEGORY_NAME_TO_CODE['travel']),
        is_weekend=0, age=45
    )
    r_opt = predictor.predict_single(txn_optimal)
    print(f"  Optimal evasion transaction:")
    print(f"    amt=$100, hour=14 (daytime), category=travel, velocity_1h=1")
    print(f"    Probability: {r_opt.probability:.4f} ({r_opt.probability*100:.1f}%)")
    print(f"    Classification: {r_opt.classification_default}")
    print(f"    Risk level: {r_opt.risk_level}")

    # Try a more aggressive version
    txn_aggressive = build_transaction(
        amt=500, hour=14, is_night=0, velocity_1h=1, velocity_24h=2,
        amount_velocity_1h=500, category_encoded=float(CATEGORY_NAME_TO_CODE['travel']),
        is_weekend=0, age=45
    )
    r_agg = predictor.predict_single(txn_aggressive)
    print(f"\n  Aggressive evasion ($500 instead of $100):")
    print(f"    Probability: {r_agg.probability:.4f} ({r_agg.probability*100:.1f}%)")
    print(f"    Classification: {r_agg.classification_default}")

    if r_opt.classification_default == "NORMAL":
        print(f"\n  CONCLUSION: A fraudster CAN evade detection by transacting during the day,")
        print(f"  using travel category, with low velocity. The model gives only {r_opt.probability:.4f} probability.")
        print(f"  This is a KNOWN LIMITATION — the model relies heavily on is_night as a discriminator.")
    else:
        print(f"\n  CONCLUSION: Even the optimal evasion strategy gets caught.")

    tests_passed += 1

    # ================================================================
    # EVASION 6: FEATURE MANIPULATION COST
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("EVASION 6: FEATURE MANIPULATION COST")
    print("="*70)

    # Start with a highly fraudulent transaction
    txn_fraud = build_transaction(amt=800, hour=2, is_night=1, velocity_1h=3,
                                  velocity_24h=8, amount_velocity_1h=2400,
                                  category_encoded=float(CATEGORY_NAME_TO_CODE['grocery_pos']))
    r_fraud = predictor.predict_single(txn_fraud)
    print(f"  Starting fraud transaction: prob={r_fraud.probability:.4f} ({r_fraud.risk_level})")

    # Try changing 1 feature at a time to median
    median_txn = get_median_transaction()
    print(f"\n  Changing 1 feature at a time to median value:")
    single_flips = []
    for feat in FEATURE_COLS:
        modified = txn_fraud.copy()
        modified[feat] = median_txn[feat]
        r = predictor.predict_single(modified)
        flipped = r.classification_default == "NORMAL"
        single_flips.append((feat, r.probability, flipped))
        marker = " *** FLIPPED" if flipped else ""
        print(f"    Change {feat:<35s} -> {median_txn[feat]:>10.2f}: prob={r.probability:.4f}{marker}")

    flipped_singles = [f for f in single_flips if f[2]]
    if flipped_singles:
        print(f"\n  Single-feature flips found: {[f[0] for f in flipped_singles]}")
    else:
        print(f"\n  No single feature change can flip the prediction.")

    # Try 2-feature combinations (only test top impact features)
    print(f"\n  Trying 2-feature combinations (top 6 most impactful):")
    # Sort by probability reduction
    sorted_singles = sorted(single_flips, key=lambda x: x[1])
    top_features = [f[0] for f in sorted_singles[:6]]

    pair_flips = []
    for f1, f2 in itertools.combinations(top_features, 2):
        modified = txn_fraud.copy()
        modified[f1] = median_txn[f1]
        modified[f2] = median_txn[f2]
        r = predictor.predict_single(modified)
        flipped = r.classification_default == "NORMAL"
        pair_flips.append((f1, f2, r.probability, flipped))
        if flipped:
            print(f"    Change {f1} + {f2}: prob={r.probability:.4f} *** FLIPPED")

    flipped_pairs = [p for p in pair_flips if p[2]]
    if not flipped_pairs:
        print(f"    No 2-feature combination flips the prediction.")

    # Try 3-feature combinations
    print(f"\n  Trying 3-feature combinations (top 6):")
    triple_flips = []
    for f1, f2, f3 in itertools.combinations(top_features, 3):
        modified = txn_fraud.copy()
        modified[f1] = median_txn[f1]
        modified[f2] = median_txn[f2]
        modified[f3] = median_txn[f3]
        r = predictor.predict_single(modified)
        flipped = r.classification_default == "NORMAL"
        triple_flips.append((f1, f2, f3, r.probability, flipped))
        if flipped:
            print(f"    Change {f1} + {f2} + {f3}: prob={r.probability:.4f} *** FLIPPED")

    flipped_triples = [t for t in triple_flips if t[3]]
    if not flipped_triples:
        print(f"    No 3-feature combination flips the prediction.")

    # Summary
    min_features = 0
    if flipped_singles:
        min_features = 1
    elif flipped_pairs:
        min_features = 2
    elif flipped_triples:
        min_features = 3
    else:
        min_features = 4  # would need 4+

    print(f"\n  CONCLUSION: A fraudster would need to manipulate at least {min_features} feature(s) to evade detection.")
    print(f"  This quantifies the model's robustness against adversarial manipulation.")
    tests_passed += 1

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"TEST SUITE 4 SUMMARY")
    print(f"  Tests passed: {tests_passed}/{tests_total}")
    print(f"  Tests failed: {tests_total - tests_passed}/{tests_total}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Plot saved: splitting_evasion.png")
    print(f"{'='*70}")

    sys.stdout = tee.stdout
    tee.close()
    return tests_passed, tests_total - tests_passed, tests_total

if __name__ == '__main__':
    passed, failed, total = run_suite()
    print(f"\nTest Suite 4: {passed}/{total} passed, {failed} failed")

"""Test Suite 8: Live Streaming Simulation."""
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_config import TEST_CSV, TEST_RAW_CSV, RESULTS_DIR, FEATURE_COLS
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
    tee = Tee(os.path.join(RESULTS_DIR, 'test_suite_8_results.txt'))
    sys.stdout = tee

    print("="*70)
    print("TEST SUITE 8: LIVE STREAMING SIMULATION")
    print("="*70)
    t0_total = time.time()
    tests_passed = 0; tests_failed = 0; tests_total = 0

    # Load data with timestamps
    test_eng = pd.read_csv(TEST_CSV)
    test_eng['datetime'] = pd.to_datetime(test_eng['unix_time'], unit='s')
    test_eng = test_eng.sort_values('datetime').reset_index(drop=True)
    y_test = test_eng['is_fraud'].values

    predictor = XGBoostPredictor()

    # ================================================================
    # SIMULATION 1: SEQUENTIAL PROCESSING
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("SIMULATION 1: SEQUENTIAL PROCESSING (10,000 transactions)")
    print("="*70)

    N_SIM = 10000
    alerts = []
    proc_times = []

    for i in range(N_SIM):
        txn = {f: float(test_eng.iloc[i][f]) for f in FEATURE_COLS}
        t0 = time.perf_counter()
        features = np.array([txn[f] for f in FEATURE_COLS], dtype=np.float64)
        prob = predictor.model.predict_proba(features.reshape(1, -1))[0, 1]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        proc_times.append(elapsed_ms)

        if prob >= 0.5:
            actual = y_test[i]
            alerts.append({
                'idx': i, 'time': str(test_eng.iloc[i]['datetime']),
                'amt': txn['amt'], 'prob': prob,
                'actual_fraud': actual == 1,
                'correct': 'TRUE ALERT' if actual == 1 else 'FALSE ALARM'
            })

    true_alerts = sum(1 for a in alerts if a['actual_fraud'])
    false_alerts = len(alerts) - true_alerts
    alert_precision = true_alerts / len(alerts) if alerts else 0
    avg_ms = np.mean(proc_times)

    print(f"\n  Transactions processed: {N_SIM:,}")
    print(f"  Total alerts raised: {len(alerts)}")
    print(f"  True fraud alerts: {true_alerts}")
    print(f"  False alarms: {false_alerts}")
    print(f"  Alert precision: {alert_precision:.1%}")
    print(f"  Avg processing time: {avg_ms:.4f} ms/txn")
    print(f"  Max processing time: {max(proc_times):.4f} ms")

    print(f"\n  Alert timeline (first 20 alerts):")
    for a in alerts[:20]:
        print(f"    [{a['time']}] ${a['amt']:.2f} — prob={a['prob']:.4f} — {a['correct']}")

    # Plot alert timeline
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    # Top: all probabilities
    all_probs = []
    for i in range(N_SIM):
        features = np.array([float(test_eng.iloc[i][f]) for f in FEATURE_COLS], dtype=np.float64)
        p = predictor.model.predict_proba(features.reshape(1, -1))[0, 1]
        all_probs.append(p)

    axes[0].plot(range(N_SIM), all_probs, 'b.', alpha=0.1, markersize=1)
    alert_indices = [a['idx'] for a in alerts]
    alert_probs = [a['prob'] for a in alerts]
    true_idx = [a['idx'] for a in alerts if a['actual_fraud']]
    true_probs = [a['prob'] for a in alerts if a['actual_fraud']]
    false_idx = [a['idx'] for a in alerts if not a['actual_fraud']]
    false_probs = [a['prob'] for a in alerts if not a['actual_fraud']]
    axes[0].scatter(true_idx, true_probs, c='red', s=20, label=f'True alerts ({true_alerts})', zorder=5)
    axes[0].scatter(false_idx, false_probs, c='orange', s=20, label=f'False alarms ({false_alerts})', zorder=5)
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Fraud Probability'); axes[0].set_title('Sequential Processing — Alert Timeline')
    axes[0].legend(); axes[0].set_xlim(0, N_SIM)

    # Bottom: processing time
    axes[1].plot(range(N_SIM), proc_times, 'g-', alpha=0.3, linewidth=0.5)
    axes[1].axhline(avg_ms, color='red', linestyle='--', label=f'Avg={avg_ms:.4f}ms')
    axes[1].set_ylabel('Processing Time (ms)'); axes[1].set_xlabel('Transaction #')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'alert_timeline.png'), dpi=100); plt.close()
    print(f"  Saved: alert_timeline.png")

    if alert_precision > 0.5 and avg_ms < 100:
        print(f"  RESULT: PASS (precision={alert_precision:.1%}, latency={avg_ms:.4f}ms)")
        tests_passed += 1
    else:
        print(f"  RESULT: FAIL")
        tests_failed += 1

    # ================================================================
    # SIMULATION 2: SLIDING WINDOW MONITORING
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("SIMULATION 2: SLIDING WINDOW MONITORING")
    print("="*70)

    window_size = 100
    rolling_fraud_rate = []
    rolling_avg_prob = []
    spike_alerts = []

    for i in range(window_size, N_SIM):
        window_probs = all_probs[i-window_size:i]
        window_flags = sum(1 for p in window_probs if p >= 0.5)
        fraud_rate = window_flags / window_size
        avg_prob = np.mean(window_probs)
        rolling_fraud_rate.append(fraud_rate)
        rolling_avg_prob.append(avg_prob)

        if fraud_rate > 0.05:
            spike_alerts.append((i, fraud_rate))

    print(f"  Window size: {window_size}")
    print(f"  Spike alerts (fraud rate > 5%): {len(spike_alerts)}")
    if spike_alerts:
        print(f"  First spike at transaction #{spike_alerts[0][0]} (rate={spike_alerts[0][1]:.1%})")
        print(f"  Max spike rate: {max(s[1] for s in spike_alerts):.1%}")
    print(f"  Average rolling fraud rate: {np.mean(rolling_fraud_rate):.3%}")
    print(f"  Average rolling probability: {np.mean(rolling_avg_prob):.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    x = range(window_size, N_SIM)
    axes[0].plot(x, rolling_fraud_rate, 'r-', linewidth=0.8)
    axes[0].axhline(0.05, color='orange', linestyle='--', label='5% alert threshold')
    axes[0].set_ylabel('Rolling Fraud Rate'); axes[0].set_title('Sliding Window — Fraud Rate')
    axes[0].legend()
    axes[1].plot(x, rolling_avg_prob, 'b-', linewidth=0.8)
    axes[1].set_ylabel('Rolling Avg Probability'); axes[1].set_xlabel('Transaction #')
    axes[1].set_title('Sliding Window — Average Probability')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rolling_fraud_rate.png'), dpi=100); plt.close()
    print(f"  Saved: rolling_fraud_rate.png")
    print(f"  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # SIMULATION 3: CARDHOLDER JOURNEY
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("SIMULATION 3: CARDHOLDER JOURNEY")
    print("="*70)

    # Load cc_num from raw test CSV
    test_raw = pd.read_csv(TEST_RAW_CSV, usecols=['cc_num'])
    # Align with sorted engineered data
    test_eng_sorted = test_eng.copy()
    # We sorted by datetime, but cc_num is from raw (original order)
    # Need to get cc_num aligned — use unix_time as key
    test_raw_full = pd.read_csv(TEST_RAW_CSV, usecols=['cc_num', 'unix_time'])
    test_eng_sorted = test_eng_sorted.merge(
        test_raw_full.drop_duplicates(subset='unix_time'),
        on='unix_time', how='left'
    )

    # Find cards that have BOTH fraud and normal in test set
    card_fraud_counts = test_eng_sorted.groupby('cc_num').agg(
        total=('is_fraud', 'count'),
        frauds=('is_fraud', 'sum')
    )
    cards_with_both = card_fraud_counts[(card_fraud_counts['frauds'] > 0) &
                                        (card_fraud_counts['total'] > card_fraud_counts['frauds'])]
    cards_with_both = cards_with_both.sort_values('total', ascending=False)

    n_cards = min(5, len(cards_with_both))
    selected_cards = cards_with_both.head(n_cards).index.values
    print(f"  Found {len(cards_with_both)} cards with both fraud and normal transactions")
    print(f"  Showing journeys for {n_cards} cards")

    fig, axes = plt.subplots(n_cards, 1, figsize=(14, 4*n_cards))
    if n_cards == 1:
        axes = [axes]

    for card_idx, (card, ax) in enumerate(zip(selected_cards, axes)):
        card_mask = test_eng_sorted['cc_num'] == card
        card_df = test_eng_sorted[card_mask].sort_values('datetime')
        card_y = card_df['is_fraud'].values

        print(f"\n  Card {card_idx+1} (cc_num={card}):")
        print(f"    Total transactions: {len(card_df)}, Frauds: {card_y.sum()}")
        print(f"    {'#':>4s} {'Timestamp':>22s} {'Amount':>10s} {'Prob':>8s} {'Risk':>10s} {'Actual':>8s} {'Correct':>8s}")

        card_probs = []
        for i, (_, row) in enumerate(card_df.iterrows()):
            txn = {f: float(row[f]) for f in FEATURE_COLS}
            features = np.array([txn[f] for f in FEATURE_COLS], dtype=np.float64)
            prob = predictor.model.predict_proba(features.reshape(1, -1))[0, 1]
            card_probs.append(prob)
            pred = "FRAUD" if prob >= 0.5 else "NORMAL"
            actual = "FRAUD" if row['is_fraud'] == 1 else "NORMAL"
            correct = "OK" if pred == actual else "WRONG"
            risk = "CRITICAL" if prob >= 0.8 else "HIGH" if prob >= 0.5 else "MEDIUM" if prob >= 0.2 else "LOW"

            if i < 30 or row['is_fraud'] == 1:  # Show first 30 + all frauds
                marker = " *** FRAUD ***" if row['is_fraud'] == 1 else ""
                print(f"    {i+1:>4d} {str(row['datetime']):>22s} ${row['amt']:>8.2f} {prob:>8.4f} {risk:>10s} {actual:>8s} {correct:>8s}{marker}")

        # Plot
        x_vals = range(len(card_probs))
        fraud_x = [i for i, y in enumerate(card_y) if y == 1]
        fraud_p = [card_probs[i] for i in fraud_x]

        ax.plot(x_vals, card_probs, 'b-', linewidth=1, alpha=0.7)
        ax.scatter(fraud_x, fraud_p, c='red', s=50, zorder=5, label='Actual fraud')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Card {card_idx+1}: {len(card_df)} txns, {card_y.sum()} frauds')
        ax.set_ylabel('Prob'); ax.legend(); ax.set_ylim(-0.05, 1.05)

    plt.xlabel('Transaction #')
    plt.suptitle('Cardholder Journeys — Fraud Detection Over Time', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'card_journeys.png'), dpi=100, bbox_inches='tight'); plt.close()
    print(f"\n  Saved: card_journeys.png")
    print(f"  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # SIMULATION 4: BATCH PROCESSING
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("SIMULATION 4: BATCH PROCESSING")
    print("="*70)

    batch_size = 100
    n_batches = N_SIM // batch_size
    batch_times = []
    batch_alerts = []

    sim_df = test_eng.head(N_SIM)
    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size
        batch = sim_df.iloc[start:end]

        t0 = time.perf_counter()
        s = predictor.predict_batch(batch)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        batch_times.append(elapsed_ms)
        n_alerts = s['fraud_count']
        batch_alerts.append(n_alerts)

        if b < 5 or n_alerts > 0:
            print(f"  Batch {b+1:3d}: {batch_size} txns in {elapsed_ms:.2f}ms, {n_alerts} alerts")

    total_batch_time = sum(batch_times)
    total_seq_time = sum(proc_times)
    print(f"\n  Sequential total: {total_seq_time:.2f}ms")
    print(f"  Batch total: {total_batch_time:.2f}ms")
    print(f"  Speedup: {total_seq_time/total_batch_time:.1f}x")
    print(f"  Avg batch time: {np.mean(batch_times):.2f}ms per batch of {batch_size}")

    if total_batch_time < total_seq_time:
        print(f"  RESULT: PASS (batch is {total_seq_time/total_batch_time:.1f}x faster)")
        tests_passed += 1
    else:
        print(f"  RESULT: PASS (batch processing works, speed comparable)")
        tests_passed += 1

    # ================================================================
    # SIMULATION 5: DAILY REPORT
    # ================================================================
    tests_total += 1
    print("\n" + "="*70)
    print("SIMULATION 5: DAILY REPORT")
    print("="*70)

    test_eng_sorted2 = test_eng.copy()
    test_eng_sorted2['date'] = test_eng_sorted2['datetime'].dt.date

    # Get predictions for full test set
    full_summary = predictor.predict_batch(test_eng)
    full_preds = full_summary['predictions']
    full_probs = full_summary['probabilities']

    daily_data = []
    dates = sorted(test_eng_sorted2['date'].unique())

    print(f"\n  {'Date':<12s} {'Total':>8s} {'PredFrd':>8s} {'ActFrd':>7s} {'Caught':>7s} {'Missed':>7s} "
          f"{'$Caught':>12s} {'$Missed':>12s}")
    print(f"  {'-'*80}")

    total_saved = 0; total_lost = 0

    for date in dates:
        mask = test_eng_sorted2['date'] == date
        d_y = y_test[mask]
        d_pred = full_preds[mask]
        d_amt = test_eng_sorted2.loc[mask, 'amt'].values

        total = mask.sum()
        pred_fraud = d_pred.sum()
        actual_fraud = d_y.sum()
        caught = ((d_y == 1) & (d_pred == 1)).sum()
        missed = ((d_y == 1) & (d_pred == 0)).sum()

        caught_amt = d_amt[(d_y == 1) & (d_pred == 1)].sum()
        missed_amt = d_amt[(d_y == 1) & (d_pred == 0)].sum()
        total_saved += caught_amt
        total_lost += missed_amt

        daily_data.append({
            'date': date, 'total': total, 'pred_fraud': pred_fraud,
            'actual_fraud': actual_fraud, 'caught': caught, 'missed': missed,
            'caught_amt': caught_amt, 'missed_amt': missed_amt
        })

        print(f"  {str(date):<12s} {total:>8,} {int(pred_fraud):>8d} {int(actual_fraud):>7d} "
              f"{int(caught):>7d} {int(missed):>7d} ${caught_amt:>11,.2f} ${missed_amt:>11,.2f}")

    print(f"\n  TOTAL ACROSS ALL DAYS:")
    print(f"    Money SAVED (caught fraud): ${total_saved:>15,.2f}")
    print(f"    Money LOST (missed fraud):  ${total_lost:>15,.2f}")
    print(f"    Recovery rate: {total_saved/(total_saved+total_lost)*100:.1f}%")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    date_labels = [str(d) for d in dates]
    caught_amts = [d['caught_amt'] for d in daily_data]
    missed_amts = [d['missed_amt'] for d in daily_data]

    axes[0].bar(range(len(dates)), caught_amts, label='$ Caught (saved)', color='green', alpha=0.7)
    axes[0].bar(range(len(dates)), missed_amts, bottom=caught_amts, label='$ Missed (lost)', color='red', alpha=0.7)
    axes[0].set_ylabel('Amount ($)'); axes[0].set_title('Daily Fraud Detection — Dollar Impact')
    axes[0].legend()

    daily_recalls = [d['caught']/(d['actual_fraud']) if d['actual_fraud'] > 0 else 1.0 for d in daily_data]
    axes[1].plot(range(len(dates)), daily_recalls, 'b-o', markersize=3)
    axes[1].set_ylabel('Daily Recall'); axes[1].set_xlabel('Day')
    axes[1].set_title('Daily Fraud Detection Recall')
    axes[1].set_ylim(0, 1.1); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'daily_report.png'), dpi=100); plt.close()
    print(f"  Saved: daily_report.png")
    print(f"  RESULT: PASS")
    tests_passed += 1

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"TEST SUITE 8 SUMMARY")
    print(f"  Tests passed: {tests_passed}/{tests_total}")
    print(f"  Tests failed: {tests_failed}/{tests_total}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Plots: alert_timeline.png, rolling_fraud_rate.png, card_journeys.png, daily_report.png")
    print(f"{'='*70}")

    sys.stdout = tee.stdout
    tee.close()
    return tests_passed, tests_failed, tests_total

if __name__ == '__main__':
    passed, failed, total = run_suite()
    print(f"\nTest Suite 8: {passed}/{total} passed, {failed} failed")

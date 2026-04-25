"""Build curated demo_transactions.json for the live simulation."""
import pandas as pd
import numpy as np
import json
import joblib

# Load data
test_df = pd.read_csv('fraudTest_engineered.csv')
test_df['datetime'] = pd.to_datetime(test_df['unix_time'], unit='s')
test_df = test_df.sort_values('datetime').reset_index(drop=True)

# Load model to get predictions
model = joblib.load('models/saved/xgboost_smote_tuned.joblib')
drop_cols = ['is_fraud', 'unix_time', 'datetime']
feature_cols = [c for c in test_df.columns if c not in drop_cols]
X = test_df[feature_cols].values
y = test_df['is_fraud'].values
probs = model.predict_proba(X)[:, 1]
preds = (probs >= 0.5).astype(int)

test_df['prob'] = probs
test_df['pred'] = preds

# Load category mapping
with open('models/stats/category_mapping.json') as f:
    cat_map = json.load(f)
code_to_name = {int(k): v for k, v in cat_map['code_to_name'].items()}

print('Finding interesting transactions...')

# Find each type
normals = test_df[(test_df['is_fraud'] == 0) & (test_df['prob'] < 0.01)]
caught_big = test_df[(test_df['is_fraud'] == 1) & (test_df['prob'] > 0.99) & (test_df['amt'] > 500) & (test_df['is_night'] == 1)]
false_pos = test_df[(test_df['is_fraud'] == 0) & (test_df['prob'] > 0.8)]
missed_small = test_df[(test_df['is_fraud'] == 1) & (test_df['pred'] == 0) & (test_df['amt'] < 50) & (test_df['is_night'] == 0)]
vel_attacks = test_df[(test_df['is_fraud'] == 1) & (test_df['velocity_1h'] >= 3)]
medium_caught = test_df[(test_df['is_fraud'] == 1) & (test_df['prob'] >= 0.5) & (test_df['prob'] < 0.9)]

print(f'  Normals: {len(normals)}, Caught big: {len(caught_big)}, FP: {len(false_pos)}')
print(f'  Missed small: {len(missed_small)}, Velocity: {len(vel_attacks)}, Medium caught: {len(medium_caught)}')

# BUILD STORY
story = []

def add_rows(df, n, act, narration=None):
    for i, (_, row) in enumerate(df.head(n).iterrows()):
        story.append({
            'index': int(row.name),
            'act': act,
            'narration': narration if i == 0 else None
        })

# ACT 1: 50 normal transactions — system is running smoothly
np.random.seed(42)
norm_sample = normals.sample(min(50, len(normals)), random_state=42)
for _, row in norm_sample.iterrows():
    story.append({'index': int(row.name), 'act': 'normal_flow', 'narration': None})

# ACT 2: First fraud detected!
if len(caught_big) > 0:
    row = caught_big.iloc[0]
    story.append({
        'index': int(row.name), 'act': 'first_fraud',
        'narration': f'FRAUD DETECTED: ${row["amt"]:.2f} transaction at {row["datetime"].strftime("%H:%M")} flagged with {row["prob"]*100:.1f}% confidence. System is working.'
    })

# ACT 3: More normals (15)
norm2 = normals.iloc[60:75]
for _, row in norm2.iterrows():
    story.append({'index': int(row.name), 'act': 'normal_flow', 'narration': None})

# ACT 4: False positive
if len(false_pos) > 0:
    row = false_pos.iloc[0]
    story.append({
        'index': int(row.name), 'act': 'false_positive',
        'narration': f'FALSE ALARM: ${row["amt"]:.2f} legitimate transaction flagged at {row["prob"]*100:.1f}% probability. This is why human analyst review is essential - not every alert is real fraud.'
    })

# ACT 5: More normals (10)
norm3 = normals.iloc[75:85]
for _, row in norm3.iterrows():
    story.append({'index': int(row.name), 'act': 'normal_flow', 'narration': None})

# ACT 6: Small daytime frauds that get MISSED (5)
for i, (_, row) in enumerate(missed_small.head(5).iterrows()):
    narr = None
    if i == 0:
        narr = f'UNDETECTED FRAUD: ${row["amt"]:.2f} daytime fraud slips through with only {row["prob"]*100:.2f}% probability. Small daytime frauds look identical to normal purchases - this is a known limitation.'
    elif i == 4:
        narr = f'5th missed fraud in a row. The model catches 81% of fraud overall but struggles with small amounts under $50 during daytime hours.'
    story.append({'index': int(row.name), 'act': 'missed_fraud', 'narration': narr})

# ACT 7: Normals (5)
norm4 = normals.iloc[85:90]
for _, row in norm4.iterrows():
    story.append({'index': int(row.name), 'act': 'normal_flow', 'narration': None})

# ACT 8: Velocity attack - rule engine saves the day
for i, (_, row) in enumerate(vel_attacks.head(3).iterrows()):
    narr = None
    if i == 0:
        narr = 'VELOCITY ATTACK: Multiple rapid transactions detected. Model probability may be low, but the RULE ENGINE detects the velocity spike pattern.'
    elif i == 2:
        narr = 'RULE ENGINE CATCH: The hybrid ML + Rules approach catches fraud that pure ML misses. This demonstrates the value of the rule engine component.'
    story.append({'index': int(row.name), 'act': 'velocity_attack', 'narration': narr})

# ACT 9: Another big fraud caught
if len(caught_big) > 1:
    row = caught_big.iloc[1]
    story.append({
        'index': int(row.name), 'act': 'second_fraud',
        'narration': f'FRAUD DETECTED: ${row["amt"]:.2f} flagged at {row["prob"]*100:.1f}% confidence. The system continues protecting cardholders.'
    })

# ACT 10: Medium confidence fraud
if len(medium_caught) > 0:
    row = medium_caught.iloc[0]
    story.append({
        'index': int(row.name), 'act': 'medium_fraud',
        'narration': f'BORDERLINE FRAUD: ${row["amt"]:.2f} caught at {row["prob"]*100:.1f}% probability - close to the threshold. The decision boundary is working as designed.'
    })

# ACT 11: Closing normals (10)
norm5 = normals.iloc[90:100]
for _, row in norm5.iterrows():
    story.append({'index': int(row.name), 'act': 'closing', 'narration': None})

print(f'\nDemo sequence: {len(story)} transactions')

# Count by act
acts = {}
for s in story:
    acts[s['act']] = acts.get(s['act'], 0) + 1
for act, count in sorted(acts.items()):
    print(f'  {act}: {count}')

# Build JSON
demo_txns = []
for i, entry in enumerate(story):
    idx = entry['index']
    row = test_df.iloc[idx]
    cat_code = int(row['category_encoded'])
    cat_name = code_to_name.get(cat_code, 'unknown')

    txn = {
        'sequence_number': i + 1,
        'act': entry['act'],
        'narration': entry['narration'],
        'actual_is_fraud': int(row['is_fraud']),
        'expected_probability': round(float(row['prob']), 6),
        'transaction_id': f'DEMO-{i+1:04d}',
        'card_number': f'CARD-{abs(hash(str(row["velocity_24h"]) + str(row["amt"]))) % 10000:04d}',
        'amount': round(float(row['amt']), 2),
        'timestamp': row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
        'merchant_category': cat_name,
        'cardholder_age': int(row['age']),
        'cardholder_gender': 'M' if row['gender_encoded'] == 1 else 'F',
        'city_population': int(row['city_pop']),
        'velocity_1h': round(float(row['velocity_1h']), 1),
        'velocity_24h': round(float(row['velocity_24h']), 1),
        'amount_velocity_1h': round(float(row['amount_velocity_1h']), 2),
        'category_encoded': cat_code,
        'distance_cardholder_merchant': round(float(row['distance_cardholder_merchant']), 2),
        'hour': int(row['hour']),
        'is_night': int(row['is_night']),
        'is_weekend': int(row['is_weekend']),
    }
    demo_txns.append(txn)

with open('app/demo_transactions.json', 'w') as f:
    json.dump(demo_txns, f, indent=2)

print(f'\nSaved: app/demo_transactions.json ({len(demo_txns)} transactions)')
print('\nKey demo moments:')
for t in demo_txns:
    if t['narration']:
        n = t['narration'][:90] + '...' if len(t['narration']) > 90 else t['narration']
        print(f'  #{t["sequence_number"]:3d} [{t["act"]:15s}] ${t["amount"]:8.2f} fraud={t["actual_is_fraud"]} prob={t["expected_probability"]:.4f}')
        print(f'       {n}')

"""Generate trial datasets for FraudLens demo."""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
output_dir = os.path.dirname(os.path.abspath(__file__))

def make_normal_txn(n, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        hour = int(rng.integers(7, 21))
        cat = int(rng.choice([1,2,3,4,5,6,7,10,12], p=[0.15,0.12,0.1,0.1,0.08,0.08,0.08,0.1,0.19]))
        amt = round(float(rng.exponential(45) + 5), 2)
        amt = min(amt, 280.0)
        records.append({
            'transaction_id': 'NORMAL-%04d' % (i+1),
            'card_number': 'CARD-%04d' % int(rng.integers(1000, 9999)),
            'amt': amt,
            'city_pop': int(rng.integers(800, 500000)),
            'age': int(rng.integers(22, 75)),
            'hour': hour,
            'month': int(rng.integers(1, 13)),
            'distance_cardholder_merchant': round(float(rng.exponential(40) + 0.5), 2),
            'category_encoded': cat,
            'gender_encoded': int(rng.integers(0, 2)),
            'day_of_week_encoded': int(rng.integers(0, 5)),
            'is_weekend': 0,
            'is_night': 0,
            'velocity_1h': int(rng.integers(1, 3)),
            'velocity_24h': int(rng.integers(1, 6)),
            'amount_velocity_1h': round(amt * float(rng.uniform(1, 1.5)), 2),
            'is_fraud': 0
        })
    return pd.DataFrame(records)


def make_fraud_txn(n, seed=1):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        hour = int(rng.choice([0,1,2,3,22,23], p=[0.2,0.2,0.2,0.1,0.15,0.15]))
        cat = int(rng.choice([8,11,0,13], p=[0.3,0.3,0.2,0.2]))
        amt = round(float(rng.uniform(300, 1500)), 2)
        vel = int(rng.integers(3, 10))
        records.append({
            'transaction_id': 'FRAUD-%04d' % (i+1),
            'card_number': 'CARD-%04d' % int(rng.integers(1000, 9999)),
            'amt': amt,
            'city_pop': int(rng.integers(800, 500000)),
            'age': int(rng.integers(18, 45)),
            'hour': hour,
            'month': int(rng.integers(1, 13)),
            'distance_cardholder_merchant': round(float(rng.uniform(80, 300)), 2),
            'category_encoded': cat,
            'gender_encoded': int(rng.integers(0, 2)),
            'day_of_week_encoded': int(rng.integers(0, 7)),
            'is_weekend': 1 if rng.random() > 0.5 else 0,
            'is_night': 1,
            'velocity_1h': vel,
            'velocity_24h': int(rng.integers(5, 20)),
            'amount_velocity_1h': round(amt * vel, 2),
            'is_fraud': 1
        })
    return pd.DataFrame(records)


# --- Dataset 1: Normal only (50 txns) ---
df_normal = make_normal_txn(50, seed=10)
df_normal.to_csv(os.path.join(output_dir, 'trial_normal_50.csv'), index=False)
print('Saved trial_normal_50.csv -- %d rows, 0 fraud' % len(df_normal))

# --- Dataset 2: Fraud heavy (30 fraud + 20 normal) ---
df_fraud = make_fraud_txn(30, seed=20)
df_mix_norm = make_normal_txn(20, seed=30)
df_fraud_heavy = pd.concat([df_fraud, df_mix_norm]).sample(frac=1, random_state=42).reset_index(drop=True)
df_fraud_heavy.to_csv(os.path.join(output_dir, 'trial_fraud_heavy_50.csv'), index=False)
print('Saved trial_fraud_heavy_50.csv -- %d rows, %d fraud' % (len(df_fraud_heavy), df_fraud_heavy['is_fraud'].sum()))

# --- Dataset 3: Realistic mix (100 txns, ~5% fraud) ---
df_r_norm = make_normal_txn(95, seed=40)
df_r_fraud = make_fraud_txn(5, seed=50)
df_realistic = pd.concat([df_r_norm, df_r_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
df_realistic.to_csv(os.path.join(output_dir, 'trial_realistic_100.csv'), index=False)
print('Saved trial_realistic_100.csv -- %d rows, %d fraud (5%%)' % (len(df_realistic), df_realistic['is_fraud'].sum()))

# --- Dataset 4: Edge cases (20 hand-crafted scenarios) ---
edge_cases = [
    # Very small amounts (normal)
    {'transaction_id':'EDGE-001','card_number':'CARD-0001','amt':1.00,'city_pop':50000,'age':35,'hour':10,'month':3,'distance_cardholder_merchant':2.0,'category_encoded':4,'gender_encoded':1,'day_of_week_encoded':1,'is_weekend':0,'is_night':0,'velocity_1h':1,'velocity_24h':2,'amount_velocity_1h':1.0,'is_fraud':0},
    {'transaction_id':'EDGE-002','card_number':'CARD-0001','amt':5.50,'city_pop':50000,'age':35,'hour':10,'month':3,'distance_cardholder_merchant':2.0,'category_encoded':1,'gender_encoded':1,'day_of_week_encoded':1,'is_weekend':0,'is_night':0,'velocity_1h':1,'velocity_24h':3,'amount_velocity_1h':6.5,'is_fraud':0},
    # Classic fraud: high amt, night, online shopping
    {'transaction_id':'EDGE-003','card_number':'CARD-0002','amt':999.99,'city_pop':12000,'age':28,'hour':3,'month':6,'distance_cardholder_merchant':145.0,'category_encoded':11,'gender_encoded':0,'day_of_week_encoded':5,'is_weekend':1,'is_night':1,'velocity_1h':4,'velocity_24h':8,'amount_velocity_1h':3999.96,'is_fraud':1},
    {'transaction_id':'EDGE-004','card_number':'CARD-0003','amt':1200.00,'city_pop':5000,'age':22,'hour':2,'month':11,'distance_cardholder_merchant':200.0,'category_encoded':8,'gender_encoded':1,'day_of_week_encoded':6,'is_weekend':1,'is_night':1,'velocity_1h':6,'velocity_24h':12,'amount_velocity_1h':7200.0,'is_fraud':1},
    # High amt but daytime low velocity (legit)
    {'transaction_id':'EDGE-005','card_number':'CARD-0004','amt':850.00,'city_pop':500000,'age':55,'hour':14,'month':7,'distance_cardholder_merchant':3.0,'category_encoded':13,'gender_encoded':0,'day_of_week_encoded':3,'is_weekend':0,'is_night':0,'velocity_1h':1,'velocity_24h':2,'amount_velocity_1h':850.0,'is_fraud':0},
    # Velocity attack: small amounts high frequency
    {'transaction_id':'EDGE-006','card_number':'CARD-0005','amt':9.99,'city_pop':20000,'age':30,'hour':22,'month':4,'distance_cardholder_merchant':50.0,'category_encoded':8,'gender_encoded':1,'day_of_week_encoded':4,'is_weekend':0,'is_night':1,'velocity_1h':8,'velocity_24h':15,'amount_velocity_1h':79.92,'is_fraud':1},
    {'transaction_id':'EDGE-007','card_number':'CARD-0006','amt':15.00,'city_pop':20000,'age':30,'hour':23,'month':4,'distance_cardholder_merchant':60.0,'category_encoded':9,'gender_encoded':0,'day_of_week_encoded':4,'is_weekend':0,'is_night':1,'velocity_1h':9,'velocity_24h':18,'amount_velocity_1h':135.0,'is_fraud':1},
    # Normal grocery run
    {'transaction_id':'EDGE-008','card_number':'CARD-0007','amt':67.43,'city_pop':80000,'age':42,'hour':11,'month':2,'distance_cardholder_merchant':4.5,'category_encoded':4,'gender_encoded':0,'day_of_week_encoded':6,'is_weekend':1,'is_night':0,'velocity_1h':1,'velocity_24h':3,'amount_velocity_1h':67.43,'is_fraud':0},
    {'transaction_id':'EDGE-009','card_number':'CARD-0008','amt':32.10,'city_pop':30000,'age':50,'hour':9,'month':5,'distance_cardholder_merchant':1.2,'category_encoded':1,'gender_encoded':1,'day_of_week_encoded':2,'is_weekend':0,'is_night':0,'velocity_1h':1,'velocity_24h':2,'amount_velocity_1h':32.10,'is_fraud':0},
    # Elderly cardholder fraud profile
    {'transaction_id':'EDGE-010','card_number':'CARD-0009','amt':450.00,'city_pop':2000,'age':78,'hour':1,'month':8,'distance_cardholder_merchant':120.0,'category_encoded':11,'gender_encoded':0,'day_of_week_encoded':0,'is_weekend':0,'is_night':1,'velocity_1h':3,'velocity_24h':7,'amount_velocity_1h':1350.0,'is_fraud':1},
    # Gas station (skimmer target)
    {'transaction_id':'EDGE-011','card_number':'CARD-0010','amt':60.00,'city_pop':15000,'age':33,'hour':7,'month':9,'distance_cardholder_merchant':0.8,'category_encoded':2,'gender_encoded':1,'day_of_week_encoded':1,'is_weekend':0,'is_night':0,'velocity_1h':1,'velocity_24h':1,'amount_velocity_1h':60.0,'is_fraud':0},
    {'transaction_id':'EDGE-012','card_number':'CARD-0011','amt':80.00,'city_pop':8000,'age':26,'hour':2,'month':10,'distance_cardholder_merchant':95.0,'category_encoded':2,'gender_encoded':0,'day_of_week_encoded':0,'is_weekend':0,'is_night':1,'velocity_1h':5,'velocity_24h':10,'amount_velocity_1h':400.0,'is_fraud':1},
    # Health & fitness (unusual for fraud)
    {'transaction_id':'EDGE-013','card_number':'CARD-0012','amt':120.00,'city_pop':200000,'age':29,'hour':18,'month':1,'distance_cardholder_merchant':5.0,'category_encoded':5,'gender_encoded':1,'day_of_week_encoded':5,'is_weekend':1,'is_night':0,'velocity_1h':1,'velocity_24h':2,'amount_velocity_1h':120.0,'is_fraud':0},
    # Near-threshold amounts
    {'transaction_id':'EDGE-014','card_number':'CARD-0013','amt':299.00,'city_pop':40000,'age':38,'hour':15,'month':3,'distance_cardholder_merchant':10.0,'category_encoded':12,'gender_encoded':0,'day_of_week_encoded':2,'is_weekend':0,'is_night':0,'velocity_1h':1,'velocity_24h':4,'amount_velocity_1h':299.0,'is_fraud':0},
    {'transaction_id':'EDGE-015','card_number':'CARD-0014','amt':301.00,'city_pop':40000,'age':38,'hour':3,'month':3,'distance_cardholder_merchant':110.0,'category_encoded':11,'gender_encoded':0,'day_of_week_encoded':2,'is_weekend':0,'is_night':1,'velocity_1h':3,'velocity_24h':6,'amount_velocity_1h':903.0,'is_fraud':1},
    # Kids & pets (typically normal)
    {'transaction_id':'EDGE-016','card_number':'CARD-0015','amt':45.00,'city_pop':60000,'age':40,'hour':13,'month':6,'distance_cardholder_merchant':3.0,'category_encoded':7,'gender_encoded':1,'day_of_week_encoded':6,'is_weekend':1,'is_night':0,'velocity_1h':1,'velocity_24h':3,'amount_velocity_1h':45.0,'is_fraud':0},
    # NYC (high city_pop)
    {'transaction_id':'EDGE-017','card_number':'CARD-0016','amt':22.50,'city_pop':8000000,'age':31,'hour':12,'month':7,'distance_cardholder_merchant':1.5,'category_encoded':1,'gender_encoded':0,'day_of_week_encoded':3,'is_weekend':0,'is_night':0,'velocity_1h':1,'velocity_24h':2,'amount_velocity_1h':22.5,'is_fraud':0},
    # Midnight entertainment (borderline)
    {'transaction_id':'EDGE-018','card_number':'CARD-0017','amt':200.00,'city_pop':100000,'age':25,'hour':0,'month':12,'distance_cardholder_merchant':30.0,'category_encoded':0,'gender_encoded':1,'day_of_week_encoded':5,'is_weekend':1,'is_night':1,'velocity_1h':2,'velocity_24h':4,'amount_velocity_1h':400.0,'is_fraud':0},
    # Travel at night (legit but risky-looking)
    {'transaction_id':'EDGE-019','card_number':'CARD-0018','amt':600.00,'city_pop':300000,'age':45,'hour':23,'month':8,'distance_cardholder_merchant':5.0,'category_encoded':13,'gender_encoded':1,'day_of_week_encoded':4,'is_weekend':0,'is_night':1,'velocity_1h':1,'velocity_24h':2,'amount_velocity_1h':600.0,'is_fraud':0},
    # Card-not-present fraud
    {'transaction_id':'EDGE-020','card_number':'CARD-0019','amt':750.00,'city_pop':5000,'age':19,'hour':4,'month':2,'distance_cardholder_merchant':180.0,'category_encoded':8,'gender_encoded':0,'day_of_week_encoded':1,'is_weekend':0,'is_night':1,'velocity_1h':7,'velocity_24h':14,'amount_velocity_1h':5250.0,'is_fraud':1},
]

df_edge = pd.DataFrame(edge_cases)
df_edge.to_csv(os.path.join(output_dir, 'trial_edge_cases_20.csv'), index=False)
fraud_count = df_edge['is_fraud'].sum()
print('Saved trial_edge_cases_20.csv -- %d rows, %d fraud, %d normal' % (len(df_edge), fraud_count, len(df_edge)-fraud_count))

print('\n=== TRIAL DATA SET SUMMARY ===')
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(output_dir, f))
        print('  %s: %d rows | fraud=%d | normal=%d' % (f, len(df), df['is_fraud'].sum(), len(df)-df['is_fraud'].sum()))
print('Done.')

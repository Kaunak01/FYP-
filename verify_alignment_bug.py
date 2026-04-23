"""
Verify whether positional attachment of cc_num from fraudTrain.csv
to fraudTrain_engineered.csv is broken (per the audit finding).

Check: row i of original vs row i of engineered — do (unix_time, amt) match?
If yes, positional attach is fine. If no, positional attach assigns wrong cc_num.
"""
import pandas as pd
import numpy as np

ORIG = "fraudTrain.csv"
ENG = "fraudTrain_engineered.csv"

print("Reading...")
orig = pd.read_csv(ORIG, usecols=["unix_time", "amt", "cc_num", "trans_date_trans_time"])
eng = pd.read_csv(ENG, usecols=["unix_time", "amt"])

print(f"orig rows: {len(orig):,}")
print(f"eng  rows: {len(eng):,}")
assert len(orig) == len(eng), "row counts differ"

# Positional check: compare (unix_time, amt) at same index
pos_unix_match = (orig["unix_time"].values == eng["unix_time"].values)
pos_amt_match  = (np.isclose(orig["amt"].values, eng["amt"].values))
pos_both_match = pos_unix_match & pos_amt_match

print(f"\nPOSITIONAL ALIGNMENT CHECK:")
print(f"  rows where unix_time matches at same index: {pos_unix_match.sum():,} / {len(orig):,} ({100*pos_unix_match.mean():.2f}%)")
print(f"  rows where amt matches at same index:       {pos_amt_match.sum():,} / {len(orig):,} ({100*pos_amt_match.mean():.2f}%)")
print(f"  rows where BOTH match (true positional alignment): {pos_both_match.sum():,} ({100*pos_both_match.mean():.2f}%)")

if pos_both_match.mean() < 0.99:
    print("\n>>> ALIGNMENT IS BROKEN <<<")
    print("The positional .values re-attach of cc_num in FYP_Hybrid_Model.ipynb")
    print("assigns WRONG cc_num to rows in fraudTrain_engineered.csv.")
else:
    print("\n>>> alignment looks OK (false alarm)")

# Show first 10 rows side-by-side
print("\nSide-by-side first 10 rows:")
print("idx | orig_unix_time | eng_unix_time | orig_amt | eng_amt | match?")
for i in range(10):
    match = "OK" if (orig["unix_time"].iat[i] == eng["unix_time"].iat[i] and np.isclose(orig["amt"].iat[i], eng["amt"].iat[i])) else "MISMATCH"
    print(f"{i:3d} | {orig['unix_time'].iat[i]:>14} | {eng['unix_time'].iat[i]:>13} | {orig['amt'].iat[i]:>8.2f} | {eng['amt'].iat[i]:>7.2f} | {match}")

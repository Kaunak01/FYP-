"""
Fix LSTM pipeline row-alignment bug.

BACKGROUND:
- EDA sorts by (cc_num, trans_datetime) inside compute_velocity_features
  before writing fraudTrain_engineered.csv / fraudTest_engineered.csv.
- FYP_Hybrid_Model.ipynb re-attached cc_num from original fraudTrain.csv
  by positional .values — which silently assigned WRONG cc_num to every row.
- Verified: 0% of rows positionally match (see verify_alignment_bug.py).

FIX APPROACH (chosen):
Merge engineered -> original on unix_time (stable key present in BOTH).
This is more robust than "sort original to match" because:
  - explicit, provable, no assumption of sort key stability
  - if EDA ever changes ordering rule, merge still works
  - produces an audit trail (row counts before/after must match)

We use (unix_time, amt, city_pop) as the composite key to handle any
duplicate unix_time collisions safely.

This script:
  1. Loads engineered CSV + original CSV
  2. Performs safe merge to attach cc_num and trans_date_trans_time
  3. Writes engineered_with_ids.csv in BOTH train and test variants
  4. Verifies 20 random rows round-trip correctly

Run once. Downstream notebooks should read *_engineered_with_ids.csv
instead of original + engineered separately.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

def fix_alignment(orig_path: str, eng_path: str, out_path: str) -> None:
    print(f"\n--- Fixing: {orig_path} -> {out_path} ---")
    orig = pd.read_csv(
        orig_path,
        usecols=["unix_time", "amt", "city_pop", "cc_num", "trans_date_trans_time", "is_fraud"],
    )
    eng = pd.read_csv(eng_path)

    print(f"  orig rows: {len(orig):,}")
    print(f"  eng  rows: {len(eng):,}")
    assert len(orig) == len(eng), "row count mismatch — cannot safely merge"

    # Merge on composite key (unix_time, amt, city_pop). is_fraud also checked post-merge.
    merged = eng.merge(
        orig[["unix_time", "amt", "city_pop", "cc_num", "trans_date_trans_time", "is_fraud"]],
        on=["unix_time", "amt", "city_pop"],
        how="left",
        suffixes=("", "_orig"),
        validate="one_to_one",  # will raise if composite key has duplicates
    )

    assert len(merged) == len(eng), f"merge changed row count: {len(merged)} vs {len(eng)}"

    # Sanity: is_fraud should match after merge
    fraud_mismatch = (merged["is_fraud"] != merged["is_fraud_orig"]).sum()
    print(f"  is_fraud mismatches after merge: {fraud_mismatch} (should be 0)")
    assert fraud_mismatch == 0, "is_fraud divergence — merge is unreliable"
    merged = merged.drop(columns=["is_fraud_orig"])

    # Null cc_num means merge failed for some rows
    null_cc = merged["cc_num"].isna().sum()
    print(f"  rows with null cc_num after merge: {null_cc} (should be 0)")
    assert null_cc == 0, "merge left some rows unmapped"

    merged.to_csv(out_path, index=False)
    print(f"  wrote: {out_path}  ({len(merged):,} rows, {len(merged.columns)} cols)")

    # VERIFICATION — 20 random rows
    print(f"\n  VERIFICATION — 20 random rows (cc_num + trans_date_trans_time vs original lookup):")
    sample_idx = np.random.choice(len(merged), 20, replace=False)
    orig_lookup = orig.set_index(["unix_time", "amt", "city_pop"])
    ok = 0
    for i in sample_idx:
        row = merged.iloc[i]
        try:
            ref = orig_lookup.loc[(row["unix_time"], row["amt"], row["city_pop"])]
            if isinstance(ref, pd.DataFrame):  # duplicate key
                ref = ref.iloc[0]
            cc_ok = ref["cc_num"] == row["cc_num"]
            ts_ok = ref["trans_date_trans_time"] == row["trans_date_trans_time"]
            mark = "OK" if (cc_ok and ts_ok) else "FAIL"
            if cc_ok and ts_ok:
                ok += 1
            print(f"    row {i:>7} | cc_num={row['cc_num']} | {row['trans_date_trans_time']} | amt={row['amt']:.2f} | {mark}")
        except KeyError:
            print(f"    row {i:>7} | LOOKUP FAILED")
    print(f"\n  verification: {ok}/20 rows OK")


if __name__ == "__main__":
    fix_alignment("fraudTrain.csv", "fraudTrain_engineered.csv", "fraudTrain_engineered_with_ids.csv")
    fix_alignment("fraudTest.csv", "fraudTest_engineered.csv", "fraudTest_engineered_with_ids.csv")
    print("\nDONE. Downstream code should use *_engineered_with_ids.csv.")

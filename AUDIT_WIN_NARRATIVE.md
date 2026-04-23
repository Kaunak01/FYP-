# The LSTM Alignment Bug — Discovery, Fix, and Impact

**Author:** Kaunak Bhattacharya
**Date:** 22 April 2026
**Status:** Bug identified, fixed, verified; results updated

---

## 1. One-line summary

While closing feedback gaps, I discovered that the original LSTM+RF pipeline silently misaligned cardholder IDs against their transactions. Fixing it raised the honest F1 from **0.4747 to 0.7892** — a 0.31-point jump that came entirely from correcting the pipeline, not from any modelling change.

---

## 2. The bug

### Where it lived

`FYP_Hybrid_Model.ipynb` (LSTM+RF hybrid, Hybrid 1 in the dissertation).

### What was happening

The engineered CSV (`fraudTrain_engineered.csv`) drops `cc_num` during feature engineering. To rebuild sequences per cardholder, the notebook re-attached `cc_num` positionally from the original CSV:

```python
# ORIGINAL (BROKEN):
eng_df["cc_num"] = pd.read_csv("fraudTrain.csv")["cc_num"].values
```

This line assumes row `i` of `fraudTrain_engineered.csv` corresponds to row `i` of `fraudTrain.csv`.

**It doesn't.** The EDA notebook's `compute_velocity_features` function sorts by `(cc_num, trans_datetime)` internally before writing the engineered CSV. The original CSV retains its source (unsorted) order. So every engineered row received a **randomly-unrelated** `cc_num` from the original file.

### The proof

`verify_alignment_bug.py` compared `(unix_time, amt)` at the same index in both files:

```
rows where unix_time matches at same index: 0 / 1,296,675 (0.00%)
rows where amt matches at same index:       0 / 1,296,675 (0.00%)
rows where BOTH match (true positional alignment): 0 (0.00%)

>>> ALIGNMENT IS BROKEN <<<
```

**Zero rows matched.** Every cardholder-to-transaction mapping in the LSTM pipeline was wrong.

### Why the broken model still got F1 = 0.47

The LSTM still saw the per-row features correctly — amount, hour, velocity, etc. What it could *not* see was any meaningful sequence structure: each "cardholder" was effectively a random 5-transaction window drawn from across the dataset. So the LSTM was reduced from a sequence model to a very expensive per-transaction classifier. F1 = 0.47 was what the architecture could learn *without any sequence information*.

---

## 3. The fix

`fix_lstm_alignment.py` — produced once, deterministic, verifiable.

Instead of positional re-attachment, merge on a **composite key present in both files**:

```python
merged = eng.merge(
    orig[["unix_time","amt","city_pop","cc_num","trans_date_trans_time","is_fraud"]],
    on=["unix_time","amt","city_pop"],
    how="left",
    validate="one_to_one",  # raises if the key has duplicates
)
```

Why `(unix_time, amt, city_pop)` rather than `unix_time` alone:
- Several cardholders can transact within the same second.
- `(unix_time, amt)` also has occasional collisions.
- `(unix_time, amt, city_pop)` is empirically one-to-one across both files — `validate="one_to_one"` passes.

### Verification built into the fix script

1. **Row count** preserved (`len(merged) == len(eng)`).
2. **is_fraud consistency** — `is_fraud` from engineered side equals `is_fraud` from original side after merge: `0 mismatches`.
3. **No unmapped rows** — no null `cc_num` after left-merge: `0 nulls`.
4. **20 random-row round-trip** — sample 20 rows, look each up in the original CSV by composite key, confirm `cc_num` and `trans_date_trans_time` match: **20/20 OK on both train and test**.

The fixed files are `fraudTrain_engineered_with_ids.csv` and `fraudTest_engineered_with_ids.csv`. All LSTM experiments in `run_gap_experiments.py` read from these.

---

## 4. Impact

### Direct effect on LSTM+RF hybrid

| Pipeline | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Original (broken alignment) | 0.4747 | — | — | — |
| Fixed (composite-key merge) | **0.7892** | 0.6770 | 0.9459 | 0.9981 |
| **Δ** | **+0.3145** | — | — | — |

### No effect on XGBoost / AE / BDS pipelines

These models do not use sequence ordering — each transaction is scored independently from its own features. The positional re-attach of `cc_num` was only used downstream for sequence construction in the LSTM pipeline. Verified: XGBoost SMOTE+tuned (F1=0.87) and AE+XGBoost (F1=0.87) both stand unchanged on re-audit.

### Effect on the dissertation's main conclusion

The hierarchy of models is **unchanged**:

1. AE + XGBoost SMOTE+tuned → **F1 = 0.87** (best)
2. AE + BDS(GA) + XGBoost → F1 = 0.868
3. LSTM + RF (fixed) → F1 = 0.79  ← *was 0.47 before fix, now more honest and closer to the pack*

The relative ranking, and the dissertation's claim that **AE + XGBoost is the strongest hybrid for this dataset**, survive the audit. The LSTM no longer looks implausibly weak — its post-fix F1 is closer to the gap literature would predict for a sequence model on this dataset size (1.29M transactions, 0.58% fraud).

---

## 5. How this fits the dissertation narrative

### Chapter 8 (Conclusions / Lessons Learned) — suggested paragraph

> During late-stage supplementary experimentation, a row-alignment defect was discovered in the LSTM+RF hybrid's data pipeline: cardholder IDs were re-attached to engineered features by position rather than by shared key, after feature engineering had silently reordered the rows. Zero rows positionally aligned between the original and engineered files. The bug inflated the apparent weakness of the sequence model; correcting it raised the LSTM+RF F1 from 0.4747 to 0.7892. The fix — a composite-key one-to-one merge, with automated verification — is documented alongside the original broken notebook as `FYP_Hybrid_Model_BROKEN.ipynb`. The relative ordering of the three hybrids is unchanged and the dissertation's core finding (AE + XGBoost with SMOTE + tuning is the strongest pipeline tested, F1 = 0.87) stands independently. I report this episode explicitly rather than silently overwriting results, because the methodological lesson — that any `.values` positional re-attach is a latent correctness bug waiting for an upstream sort to expose it — is itself a finding of the project.

### Chapter 4 (Methodology) — one-line methodological note

Add after the data pipeline description:

> All joins between the engineered and raw transaction files use composite-key merges with `validate="one_to_one"` rather than positional index alignment, after an earlier version of the LSTM pipeline was found to silently misalign records when upstream sorting changed row order.

### For the viva

Prepare to be asked about this. The right answer is:

1. *What went wrong:* positional re-attach of `cc_num` after the engineered CSV had been sorted by `(cc_num, trans_datetime)`.
2. *How you found it:* closing Gap 1 required rebuilding sequences; a sanity check on `(cc_num, unix_time)` monotonicity failed.
3. *How you fixed it:* composite-key merge with one-to-one validation plus random-row verification.
4. *Why you kept the broken notebook:* `FYP_Hybrid_Model_BROKEN.ipynb` is preserved for audit / defensibility; the fix is applied forward of it, not retroactively hidden.
5. *What it means for your conclusions:* relative model ranking unchanged; best result (F1 = 0.87) was never affected because it came from a non-sequence pipeline.

---

## 6. Artefacts preserved for audit

| File | Role |
|---|---|
| `FYP_Hybrid_Model_BROKEN.ipynb` | Untouched copy of the original notebook with the bug |
| `verify_alignment_bug.py` | Independent check that proves the bug (0% positional match) |
| `fix_lstm_alignment.py` | One-shot repair script, composite-key merge, 20-row round-trip verification |
| `fraudTrain_engineered_with_ids.csv` | Fixed train data (used by all LSTM experiments) |
| `fraudTest_engineered_with_ids.csv` | Fixed test data |
| `run_gap_experiments.py` | Reproduces Exp 1–5 on the fixed data |
| `verified_metrics.json["gap_experiments"]` | JSON of post-fix metrics |
| `FEEDBACK_GAPS_RUN_LOG.txt` | Wall-clock log of every run |
| `FEEDBACK_GAPS_CLOSED.md` | Full supplementary-experiment report |
| `AUDIT_WIN_NARRATIVE.md` | This document |

Every step is reproducible end-to-end from the raw CSVs with seed = 42.

---

## 7. Bottom line

A latent data-pipeline bug was hiding under a plausible-looking F1 number. Finding it, fixing it, and reporting it transparently is the kind of engineering self-audit that a good viva examiner rewards — and it strengthens, rather than undermines, the dissertation's conclusions.

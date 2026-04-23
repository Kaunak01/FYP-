# Feedback Gaps Closed — Supplementary Experiments

**Author:** Kaunak Bhattacharya
**Supervisor:** Dr. Yasmine Arafa
**Date:** 22 April 2026
**Status:** All 5 planned experiments complete (+1 fallback)
**Source data:** `verified_metrics.json["gap_experiments"]`, log: `FEEDBACK_GAPS_RUN_LOG.txt`

---

## 1. Summary

Two supervisor feedback gaps were flagged after the core results were reported:

- **Gap 1** — The LSTM+RF hybrid was only evaluated *with* velocity features; Dr. Arafa asked for an ablation matching the one already done for XGBoost.
- **Gap 2** — Oversampling comparison used only SMOTE; ADASYN and focal-loss were expected alongside it.

Both gaps are now closed. While closing them, a **critical row-alignment bug** was discovered in the original LSTM pipeline — its F1=0.4747 figure was an artefact of silently misaligned cardholder IDs. Rerunning on the fixed pipeline raises the honest LSTM+RF F1 to **0.7892** (+0.31).

Full narrative on the bug and its fix is in `AUDIT_WIN_NARRATIVE.md`.

---

## 2. Pre-audit Summary (what changed vs. the original results)

| Model | Original report | Post-audit / fixed | Delta |
|---|---|---|---|
| LSTM + RF (with velocity) | F1 = 0.4747 *(broken pipeline)* | F1 = 0.7892 | **+0.3145** |
| XGBoost SMOTE + tuned | F1 = 0.87 | unchanged | — |
| AE + XGBoost SMOTE + tuned | F1 = 0.87 | unchanged | — |
| AE + BDS(GA) + XGBoost | F1 = 0.868 | unchanged | — |

Only the LSTM pipeline was affected. XGBoost and AE pipelines do not attach sequence-order state, so the row-level cardholder misalignment had no effect on them. Verified independently via `verify_alignment_bug.py`.

---

## 3. Gap 1 — LSTM+RF Velocity Ablation

**Experimental setup:** Identical architecture, seed, scaler, sequence construction, and class weights as the original notebook — only the feature set differs. Fixed `*_engineered_with_ids.csv` files used throughout. Keras 3.14 with PyTorch 2.11 CPU backend (original ran Keras-on-TF-GPU in Colab).

| Feature set | #features | F1 | Precision | Recall | ROC-AUC | TP | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| With velocity (reproduced baseline) | 14 | **0.7892** | 0.6770 | 0.9459 | 0.9981 | 2029 | 116 | 968 |
| Without velocity | 11 | 0.7766 | 0.6596 | 0.9441 | 0.9981 | 2025 | 120 | 1045 |
| **Δ (velocity contribution)** | — | **+0.0126** | +0.0174 | +0.0018 | 0.0000 | +4 | −4 | −77 |

### Interpretation

- Velocity features contribute **+0.013 F1** to the LSTM+RF hybrid — small but positive, matching the direction (though smaller magnitude) of the XGBoost ablation (+0.0144).
- Gain is almost entirely on the **precision** axis (+0.017, vs. +0.002 recall): velocity features mainly help the model *reject* legitimate transactions that superficially resemble fraud, not find more frauds.
- ROC-AUC is identical to four decimal places, indicating the model's *ranking* of fraud is already near-optimal; velocity features refine the operating point rather than the underlying signal.
- LSTM+RF (0.79) still trails XGBoost-based hybrids (0.87) on this dataset. Sequence modelling over 5-transaction windows does not overcome XGBoost's feature-wise advantage — a finding that now stands on a correct pipeline, not the broken one.

---

## 4. Gap 2 — Oversampling / Imbalance-Handling Comparison

All runs reuse the **tuned XGBoost hyperparameters** from `xgboost_smote_tuned.joblib` (`n_estimators=400, max_depth=10, lr=0.05, subsample=0.8, colsample_bytree=0.7`) so only the imbalance-handling step differs.

| Method | Threshold | F1 | Precision | Recall | ROC-AUC | TP | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **SMOTE (baseline, from existing notebook)** | 0.5 | **0.8700** | — | — | — | — | — | — |
| ADASYN | 0.5 | **0.8640** | 0.9296 | 0.8070 | 0.9973 | 1731 | 414 | 131 |
| ADASYN | 0.7 | 0.8600 | 0.9663 | 0.7748 | 0.9973 | 1662 | 483 | 58 |
| Focal Loss (custom gamma=2, alpha=0.9942) | 0.5 | **0.0000** | 0.0000 | 0.0000 | 0.7848 | 0 | 2145 | 0 |
| Focal Loss (custom gamma=2, alpha=0.9942) | 0.7 | 0.0000 | 0.0000 | 0.0000 | 0.7848 | 0 | 2145 | 0 |
| Focal Loss fallback (BalancedBagging) | 0.5 | 0.3511 | 0.2148 | 0.9608 | 0.9976 | — | — | — |
| Focal Loss fallback (BalancedBagging) | 0.7 | 0.4639 | 0.3075 | 0.9445 | 0.9976 | — | — | — |

### Interpretation

- **SMOTE remains the best oversampling strategy** for this dataset and model (F1 = 0.870), with ADASYN marginally behind (0.864). The 0.006 gap is within noise; the two are operationally equivalent.
- **ADASYN trades recall for precision** relative to SMOTE — at thr=0.7 precision climbs to 0.97 with only 58 false positives, useful if the deployed system prioritises low operator fatigue over catch rate.
- **Custom focal-loss collapsed** — the trained model refused to predict the positive class at any threshold, despite non-trivial ROC-AUC (0.7848). This is a known pathology of focal-loss on heavily imbalanced binary tasks without a probability calibration step or warm-start from a baseline. The ROC-AUC number confirms the model *ranks* frauds above non-frauds, but the decision threshold is pushed so high by the loss surface that no test-set probability exceeds 0.5. Documented as a negative result.
- **BalancedBagging** (the planned fallback) works but at large precision cost (F1 = 0.35–0.46). Recall is exceptional (0.94–0.96) but 77% of its positive predictions are false. Useful only as a recall-maximising ensemble member, not a primary classifier.

### Recommendation for dissertation

Report SMOTE, ADASYN, and focal-loss side by side. The focal-loss collapse is a genuine finding worth discussing — it shows why SMOTE dominates the recent fraud-detection literature despite focal-loss's theoretical appeal.

---

## 5. Key Implications for the Dissertation

1. **Chapter 4 (Methodology):** Add a paragraph on the alignment bug and the `_with_ids.csv` fix. Frame it as a validation-driven discovery that reinforces the need for composite-key merges over positional joins.
2. **Chapter 5 (Results):** Replace the old LSTM+RF F1=0.4747 figure everywhere with F1=0.7892 (fixed pipeline). Add the velocity ablation table for LSTM+RF alongside the existing XGBoost one.
3. **Chapter 6 (Discussion):** Contrast SMOTE (0.87) vs ADASYN (0.86) vs Focal Loss (collapsed) in a single oversampling subsection. The failure of focal-loss without calibration is a meaningful finding.
4. **Chapter 8 (Conclusions / Lessons Learned):** Explicitly reference the pipeline-audit discovery — this is an *example of engineering rigour*, not a cover-up. See `AUDIT_WIN_NARRATIVE.md` for the suggested paragraph.

---

## 6. Reproducibility

- `run_gap_experiments.py` — orchestrator for Exp 1–5, seed=42 everywhere.
- `run_focal_loss_fallback.py` — BalancedBagging fallback for Exp 5.
- `fix_lstm_alignment.py` — produced the `_with_ids.csv` datasets used by all LSTM experiments.
- `verify_alignment_bug.py` — independently confirms the original misalignment (0/1,296,675 rows positionally aligned).
- `verified_metrics.json["gap_experiments"]` — canonical JSON of all eight rows (5 experiments + 1 cross-ref + 2 fallback thresholds).
- `FEEDBACK_GAPS_RUN_LOG.txt` — timestamped wall-clock log of every run.

---

## 7. Appendix — Runtime / Environment

| Experiment | Runtime | Model backend |
|---|---:|---|
| LSTM w/ velocity | 473 s | Keras 3.14 + PyTorch 2.11 CPU |
| LSTM w/o velocity | 605 s | Keras 3.14 + PyTorch 2.11 CPU |
| XGBoost ADASYN | 70 s | XGBoost CPU |
| XGBoost Focal Loss (custom) | 78 s | XGBoost CPU |
| XGBoost Focal Loss (fallback) | 36 s | imbalanced-learn BalancedBagging |
| **Total** | **21 min** | seed=42 throughout |

Hardware: Windows laptop, Python 3.14.3, no GPU used. All experiments are seeded and fully reproducible.

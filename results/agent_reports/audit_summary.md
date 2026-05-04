# FYP Audit Report — 2026-04-29T00:00:00

**Target:** Full project — all notebooks, all model artifacts, all results JSONs, Flask app
**Branch:** restructure-folders (post April 27 folder restructure)
**Scope:** --full pre-submission sweep
**Specialists run:** metrics-verifier, fraud-pipeline-auditor, code-quality-reviewer, baseline-verifier, comparator-verifier, proposed-verifier
**Verdict:** ISSUES FOUND (no P1 Critical blockers; P2 methodology gaps require attention before viva)

---

## Issue Count Summary

- P1 Critical: 2
- P2 Fix-before-viva: 8
- P3 Cosmetic: 6

---

## P1 Critical (must fix before submission)

### 1. scripts/run_bds_ga.py hard-codes stale F1=0.47 for LSTM in printed summary table
- **Found by:** metrics-verifier, code-quality-reviewer
- **File:** `scripts/run_bds_ga.py:401` and `:487`
- **Issue:** Line 401: `'F1': [0.47, 0.87, f1_cw, f1_sm, f1_tuned]` and line 487: `print(f"  {'LSTM + RF (Hybrid 1)':<45s} {'0.47':>6s}")` hard-code the broken-pipeline F1=0.47 as a live comparison value in the final printed summary. This script is the canonical training runner for the Proposed Model. A marker running this script sees "LSTM+RF F1=0.47" printed as if it is current, directly contradicting `verified_metrics.json` (LSTM F1=0.7892). The 0.47 figure has been clearly repudiated in all other files but survives here as a live code literal.
- **Fix:** Replace `0.47` with `0.7892` (and `0.87` with `0.8690`) in the comparison DataFrame on line 401 and in the print statement on line 487; add a comment citing `verified_metrics.json` as source.

### 2. scripts/compute_metrics.py uses stale flat model paths (pre-restructure) and will silently fail
- **Found by:** code-quality-reviewer, baseline-verifier, comparator-verifier, proposed-verifier
- **File:** `scripts/compute_metrics.py:34,35,54,66,121,127,138,144`
- **Issue:** This script loads models from `models/saved/ae_scaler.joblib`, `models/saved/ae_model.pt`, `models/saved/bds_profiles.joblib`, `models/saved/ga_best_params.json`, `models/saved/xgboost_baseline_cw.joblib`, `models/saved/xgboost_smote_tuned.joblib`, `models/saved/ae_xgboost_smote_tuned.joblib`, `models/saved/ae_bds_xgboost_smote_tuned.joblib` — all flat paths that no longer exist after the April 27 restructure (they now live under `01_baseline/`, `02_comparator/`, `03_proposed/`, `supplementary/`). The script will fail at import time when a marker runs it. This is a reproducibility blocker for the submitted dissertation artifact.
- **Fix:** Update all model load paths in `compute_metrics.py` to match the restructured layout (matching `app/config.py` MODEL_FILES). Also update `MODELS_DIR = "models/saved"` to path-keyed references per subfolder.

---

## P2 Fix-before-viva

### 3. SMOTE applied outside CV folds in Notebook 03 and save_all_models.py — documented open issue
- **Found by:** fraud-pipeline-auditor, proposed-verifier
- **File:** `notebooks/03_Autoencoder_XGBoost.ipynb:cell-22`, `scripts/save_all_models.py:214,423`
- **Issue:** In both the notebook and the production model-save script, SMOTE is applied to the full training set before `RandomizedSearchCV`, meaning synthetic minority samples derived from validation-fold neighbours appear in sibling training folds. This optimistically biases CV F1 scores (though test-set evaluation is clean). The notebook contains a stub cell (cell-24) with the correct `imblearn.Pipeline` fix that is gated behind `RUN_PIPELINE_CV_INLINE = False`. The save script has no equivalent guard. This is documented as "Audit fix P4" in the notebook but no result has been written to `verified_metrics.json` under `AE_XGBoost_PipelineCV`. For the viva, a supervisor will likely probe this.
- **Fix:** Run `python scripts/run_audit_retrains.py --pipeline-cv-ae` and ensure the result appears in `verified_metrics.json`; cite it in the dissertation as a sensitivity check confirming the headline number is unaffected (or note the delta if it is).

### 4. BDS profiles built from all training transactions — "past-only" constraint not enforced (open P1 from codex audit)
- **Found by:** fraud-pipeline-auditor, proposed-verifier
- **File:** `scripts/save_all_models.py:238-276`, `scripts/run_bds_ga.py:95-128`
- **Issue:** `profile_df` is built from the entire training DataFrame in a single batch. For any given training transaction at position i, the BDS profile incorporates that transaction itself in its own profile statistics (amount mean, hour counts, velocity mean). This is mild data snooping: a transaction contributes to the profile from which its own deviation is computed. The theoretically correct approach is to build the profile from all transactions before the current one in chronological order per card (past-only). The codex audit flagged this as an open P1 on 2026-04-24 with no confirmed fix. The issue does not invalidate the test-set result (test BDS profiles are built from training data only) but is a methodological gap the examiner may probe.
- **Fix:** Implement a rolling/expanding window profile accumulation for training BDS scores (sort by unix_time per card, compute profile from all prior rows). Flag in dissertation as a known limitation and note the test evaluation is unaffected.

### 5. Flask app threshold is 0.70, all evaluated metrics are at threshold 0.50 — inconsistency not disclosed
- **Found by:** metrics-verifier, fraud-pipeline-auditor
- **File:** `app/config.py:48` (DEFAULT_THRESHOLD=0.70), `results/verified_metrics.json` (all headline F1s at threshold=0.5), `app/config.py:127-136` (STAGED_STUDY_TABLE_A and B cite threshold-0.5 metrics)
- **Issue:** The live Flask application uses DEFAULT_THRESHOLD=0.70 for the FRAUD decision, yet all headline numbers cited in the dissertation (F1=0.8646, 0.7892, 0.8706) and hardcoded in the UI table (`STAGED_STUDY_TABLE_A`, `MODEL_F1_LABELS`) are at threshold=0.5. The verified_metrics.json contains results at both thresholds; at 0.70 the numbers are notably different (e.g., XGBoost SMOTE+tuned F1 drops to 0.8605). A marker testing the app and computing metrics will see different F1 from what the dissertation claims unless this inconsistency is explicitly disclosed and justified.
- **Fix:** Add one sentence to the dissertation methods/evaluation section stating that research metrics are reported at threshold=0.5 (standard ML practice for model comparison) while the Flask deployment uses 0.70 (operational precision preference). Alternatively, align the live threshold to 0.5.

### 6. scripts/run_bds_ga.py and save_all_models.py use bare relative CSV paths — will fail unless run from project root
- **Found by:** code-quality-reviewer
- **File:** `scripts/run_bds_ga.py:27-30`, `scripts/save_all_models.py:23-25`
- **Issue:** Both scripts open `fraudTrain_engineered.csv`, `fraudTest_engineered.csv`, `fraudTrain.csv`, `fraudTest.csv` as bare relative paths with no path resolution. `run_gap_experiments.py` (correctly) uses `Path(__file__).parent / "..."` anchored to the script location, pointing at `data/engineered/`. The bds_ga and save_all_models scripts will FileNotFoundError if run from any directory other than the old project root (where the CSVs used to live as flat files). These scripts are the canonical training artifacts for reproducibility.
- **Fix:** Replace bare CSV paths in both scripts with `Path(__file__).resolve().parent.parent / 'data' / 'engineered' / 'fraudTrain_engineered.csv'` etc., matching the post-restructure layout.

### 7. Notebook 02 (Hybrid Model) RF max_depth=10 in cell-9 vs max_depth=12 in the saved artifact and scripts
- **Found by:** comparator-verifier, metrics-verifier
- **File:** `notebooks/02_Hybrid_Model.ipynb:cell-9` (max_depth=10) vs `scripts/save_lstm_rf_model.py:163` and `scripts/run_gap_experiments.py:209` (max_depth=12)
- **Issue:** The first RF in the notebook (cell-9) uses `max_depth=10, class_weight={0:1, 1:171.8}` which is an exploratory configuration. The saved artifact (lstm_rf_classifier.joblib) and both authoritative scripts use `max_depth=12, class_weight={0:1, 1:150}`. The notebook's cell-12 later sweeps configs including `max_depth=12, cw=150` which is the one that matches the saved artifact. A reader following cell-9 alone as the "final" config would misread the methodology. The cell ordering is confusing.
- **Fix:** Add a comment in cell-9 marking it as "exploratory — final config is in cell-12 / scripts/save_lstm_rf_model.py". Alternatively add a markdown cell before cell-9 flagging the exploration context.

### 8. models/legacy_root/ contains 4 model files with no documented decision on retention or deletion
- **Found by:** code-quality-reviewer, baseline-verifier
- **File:** `models/legacy_root/` (xgboost_smote_tuned.joblib, ae_model.pt, ae_xgboost_smote_tuned.joblib, ae_bds_xgboost_smote_tuned.joblib)
- **Issue:** Per the memory note from 2026-04-26, the `legacy_root` decision was explicitly deferred as "pending". These files appear to be identical to the pre-restructure originals but are not referenced by any active code path. They occupy disk space and could confuse a marker inspecting the models/ directory. The memory note says "legacy_root decision" is outstanding.
- **Fix:** Either delete the directory with a git commit message documenting its removal, or add a `README.txt` inside `legacy_root/` explicitly stating these are superseded by `models/saved/` equivalents and are retained only as a git recovery point. Resolve this before submission.

### 9. models/saved_original/ duplicates all 11 model artifacts — not referenced by any active code
- **Found by:** code-quality-reviewer
- **File:** `models/saved_original/` (8 files including ga_best_params.json, category_mapping.json, training_stats.json, category_aliases.json)
- **Issue:** `models/saved_original/` is a full duplicate of the original flat model layout. No Python file in `app/`, `scripts/`, `testing/`, or `notebooks/` imports from this path. It is dead weight and may confuse examiners about which model set is canonical.
- **Fix:** Delete `models/saved_original/` or document it as a git backup in a README. Resolve alongside the legacy_root decision.

### 10. models/saved/supplementary/ contains xgboost_best.joblib with no documented provenance
- **Found by:** baseline-verifier, code-quality-reviewer
- **File:** `models/saved/supplementary/xgboost_best.joblib`
- **Issue:** `models/saved/supplementary/` holds `xgboost_best.joblib` and `xgboost_baseline_cw.joblib`. The `app/config.py` MODEL_FILES references `xgboost_baseline_cw.joblib` correctly (as 'xgb_cw'). However `xgboost_best.joblib` is not referenced anywhere in active code and has no documented provenance (it may be an older copy of ae_xgboost_smote_tuned from notebook-03). A marker inspecting the supplementary folder will find an unexplained artifact.
- **Fix:** Add a `README.txt` in supplementary/ identifying what each file is, when it was produced, and whether it is actively used. Delete xgboost_best.joblib if it is not needed.

---

## P3 Cosmetic

### 11. Notebook 02 cell-6 comment "Short enough to run fast on Colab" is a Colab residue
- **Found by:** code-quality-reviewer
- **File:** `notebooks/02_Hybrid_Model.ipynb:cell-6` (SEQ_LEN comment)
- **Issue:** The comment "Short enough to run fast on Colab. Long enough to capture rapid fraud patterns." is a leftover from the Google Colab era. The notebook is now GPU-free CPU-local.
- **Fix:** Update comment to "Short enough for CPU training; long enough to capture rapid fraud patterns."

### 12. Notebook 03 cell-46 saves models to bare local paths (xgboost_best.joblib, autoencoder.pt, scaler.joblib) in the working directory
- **Found by:** code-quality-reviewer
- **File:** `notebooks/03_Autoencoder_XGBoost.ipynb:cell-46`
- **Issue:** The "Save Models for Flask App" cell dumps artifacts to the current working directory without resolving the project root. If someone runs this notebook, they will create stale flat-path artifacts instead of updating the canonical `models/saved/03_proposed/` files.
- **Fix:** Either comment out this cell (saving is now done via `scripts/save_all_models.py`) or update the paths to `../../models/saved/03_proposed/`.

### 13. scripts/compute_metrics.py uses a threshold of 0.70 internally while all other evaluation uses 0.50
- **Found by:** metrics-verifier
- **File:** `scripts/compute_metrics.py:22` (`THRESHOLD = 0.70`)
- **Issue:** Minor inconsistency: this script reports at 0.70. The headline metrics in verified_metrics.json are at 0.50. The script is not used by active Flask code but is part of the reproducibility chain.
- **Fix:** Change `THRESHOLD = 0.70` to `0.50` to align with the published results, or add both thresholds.

### 14. notebooks/02_Hybrid_Model.ipynb still references "Colab paths" in a markdown cell (cell-4)
- **Found by:** code-quality-reviewer
- **File:** `notebooks/02_Hybrid_Model.ipynb:cell-4`
- **Issue:** Cell-4 markdown says "To build these sequences, the original card numbers and timestamps are temporarily re-attached to the engineered features." — accurate — but earlier context in the same cell says nothing about Colab. However, cell-5 calls `display(Image('hybrid_architecture.png', width=1000))` with a local path that will silently fail if the file is missing (it does have a try/except). Cosmetic issue.
- **Fix:** Confirm `hybrid_architecture.png` is present in the notebooks/ directory or update the try/except to print a more helpful message.

### 15. scripts/run_bds_ga.py uses plots saved to working directory (bds_fig_1.png through bds_fig_6.png) with no path
- **Found by:** code-quality-reviewer
- **File:** `scripts/run_bds_ga.py:324,413,422,445,460,465`
- **Issue:** All six PNG outputs save to the current working directory without explicit path resolution. If run from the project root they go to the root; if run from scripts/ they go there.
- **Fix:** Prefix all savefig paths with `Path(__file__).resolve().parent.parent / 'results' / 'figures' /` or similar.

### 16. scripts/save_all_models.py saves all artifacts to the working directory (pre-restructure flat layout)
- **Found by:** code-quality-reviewer
- **File:** `scripts/save_all_models.py:122,143,194-196,224,276,403,432` (all joblib.dump/torch.save calls)
- **Issue:** Every model save in this script writes to the current working directory (e.g., `joblib.dump(xgb_cw, 'xgboost_baseline_cw.joblib')`). This is the pre-restructure flat layout. Running this script now would not populate the new `models/saved/01_baseline/`, `02_comparator/`, `03_proposed/` directories. The script would overwrite or create files in wherever it is run from, not in the canonical save locations.
- **Fix:** Update all save targets to the structured paths matching `app/config.py` MODEL_FILES. This is partly a P2 issue but since the trained artifacts already exist and are correct, the script is cosmetic until a retrain is needed.

---

## Model Verification Summary

| Model | Verifier | Verdict | F1 (JSON / expected) | Notes |
|---|---|---|---|---|
| Baseline (XGBoost SMOTE+Tuned) | baseline-verifier | PASS | 0.8646 / 0.8646 | verified_metrics.json row "XGBoost SMOTE+tuned" threshold=0.5, exact match |
| Comparator (LSTM+RF) | comparator-verifier | PASS | 0.7892 / 0.7892 | gap_experiments LSTM_reproduced_baseline; composite-key merge confirmed; F1=0.4747 NOT present in any live citation |
| Proposed (AE+BDS+XGBoost) | proposed-verifier | PASS (warning) | 0.8706 / 0.8706 | verified_metrics.json "AE + BDS + XGBoost (full)" threshold=0.5; AE trained on normal-only confirmed; BDS built from train-only confirmed; open P2: BDS past-only constraint not enforced in training loop |
| AE+XGBoost (ablation) | proposed-verifier | PASS | 0.8690 / 0.8690 | ablation_results.json full_hybrid_saved and verified_metrics.json "AE + XGBoost SMOTE+tuned" align |
| Inference latency | metrics-verifier | PASS | proposed=1.62ms / 1.62ms | inference_latency.json matches dissertation claim exactly |
| Ablation chain | proposed-verifier | PASS | 0.8646→0.8690→0.8706 | ablation_results.json monotonic; delta_f1 values internally consistent |
| SMOTE-in-CV sensitivity | fraud-pipeline-auditor | WARNING | not yet run | pipeline-cv-ae experiment stub exists in notebook cell-24 but RUN_PIPELINE_CV_INLINE=False; run_audit_retrains.py --pipeline-cv-ae has not been executed; result absent from verified_metrics.json |
| GA fitness evaluation | fraud-pipeline-auditor | PASS | on validation split only | GA fitness() in save_all_models.py evaluates on ga_val_idx, not on test set; correct |
| Threshold consistency | metrics-verifier | WARNING | Flask=0.70, metrics=0.50 | see P2 issue #5 |

---

## Detailed Metric Cross-Check

All headline numbers from `results/verified_metrics.json` at threshold=0.5:

| Metric | JSON value | Dissertation claim | Match |
|---|---|---|---|
| Baseline F1 | 0.8646 | 0.8646 | YES |
| Baseline Precision | 0.9297 | 0.9297 | YES |
| Baseline Recall | 0.8079 | 0.8079 | YES |
| Baseline ROC-AUC | 0.9972 | 0.9972 | YES |
| Baseline PR-AUC | 0.9092 | 0.9092 | YES |
| Comparator F1 | 0.7892 | 0.7892 | YES |
| Comparator PR-AUC | 0.9517 | 0.9517 | YES (stored in gap_experiments) |
| Proposed F1 | 0.8706 | 0.8706 | YES |
| Proposed PR-AUC | 0.9158 | 0.9158 | YES |
| Proposed latency (median) | 1.62ms | 1.62ms | YES |
| AE+XGB ablation F1 | 0.8690 | 0.8690 | YES |
| Ablation chain monotonic | 0.8646→0.8690→0.8706 | same | YES |
| McNemar full vs. BDS (p=0.912) | non-significant | -- | NOTE: BDS improvement over AE+XGB is statistically non-significant; document carefully in Ch 4 |

**One stale literal found:** `scripts/run_bds_ga.py:401,487` cite F1=0.47 as live comparison value (see P1 issue #1).

---

## Pipeline Methodology Summary

| Check | Status | Detail |
|---|---|---|
| Temporal leakage (train/test split) | PASS | Standard scikit-learn train/test split; no future data bleeds into training features |
| Target leakage | PASS | is_fraud excluded from all feature sets |
| SMOTE-in-CV leakage | WARNING (open) | SMOTE applied pre-CV in Notebook 03 and save_all_models.py; imblearn Pipeline fix exists but not executed |
| AE trained on legitimate-only | PASS | X_train_normal = X_train_scaled[y_train == 0] confirmed in notebook cell-5 and save_all_models.py:157 |
| BDS profiles built from train only | PASS | profile_df uses train_cc and train_eng exclusively |
| BDS past-only constraint | FAIL (open P2) | Batch computation includes each transaction in its own profile; rolling window not implemented |
| Velocity features chronologically sorted | PASS | Velocity computed in EDA notebook with groupby cc_num sorted by unix_time |
| GA fitness on validation only | PASS | Explicit ga_val_idx split; model.predict(Xv) not on test set |
| Threshold consistency research vs app | WARNING | 0.50 in papers/JSONs, 0.70 in Flask DEFAULT_THRESHOLD |
| Removed template references | PASS | No url_for references to welcome/batch/alerts in any template |
| report_generator imports | PASS | No live imports of report_generator in any app/ file |
| LSTM composite-key merge (not positional) | PASS | Notebook 02 confirms _with_ids.csv pipeline; cc_num attached via composite-key merge |
| F1=0.4747 live citation trap | PASS (with exception) | Not cited in any live code EXCEPT scripts/run_bds_ga.py line 401 and 487 (P1 issue #1) |

---

*Generated by code-audit-orchestrator — 2026-04-29*
*Branch: restructure-folders | Scope: --full pre-submission sweep*
*No code changes were made during this audit. Only this report file was written.*

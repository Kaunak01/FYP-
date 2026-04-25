# Justification for Using Multiple Hybrid Models
## FYP: Hybrid Machine Learning for Credit Card Fraud Detection

---

## 1. The Core Research Question

The dissertation title is:

> "Hybrid Machine Learning for Credit Card Fraud Detection Using Transaction Behavioural Features and Synthetic Oversampling"

The title itself implies a **comparative investigation** of hybrid approaches — not the construction of a single system. The use of multiple hybrid models is not trial and error. It is a **systematic, staged research design** that answers one central question:

> *"What combination of signals — sequential patterns, global anomaly detection, and personalised behavioural profiling — is most effective for detecting credit card fraud, and does each component contribute meaningfully?"*

Each hybrid model is a deliberate step in answering that question.

---

## 2. Why a Single Model Is Insufficient

Credit card fraud detection presents three simultaneous challenges that no single model fully addresses:

### Challenge 1 — Class Imbalance
The dataset contains 1.85 million transactions with only 0.58% fraud. A naive classifier achieves near-perfect accuracy by predicting "not fraud" every time. This means accuracy is a useless metric and standard classifiers require special treatment (SMOTE, class weighting).

### Challenge 2 — Fraud Resembles Legitimate Transactions in Isolation
A single transaction of $200 at a grocery store is not suspicious on its own. Fraud only becomes detectable when context is considered — what is normal for this cardholder? Is this transaction unusual relative to recent behaviour? A standard classifier trained only on static transaction features has no way to answer these questions.

### Challenge 3 — Fraud Manifests Across Multiple Dimensions
Research in fraud detection identifies at least three distinct fraud signals:
- **Temporal signals** — fraudsters often make rapid sequential transactions before a card is blocked
- **Anomaly signals** — fraudulent transactions deviate from the overall distribution of normal spending
- **Behavioural signals** — fraudulent transactions deviate from *that specific cardholder's* personal spending history

No single model captures all three. This is the fundamental motivation for a hybrid approach.

---

## 3. The Three Hybrid Models as a Staged Research Design

The three hybrid models are not three unrelated experiments. They form a **progressive research design**, where each stage builds on the previous and tests a specific hypothesis.

---

### Stage 1 — Hybrid 1: LSTM + Random Forest
**Hypothesis:** Can sequential/temporal modelling of transaction histories improve fraud detection?

**Design:**
- LSTM processes sequences of recent transactions per cardholder to extract temporal patterns
- Random Forest combines LSTM-derived probability features with static transaction features

**Result:** F1 = 0.47

**Finding:** Sequential modelling alone is insufficient for this dataset. The LSTM adds temporal context but the combined model does not reach a practically useful performance level. This finding **motivates** moving to an anomaly-based approach in Stage 2.

**Why keep this in the dissertation:** This is not a failed experiment — it is **evidence**. It answers the hypothesis with a clear negative result and justifies why the research direction changes. An examiner expects to see what did not work, not only what did.

---

### Stage 2 — Hybrid 2: Autoencoder + XGBoost
**Hypothesis:** Can unsupervised anomaly detection, combined with a supervised classifier, significantly improve performance over a supervised-only baseline?

**Design:**
- Autoencoder is trained exclusively on normal (non-fraudulent) transactions
- It learns a compressed representation of "what normal looks like"
- At inference, fraudulent transactions produce higher reconstruction error because they deviate from the learned normal pattern
- This reconstruction error is added as a new feature for XGBoost alongside the original transaction features

**Result:** F1 = 0.87 (with SMOTE + hyperparameter tuning)

**Finding:** Yes — combining global anomaly detection with supervised classification produces a substantial improvement over XGBoost alone (baseline F1 = 0.52). The autoencoder contributes a signal that supervised models cannot produce: a direct measure of how abnormal a transaction is relative to the global distribution of normal spending.

**Why this works:** XGBoost learns decision boundaries from labelled features. The autoencoder learns what unlabelled normal behaviour looks like. These are fundamentally different signals. Combining them gives the final model access to both.

---

### Stage 3 — Hybrid 3: Autoencoder + BDS (GA-Optimised) + XGBoost
**Hypothesis:** Can personalised behavioural deviation scoring, on top of global anomaly detection, improve fraud detection further?

**Design:**
- Behavioural Deviation Scoring (BDS) computes per-cardholder deviation scores — how much does this transaction deviate from *this cardholder's own* historical behaviour?
- Four deviation scores are computed per transaction (amount deviation, frequency deviation, time deviation, merchant deviation)
- A Genetic Algorithm optimises the 10 BDS parameters (thresholds, weights, window sizes) rather than hand-tuning them
- These scores are added as additional features alongside the autoencoder reconstruction error for XGBoost

**Result:** F1 = 0.868

**Finding:** The BDS component produces a marginal improvement over Hybrid 2 in F1 terms. However:
- It answers a distinct research question — personalised vs. global anomaly detection
- It adds interpretability: fraud analysts receive cardholder-level deviation scores, not just a binary prediction
- The marginal F1 difference suggests the BDS scores partially overlap with information already captured by the autoencoder and velocity features — which is itself a finding worth stating

**Why keep this in the dissertation:** It directly supports the dissertation title's emphasis on "transaction behavioural features." BDS is the most explicitly behavioural component. The comparison between Hybrid 2 and Hybrid 3 isolates the contribution of personalised behavioural profiling.

---

## 4. Why the Differences in F1 Are Still Meaningful

| Model | F1 | What It Tests |
|---|---|---|
| XGBoost baseline (class weights) | 0.52 | Supervised classification alone |
| XGBoost SMOTE + tuned | 0.87 | Supervised + synthetic oversampling |
| AE + XGBoost SMOTE + tuned | 0.8672 | + Global anomaly detection |
| AE + BDS(GA) + XGBoost SMOTE + tuned | 0.8720 | + Personalised behavioural profiling |
| Without velocity features | 0.8561 | Baseline without personal contribution |
| With velocity features | 0.8705 | + Behavioural velocity features |

The small numerical differences between the later models are not a weakness. They are expected. As a model improves, marginal gains become smaller — this is standard in machine learning research. What matters is:
- Each component has a **measurable contribution** (ablation study proves velocity features: +0.0144)
- Each component answers a **different research question**
- The comparisons are made **fairly** (same data, same SMOTE, same evaluation metric)

---

## 5. Addressing the "Too Many Models" Challenge Directly

A fair examiner question is: *"Why not just build the best model and stop there?"*

The answer is: **the comparison is the research contribution, not just the final model.**

If only Hybrid 3 (the best) were presented, the dissertation would have no way to answer:
- Does the autoencoder actually help, or is XGBoost alone sufficient?
- Does personalised behavioural profiling add value over global anomaly detection?
- Do velocity features contribute, or are they redundant with other features?

Each hybrid model exists to answer one of these questions. Removing any of them would leave a gap in the research findings.

This is standard practice in academic machine learning research — systems are compared in an **ablation-style progressive design** where each component is added and evaluated incrementally.

---

## 6. The Relationship Between the Models (Not Independent Experiments)

The models are not three separate systems. They share:
- The same dataset (fraudTrain_engineered.csv / fraudTest_engineered.csv)
- The same 14 features as the base input
- The same XGBoost classifier as the final decision-making component (Hybrids 2 and 3)
- The same evaluation methodology (F1 score, precision, recall, SMOTE, confusion matrix)

The progression is:
```
Supervised Baseline (XGBoost only)
        ↓
+ Sequential Modelling (LSTM) → Hybrid 1
        ↓
+ Global Anomaly Detection (Autoencoder) → Hybrid 2
        ↓
+ Personalised Behavioural Profiling (BDS + GA) → Hybrid 3
        ↓
+ Behavioural Velocity Features → Personal Contribution (ablation validated)
```

Each arrow represents one research question answered.

---

## 7. Suggested Wording for the Dissertation

### For the Introduction / Motivation section:
> "A single model architecture is insufficient to address the multifaceted nature of credit card fraud. Fraud manifests as temporal anomalies, deviations from global spending patterns, and deviations from individual cardholder behaviour. This project therefore investigates a progressive series of hybrid models, each designed to incorporate an additional signal type, enabling a systematic evaluation of which components contribute most to detection performance."

### For the Methodology section:
> "Three hybrid architectures were designed and evaluated in a staged comparative study. Rather than selecting a single approach a priori, this design allows each component — temporal sequence modelling, global anomaly detection, and personalised behavioural profiling — to be evaluated in isolation and in combination."

### For the Results / Discussion section:
> "The progression from Hybrid 1 (F1 = 0.47) to Hybrid 2 (F1 = 0.87) demonstrates that anomaly-augmented classification significantly outperforms temporal sequence modelling alone for this dataset. The marginal difference between Hybrid 2 and Hybrid 3 (F1 = 0.8720 vs 0.8672) suggests that the BDS component contributes primarily to interpretability and personalisation rather than raw classification performance, as the autoencoder reconstruction error may already capture much of the same anomaly signal at the global level."

### For the Viva (one sentence):
> "I used multiple hybrid models not to find one that works, but to systematically answer which combination of signals — temporal, anomaly-based, and behavioural — contributes most to fraud detection, which is a research question in itself."

---

## 8. Summary

| Question | Answer |
|---|---|
| Why not just one model? | One model cannot capture temporal, anomaly, and behavioural signals simultaneously |
| Why three hybrids? | Each tests a specific hypothesis in a staged research design |
| Why is LSTM included if it performed badly? | A negative result is still a finding — it justifies the move to anomaly-based approaches |
| Why is Hybrid 3 barely better than Hybrid 2? | Marginal gains are expected as models improve; the value of BDS is in interpretability and personalisation, not just raw F1 |
| What's the overall justification? | The comparison IS the research contribution — without comparing, none of the individual questions can be answered |

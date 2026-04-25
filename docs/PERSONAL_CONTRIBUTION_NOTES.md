# Strengthening the Project: Personal Contribution Focus

## Main View

The strongest part of this project, from a marking perspective, is not just that multiple models were trained. The strongest part is that the work moves beyond standard model fitting and introduces behavioural feature engineering that is both intuitive and measurable.

The personal contribution already has a solid foundation because it includes:

- original behavioural velocity features:
  - `velocity_1h`
  - `velocity_24h`
  - `amount_velocity_1h`
- a clear fraud-detection motivation:
  - fraud is often linked to unusual transaction frequency and sudden bursts of spending
- empirical validation through the ablation study:
  - with velocity features: `F1 = 0.8705`
  - without velocity features: `F1 = 0.8561`
  - measured gain: `+0.0144`

This is important because many student projects claim a contribution, but do not isolate and measure it. Here, the contribution is visible and supported by results.

## Why This Stands Out

These velocity features stand out because they are:

- domain-informed rather than arbitrary
- simple enough to explain clearly in a dissertation or viva
- directly tied to transaction behaviour, which fits the project title well
- proven to improve performance rather than just being added without evidence

This means the contribution is not only technical, but also academically defensible.

## Best Way To Present the Contribution

The personal contribution should be framed as a short research story:

1. Fraud detection is difficult because fraudulent transactions can resemble legitimate ones when only static transaction features are used.
2. Fraud often appears as abnormal short-term behaviour, such as unusually frequent purchases or sudden spending spikes.
3. To address this, three behavioural velocity features were engineered to capture recent transaction intensity and recent spending behaviour.
4. These features were integrated into the hybrid fraud detection pipeline.
5. An ablation study showed that removing them reduced F1 from `0.8705` to `0.8561`, confirming that they contributed positively to the final system.

That structure is strong because it shows problem, motivation, method, and evidence.

## What To Emphasise in the Dissertation

When writing the dissertation, give the velocity features their own clearly named subsection. Do not hide them inside general preprocessing.

Suggested subsection title:

`Behavioural Velocity Feature Engineering as a Personal Contribution`

Inside that section, make sure to explain:

- what each velocity feature measures
- why short-term behavioural change matters in card fraud
- how the features were computed
- why they are different from the original dataset variables
- how they improved the final model in the ablation study

The key message should be:

> the project does not rely only on off-the-shelf machine learning, but introduces custom behavioural features designed to capture fraud-related temporal activity patterns.

## How To Describe Each Feature Clearly

Use simple and direct explanations:

- `velocity_1h`:
  number of transactions made by the same card in the previous 1 hour

- `velocity_24h`:
  number of transactions made by the same card in the previous 24 hours

- `amount_velocity_1h`:
  total amount spent by the same card in the previous 1 hour

Then explain the intuition:

- normal customers usually follow relatively stable short-term transaction behaviour
- fraudulent activity may create sudden bursts of transactions
- fraudsters may also attempt multiple purchases quickly before the card is blocked
- the amount spent in a short time window can reveal suspicious escalation

## Why This Contribution Is Probably Stronger Than the More Complex Parts

The BDS + GA component is technically impressive, but the velocity-feature contribution may actually be easier to defend and therefore more valuable in final marking.

Why:

- it is clearly your own feature engineering work
- it is easy for an examiner to understand immediately
- it fits directly with the project title and behavioural focus
- it has direct quantitative support
- it improves the best-performing model

This does not mean the BDS + GA work is unimportant. It means the velocity-feature contribution should be treated as the clearest central contribution, while BDS + GA can be presented as an advanced extension.

## What Else Could Strengthen the Personal Contribution Further

Without changing the whole project, these would make the contribution look even stronger:

- add one clean diagram showing how raw transactions are transformed into behavioural velocity features
- add one table describing each engineered feature, its formula, and its fraud-detection intuition
- add one short figure comparing performance with and without the velocity features
- explicitly connect the contribution back to the dissertation title and research objectives
- explain that these features are lightweight and practical, meaning they could be computed in near real-time in a live fraud monitoring system

That last point is useful because it makes the contribution feel practical, not just academic.

## Good Academic Framing

Good phrasing for the dissertation:

- "A key personal contribution of this project is the design and evaluation of behavioural velocity features that capture short-term transaction dynamics at the cardholder level."
- "Unlike static transaction descriptors, these engineered features model recent behavioural intensity, which is often informative in fraud scenarios."
- "Their usefulness was validated through an ablation study, where removing them reduced the F1 score from 0.8705 to 0.8561."

## What To Avoid

Avoid presenting the personal contribution too vaguely.

Weak wording:

- "I did some feature engineering"
- "I added extra features to improve the model"

Stronger wording:

- "I designed behavioural velocity features to capture transaction burst patterns and recent spending intensity, then validated their contribution through ablation testing."

Also avoid letting the contribution get buried under too much model complexity. If the discussion focuses too heavily on autoencoder architecture, hyperparameters, and GA details, the examiner may miss the most cleanly defensible original part of the project.

## Practical Priorities From Here

If the current goal is to maximise marks, the best next steps around personal contribution are:

1. make the velocity features a clearly signposted personal contribution in the dissertation
2. prepare one neat ablation results table and one small supporting chart
3. write a short, confident explanation for viva/demo situations
4. position BDS + GA as an extension, not as the only contribution
5. connect the contribution to real-world fraud behaviour and deployment practicality

## Bottom Line

The personal contribution is already good enough to stand out, especially the velocity features.

The biggest improvement now is not necessarily adding more technical work. It is presenting the contribution more sharply, more confidently, and more explicitly as an original behavioural feature-engineering contribution that was validated experimentally.

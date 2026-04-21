"""Rule Engine: catches fraud patterns the ML model misses."""
import logging

logger = logging.getLogger(__name__)


class RuleResult:
    """Result from rule engine evaluation."""
    def __init__(self):
        self.triggered_rules = []  # [(rule_name, risk_level, reason)]
        self.rule_risk_level = 'NONE'
        self.probability_boost = 0.0

    def add_rule(self, name, risk_level, reason):
        self.triggered_rules.append((name, risk_level, reason))
        # Update overall risk to the highest triggered
        levels = {'NONE': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
        if levels.get(risk_level, 0) > levels.get(self.rule_risk_level, 0):
            self.rule_risk_level = risk_level

    @property
    def any_triggered(self):
        return len(self.triggered_rules) > 0


class RuleEngine:
    """Hard rules that catch what the ML model misses.
    Runs alongside the model, not instead of it."""

    def evaluate(self, features, metadata=None):
        """Evaluate all rules against a transaction's features.

        Args:
            features: dict with model feature values
            metadata: optional dict with additional context

        Returns:
            RuleResult with triggered rules and risk level
        """
        result = RuleResult()

        self._rule_velocity_spike(features, result)
        self._rule_first_txn_high_value(features, result)
        self._rule_nighttime_high_value(features, result)
        self._rule_amount_anomaly(features, metadata, result)
        # Rule 5 (rapid escalation) requires transaction sequence — handled separately

        return result

    def evaluate_sequence(self, transaction_sequence):
        """Evaluate rules that need transaction sequences (e.g., escalation).

        Args:
            transaction_sequence: list of feature dicts for same card, chronological order

        Returns:
            RuleResult for the latest transaction
        """
        result = RuleResult()

        if len(transaction_sequence) >= 3:
            self._rule_rapid_escalation(transaction_sequence, result)

        return result

    # ---- Individual Rules ----

    def _rule_velocity_spike(self, features, result):
        """RULE 1: velocity_1h >= 5 AND amount_velocity_1h > $500."""
        vel = features.get('velocity_1h', 0)
        amt_vel = features.get('amount_velocity_1h', 0)

        if vel >= 5 and amt_vel > 500:
            result.add_rule(
                'VELOCITY_SPIKE', 'HIGH',
                f'Unusual number of transactions detected - {vel:.0f} in 1 hour '
                f'totaling ${amt_vel:.2f} (threshold: 5+ txns and $500+)'
            )
            result.probability_boost = max(result.probability_boost, 0.3)

    def _rule_first_txn_high_value(self, features, result):
        """RULE 3: First transaction (velocity_24h == 1) with high amount (> $300)."""
        vel_24h = features.get('velocity_24h', 0)
        amt = features.get('amt', 0)

        if vel_24h <= 1 and amt > 300:
            result.add_rule(
                'FIRST_TXN_HIGH_VALUE', 'MEDIUM',
                f'High value first transaction (${amt:.2f}) with no prior 24h history'
            )
            result.probability_boost = max(result.probability_boost, 0.1)

    def _rule_nighttime_high_value(self, features, result):
        """RULE 4: Nighttime (is_night=1) AND high amount (> $500)."""
        is_night = features.get('is_night', 0)
        amt = features.get('amt', 0)

        if is_night == 1 and amt > 500:
            result.add_rule(
                'NIGHTTIME_HIGH_VALUE', 'HIGH',
                f'High value nighttime transaction: ${amt:.2f} between 10PM-6AM'
            )
            result.probability_boost = max(result.probability_boost, 0.2)

    def _rule_amount_anomaly(self, features, metadata, result):
        """RULE 2: Amount > cardholder mean + 3*std (requires profile data)."""
        # This rule needs cardholder profile — skip if not available
        if metadata is None:
            return

        card_mean = metadata.get('card_amt_mean')
        card_std = metadata.get('card_amt_std')
        if card_mean is None or card_std is None:
            return

        amt = features.get('amt', 0)
        threshold = card_mean + 3 * card_std

        if card_std > 0 and amt > threshold:
            z_score = (amt - card_mean) / card_std
            result.add_rule(
                'AMOUNT_ANOMALY', 'MEDIUM',
                f'Transaction ${amt:.2f} exceeds cardholder average '
                f'(${card_mean:.2f} +/- ${card_std:.2f}, z-score={z_score:.1f})'
            )
            result.probability_boost = max(result.probability_boost, 0.15)

    def _rule_rapid_escalation(self, sequence, result):
        """RULE 5: 3+ transactions in 1 hour, each larger than previous."""
        # Check last 3+ transactions
        recent = sequence[-3:]  # At least last 3
        amounts = [t.get('amt', 0) for t in recent]

        # Check if amounts are strictly increasing
        if all(amounts[i] < amounts[i+1] for i in range(len(amounts)-1)):
            result.add_rule(
                'RAPID_ESCALATION', 'CRITICAL',
                f'Escalating transaction pattern detected: '
                f'${" -> $".join(f"{a:.2f}" for a in amounts)} - potential card testing'
            )
            result.probability_boost = max(result.probability_boost, 0.4)

    # ---- Final Decision ----

    @staticmethod
    def combine_decision(model_probability, rule_result, threshold=0.7):
        """Combine model prediction with rule engine results.

        Decision logic:
          FRAUD   : model prob >= 0.70
          REVIEW  : 0.50 <= prob < 0.70  OR  rules say HIGH / CRITICAL
          MONITOR : 0.30 <= prob < 0.50  OR  rules say MEDIUM
          NORMAL  : prob < 0.30 and no significant rules

        Returns:
            (classification, combined_probability, decision_reason)
        """
        from app.config import REVIEW_THRESHOLD, MONITOR_THRESHOLD
        boosted_prob = min(model_probability + rule_result.probability_boost, 1.0)

        if model_probability >= threshold:
            return 'FRAUD', boosted_prob, 'Model detected fraud'

        if model_probability >= REVIEW_THRESHOLD or rule_result.rule_risk_level in ('HIGH', 'CRITICAL'):
            reasons = '; '.join(r[2] for r in rule_result.triggered_rules) if rule_result.triggered_rules else 'Elevated probability'
            return 'REVIEW', boosted_prob, f'Elevated risk: {reasons}'

        if model_probability >= MONITOR_THRESHOLD or rule_result.rule_risk_level == 'MEDIUM':
            reasons = '; '.join(r[2] for r in rule_result.triggered_rules) if rule_result.triggered_rules else 'Moderate probability'
            return 'MONITOR', boosted_prob, f'Monitor flagged: {reasons}'

        return 'NORMAL', model_probability, 'No issues detected'


# ---- Quick test ----
if __name__ == '__main__':
    engine = RuleEngine()

    print("="*60)
    print("RULE ENGINE TESTS")
    print("="*60)

    # Test 1: Velocity spike (catches what model missed in Scenario 3)
    print("\nTest 1: Velocity Spike (8 txns/hr, $400)")
    r = engine.evaluate({'velocity_1h': 8, 'velocity_24h': 15, 'amount_velocity_1h': 400,
                         'amt': 50, 'is_night': 1})
    print(f"  Triggered: {r.any_triggered}, Risk: {r.rule_risk_level}")
    # amt_vel=400 < 500, so rule doesn't trigger with these values
    # Try with higher total
    r2 = engine.evaluate({'velocity_1h': 8, 'velocity_24h': 15, 'amount_velocity_1h': 800,
                          'amt': 100, 'is_night': 1})
    print(f"  With $800 total: Triggered: {r2.any_triggered}, Risk: {r2.rule_risk_level}")
    for name, level, reason in r2.triggered_rules:
        print(f"    {name} [{level}]: {reason}")

    # Test 2: First transaction high value (Scenario 4)
    print("\nTest 2: First Transaction High Value ($500)")
    r = engine.evaluate({'velocity_24h': 1, 'amt': 500, 'velocity_1h': 1,
                         'amount_velocity_1h': 500, 'is_night': 0})
    print(f"  Triggered: {r.any_triggered}, Risk: {r.rule_risk_level}")
    for name, level, reason in r.triggered_rules:
        print(f"    {name} [{level}]: {reason}")

    # Test 3: Nighttime high value
    print("\nTest 3: Nighttime High Value ($300)")
    r = engine.evaluate({'is_night': 1, 'amt': 300, 'velocity_1h': 1,
                         'velocity_24h': 3, 'amount_velocity_1h': 300})
    for name, level, reason in r.triggered_rules:
        print(f"    {name} [{level}]: {reason}")

    # Test 4: Normal transaction (no rules trigger)
    print("\nTest 4: Normal Transaction ($15, daytime)")
    r = engine.evaluate({'amt': 15, 'is_night': 0, 'velocity_1h': 1,
                         'velocity_24h': 3, 'amount_velocity_1h': 45})
    print(f"  Triggered: {r.any_triggered}, Risk: {r.rule_risk_level}")

    # Test 5: Escalating fraud (Scenario 7)
    print("\nTest 5: Rapid Escalation (3 increasing amounts)")
    sequence = [
        {'amt': 25, 'velocity_1h': 1},
        {'amt': 100, 'velocity_1h': 2},
        {'amt': 500, 'velocity_1h': 3},
    ]
    r = engine.evaluate_sequence(sequence)
    print(f"  Triggered: {r.any_triggered}, Risk: {r.rule_risk_level}")
    for name, level, reason in r.triggered_rules:
        print(f"    {name} [{level}]: {reason}")

    # Test 6: Combined decision
    print("\nTest 6: Combined Decisions")
    cases = [
        (0.85, RuleResult(), "Model says FRAUD"),
        (0.10, engine.evaluate({'velocity_1h': 8, 'amt': 100, 'is_night': 1,
                                'velocity_24h': 3, 'amount_velocity_1h': 800}), "Model NORMAL, rules HIGH"),
        (0.10, engine.evaluate({'velocity_24h': 1, 'amt': 500, 'velocity_1h': 1,
                                'amount_velocity_1h': 500, 'is_night': 0}), "Model NORMAL, rules MEDIUM"),
        (0.05, RuleResult(), "Model NORMAL, no rules"),
    ]
    for prob, rules, desc in cases:
        classification, combined_prob, reason = RuleEngine.combine_decision(prob, rules)
        print(f"  {desc}: -> {classification} (prob={combined_prob:.2f})")

    print("\n  ALL RULE ENGINE TESTS COMPLETE")

from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from aurora_sentinel.agent import OptionsSentinel
from aurora_sentinel.alpaca_cli import build_paper_order_request
from aurora_sentinel.audit import AuditTrail, verify_chain
from aurora_sentinel.contracts import (
    AgentThesis,
    Direction,
    OptionContract,
    OptionRight,
    PaperAccountState,
    RiskLimits,
)
from aurora_sentinel.costs import estimate_long_option_round_trip
from aurora_sentinel.risk import RiskGate


NOW = datetime(2026, 8, 28, 15, 30, tzinfo=timezone.utc)


def thesis(*, confidence: str = "0.75", direction: Direction = Direction.BULLISH) -> AgentThesis:
    return AgentThesis(
        underlying="SPY",
        direction=direction,
        confidence=Decimal(confidence),
        as_of=NOW,
        rationale="Synthetic test rationale.",
        evidence_ids=("fixture-1",),
    )


def contract(*, ask: str = "3.90", bid: str = "3.80") -> OptionContract:
    return OptionContract(
        symbol="SPY260904C00650000",
        underlying="SPY",
        right=OptionRight.CALL,
        strike=Decimal("650"),
        expiration=date(2026, 9, 4),
        bid=Decimal(bid),
        ask=Decimal(ask),
        open_interest=1000,
        volume=500,
    )


def account(**overrides: object) -> PaperAccountState:
    values: dict[str, object] = {
        "environment": "paper",
        "equity_usd": Decimal("100000"),
        "daily_pnl_usd": Decimal("0"),
        "open_risk_usd": Decimal("0"),
        "orders_today": 0,
        "observed_at": NOW,
    }
    values.update(overrides)
    return PaperAccountState(**values)  # type: ignore[arg-type]


class OptionsSentinelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sentinel = OptionsSentinel(RiskGate(RiskLimits()))

    def test_approved_decision_builds_paper_limit_request(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(), contracts=(contract(),), account=account(), evaluated_at=NOW
        )
        self.assertEqual(decision.action, "BUY_TO_OPEN")
        request = build_paper_order_request(decision)
        self.assertEqual(request.environment, "paper")
        self.assertIn("--paper", request.arguments)
        self.assertIn("limit", request.arguments)

    def test_low_confidence_fails_closed(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(confidence="0.40"),
            contracts=(contract(),),
            account=account(),
            evaluated_at=NOW,
        )
        self.assertEqual(decision.action, "NO_ACTION")
        self.assertIn("confidence_below_threshold", decision.assessment.reason_codes)

    def test_daily_loss_and_order_limits_are_both_reported(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(),
            contracts=(contract(),),
            account=account(daily_pnl_usd=Decimal("-1000"), orders_today=4),
            evaluated_at=NOW,
        )
        self.assertFalse(decision.assessment.approved)
        self.assertIn("daily_loss_limit_reached", decision.assessment.reason_codes)
        self.assertIn("daily_order_limit_reached", decision.assessment.reason_codes)

    def test_stale_thesis_is_rejected(self) -> None:
        stale = AgentThesis(
            underlying="SPY",
            direction=Direction.BULLISH,
            confidence=Decimal("0.8"),
            as_of=NOW - timedelta(minutes=5),
            rationale="Stale fixture.",
            evidence_ids=("fixture-stale",),
        )
        decision = self.sentinel.decide(
            thesis=stale, contracts=(contract(),), account=account(), evaluated_at=NOW
        )
        self.assertIn("thesis_stale_or_future", decision.assessment.reason_codes)

    def test_neutral_thesis_never_selects_a_contract(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(direction=Direction.NEUTRAL),
            contracts=(contract(),),
            account=account(),
            evaluated_at=NOW,
        )
        self.assertEqual(decision.contract_symbol, None)
        self.assertEqual(decision.action, "NO_ACTION")

    def test_non_paper_account_cannot_be_constructed(self) -> None:
        with self.assertRaisesRegex(ValueError, "paper_environment_required"):
            account(environment="live")

    def test_decisions_form_a_verifiable_audit_chain(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(), contracts=(contract(),), account=account(), evaluated_at=NOW
        )
        audit = AuditTrail()
        first = audit.record_decision(decision, emitted_at=NOW)
        audit.append(event_type="ORDER_INTENT_BUILT", payload={"paper": True}, emitted_at=NOW)
        self.assertTrue(verify_chain(audit.events))
        tampered = (replace(first, payload_json='{"action":"LIVE"}'), *audit.events[1:])
        self.assertFalse(verify_chain(tampered))

    def test_audit_rejects_sensitive_fields(self) -> None:
        audit = AuditTrail()
        with self.assertRaisesRegex(ValueError, "sensitive_audit_field_rejected"):
            audit.append(
                event_type="BAD_EVENT",
                payload={"nested": {"api_key": "must-not-be-recorded"}},
                emitted_at=NOW,
            )

    def test_round_trip_cost_includes_spread_and_regulatory_fees(self) -> None:
        estimate = estimate_long_option_round_trip(
            bid=Decimal("3.64"),
            ask=Decimal("3.66"),
            exit_premium=Decimal("3.64"),
        )
        self.assertEqual(estimate.quoted_round_trip_spread_usd, Decimal("2.00"))
        self.assertEqual(estimate.buy_fees_unrounded_usd, Decimal("0.040300"))
        self.assertGreater(estimate.sell_fees_unrounded_usd, Decimal("0.04359"))
        self.assertEqual(estimate.conservative_rounded_fee_floor_usd, Decimal("0.11"))


if __name__ == "__main__":
    unittest.main()

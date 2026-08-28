from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

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
from aurora_sentinel.market_data import (
    AlpacaReadOnlyDataClient,
    MarketClock,
    next_friday_in_window,
    normalize_option_chain,
)
from aurora_sentinel.model_gate import ModelEvidence, load_model_evidence
from aurora_sentinel.paper_account import (
    API_KEY_ENV,
    AlpacaPaperCredentials,
    VerifiedPaperAccount,
    credential_target,
    load_paper_credentials,
    verify_competition_account,
)
from aurora_sentinel.paper_orders import (
    PAPER_ORDERS_URL,
    AlpacaPaperOrderClient,
    prepare_paper_order,
)
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
        quote_observed_at=NOW,
    )


def account(**overrides: object) -> PaperAccountState:
    values: dict[str, object] = {
        "environment": "paper",
        "equity_usd": Decimal("100000"),
        "daily_pnl_usd": Decimal("0"),
        "open_risk_usd": Decimal("0"),
        "orders_today": 0,
        "market_open": True,
        "observed_at": NOW,
    }
    values.update(overrides)
    return PaperAccountState(**values)  # type: ignore[arg-type]


def validated_model(**overrides: object) -> ModelEvidence:
    values: dict[str, object] = {
        "model_id": "a" * 64,
        "track": "CONTEST_EXPERIMENTAL_IEX",
        "feature_set": "MARKET_CORE_M15_V1",
        "target": "forward_close_return_4",
        "trained_at": NOW,
        "validation_mae": Decimal("0.0010"),
        "validation_baseline_mae": Decimal("0.0012"),
        "test_mae": Decimal("0.0011"),
        "test_baseline_mae": Decimal("0.0013"),
        "validation_net_return_proxy": Decimal("0.02"),
        "test_net_return_proxy": Decimal("0.01"),
        "paper_eligible": True,
        "reason_codes": ("synthetic_test_fixture",),
    }
    values.update(overrides)
    return ModelEvidence(**values)  # type: ignore[arg-type]


class OptionsSentinelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sentinel = OptionsSentinel(RiskGate(RiskLimits()))

    def test_approved_decision_builds_paper_limit_request(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(), contracts=(contract(),), account=account(), evaluated_at=NOW,
            model_evidence=validated_model(),
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
            model_evidence=validated_model(),
        )
        self.assertEqual(decision.action, "NO_ACTION")
        self.assertIn("confidence_below_threshold", decision.assessment.reason_codes)

    def test_daily_loss_and_order_limits_are_both_reported(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(),
            contracts=(contract(),),
            account=account(daily_pnl_usd=Decimal("-1000"), orders_today=4),
            evaluated_at=NOW,
            model_evidence=validated_model(),
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
            thesis=stale, contracts=(contract(),), account=account(), evaluated_at=NOW,
            model_evidence=validated_model(),
        )
        self.assertIn("thesis_stale_or_future", decision.assessment.reason_codes)

    def test_neutral_thesis_never_selects_a_contract(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(direction=Direction.NEUTRAL),
            contracts=(contract(),),
            account=account(),
            evaluated_at=NOW,
            model_evidence=validated_model(),
        )
        self.assertEqual(decision.contract_symbol, None)
        self.assertEqual(decision.action, "NO_ACTION")

    def test_non_paper_account_cannot_be_constructed(self) -> None:
        with self.assertRaisesRegex(ValueError, "paper_environment_required"):
            account(environment="live")

    def test_closed_market_and_stale_quote_fail_closed(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(),
            contracts=(
                replace(
                    contract(),
                    quote_observed_at=NOW - timedelta(minutes=5),
                ),
            ),
            account=account(market_open=False),
            evaluated_at=NOW,
            model_evidence=validated_model(),
        )
        self.assertIn("market_closed", decision.assessment.reason_codes)
        self.assertIn("option_quote_stale_or_future", decision.assessment.reason_codes)

    def test_low_open_interest_fails_closed(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(),
            contracts=(replace(contract(), open_interest=10),),
            account=account(),
            evaluated_at=NOW,
            model_evidence=validated_model(),
        )
        self.assertIn("open_interest_below_threshold", decision.assessment.reason_codes)

    def test_decisions_form_a_verifiable_audit_chain(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(), contracts=(contract(),), account=account(), evaluated_at=NOW,
            model_evidence=validated_model(),
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

    def test_credential_targets_are_account_scoped(self) -> None:
        self.assertEqual(
            credential_target("PA3HAW9279NN", "API_KEY"),
            "AURORA/Alpaca/Paper/PA3HAW9279NN/API_KEY",
        )
        with self.assertRaisesRegex(ValueError, "paper_account_number_required"):
            credential_target("LIVE123", "API_KEY")

    def test_partial_environment_credentials_fail_closed(self) -> None:
        with patch.dict(os.environ, {API_KEY_ENV: "P" * 26}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "partial_alpaca_environment_rejected"):
                load_paper_credentials("PA3HAW9279NN")

    def test_credentials_can_be_loaded_without_entering_the_repository(self) -> None:
        values = {
            credential_target("PA3HAW9279NN", "API_KEY"): "P" * 26,
            credential_target("PA3HAW9279NN", "SECRET_KEY"): "S" * 44,
        }
        with patch.dict(os.environ, {}, clear=True):
            credentials = load_paper_credentials(
                "PA3HAW9279NN", credential_reader=values.__getitem__
            )
        self.assertEqual(credentials.account_number, "PA3HAW9279NN")
        self.assertEqual(credentials.base_url, "https://paper-api.alpaca.markets")

    def test_live_endpoint_cannot_be_injected(self) -> None:
        with self.assertRaisesRegex(ValueError, "paper_endpoint_required"):
            AlpacaPaperCredentials(
                "PA3HAW9279NN",
                "P" * 26,
                "S" * 44,
                base_url="https://api.alpaca.markets",
            )

    def test_competition_account_must_be_active_fresh_and_unblocked(self) -> None:
        payload: dict[str, object] = {
            "account_number": "PA3HAW9279NN",
            "status": "ACTIVE",
            "cash": "100000",
            "buying_power": "400000",
            "portfolio_value": "100000",
            "trading_blocked": False,
            "account_blocked": False,
        }
        verified = verify_competition_account(
            payload, expected_account_number="PA3HAW9279NN"
        )
        self.assertEqual(verified.cash_usd, Decimal("100000"))
        with self.assertRaisesRegex(RuntimeError, "competition_starting_balance_mismatch"):
            verify_competition_account(
                {**payload, "cash": "99999"},
                expected_account_number="PA3HAW9279NN",
            )

    def test_option_chain_normalization_preserves_quote_time_and_open_interest(self) -> None:
        chain = normalize_option_chain(
            underlying="SPY",
            expiration=date(2026, 9, 4),
            contract_payloads=[
                {
                    "symbol": "SPY260904C00777000",
                    "type": "call",
                    "strike_price": "777",
                    "expiration_date": "2026-09-04",
                    "open_interest": "769",
                    "tradable": True,
                    "status": "active",
                }
            ],
            snapshot_payloads={
                "SPY260904C00777000": {
                    "latestQuote": {
                        "bp": 2.46,
                        "ap": 2.47,
                        "t": NOW.isoformat(),
                    },
                    "latestTrade": {"s": 5},
                }
            },
            captured_at=NOW,
        )
        self.assertEqual(len(chain.contracts), 1)
        self.assertEqual(chain.contracts[0].open_interest, 769)
        self.assertEqual(chain.contracts[0].quote_observed_at, NOW)

    def test_expiration_picker_stays_inside_risk_window(self) -> None:
        expiration = next_friday_in_window(date(2026, 8, 27))
        self.assertEqual(expiration, date(2026, 9, 4))
        self.assertGreaterEqual((expiration - date(2026, 8, 27)).days, 2)
        self.assertLessEqual((expiration - date(2026, 8, 27)).days, 14)

    def test_market_client_rejects_lookalike_hosts(self) -> None:
        credentials = AlpacaPaperCredentials("PA3HAW9279NN", "P" * 26, "S" * 44)
        client = AlpacaReadOnlyDataClient(credentials)
        with self.assertRaisesRegex(ValueError, "alpaca_read_only_endpoint_required"):
            client._get_json("https://data.alpaca.markets.evil.invalid/v2/stocks/SPY")

    def test_paper_order_gateway_is_disabled_by_default(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(), contracts=(contract(),), account=account(), evaluated_at=NOW,
            model_evidence=validated_model(),
        )
        prepared = prepare_paper_order(decision, account_number="PA3HAW9279NN")
        credentials = AlpacaPaperCredentials("PA3HAW9279NN", "P" * 26, "S" * 44)
        client = AlpacaPaperOrderClient(credentials)
        verified = VerifiedPaperAccount(
            account_number="PA3HAW9279NN",
            status="ACTIVE",
            cash_usd=Decimal("100000"),
            buying_power_usd=Decimal("400000"),
            portfolio_value_usd=Decimal("100000"),
            trading_blocked=False,
            account_blocked=False,
        )
        clock = MarketClock(True, NOW, NOW + timedelta(hours=1), NOW + timedelta(hours=6))
        with self.assertRaisesRegex(RuntimeError, "paper_order_submission_disabled"):
            client.submit(prepared, account=verified, clock=clock)

    def test_paper_order_gateway_posts_only_the_prepared_limit_payload(self) -> None:
        decision = self.sentinel.decide(
            thesis=thesis(), contracts=(contract(),), account=account(), evaluated_at=NOW,
            model_evidence=validated_model(),
        )
        prepared = prepare_paper_order(decision, account_number="PA3HAW9279NN")
        requests: list[object] = []

        class FakeResponse:
            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "id": "paper-order-1",
                        "client_order_id": prepared.payload()["client_order_id"],
                        "symbol": prepared.symbol,
                        "status": "accepted",
                    }
                ).encode("utf-8")

        def fake_opener(request: object, *, timeout: int) -> FakeResponse:
            requests.append(request)
            self.assertEqual(timeout, 20)
            self.assertEqual(getattr(request, "full_url"), PAPER_ORDERS_URL)
            self.assertEqual(getattr(request, "method"), "POST")
            return FakeResponse()

        credentials = AlpacaPaperCredentials("PA3HAW9279NN", "P" * 26, "S" * 44)
        client = AlpacaPaperOrderClient(
            credentials, submission_enabled=True, opener=fake_opener
        )
        verified = VerifiedPaperAccount(
            account_number="PA3HAW9279NN",
            status="ACTIVE",
            cash_usd=Decimal("100000"),
            buying_power_usd=Decimal("400000"),
            portfolio_value_usd=Decimal("100000"),
            trading_blocked=False,
            account_blocked=False,
        )
        clock = MarketClock(True, NOW, NOW + timedelta(hours=1), NOW + timedelta(hours=6))
        receipt = client.submit(prepared, account=verified, clock=clock)
        self.assertEqual(receipt.order_id, "paper-order-1")
        self.assertEqual(len(requests), 1)
        self.assertEqual(prepared.payload()["type"], "limit")
        self.assertEqual(prepared.payload()["qty"], "1")

    def test_missing_or_failed_model_evidence_prevents_any_order_intent(self) -> None:
        missing = self.sentinel.decide(
            thesis=thesis(), contracts=(contract(),), account=account(), evaluated_at=NOW
        )
        self.assertEqual(missing.action, "NO_ACTION")
        self.assertIn("model_evidence_missing", missing.assessment.reason_codes)

        failed = self.sentinel.decide(
            thesis=thesis(),
            contracts=(contract(),),
            account=account(),
            evaluated_at=NOW,
            model_evidence=validated_model(
                paper_eligible=False,
                test_mae=Decimal("0.0020"),
            ),
        )
        self.assertEqual(failed.action, "NO_ACTION")
        self.assertIn("test_did_not_beat_baseline", failed.assessment.reason_codes)
        self.assertIn("model_not_paper_eligible", failed.assessment.reason_codes)

    def test_order_preparation_rejects_forged_approved_decision_without_model(self) -> None:
        forged = self.sentinel.decide(
            thesis=thesis(),
            contracts=(contract(),),
            account=account(),
            evaluated_at=NOW,
            model_evidence=validated_model(),
        )
        forged = replace(
            forged,
            model_id=None,
            model_evidence_id=None,
            model_validated_for_paper=False,
        )
        with self.assertRaisesRegex(ValueError, "validated_model_evidence_required"):
            prepare_paper_order(forged, account_number="PA3HAW9279NN")

    def test_private_artifact_loader_preserves_failed_model_verdict(self) -> None:
        payload = {
            "schema_version": "aurora.contest-iex-model.v1",
            "track": "CONTEST_EXPERIMENTAL_IEX",
            "core_model_evidence": False,
            "paper_only": True,
            "live_trading_allowed": False,
            "model_id": "b" * 64,
            "trained_at": NOW.isoformat(),
            "feature_set": "MARKET_CORE_M15_V1",
            "target": "forward_close_return_4",
            "validation": {
                "mae": 0.001,
                "baseline_mae": 0.002,
                "net_return_proxy": 0.01,
            },
            "test": {
                "mae": 0.003,
                "baseline_mae": 0.002,
                "net_return_proxy": -0.01,
            },
            "paper_eligible": False,
            "reason_codes": ["test_did_not_beat_baseline"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            evidence = load_model_evidence(path)
        decision = self.sentinel.decide(
            thesis=thesis(),
            contracts=(contract(),),
            account=account(),
            evaluated_at=NOW,
            model_evidence=evidence,
        )
        self.assertEqual(decision.action, "NO_ACTION")
        self.assertIn("test_did_not_beat_baseline", decision.assessment.reason_codes)
        self.assertIn("model_not_paper_eligible", decision.assessment.reason_codes)


if __name__ == "__main__":
    unittest.main()

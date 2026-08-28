"""Fail-closed Paper order gateway. No live endpoint exists in this module."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable
from urllib.request import Request, urlopen

from .agent import AgentDecision
from .market_data import MarketClock
from .paper_account import AlpacaPaperCredentials, VerifiedPaperAccount


PAPER_ORDERS_URL = "https://paper-api.alpaca.markets/v2/orders"


@dataclass(frozen=True, slots=True)
class PreparedPaperOrder:
    account_number: str
    symbol: str
    quantity: int
    limit_price: Decimal
    payload_json: str
    intent_hash: str

    def payload(self) -> dict[str, object]:
        parsed = json.loads(self.payload_json)
        if not isinstance(parsed, dict):
            raise RuntimeError("invalid_prepared_order_payload")
        return parsed


@dataclass(frozen=True, slots=True)
class PaperOrderReceipt:
    order_id: str
    client_order_id: str
    symbol: str
    status: str
    submitted_at: str | None


def prepare_paper_order(
    decision: AgentDecision,
    *,
    account_number: str,
) -> PreparedPaperOrder:
    if not account_number.startswith("PA"):
        raise ValueError("paper_account_number_required")
    if decision.environment != "paper":
        raise ValueError("paper_environment_required")
    if not decision.assessment.approved or decision.action != "BUY_TO_OPEN":
        raise ValueError("approved_buy_to_open_decision_required")
    if (
        not decision.model_validated_for_paper
        or decision.model_id is None
        or decision.model_evidence_id is None
    ):
        raise ValueError("validated_model_evidence_required")
    if decision.contract_symbol is None or decision.limit_price is None:
        raise ValueError("incomplete_order_intent")
    if decision.quantity != 1 or decision.limit_price <= 0:
        raise ValueError("single_positive_limit_order_required")
    canonical_intent = json.dumps(
        {
            "account_number": account_number,
            "environment": "paper",
            "limit_price": str(decision.limit_price),
            "model_evidence_id": decision.model_evidence_id,
            "model_id": decision.model_id,
            "quantity": decision.quantity,
            "symbol": decision.contract_symbol,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    intent_hash = hashlib.sha256(canonical_intent.encode("utf-8")).hexdigest()
    payload = {
        "client_order_id": f"aurora-{intent_hash[:24]}",
        "limit_price": str(decision.limit_price),
        "qty": "1",
        "side": "buy",
        "symbol": decision.contract_symbol,
        "time_in_force": "day",
        "type": "limit",
    }
    return PreparedPaperOrder(
        account_number=account_number,
        symbol=decision.contract_symbol,
        quantity=1,
        limit_price=decision.limit_price,
        payload_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        intent_hash=intent_hash,
    )


class AlpacaPaperOrderClient:
    def __init__(
        self,
        credentials: AlpacaPaperCredentials,
        *,
        submission_enabled: bool = False,
        opener: Callable[..., object] = urlopen,
    ) -> None:
        self._credentials = credentials
        self._submission_enabled = submission_enabled
        self._opener = opener

    def submit(
        self,
        prepared: PreparedPaperOrder,
        *,
        account: VerifiedPaperAccount,
        clock: MarketClock,
    ) -> PaperOrderReceipt:
        if not self._submission_enabled:
            raise RuntimeError("paper_order_submission_disabled")
        if prepared.account_number != self._credentials.account_number:
            raise RuntimeError("paper_account_credential_mismatch")
        if account.account_number != prepared.account_number or account.status != "ACTIVE":
            raise RuntimeError("paper_account_identity_mismatch")
        if account.trading_blocked or account.account_blocked:
            raise RuntimeError("paper_account_blocked")
        if not clock.is_open:
            raise RuntimeError("market_closed")
        if prepared.quantity != 1 or prepared.limit_price <= 0:
            raise RuntimeError("invalid_prepared_order")

        request = Request(
            PAPER_ORDERS_URL,
            data=prepared.payload_json.encode("utf-8"),
            headers={
                "APCA-API-KEY-ID": self._credentials.api_key,
                "APCA-API-SECRET-KEY": self._credentials.secret_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with self._opener(request, timeout=20) as response:  # type: ignore[attr-defined]
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or not payload.get("id"):
            raise RuntimeError("invalid_alpaca_order_response")
        return PaperOrderReceipt(
            order_id=str(payload["id"]),
            client_order_id=str(payload.get("client_order_id", "")),
            symbol=str(payload.get("symbol", prepared.symbol)),
            status=str(payload.get("status", "unknown")),
            submitted_at=(
                str(payload["submitted_at"]) if payload.get("submitted_at") else None
            ),
        )

"""Paper-only command boundary for the Alpaca CLI integration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

from .agent import AgentDecision


@dataclass(frozen=True, slots=True)
class RedactedCliRequest:
    executable: str
    arguments: tuple[str, ...]
    environment: str

    def render_for_audit(self) -> str:
        return json.dumps(
            {"executable": self.executable, "arguments": self.arguments, "environment": self.environment},
            sort_keys=True,
            separators=(",", ":"),
        )


def build_paper_order_request(decision: AgentDecision) -> RedactedCliRequest:
    if decision.environment != "paper":
        raise ValueError("paper_environment_required")
    if decision.action != "BUY_TO_OPEN" or not decision.assessment.approved:
        raise ValueError("approved_buy_to_open_decision_required")
    if (
        not decision.model_validated_for_paper
        or decision.model_id is None
        or decision.model_evidence_id is None
    ):
        raise ValueError("validated_model_evidence_required")
    if decision.contract_symbol is None or decision.limit_price is None or decision.quantity != 1:
        raise ValueError("incomplete_order_intent")
    if decision.limit_price <= Decimal("0"):
        raise ValueError("positive_limit_price_required")
    return RedactedCliRequest(
        executable="alpaca",
        arguments=(
            "orders",
            "submit",
            "--paper",
            "--symbol",
            decision.contract_symbol,
            "--qty",
            "1",
            "--side",
            "buy",
            "--type",
            "limit",
            "--limit-price",
            str(decision.limit_price),
            "--time-in-force",
            "day",
        ),
        environment="paper",
    )

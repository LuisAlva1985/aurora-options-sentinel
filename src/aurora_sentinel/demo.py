"""Deterministic offline demonstration. It never invokes Alpaca."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal

from .agent import OptionsSentinel
from .alpaca_cli import build_paper_order_request
from .contracts import AgentThesis, Direction, OptionContract, OptionRight, PaperAccountState, RiskLimits
from .risk import RiskGate


def main() -> None:
    now = datetime(2026, 8, 28, 15, 30, tzinfo=timezone.utc)
    thesis = AgentThesis(
        underlying="SPY",
        direction=Direction.BULLISH,
        confidence=Decimal("0.74"),
        as_of=now,
        rationale="Synthetic demo thesis: bounded upside participation with premium-defined risk.",
        evidence_ids=("synthetic-demo-observation",),
    )
    contract = OptionContract(
        symbol="SPY260904C00650000",
        underlying="SPY",
        right=OptionRight.CALL,
        strike=Decimal("650"),
        expiration=date(2026, 9, 4),
        bid=Decimal("3.80"),
        ask=Decimal("3.90"),
        open_interest=1500,
        volume=800,
    )
    account = PaperAccountState(
        environment="paper",
        equity_usd=Decimal("100000"),
        daily_pnl_usd=Decimal("0"),
        open_risk_usd=Decimal("0"),
        orders_today=0,
        observed_at=now,
    )
    decision = OptionsSentinel(RiskGate(RiskLimits())).decide(
        thesis=thesis,
        contracts=(contract,),
        account=account,
        evaluated_at=now,
    )
    request = build_paper_order_request(decision)
    print(json.dumps({"decision": decision.action, "request": json.loads(request.render_for_audit())}, indent=2))


if __name__ == "__main__":
    main()

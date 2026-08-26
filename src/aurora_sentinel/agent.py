"""Contract selection and decision assembly for the competition MVP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .contracts import AgentThesis, Direction, OptionContract, OptionRight, PaperAccountState
from .risk import RiskAssessment, RiskGate


@dataclass(frozen=True, slots=True)
class AgentDecision:
    action: str
    contract_symbol: str | None
    limit_price: Decimal | None
    quantity: int
    assessment: RiskAssessment
    rationale: str
    environment: str = "paper"
    order_type: str = "limit"
    time_in_force: str = "day"


class OptionsSentinel:
    def __init__(self, risk_gate: RiskGate) -> None:
        self._risk_gate = risk_gate

    def decide(
        self,
        *,
        thesis: AgentThesis,
        contracts: tuple[OptionContract, ...],
        account: PaperAccountState,
        evaluated_at: datetime,
    ) -> AgentDecision:
        if thesis.direction is Direction.NEUTRAL:
            assessment = RiskAssessment(False, ("neutral_thesis_no_action",))
            return AgentDecision("NO_ACTION", None, None, 0, assessment, thesis.rationale)
        required_right = OptionRight.CALL if thesis.direction is Direction.BULLISH else OptionRight.PUT
        eligible = tuple(
            contract
            for contract in contracts
            if contract.underlying == thesis.underlying and contract.right is required_right
        )
        if not eligible:
            assessment = RiskAssessment(False, ("no_direction_compatible_contract",))
            return AgentDecision("NO_ACTION", None, None, 0, assessment, thesis.rationale)
        ranked = sorted(
            eligible,
            key=lambda contract: (
                contract.spread_pct,
                -contract.open_interest,
                -contract.volume,
                contract.premium_risk_usd,
                contract.symbol,
            ),
        )
        candidate = ranked[0]
        assessment = self._risk_gate.assess(
            thesis=thesis,
            contract=candidate,
            account=account,
            evaluated_at=evaluated_at,
        )
        if not assessment.approved:
            return AgentDecision(
                "NO_ACTION", candidate.symbol, None, 0, assessment, thesis.rationale
            )
        return AgentDecision(
            "BUY_TO_OPEN",
            candidate.symbol,
            candidate.ask,
            1,
            assessment,
            thesis.rationale,
        )

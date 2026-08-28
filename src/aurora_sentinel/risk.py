"""Fail-closed risk gate for Paper options proposals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .contracts import AgentThesis, Direction, OptionContract, OptionRight, PaperAccountState, RiskLimits


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    approved: bool
    reason_codes: tuple[str, ...]


class RiskGate:
    """Evaluate every hard gate; absence of reasons is the only approval path."""

    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits

    def assess(
        self,
        *,
        thesis: AgentThesis,
        contract: OptionContract,
        account: PaperAccountState,
        evaluated_at: datetime,
    ) -> RiskAssessment:
        reasons: list[str] = []
        if account.environment != "paper":
            reasons.append("paper_environment_required")
        if not account.market_open:
            reasons.append("market_closed")
        if thesis.underlying != self._limits.allowed_underlying:
            reasons.append("underlying_not_allowed")
        if contract.underlying != thesis.underlying:
            reasons.append("contract_underlying_mismatch")
        if thesis.direction is Direction.NEUTRAL:
            reasons.append("neutral_thesis_no_action")
        expected_right = OptionRight.CALL if thesis.direction is Direction.BULLISH else OptionRight.PUT
        if thesis.direction is not Direction.NEUTRAL and contract.right is not expected_right:
            reasons.append("option_right_direction_mismatch")
        if thesis.confidence < self._limits.minimum_confidence:
            reasons.append("confidence_below_threshold")
        if not contract.tradable:
            reasons.append("contract_not_tradable")
        if contract.premium_risk_usd > self._limits.max_premium_per_trade_usd:
            reasons.append("premium_risk_above_limit")
        if contract.spread_pct > self._limits.max_bid_ask_spread_pct:
            reasons.append("spread_above_limit")
        if contract.open_interest < self._limits.minimum_open_interest:
            reasons.append("open_interest_below_threshold")
        if account.daily_pnl_usd <= -self._limits.max_daily_loss_usd:
            reasons.append("daily_loss_limit_reached")
        if account.open_risk_usd + contract.premium_risk_usd > self._limits.max_open_risk_usd:
            reasons.append("open_risk_above_limit")
        if account.orders_today >= self._limits.max_orders_per_day:
            reasons.append("daily_order_limit_reached")
        days_to_expiry = (contract.expiration - evaluated_at.date()).days
        if not self._limits.min_days_to_expiry <= days_to_expiry <= self._limits.max_days_to_expiry:
            reasons.append("expiry_outside_allowed_window")
        thesis_age = (evaluated_at - thesis.as_of).total_seconds()
        account_age = (evaluated_at - account.observed_at).total_seconds()
        option_quote_age = (evaluated_at - contract.quote_observed_at).total_seconds()
        if thesis_age < 0 or thesis_age > self._limits.max_observation_age_seconds:
            reasons.append("thesis_stale_or_future")
        if account_age < 0 or account_age > self._limits.max_observation_age_seconds:
            reasons.append("account_state_stale_or_future")
        if option_quote_age < 0 or option_quote_age > self._limits.max_observation_age_seconds:
            reasons.append("option_quote_stale_or_future")
        return RiskAssessment(approved=not reasons, reason_codes=tuple(reasons))

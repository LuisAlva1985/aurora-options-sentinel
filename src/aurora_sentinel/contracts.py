"""Strict, dependency-free contracts for the competition agent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import StrEnum


class Direction(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class OptionRight(StrEnum):
    CALL = "CALL"
    PUT = "PUT"


@dataclass(frozen=True, slots=True)
class AgentThesis:
    underlying: str
    direction: Direction
    confidence: Decimal
    as_of: datetime
    rationale: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("thesis_timestamp_must_be_timezone_aware")
        if not Decimal("0") <= self.confidence <= Decimal("1"):
            raise ValueError("confidence_out_of_range")
        if not self.underlying or not self.rationale.strip() or not self.evidence_ids:
            raise ValueError("thesis_traceability_incomplete")


@dataclass(frozen=True, slots=True)
class OptionContract:
    symbol: str
    underlying: str
    right: OptionRight
    strike: Decimal
    expiration: date
    bid: Decimal
    ask: Decimal
    open_interest: int
    volume: int
    quote_observed_at: datetime
    tradable: bool = True

    def __post_init__(self) -> None:
        if not self.symbol or not self.underlying:
            raise ValueError("contract_identity_missing")
        if self.strike <= 0 or self.bid < 0 or self.ask <= 0 or self.ask < self.bid:
            raise ValueError("invalid_option_quote")
        if self.open_interest < 0 or self.volume < 0:
            raise ValueError("invalid_liquidity_count")
        if self.quote_observed_at.tzinfo is None or self.quote_observed_at.utcoffset() is None:
            raise ValueError("option_quote_timestamp_must_be_timezone_aware")

    @property
    def spread_pct(self) -> Decimal:
        midpoint = (self.bid + self.ask) / Decimal("2")
        if midpoint == 0:
            return Decimal("Infinity")
        return (self.ask - self.bid) / midpoint

    @property
    def premium_risk_usd(self) -> Decimal:
        return self.ask * Decimal("100")


@dataclass(frozen=True, slots=True)
class PaperAccountState:
    environment: str
    equity_usd: Decimal
    daily_pnl_usd: Decimal
    open_risk_usd: Decimal
    orders_today: int
    market_open: bool
    observed_at: datetime

    def __post_init__(self) -> None:
        if self.environment != "paper":
            raise ValueError("paper_environment_required")
        if self.equity_usd <= 0 or self.open_risk_usd < 0 or self.orders_today < 0:
            raise ValueError("invalid_account_state")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("account_timestamp_must_be_timezone_aware")


@dataclass(frozen=True, slots=True)
class RiskLimits:
    allowed_underlying: str = "SPY"
    minimum_confidence: Decimal = Decimal("0.65")
    max_premium_per_trade_usd: Decimal = Decimal("500")
    max_daily_loss_usd: Decimal = Decimal("1000")
    max_open_risk_usd: Decimal = Decimal("1500")
    max_orders_per_day: int = 4
    minimum_open_interest: int = 100
    min_days_to_expiry: int = 2
    max_days_to_expiry: int = 14
    max_bid_ask_spread_pct: Decimal = Decimal("0.15")
    max_observation_age_seconds: int = 90

    def __post_init__(self) -> None:
        decimals = (
            self.minimum_confidence,
            self.max_premium_per_trade_usd,
            self.max_daily_loss_usd,
            self.max_open_risk_usd,
            self.max_bid_ask_spread_pct,
        )
        if any(value <= 0 for value in decimals):
            raise ValueError("risk_limits_must_be_positive")
        if (
            self.max_orders_per_day <= 0
            or self.minimum_open_interest <= 0
            or self.min_days_to_expiry < 0
        ):
            raise ValueError("invalid_risk_limit_count")
        if self.max_days_to_expiry < self.min_days_to_expiry:
            raise ValueError("invalid_expiry_window")


def utc_now_for_demo() -> datetime:
    """Explicit clock boundary used only by the local demonstration."""

    return datetime.now(timezone.utc)

"""Explicit option spread and pass-through fee model for Paper evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING


@dataclass(frozen=True, slots=True)
class OptionFeeSchedule:
    """Rates from Alpaca's brokerage fee schedule revised 2026-07-20."""

    commission_rate: Decimal = Decimal("0")
    sec_sell_rate_on_trade_value: Decimal = Decimal("0.0000206")
    taf_sell_per_contract: Decimal = Decimal("0.00329")
    cat_per_equivalent_share: Decimal = Decimal("0.000003")
    orf_per_contract: Decimal = Decimal("0.015")
    occ_per_contract: Decimal = Decimal("0.025")
    contract_multiplier: int = 100


@dataclass(frozen=True, slots=True)
class FrictionEstimate:
    contracts: int
    quoted_round_trip_spread_usd: Decimal
    buy_fees_unrounded_usd: Decimal
    sell_fees_unrounded_usd: Decimal
    regulatory_round_trip_unrounded_usd: Decimal
    conservative_rounded_fee_floor_usd: Decimal

    @property
    def total_friction_unrounded_usd(self) -> Decimal:
        return self.quoted_round_trip_spread_usd + self.regulatory_round_trip_unrounded_usd


def estimate_long_option_round_trip(
    *,
    bid: Decimal,
    ask: Decimal,
    exit_premium: Decimal,
    contracts: int = 1,
    schedule: OptionFeeSchedule = OptionFeeSchedule(),
) -> FrictionEstimate:
    """Estimate a buy-at-ask/sell-at-bid round trip without predicting price movement.

    The conservative rounded floor applies Alpaca's daily per-fee-type cent
    rounding as if this were the account's only option round trip that day.
    """

    if bid < 0 or ask <= 0 or ask < bid or exit_premium < 0:
        raise ValueError("invalid_option_price_for_cost_model")
    if contracts <= 0:
        raise ValueError("contracts_must_be_positive")
    quantity = Decimal(contracts)
    multiplier = Decimal(schedule.contract_multiplier)
    equivalent_shares = quantity * multiplier
    buy_notional = ask * equivalent_shares
    sell_notional = exit_premium * equivalent_shares
    spread = (ask - bid) * equivalent_shares

    buy_components = (
        buy_notional * schedule.commission_rate,
        equivalent_shares * schedule.cat_per_equivalent_share,
        quantity * schedule.orf_per_contract,
        quantity * schedule.occ_per_contract,
    )
    sell_components = (
        sell_notional * schedule.commission_rate,
        sell_notional * schedule.sec_sell_rate_on_trade_value,
        quantity * schedule.taf_sell_per_contract,
        equivalent_shares * schedule.cat_per_equivalent_share,
        quantity * schedule.orf_per_contract,
        quantity * schedule.occ_per_contract,
    )
    buy_fees = sum(buy_components, start=Decimal("0"))
    sell_fees = sum(sell_components, start=Decimal("0"))

    daily_fee_type_totals = (
        (buy_notional + sell_notional) * schedule.commission_rate,
        sell_notional * schedule.sec_sell_rate_on_trade_value,
        quantity * schedule.taf_sell_per_contract,
        equivalent_shares * schedule.cat_per_equivalent_share * Decimal("2"),
        quantity * schedule.orf_per_contract * Decimal("2"),
        quantity * schedule.occ_per_contract * Decimal("2"),
    )
    rounded_floor = sum(
        (_round_up_cent(amount) for amount in daily_fee_type_totals if amount > 0),
        start=Decimal("0"),
    )
    return FrictionEstimate(
        contracts=contracts,
        quoted_round_trip_spread_usd=spread,
        buy_fees_unrounded_usd=buy_fees,
        sell_fees_unrounded_usd=sell_fees,
        regulatory_round_trip_unrounded_usd=buy_fees + sell_fees,
        conservative_rounded_fee_floor_usd=rounded_floor,
    )


def _round_up_cent(amount: Decimal) -> Decimal:
    return (amount * Decimal("100")).to_integral_value(rounding=ROUND_CEILING) / Decimal("100")

"""Read-only Alpaca market observations normalized for the risk kernel."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from math import floor
from typing import Callable
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .contracts import OptionContract, OptionRight
from .paper_account import AlpacaPaperCredentials, load_paper_credentials


DATA_API_BASE_URL = "https://data.alpaca.markets"
TRADING_API_BASE_URL = "https://paper-api.alpaca.markets"
ALLOWED_UNDERLYING = "SPY"
STOCK_FEED = "iex"
OPTION_FEED = "indicative"


@dataclass(frozen=True, slots=True)
class MarketClock:
    is_open: bool
    observed_at: datetime
    next_open: datetime
    next_close: datetime


@dataclass(frozen=True, slots=True)
class StockObservation:
    symbol: str
    latest_trade: Decimal
    bid: Decimal
    ask: Decimal
    quote_observed_at: datetime
    feed: str

    def __post_init__(self) -> None:
        if self.symbol != ALLOWED_UNDERLYING or self.feed != STOCK_FEED:
            raise ValueError("stock_observation_source_not_allowed")
        if self.latest_trade <= 0 or self.bid <= 0 or self.ask < self.bid:
            raise ValueError("invalid_stock_observation")
        if self.quote_observed_at.tzinfo is None or self.quote_observed_at.utcoffset() is None:
            raise ValueError("stock_quote_timestamp_must_be_timezone_aware")

    @property
    def spread_pct(self) -> Decimal:
        midpoint = (self.bid + self.ask) / Decimal("2")
        return (self.ask - self.bid) / midpoint


@dataclass(frozen=True, slots=True)
class OptionChainObservation:
    underlying: str
    expiration: date
    feed: str
    captured_at: datetime
    contracts: tuple[OptionContract, ...]


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("market_timestamp_missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("market_timestamp_must_be_timezone_aware")
    return parsed


def next_friday_in_window(
    observed_date: date,
    *,
    minimum_days: int = 2,
    maximum_days: int = 14,
) -> date:
    for days in range(minimum_days, maximum_days + 1):
        candidate = observed_date + timedelta(days=days)
        if candidate.weekday() == 4:
            return candidate
    raise RuntimeError("no_friday_in_expiry_window")


def normalize_option_chain(
    *,
    underlying: str,
    expiration: date,
    contract_payloads: list[dict[str, object]],
    snapshot_payloads: dict[str, object],
    captured_at: datetime,
) -> OptionChainObservation:
    if underlying != ALLOWED_UNDERLYING:
        raise ValueError("underlying_not_allowed")
    normalized: list[OptionContract] = []
    for metadata in contract_payloads:
        if metadata.get("underlying_symbol") not in {None, underlying}:
            continue
        if metadata.get("expiration_date") != expiration.isoformat():
            continue
        symbol = str(metadata.get("symbol", ""))
        snapshot = snapshot_payloads.get(symbol)
        if not isinstance(snapshot, dict):
            continue
        quote = snapshot.get("latestQuote")
        if not isinstance(quote, dict):
            continue
        bid = Decimal(str(quote.get("bp", "0")))
        ask = Decimal(str(quote.get("ap", "0")))
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        option_type = metadata.get("type")
        if option_type == "call":
            right = OptionRight.CALL
        elif option_type == "put":
            right = OptionRight.PUT
        else:
            continue
        normalized.append(
            OptionContract(
                symbol=symbol,
                underlying=underlying,
                right=right,
                strike=Decimal(str(metadata.get("strike_price"))),
                expiration=date.fromisoformat(str(metadata.get("expiration_date"))),
                bid=bid,
                ask=ask,
                open_interest=int(metadata.get("open_interest") or 0),
                volume=0,
                quote_observed_at=_parse_timestamp(quote.get("t")),
                tradable=bool(metadata.get("tradable"))
                and metadata.get("status") == "active",
            )
        )
    return OptionChainObservation(
        underlying=underlying,
        expiration=expiration,
        feed=OPTION_FEED,
        captured_at=captured_at,
        contracts=tuple(sorted(normalized, key=lambda item: item.symbol)),
    )


class AlpacaReadOnlyDataClient:
    def __init__(
        self,
        credentials: AlpacaPaperCredentials,
        *,
        opener: Callable[..., object] = urlopen,
    ) -> None:
        self._credentials = credentials
        self._opener = opener

    def _get_json(self, url: str) -> dict[str, object]:
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in {"data.alpaca.markets", "paper-api.alpaca.markets"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
        ):
            raise ValueError("alpaca_read_only_endpoint_required")
        request = Request(
            url,
            headers={
                "APCA-API-KEY-ID": self._credentials.api_key,
                "APCA-API-SECRET-KEY": self._credentials.secret_key,
                "Accept": "application/json",
            },
            method="GET",
        )
        with self._opener(request, timeout=20) as response:  # type: ignore[attr-defined]
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("invalid_alpaca_market_response")
        return payload

    def get_clock(self) -> MarketClock:
        payload = self._get_json(f"{TRADING_API_BASE_URL}/v2/clock")
        return MarketClock(
            is_open=bool(payload.get("is_open")),
            observed_at=_parse_timestamp(payload.get("timestamp")),
            next_open=_parse_timestamp(payload.get("next_open")),
            next_close=_parse_timestamp(payload.get("next_close")),
        )

    def get_stock_observation(self, symbol: str = ALLOWED_UNDERLYING) -> StockObservation:
        if symbol != ALLOWED_UNDERLYING:
            raise ValueError("underlying_not_allowed")
        query = urlencode({"feed": STOCK_FEED})
        payload = self._get_json(
            f"{DATA_API_BASE_URL}/v2/stocks/{symbol}/snapshot?{query}"
        )
        trade = payload.get("latestTrade")
        quote = payload.get("latestQuote")
        if not isinstance(trade, dict) or not isinstance(quote, dict):
            raise RuntimeError("incomplete_stock_snapshot")
        return StockObservation(
            symbol=symbol,
            latest_trade=Decimal(str(trade.get("p"))),
            bid=Decimal(str(quote.get("bp"))),
            ask=Decimal(str(quote.get("ap"))),
            quote_observed_at=_parse_timestamp(quote.get("t")),
            feed=STOCK_FEED,
        )

    def get_option_chain(
        self,
        *,
        expiration: date,
        strike_low: Decimal,
        strike_high: Decimal,
        captured_at: datetime | None = None,
    ) -> OptionChainObservation:
        if strike_low <= 0 or strike_high <= strike_low:
            raise ValueError("invalid_strike_window")
        shared = {
            "expiration_date": expiration.isoformat(),
            "strike_price_gte": str(strike_low),
            "strike_price_lte": str(strike_high),
            "limit": 1000,
        }
        contract_query = urlencode({"underlying_symbols": ALLOWED_UNDERLYING, **shared})
        snapshot_query = urlencode({"feed": OPTION_FEED, **shared})
        contracts_payload = self._get_json(
            f"{TRADING_API_BASE_URL}/v2/options/contracts?{contract_query}"
        )
        snapshots_payload = self._get_json(
            f"{DATA_API_BASE_URL}/v1beta1/options/snapshots/"
            f"{ALLOWED_UNDERLYING}?{snapshot_query}"
        )
        contracts = contracts_payload.get("option_contracts")
        snapshots = snapshots_payload.get("snapshots")
        if not isinstance(contracts, list) or not isinstance(snapshots, dict):
            raise RuntimeError("incomplete_option_chain")
        return normalize_option_chain(
            underlying=ALLOWED_UNDERLYING,
            expiration=expiration,
            contract_payloads=[item for item in contracts if isinstance(item, dict)],
            snapshot_payloads=snapshots,
            captured_at=captured_at or datetime.now(timezone.utc),
        )


def main() -> None:
    account_number = os.environ.get("AURORA_ALPACA_PAPER_ACCOUNT", "PA3HAW9279NN")
    client = AlpacaReadOnlyDataClient(load_paper_credentials(account_number))
    clock = client.get_clock()
    stock = client.get_stock_observation()
    expiration = next_friday_in_window(clock.observed_at.date())
    center = Decimal(floor(stock.latest_trade))
    chain = client.get_option_chain(
        expiration=expiration,
        strike_low=center - Decimal("15"),
        strike_high=center + Decimal("15"),
        captured_at=clock.observed_at,
    )
    viable = tuple(
        contract
        for contract in chain.contracts
        if contract.open_interest >= 100
        and contract.spread_pct <= Decimal("0.15")
        and contract.premium_risk_usd <= Decimal("500")
    )
    print(
        json.dumps(
            {
                "environment": "paper",
                "market_open": clock.is_open,
                "next_open": clock.next_open.isoformat(),
                "underlying": stock.symbol,
                "last_trade": str(stock.latest_trade),
                "stock_spread_pct": str(stock.spread_pct.quantize(Decimal("0.00000001"))),
                "stock_quote_at": stock.quote_observed_at.isoformat(),
                "option_feed": chain.feed,
                "expiration": chain.expiration.isoformat(),
                "quoted_contracts": len(chain.contracts),
                "prefilter_viable_contracts": len(viable),
                "decision": "NO_ACTION" if not clock.is_open else "OBSERVATION_READY",
                "reason": "market_closed" if not clock.is_open else "awaiting_fresh_thesis",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

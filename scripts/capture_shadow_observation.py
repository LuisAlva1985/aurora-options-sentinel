"""Capture a private, read-only forward observation; this script cannot place orders."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from math import floor
from pathlib import Path
from statistics import median

from aurora_sentinel.costs import estimate_long_option_round_trip
from aurora_sentinel.market_data import AlpacaReadOnlyDataClient, next_friday_in_window
from aurora_sentinel.model_gate import assess_model_evidence, load_model_evidence
from aurora_sentinel.paper_account import (
    AlpacaPaperAccountClient,
    load_paper_credentials,
    verify_competition_account,
)


ACCOUNT_NUMBER = "PA3HAW9279NN"
DEFAULT_MODEL_EVIDENCE = Path(
    "models-private/contest-iex-generation-2/model-artifact.json"
)


def main() -> None:
    captured_at = datetime.now(timezone.utc)
    credentials = load_paper_credentials(ACCOUNT_NUMBER)
    verified = verify_competition_account(
        AlpacaPaperAccountClient(credentials).get_account(),
        expected_account_number=ACCOUNT_NUMBER,
    )
    client = AlpacaReadOnlyDataClient(credentials)
    clock = client.get_clock()
    stock = client.get_stock_observation()
    expiration = next_friday_in_window(clock.observed_at.date())
    center = Decimal(floor(stock.latest_trade))
    chain = client.get_option_chain(
        expiration=expiration,
        strike_low=center - Decimal("15"),
        strike_high=center + Decimal("15"),
        captured_at=captured_at,
    )
    model_path = Path(
        os.environ.get("AURORA_MODEL_EVIDENCE_PATH", str(DEFAULT_MODEL_EVIDENCE))
    )
    evidence = load_model_evidence(model_path) if model_path.is_file() else None
    model_reasons = assess_model_evidence(evidence, evaluated_at=captured_at)
    viable = tuple(
        contract
        for contract in chain.contracts
        if contract.open_interest >= 100
        and contract.spread_pct <= Decimal("0.15")
        and contract.premium_risk_usd <= Decimal("500")
    )
    spread_values = [contract.spread_pct for contract in chain.contracts]
    viable_costs = [
        estimate_long_option_round_trip(
            bid=contract.bid,
            ask=contract.ask,
            exit_premium=contract.bid,
        ).total_friction_unrounded_usd
        for contract in viable
    ]
    quote_age_seconds = (captured_at - stock.quote_observed_at).total_seconds()
    reasons: list[str] = list(model_reasons)
    if not clock.is_open:
        reasons.append("market_closed")
    if quote_age_seconds < 0 or quote_age_seconds > 90:
        reasons.append("stock_quote_stale_or_future")
    if not viable:
        reasons.append("no_viable_option_contract")
    payload = {
        "schema_version": "aurora.forward-shadow-observation.v1",
        "captured_at": captured_at.isoformat(),
        "environment": "paper",
        "account_number": verified.account_number,
        "account_status": verified.status,
        "portfolio_value_usd": str(verified.portfolio_value_usd),
        "model_id": evidence.model_id if evidence else None,
        "model_evidence_id": evidence.evidence_id if evidence else None,
        "model_paper_eligible": evidence.paper_eligible if evidence else False,
        "market_open": clock.is_open,
        "next_open": clock.next_open.isoformat(),
        "stock": {
            "symbol": stock.symbol,
            "feed": stock.feed,
            "last_trade": str(stock.latest_trade),
            "bid": str(stock.bid),
            "ask": str(stock.ask),
            "spread_pct": str(stock.spread_pct),
            "quote_observed_at": stock.quote_observed_at.isoformat(),
            "quote_age_seconds": quote_age_seconds,
        },
        "options": {
            "feed": chain.feed,
            "expiration": expiration.isoformat(),
            "quoted_contracts": len(chain.contracts),
            "viable_contracts": len(viable),
            "median_quoted_spread_pct": str(median(spread_values)) if spread_values else None,
            "minimum_viable_round_trip_friction_usd": str(min(viable_costs)) if viable_costs else None,
            "median_viable_round_trip_friction_usd": str(median(viable_costs)) if viable_costs else None,
        },
        "decision": "NO_ACTION",
        "reason_codes": tuple(dict.fromkeys(reasons)),
        "order_prepared": False,
        "order_submitted": False,
    }
    output_root = Path("artifacts/forward-shadow")
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = captured_at.strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_root / f"{timestamp}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact": str(output_path.resolve()),
                "decision": payload["decision"],
                "reason_codes": payload["reason_codes"],
                "market_open": payload["market_open"],
                "quoted_contracts": payload["options"]["quoted_contracts"],
                "viable_contracts": payload["options"]["viable_contracts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

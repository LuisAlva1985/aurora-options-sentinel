"""Read-only Alpaca Paper account boundary with Windows credential storage."""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable
from urllib.request import Request, urlopen


PAPER_API_BASE_URL = "https://paper-api.alpaca.markets"
API_KEY_ENV = "APCA_API_KEY_ID"
SECRET_KEY_ENV = "APCA_API_SECRET_KEY"


@dataclass(frozen=True, slots=True)
class AlpacaPaperCredentials:
    account_number: str
    api_key: str
    secret_key: str
    base_url: str = PAPER_API_BASE_URL

    def __post_init__(self) -> None:
        if not self.account_number.startswith("PA"):
            raise ValueError("paper_account_number_required")
        if len(self.api_key) < 20 or len(self.secret_key) < 32:
            raise ValueError("alpaca_credentials_invalid")
        if self.base_url != PAPER_API_BASE_URL:
            raise ValueError("paper_endpoint_required")


@dataclass(frozen=True, slots=True)
class VerifiedPaperAccount:
    account_number: str
    status: str
    cash_usd: Decimal
    buying_power_usd: Decimal
    portfolio_value_usd: Decimal
    trading_blocked: bool
    account_blocked: bool


def credential_target(account_number: str, kind: str) -> str:
    if not account_number.startswith("PA"):
        raise ValueError("paper_account_number_required")
    if kind not in {"API_KEY", "SECRET_KEY"}:
        raise ValueError("unsupported_credential_kind")
    return f"AURORA/Alpaca/Paper/{account_number}/{kind}"


def _read_windows_generic_credential(target: str) -> str:
    if os.name != "nt":
        raise RuntimeError("windows_credential_manager_unavailable")

    from ctypes import wintypes

    class Credential(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    credential_pointer = ctypes.POINTER(Credential)()
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    cred_read = advapi32.CredReadW
    cred_read.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(Credential)),
    )
    cred_read.restype = wintypes.BOOL
    cred_free = advapi32.CredFree
    cred_free.argtypes = (ctypes.c_void_p,)
    cred_free.restype = None

    if not cred_read(target, 1, 0, ctypes.byref(credential_pointer)):
        raise OSError(ctypes.get_last_error(), f"credential_not_found:{target}")
    try:
        credential = credential_pointer.contents
        raw_value = ctypes.string_at(
            credential.CredentialBlob, credential.CredentialBlobSize
        )
        return raw_value.decode("utf-8")
    finally:
        cred_free(credential_pointer)


def load_paper_credentials(
    account_number: str,
    *,
    credential_reader: Callable[[str], str] | None = None,
) -> AlpacaPaperCredentials:
    """Load both secrets from environment or Windows Credential Manager.

    Environment variables are accepted for CI. A partial environment is rejected
    instead of silently mixing two credential sources.
    """

    env_api_key = os.environ.get(API_KEY_ENV)
    env_secret_key = os.environ.get(SECRET_KEY_ENV)
    if bool(env_api_key) != bool(env_secret_key):
        raise RuntimeError("partial_alpaca_environment_rejected")
    if env_api_key and env_secret_key:
        return AlpacaPaperCredentials(account_number, env_api_key, env_secret_key)

    reader = credential_reader or _read_windows_generic_credential
    api_key = reader(credential_target(account_number, "API_KEY"))
    secret_key = reader(credential_target(account_number, "SECRET_KEY"))
    return AlpacaPaperCredentials(account_number, api_key, secret_key)


class AlpacaPaperAccountClient:
    """Small read-only client whose URL cannot be changed to the live endpoint."""

    def __init__(
        self,
        credentials: AlpacaPaperCredentials,
        *,
        opener: Callable[..., object] = urlopen,
    ) -> None:
        self._credentials = credentials
        self._opener = opener

    def get_account(self) -> dict[str, object]:
        request = Request(
            f"{PAPER_API_BASE_URL}/v2/account",
            headers={
                "APCA-API-KEY-ID": self._credentials.api_key,
                "APCA-API-SECRET-KEY": self._credentials.secret_key,
                "Accept": "application/json",
            },
            method="GET",
        )
        with self._opener(request, timeout=15) as response:  # type: ignore[attr-defined]
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("invalid_alpaca_account_response")
        return payload


def verify_competition_account(
    payload: dict[str, object],
    *,
    expected_account_number: str,
    expected_starting_cash_usd: Decimal = Decimal("100000"),
) -> VerifiedPaperAccount:
    if payload.get("account_number") != expected_account_number:
        raise RuntimeError("unexpected_paper_account")
    if payload.get("status") != "ACTIVE":
        raise RuntimeError("paper_account_not_active")
    cash = Decimal(str(payload.get("cash")))
    portfolio_value = Decimal(str(payload.get("portfolio_value")))
    if cash != expected_starting_cash_usd or portfolio_value != expected_starting_cash_usd:
        raise RuntimeError("competition_starting_balance_mismatch")
    if bool(payload.get("trading_blocked")) or bool(payload.get("account_blocked")):
        raise RuntimeError("paper_account_blocked")
    return VerifiedPaperAccount(
        account_number=expected_account_number,
        status="ACTIVE",
        cash_usd=cash,
        buying_power_usd=Decimal(str(payload.get("buying_power"))),
        portfolio_value_usd=portfolio_value,
        trading_blocked=False,
        account_blocked=False,
    )


def main() -> None:
    account_number = os.environ.get("AURORA_ALPACA_PAPER_ACCOUNT", "PA3HAW9279NN")
    credentials = load_paper_credentials(account_number)
    client = AlpacaPaperAccountClient(credentials)
    verified = verify_competition_account(
        client.get_account(), expected_account_number=account_number
    )
    print(
        json.dumps(
            {
                "account_number": verified.account_number,
                "status": verified.status,
                "cash_usd": str(verified.cash_usd),
                "buying_power_usd": str(verified.buying_power_usd),
                "portfolio_value_usd": str(verified.portfolio_value_usd),
                "environment": "paper",
                "credentials": "windows_credential_manager",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

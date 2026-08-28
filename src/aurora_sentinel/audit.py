"""Tamper-evident, secret-rejecting audit trail for agent decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Mapping, Sequence

from .agent import AgentDecision


_GENESIS_HASH = "0" * 64
_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)


@dataclass(frozen=True, slots=True)
class AuditEvent:
    sequence: int
    emitted_at: datetime
    event_type: str
    payload_json: str
    previous_hash: str
    event_hash: str


class AuditTrail:
    """Append-only in-memory chain suitable for later JSONL persistence."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)

    def append(
        self,
        *,
        event_type: str,
        payload: Mapping[str, object],
        emitted_at: datetime,
    ) -> AuditEvent:
        if emitted_at.tzinfo is None or emitted_at.utcoffset() is None:
            raise ValueError("audit_timestamp_must_be_timezone_aware")
        if not event_type.strip():
            raise ValueError("audit_event_type_required")
        _reject_sensitive_keys(payload)
        payload_json = json.dumps(payload, cls=_AuditJsonEncoder, sort_keys=True, separators=(",", ":"))
        sequence = len(self._events) + 1
        previous_hash = self._events[-1].event_hash if self._events else _GENESIS_HASH
        event_hash = _event_hash(sequence, emitted_at, event_type, payload_json, previous_hash)
        event = AuditEvent(sequence, emitted_at, event_type, payload_json, previous_hash, event_hash)
        self._events.append(event)
        return event

    def record_decision(self, decision: AgentDecision, *, emitted_at: datetime) -> AuditEvent:
        payload: dict[str, object] = {
            "action": decision.action,
            "approved": decision.assessment.approved,
            "contract_symbol": decision.contract_symbol,
            "environment": decision.environment,
            "limit_price": decision.limit_price,
            "model_evidence_id": decision.model_evidence_id,
            "model_id": decision.model_id,
            "model_validated_for_paper": decision.model_validated_for_paper,
            "quantity": decision.quantity,
            "reason_codes": decision.assessment.reason_codes,
        }
        return self.append(event_type="AGENT_DECISION", payload=payload, emitted_at=emitted_at)


def verify_chain(events: Sequence[AuditEvent]) -> bool:
    previous_hash = _GENESIS_HASH
    for expected_sequence, event in enumerate(events, start=1):
        if event.sequence != expected_sequence or event.previous_hash != previous_hash:
            return False
        expected_hash = _event_hash(
            event.sequence,
            event.emitted_at,
            event.event_type,
            event.payload_json,
            event.previous_hash,
        )
        if event.event_hash != expected_hash:
            return False
        previous_hash = event.event_hash
    return True


def _event_hash(
    sequence: int,
    emitted_at: datetime,
    event_type: str,
    payload_json: str,
    previous_hash: str,
) -> str:
    canonical = "|".join(
        (str(sequence), emitted_at.isoformat(), event_type, payload_json, previous_hash)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reject_sensitive_keys(value: object, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS):
                raise ValueError(f"sensitive_audit_field_rejected:{path}.{key}")
            _reject_sensitive_keys(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_sensitive_keys(nested, path=f"{path}[{index}]")


class _AuditJsonEncoder(json.JSONEncoder):
    def default(self, value: object) -> object:
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, tuple):
            return list(value)
        return super().default(value)

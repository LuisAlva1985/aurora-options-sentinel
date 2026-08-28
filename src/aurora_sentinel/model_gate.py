"""Model provenance gate separating AURORA Core from the contest experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal


CONTEST_MODEL_TRACK = "CONTEST_EXPERIMENTAL_IEX"


@dataclass(frozen=True, slots=True)
class ModelEvidence:
    """Minimal immutable evidence required before a signal can reach Paper routing."""

    model_id: str
    track: str
    feature_set: str
    target: str
    trained_at: datetime
    validation_mae: Decimal
    validation_baseline_mae: Decimal
    test_mae: Decimal
    test_baseline_mae: Decimal
    validation_net_return_proxy: Decimal
    test_net_return_proxy: Decimal
    paper_eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.model_id) != 64 or any(c not in "0123456789abcdef" for c in self.model_id):
            raise ValueError("sha256_model_id_required")
        if self.trained_at.tzinfo is None or self.trained_at.utcoffset() is None:
            raise ValueError("model_timestamp_must_be_timezone_aware")
        if any(
            value < 0
            for value in (
                self.validation_mae,
                self.validation_baseline_mae,
                self.test_mae,
                self.test_baseline_mae,
            )
        ):
            raise ValueError("model_mae_must_be_non_negative")
        if not self.reason_codes:
            raise ValueError("model_reason_codes_required")

    @property
    def evidence_id(self) -> str:
        payload = {
            "feature_set": self.feature_set,
            "model_id": self.model_id,
            "paper_eligible": self.paper_eligible,
            "reason_codes": self.reason_codes,
            "target": self.target,
            "test_baseline_mae": str(self.test_baseline_mae),
            "test_mae": str(self.test_mae),
            "test_net_return_proxy": str(self.test_net_return_proxy),
            "track": self.track,
            "trained_at": self.trained_at.isoformat(),
            "validation_baseline_mae": str(self.validation_baseline_mae),
            "validation_mae": str(self.validation_mae),
            "validation_net_return_proxy": str(self.validation_net_return_proxy),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ClassifierModelEvidence:
    """Evidence for the cost-aware CALL/PUT/NO_ACTION contest classifier."""

    model_id: str
    track: str
    feature_set: str
    target: str
    trained_at: datetime
    validation_macro_f1: Decimal
    validation_baseline_macro_f1: Decimal
    test_macro_f1: Decimal
    test_baseline_macro_f1: Decimal
    validation_net_return_proxy: Decimal
    test_net_return_proxy: Decimal
    paper_eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.model_id) != 64 or any(c not in "0123456789abcdef" for c in self.model_id):
            raise ValueError("sha256_model_id_required")
        if self.trained_at.tzinfo is None or self.trained_at.utcoffset() is None:
            raise ValueError("model_timestamp_must_be_timezone_aware")
        scores = (
            self.validation_macro_f1,
            self.validation_baseline_macro_f1,
            self.test_macro_f1,
            self.test_baseline_macro_f1,
        )
        if any(value < 0 or value > 1 for value in scores):
            raise ValueError("classifier_score_out_of_range")
        if not self.reason_codes:
            raise ValueError("model_reason_codes_required")

    @property
    def evidence_id(self) -> str:
        payload = {
            "feature_set": self.feature_set,
            "model_id": self.model_id,
            "paper_eligible": self.paper_eligible,
            "reason_codes": self.reason_codes,
            "target": self.target,
            "test_baseline_macro_f1": str(self.test_baseline_macro_f1),
            "test_macro_f1": str(self.test_macro_f1),
            "test_net_return_proxy": str(self.test_net_return_proxy),
            "track": self.track,
            "trained_at": self.trained_at.isoformat(),
            "validation_baseline_macro_f1": str(self.validation_baseline_macro_f1),
            "validation_macro_f1": str(self.validation_macro_f1),
            "validation_net_return_proxy": str(self.validation_net_return_proxy),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assess_model_evidence(
    evidence: ModelEvidence | ClassifierModelEvidence | None,
    *,
    evaluated_at: datetime,
    maximum_artifact_age: timedelta = timedelta(days=14),
) -> tuple[str, ...]:
    """Return stable rejection reasons; an empty tuple is the only approval path."""

    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise ValueError("evaluation_timestamp_must_be_timezone_aware")
    if evidence is None:
        return ("model_evidence_missing",)
    reasons: list[str] = []
    if evidence.track != CONTEST_MODEL_TRACK:
        reasons.append("model_track_not_competition_iex")
    if evidence.feature_set != "MARKET_CORE_M15_V1":
        reasons.append("model_feature_set_mismatch")
    if evidence.target not in (
        "forward_close_return_4",
        "forward_close_return_4_cost_aware_class",
    ):
        reasons.append("model_target_mismatch")
    age = evaluated_at - evidence.trained_at
    if age < timedelta(0) or age > maximum_artifact_age:
        reasons.append("model_artifact_stale_or_future")
    if isinstance(evidence, ModelEvidence):
        if evidence.validation_mae >= evidence.validation_baseline_mae:
            reasons.append("validation_did_not_beat_baseline")
        if evidence.test_mae >= evidence.test_baseline_mae:
            reasons.append("test_did_not_beat_baseline")
    else:
        if evidence.validation_macro_f1 <= evidence.validation_baseline_macro_f1:
            reasons.append("validation_macro_f1_did_not_beat_majority")
        if evidence.test_macro_f1 <= evidence.test_baseline_macro_f1:
            reasons.append("sealed_test_macro_f1_did_not_beat_majority")
    if evidence.validation_net_return_proxy <= 0:
        reasons.append("validation_net_return_proxy_not_positive")
    if evidence.test_net_return_proxy <= 0:
        reasons.append("test_net_return_proxy_not_positive")
    if not evidence.paper_eligible:
        reasons.append("model_not_paper_eligible")
    return tuple(dict.fromkeys(reasons))


def load_model_evidence(path: Path) -> ModelEvidence | ClassifierModelEvidence:
    """Load only the aggregate evidence needed by the order gate, never the estimator."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version not in (
        "aurora.contest-iex-model.v1",
        "aurora.contest-iex-classifier.v1",
    ):
        raise ValueError("contest_model_schema_required")
    if payload.get("core_model_evidence") is not False:
        raise ValueError("contest_model_must_not_claim_core_evidence")
    if payload.get("paper_only") is not True or payload.get("live_trading_allowed") is not False:
        raise ValueError("paper_only_model_required")
    validation = payload.get("validation")
    test = payload.get("test") if schema_version == "aurora.contest-iex-model.v1" else payload.get("sealed_test")
    if not isinstance(validation, dict) or not isinstance(test, dict):
        raise ValueError("model_evaluation_evidence_required")
    shared = {
        "model_id": str(payload["model_id"]),
        "track": str(payload["track"]),
        "feature_set": str(payload["feature_set"]),
        "target": str(payload["target"]),
        "trained_at": datetime.fromisoformat(str(payload["trained_at"])),
        "validation_net_return_proxy": Decimal(str(validation["net_return_proxy"])),
        "test_net_return_proxy": Decimal(str(test["net_return_proxy"])),
        "paper_eligible": payload.get("paper_eligible") is True,
        "reason_codes": tuple(str(code) for code in payload.get("reason_codes", ())),
    }
    if schema_version == "aurora.contest-iex-classifier.v1":
        return ClassifierModelEvidence(
            **shared,
            validation_macro_f1=Decimal(str(validation["macro_f1"])),
            validation_baseline_macro_f1=Decimal(str(validation["majority_macro_f1"])),
            test_macro_f1=Decimal(str(test["macro_f1"])),
            test_baseline_macro_f1=Decimal(str(test["majority_macro_f1"])),
        )
    return ModelEvidence(
        **shared,
        validation_mae=Decimal(str(validation["mae"])),
        validation_baseline_mae=Decimal(str(validation["baseline_mae"])),
        test_mae=Decimal(str(test["mae"])),
        test_baseline_mae=Decimal(str(test["baseline_mae"])),
    )

"""Dependency-free temporal trainer for the isolated IEX competition track.

This module deliberately does not import, read, or identify the private SIP dataset used
by AURORA Core.  The 2025 calendar year is a hard blackout in the companion runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import fmean
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo


FEATURE_SET = "MARKET_CORE_M15_V1"
TARGET = "forward_close_return_4"
FEATURE_NAMES = (
    "close_return_1",
    "close_return_4",
    "range_pct_1",
    "body_pct_1",
    "close_vs_sma_4",
    "realized_vol_4",
)
_NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class Example:
    as_of: datetime
    features: tuple[float, ...]
    target: float


@dataclass(frozen=True, slots=True)
class RidgeModel:
    l2: float
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    intercept: float
    weights: tuple[float, ...]

    def predict_one(self, features: Sequence[float]) -> float:
        standardized = (
            (value - mean) / scale
            for value, mean, scale in zip(
                features, self.feature_means, self.feature_scales, strict=True
            )
        )
        return self.intercept + sum(
            weight * value for weight, value in zip(self.weights, standardized, strict=True)
        )


@dataclass(frozen=True, slots=True)
class Evaluation:
    mae: float
    baseline_mae: float
    net_return_proxy: float
    gross_return_proxy: float
    maximum_drawdown_proxy: float
    trades: int
    threshold: float


@dataclass(frozen=True, slots=True)
class TrainingRun:
    model: RidgeModel
    model_id: str
    selected_l2: float
    selected_threshold: float
    train_count: int
    validation_count: int
    test_count: int
    validation: Evaluation
    test: Evaluation
    paper_eligible: bool
    reason_codes: tuple[str, ...]

    def as_dict(self, *, trained_at: datetime, source_sha256: str) -> dict[str, object]:
        return {
            "schema_version": "aurora.contest-iex-model.v1",
            "track": "CONTEST_EXPERIMENTAL_IEX",
            "core_model_evidence": False,
            "paper_only": True,
            "live_trading_allowed": False,
            "model_id": self.model_id,
            "trained_at": trained_at.isoformat(),
            "source_sha256": source_sha256,
            "source_feed": "iex",
            "source_window": {"start": "2021-01-01", "end_exclusive": "2025-01-01"},
            "core_holdout_2025_blackout": True,
            "feature_set": FEATURE_SET,
            "feature_names": FEATURE_NAMES,
            "target": TARGET,
            "target_horizon_bars": 4,
            "split": {
                "train": "2021-01-01/2023-01-01",
                "validation": "2023-01-01/2024-01-01",
                "test": "2024-01-01/2025-01-01",
                "purge_boundary_targets": True,
            },
            "model": {
                "family": "RIDGE_LINEAR_REGRESSION",
                "selected_l2": self.selected_l2,
                "selected_signal_threshold": self.selected_threshold,
                "feature_means": self.model.feature_means,
                "feature_scales": self.model.feature_scales,
                "intercept": self.model.intercept,
                "weights": self.model.weights,
            },
            "counts": {
                "train": self.train_count,
                "validation": self.validation_count,
                "test": self.test_count,
            },
            "validation": _evaluation_dict(self.validation),
            "test": _evaluation_dict(self.test),
            "economic_proxy": {
                "round_trip_cost_bps": 2.0,
                "instrument": "SPY_UNDERLYING_PROXY",
                "overlap": "NON_OVERLAPPING_4_BAR_SIGNALS",
                "status": "UNVERIFIED_PROXY_NOT_OPTIONS_PNL",
            },
            "paper_eligible": self.paper_eligible,
            "reason_codes": self.reason_codes,
        }


def parse_alpaca_bars(payloads: Iterable[dict[str, object]]) -> tuple[Bar, ...]:
    bars: list[Bar] = []
    for item in payloads:
        timestamp = datetime.fromisoformat(str(item["t"]).replace("Z", "+00:00"))
        bar = Bar(
            timestamp=timestamp,
            open=float(item["o"]),
            high=float(item["h"]),
            low=float(item["l"]),
            close=float(item["c"]),
            volume=float(item["v"]),
        )
        if min(bar.open, bar.high, bar.low, bar.close) <= 0:
            raise ValueError("non_positive_bar_price")
        bars.append(bar)
    return tuple(sorted(bars, key=lambda bar: bar.timestamp))


def build_examples(bars: Sequence[Bar]) -> tuple[Example, ...]:
    """Build point-in-time features and a four-bar future target within each session."""

    sessions: dict[str, list[Bar]] = {}
    for bar in bars:
        local = bar.timestamp.astimezone(_NEW_YORK)
        if local.weekday() >= 5 or not (9 * 60 + 30 <= local.hour * 60 + local.minute < 16 * 60):
            continue
        sessions.setdefault(local.date().isoformat(), []).append(bar)

    examples: list[Example] = []
    for session_key in sorted(sessions):
        session = sorted(sessions[session_key], key=lambda bar: bar.timestamp)
        for index in range(4, len(session) - 4):
            current = session[index]
            previous = session[index - 1]
            lag4 = session[index - 4]
            future = session[index + 4]
            if future.timestamp - current.timestamp != timedelta(minutes=60):
                continue
            closes4 = [bar.close for bar in session[index - 3 : index + 1]]
            returns4 = [
                session[j].close / session[j - 1].close - 1.0
                for j in range(index - 3, index + 1)
            ]
            mean_return = fmean(returns4)
            realized_vol = math.sqrt(fmean((value - mean_return) ** 2 for value in returns4))
            features = (
                current.close / previous.close - 1.0,
                current.close / lag4.close - 1.0,
                (current.high - current.low) / current.close,
                (current.close - current.open) / current.open,
                current.close / fmean(closes4) - 1.0,
                realized_vol,
            )
            if all(math.isfinite(value) for value in features):
                examples.append(
                    Example(
                        as_of=current.timestamp,
                        features=features,
                        target=future.close / current.close - 1.0,
                    )
                )
    return tuple(examples)


def train_temporal_ridge(examples: Sequence[Example]) -> TrainingRun:
    train = tuple(item for item in examples if item.as_of.year in (2021, 2022))
    validation = tuple(item for item in examples if item.as_of.year == 2023)
    test = tuple(item for item in examples if item.as_of.year == 2024)
    if min(len(train), len(validation), len(test)) < 500:
        raise ValueError("insufficient_temporal_examples")

    baseline = fmean(item.target for item in train)
    candidates = (0.000001, 0.001, 0.01, 0.1, 1.0)
    fitted = tuple(fit_ridge(train, l2=value) for value in candidates)
    model = min(
        fitted,
        key=lambda candidate: (
            _mae(validation, candidate),
            candidate.l2,
        ),
    )
    thresholds = (0.0, 0.00025, 0.0005, 0.001, 0.0015, 0.002)
    threshold = max(
        thresholds,
        key=lambda value: (
            _economic_proxy(validation, model, threshold=value, cost=0.0002)[0],
            -value,
        ),
    )
    validation_result = evaluate(
        validation, model, baseline=baseline, threshold=threshold, cost=0.0002
    )
    test_result = evaluate(test, model, baseline=baseline, threshold=threshold, cost=0.0002)
    reasons: list[str] = []
    if validation_result.mae >= validation_result.baseline_mae:
        reasons.append("validation_did_not_beat_baseline")
    if test_result.mae >= test_result.baseline_mae:
        reasons.append("test_did_not_beat_baseline")
    if validation_result.net_return_proxy <= 0:
        reasons.append("validation_net_return_proxy_not_positive")
    if test_result.net_return_proxy <= 0:
        reasons.append("test_net_return_proxy_not_positive")
    if min(validation_result.trades, test_result.trades) < 30:
        reasons.append("insufficient_non_overlapping_trades")
    if not reasons:
        reasons.append("predictive_and_proxy_economic_gates_passed")
    paper_eligible = reasons == ["predictive_and_proxy_economic_gates_passed"]
    model_id = _model_id(model, threshold=threshold)
    return TrainingRun(
        model=model,
        model_id=model_id,
        selected_l2=model.l2,
        selected_threshold=threshold,
        train_count=len(train),
        validation_count=len(validation),
        test_count=len(test),
        validation=validation_result,
        test=test_result,
        paper_eligible=paper_eligible,
        reason_codes=tuple(reasons),
    )


def fit_ridge(examples: Sequence[Example], *, l2: float) -> RidgeModel:
    if not examples or l2 < 0:
        raise ValueError("invalid_ridge_fit_request")
    width = len(examples[0].features)
    means = tuple(fmean(item.features[j] for item in examples) for j in range(width))
    scales = tuple(
        max(
            math.sqrt(fmean((item.features[j] - means[j]) ** 2 for item in examples)),
            1e-12,
        )
        for j in range(width)
    )
    dimension = width + 1
    gram = [[0.0] * dimension for _ in range(dimension)]
    rhs = [0.0] * dimension
    for item in examples:
        row = [1.0] + [
            (item.features[j] - means[j]) / scales[j] for j in range(width)
        ]
        for i in range(dimension):
            rhs[i] += row[i] * item.target
            for j in range(dimension):
                gram[i][j] += row[i] * row[j]
    for index in range(1, dimension):
        gram[index][index] += l2
    coefficients = _solve_linear_system(gram, rhs)
    return RidgeModel(l2, means, scales, coefficients[0], tuple(coefficients[1:]))


def evaluate(
    examples: Sequence[Example],
    model: RidgeModel,
    *,
    baseline: float,
    threshold: float,
    cost: float,
) -> Evaluation:
    net, gross, drawdown, trades = _economic_proxy(
        examples, model, threshold=threshold, cost=cost
    )
    return Evaluation(
        mae=_mae(examples, model),
        baseline_mae=fmean(abs(item.target - baseline) for item in examples),
        net_return_proxy=net,
        gross_return_proxy=gross,
        maximum_drawdown_proxy=drawdown,
        trades=trades,
        threshold=threshold,
    )


def _economic_proxy(
    examples: Sequence[Example],
    model: RidgeModel,
    *,
    threshold: float,
    cost: float,
) -> tuple[float, float, float, int]:
    gross = 0.0
    net = 0.0
    peak = 0.0
    drawdown = 0.0
    trades = 0
    next_allowed = datetime.min.replace(tzinfo=timezone.utc)
    for item in sorted(examples, key=lambda value: value.as_of):
        if item.as_of < next_allowed:
            continue
        prediction = model.predict_one(item.features)
        if abs(prediction) < threshold or prediction == 0:
            continue
        outcome = item.target if prediction > 0 else -item.target
        gross += outcome
        net += outcome - cost
        trades += 1
        peak = max(peak, net)
        drawdown = min(drawdown, net - peak)
        next_allowed = item.as_of + timedelta(minutes=60)
    return net, gross, abs(drawdown), trades


def _mae(examples: Sequence[Example], model: RidgeModel) -> float:
    return fmean(abs(item.target - model.predict_one(item.features)) for item in examples)


def _model_id(model: RidgeModel, *, threshold: float) -> str:
    payload = {
        "family": "RIDGE_LINEAR_REGRESSION",
        "feature_means": model.feature_means,
        "feature_scales": model.feature_scales,
        "feature_set": FEATURE_SET,
        "intercept": model.intercept,
        "l2": model.l2,
        "target": TARGET,
        "threshold": threshold,
        "weights": model.weights,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float]:
    size = len(values)
    augmented = [row[:] + [value] for row, value in zip(matrix, values, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-18:
            raise ValueError("singular_ridge_system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[row][-1] for row in range(size)]


def _evaluation_dict(value: Evaluation) -> dict[str, object]:
    return {
        "mae": value.mae,
        "baseline_mae": value.baseline_mae,
        "net_return_proxy": value.net_return_proxy,
        "gross_return_proxy": value.gross_return_proxy,
        "maximum_drawdown_proxy": value.maximum_drawdown_proxy,
        "trades": value.trades,
        "threshold": value.threshold,
    }

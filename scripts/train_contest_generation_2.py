"""Execute the frozen generation-2 contest experiment exactly once."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean

from aurora_sentinel.experimental_model import build_examples, parse_alpaca_bars
from aurora_sentinel.paper_account import load_paper_credentials


ACCOUNT_NUMBER = "PA3HAW9279NN"
DATA_URL = "https://data.alpaca.markets/v2/stocks/SPY/bars"
DEVELOPMENT_DATA = Path("models-private/contest-iex/spy-iex-m15-2021-2024.json")
OUTPUT_ROOT = Path("models-private/contest-iex-generation-2")
TEST_START = "2026-01-01T00:00:00Z"
TEST_END_EXCLUSIVE = "2026-08-27T00:00:00Z"


def fetch_sealed_test() -> list[dict[str, object]]:
    credentials = load_paper_credentials(ACCOUNT_NUMBER)
    bars: list[dict[str, object]] = []
    page_token: str | None = None
    while True:
        parameters = {
            "timeframe": "15Min",
            "feed": "iex",
            "adjustment": "raw",
            "start": TEST_START,
            "end": TEST_END_EXCLUSIVE,
            "limit": "10000",
        }
        if page_token:
            parameters["page_token"] = page_token
        request = urllib.request.Request(
            DATA_URL + "?" + urllib.parse.urlencode(parameters),
            headers={
                "APCA-API-KEY-ID": credentials.api_key,
                "APCA-API-SECRET-KEY": credentials.secret_key,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        page = payload.get("bars") or []
        if not isinstance(page, list):
            raise RuntimeError("invalid_alpaca_bars_response")
        bars.extend(page)
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    if any(str(item.get("t", ""))[:4] == "2025" for item in bars):
        raise RuntimeError("core_holdout_blackout_violated")
    return bars


def labels(examples: list[object]) -> list[int]:
    return [1 if item.target > 0.0005 else -1 if item.target < -0.0005 else 0 for item in examples]


def macro_f1(actual: list[int], predicted: list[int]) -> float:
    scores: list[float] = []
    for category in (-1, 0, 1):
        true_positive = sum(a == category and p == category for a, p in zip(actual, predicted, strict=True))
        false_positive = sum(a != category and p == category for a, p in zip(actual, predicted, strict=True))
        false_negative = sum(a == category and p != category for a, p in zip(actual, predicted, strict=True))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return fmean(scores)


def majority_f1(actual: list[int], majority: int) -> float:
    return macro_f1(actual, [majority] * len(actual))


def decisions_from_probabilities(
    probabilities: list[list[float]], classes: list[int], *, threshold: float
) -> list[int]:
    decisions: list[int] = []
    for row in probabilities:
        index = max(range(len(row)), key=lambda value: row[value])
        category = classes[index]
        decisions.append(category if category != 0 and row[index] >= threshold else 0)
    return decisions


def economic_proxy(
    examples: list[object], decisions: list[int], *, cost: float = 0.0002
) -> dict[str, object]:
    next_allowed = datetime.min.replace(tzinfo=timezone.utc)
    gross = 0.0
    net = 0.0
    peak = 0.0
    drawdown = 0.0
    trades = 0
    for item, decision in sorted(
        zip(examples, decisions, strict=True), key=lambda pair: pair[0].as_of
    ):
        if decision == 0 or item.as_of < next_allowed:
            continue
        outcome = item.target if decision > 0 else -item.target
        gross += outcome
        net += outcome - cost
        trades += 1
        peak = max(peak, net)
        drawdown = min(drawdown, net - peak)
        next_allowed = item.as_of + timedelta(minutes=60)
    return {
        "gross_return_proxy": gross,
        "net_return_proxy": net,
        "maximum_drawdown_proxy": abs(drawdown),
        "trades": trades,
    }


def main() -> int:
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError("scientific_training_dependencies_missing") from exc

    if not DEVELOPMENT_DATA.is_file():
        raise RuntimeError("generation_1_private_development_data_missing")
    development_payload = json.loads(DEVELOPMENT_DATA.read_text(encoding="utf-8"))
    development = build_examples(parse_alpaca_bars(development_payload))
    train = [item for item in development if item.as_of.year in (2021, 2022, 2023)]
    validation = [item for item in development if item.as_of.year == 2024]
    x_train = np.asarray([item.features for item in train], dtype=float)
    y_train = np.asarray(labels(train), dtype=int)
    x_validation = np.asarray([item.features for item in validation], dtype=float)
    y_validation = labels(validation)
    majority = max((-1, 0, 1), key=lambda category: (sum(y_train == category), category))

    candidates: list[tuple[str, dict[str, object], object]] = []
    for config in (
        {"n_estimators": 150, "max_depth": 5, "min_samples_leaf": 40, "max_features": 1.0},
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 30, "max_features": 0.75},
        {"n_estimators": 250, "max_depth": 12, "min_samples_leaf": 20, "max_features": 0.75},
    ):
        candidates.append(
            (
                "RANDOM_FOREST_CLASSIFIER",
                config,
                RandomForestClassifier(
                    **config,
                    class_weight="balanced_subsample",
                    random_state=1729,
                    n_jobs=-1,
                ),
            )
        )
    for config in (
        {"max_iter": 100, "max_leaf_nodes": 7, "learning_rate": 0.05, "l2_regularization": 0.1},
        {"max_iter": 150, "max_leaf_nodes": 15, "learning_rate": 0.05, "l2_regularization": 1.0},
        {"max_iter": 200, "max_leaf_nodes": 31, "learning_rate": 0.03, "l2_regularization": 3.0},
    ):
        candidates.append(
            (
                "HIST_GRADIENT_BOOSTING_CLASSIFIER",
                config,
                HistGradientBoostingClassifier(**config, random_state=1729),
            )
        )

    reviewed: list[tuple[float, float, str, int, float, object, dict[str, object]]] = []
    candidate_report: list[dict[str, object]] = []
    thresholds = (0.45, 0.5, 0.55, 0.6)
    for index, (family, config, estimator) in enumerate(candidates):
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_validation).tolist()
        classes = [int(value) for value in estimator.classes_.tolist()]
        best: tuple[float, float, float, dict[str, object], list[int]] | None = None
        for threshold in thresholds:
            decisions = decisions_from_probabilities(probabilities, classes, threshold=threshold)
            proxy = economic_proxy(validation, decisions)
            if int(proxy["trades"]) < 100:
                continue
            score = macro_f1(y_validation, decisions)
            candidate = (float(proxy["net_return_proxy"]), score, -threshold, proxy, decisions)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        if best is None:
            threshold = 0.45
            decisions = decisions_from_probabilities(probabilities, classes, threshold=threshold)
            proxy = economic_proxy(validation, decisions)
            score = macro_f1(y_validation, decisions)
        else:
            _, score, negative_threshold, proxy, decisions = best
            threshold = -negative_threshold
        report = {
            "family": family,
            "configuration": config,
            "threshold": threshold,
            "macro_f1": score,
            "majority_macro_f1": majority_f1(y_validation, majority),
            **proxy,
        }
        candidate_report.append(report)
        reviewed.append(
            (
                float(proxy["net_return_proxy"]),
                score,
                family,
                -index,
                threshold,
                estimator,
                report,
            )
        )
    selected = max(reviewed, key=lambda item: item[:4])
    _, _, selected_family, _, selected_threshold, estimator, validation_report = selected

    sealed_bars = fetch_sealed_test()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sealed_canonical = json.dumps(sealed_bars, sort_keys=True, separators=(",", ":"))
    (OUTPUT_ROOT / "spy-iex-m15-2026-sealed-test.json").write_text(
        sealed_canonical, encoding="utf-8"
    )
    sealed_examples = list(build_examples(parse_alpaca_bars(sealed_bars)))
    x_sealed = np.asarray([item.features for item in sealed_examples], dtype=float)
    y_sealed = labels(sealed_examples)
    probabilities = estimator.predict_proba(x_sealed).tolist()
    classes = [int(value) for value in estimator.classes_.tolist()]
    sealed_decisions = decisions_from_probabilities(
        probabilities, classes, threshold=selected_threshold
    )
    sealed_report = {
        "macro_f1": macro_f1(y_sealed, sealed_decisions),
        "majority_macro_f1": majority_f1(y_sealed, majority),
        **economic_proxy(sealed_examples, sealed_decisions),
    }
    reasons: list[str] = []
    if float(validation_report["net_return_proxy"]) <= 0:
        reasons.append("validation_net_return_proxy_not_positive")
    if float(sealed_report["net_return_proxy"]) <= 0:
        reasons.append("sealed_test_net_return_proxy_not_positive")
    if float(validation_report["macro_f1"]) <= float(validation_report["majority_macro_f1"]):
        reasons.append("validation_macro_f1_did_not_beat_majority")
    if float(sealed_report["macro_f1"]) <= float(sealed_report["majority_macro_f1"]):
        reasons.append("sealed_test_macro_f1_did_not_beat_majority")
    if int(validation_report["trades"]) < 100:
        reasons.append("insufficient_validation_decisions")
    if int(sealed_report["trades"]) < 100:
        reasons.append("insufficient_sealed_test_decisions")
    if float(sealed_report["maximum_drawdown_proxy"]) > 0.15:
        reasons.append("sealed_test_drawdown_above_limit")
    if not reasons:
        reasons.append("generation_2_all_gates_passed")
    model_path = OUTPUT_ROOT / "classifier.joblib"
    joblib.dump(estimator, model_path)
    model_id = hashlib.sha256(model_path.read_bytes()).hexdigest()
    artifact = {
        "schema_version": "aurora.contest-iex-classifier.v1",
        "generation": 2,
        "track": "CONTEST_EXPERIMENTAL_IEX",
        "core_model_evidence": False,
        "paper_only": True,
        "live_trading_allowed": False,
        "model_id": model_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_set": "MARKET_CORE_M15_V1",
        "target": "forward_close_return_4_cost_aware_class",
        "core_holdout_2025_blackout": True,
        "selected_family": selected_family,
        "selected_threshold": selected_threshold,
        "candidate_report": candidate_report,
        "validation": validation_report,
        "sealed_test": sealed_report,
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "sealed_test": len(sealed_examples),
            "sealed_test_raw_bars": len(sealed_bars),
        },
        "sealed_test_sha256": hashlib.sha256(sealed_canonical.encode("utf-8")).hexdigest(),
        "sealed_test_openings_used": 1,
        "paper_eligible": reasons == ["generation_2_all_gates_passed"],
        "reason_codes": reasons,
    }
    artifact_path = OUTPUT_ROOT / "model-artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_path": str(artifact_path.resolve()),
                "model_id": model_id,
                "selected_family": selected_family,
                "selected_threshold": selected_threshold,
                "paper_eligible": artifact["paper_eligible"],
                "reason_codes": reasons,
                "validation": validation_report,
                "sealed_test": sealed_report,
                "counts": artifact["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(artifact["paper_eligible"]) else 2


if __name__ == "__main__":
    sys.exit(main())

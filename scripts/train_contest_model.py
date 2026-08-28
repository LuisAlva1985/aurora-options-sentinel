"""Acquire authorized IEX bars and train the isolated contest model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean

from aurora_sentinel.experimental_model import (
    build_examples,
    parse_alpaca_bars,
    train_temporal_ridge,
)
from aurora_sentinel.paper_account import load_paper_credentials


DATA_URL = "https://data.alpaca.markets/v2/stocks/SPY/bars"
ACCOUNT_NUMBER = "PA3HAW9279NN"
START = "2021-01-01T00:00:00Z"
END_EXCLUSIVE = "2025-01-01T00:00:00Z"


def _fetch() -> list[dict[str, object]]:
    credentials = load_paper_credentials(ACCOUNT_NUMBER)
    bars: list[dict[str, object]] = []
    page_token: str | None = None
    while True:
        parameters = {
            "timeframe": "15Min",
            "feed": "iex",
            "adjustment": "raw",
            "start": START,
            "end": END_EXCLUSIVE,
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


def _proxy(
    examples: list[object], predictions: list[float], *, threshold: float, cost: float
) -> dict[str, object]:
    next_allowed = datetime.min.replace(tzinfo=timezone.utc)
    gross = 0.0
    net = 0.0
    peak = 0.0
    maximum_drawdown = 0.0
    trades = 0
    for item, prediction in sorted(
        zip(examples, predictions, strict=True), key=lambda pair: pair[0].as_of
    ):
        if item.as_of < next_allowed or abs(prediction) < threshold or prediction == 0:
            continue
        outcome = item.target if prediction > 0 else -item.target
        gross += outcome
        net += outcome - cost
        trades += 1
        peak = max(peak, net)
        maximum_drawdown = min(maximum_drawdown, net - peak)
        next_allowed = item.as_of + timedelta(minutes=60)
    return {
        "gross_return_proxy": gross,
        "net_return_proxy": net,
        "maximum_drawdown_proxy": abs(maximum_drawdown),
        "trades": trades,
        "threshold": threshold,
    }


def _train_random_forest(examples: tuple[object, ...], output_root: Path) -> dict[str, object]:
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise RuntimeError("scientific_training_dependencies_missing") from exc

    train = [item for item in examples if item.as_of.year in (2021, 2022)]
    validation = [item for item in examples if item.as_of.year == 2023]
    test = [item for item in examples if item.as_of.year == 2024]
    x_train = np.asarray([item.features for item in train], dtype=float)
    y_train = np.asarray([item.target for item in train], dtype=float)
    x_validation = np.asarray([item.features for item in validation], dtype=float)
    y_validation = np.asarray([item.target for item in validation], dtype=float)
    x_test = np.asarray([item.features for item in test], dtype=float)
    y_test = np.asarray([item.target for item in test], dtype=float)
    baseline = float(np.mean(y_train))
    configurations = (
        {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 50, "max_features": 1.0},
        {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 50, "max_features": 1.0},
        {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 50, "max_features": 1.0},
        {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 20, "max_features": 0.75},
        {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 20, "max_features": 0.75},
    )
    candidates: list[tuple[float, int, object, list[float]]] = []
    candidate_report: list[dict[str, object]] = []
    for index, configuration in enumerate(configurations):
        estimator = RandomForestRegressor(
            **configuration,
            criterion="squared_error",
            random_state=1729,
            n_jobs=-1,
        )
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_validation).tolist()
        mae = fmean(abs(item.target - prediction) for item, prediction in zip(validation, predictions, strict=True))
        candidates.append((mae, index, estimator, predictions))
        candidate_report.append({"configuration": configuration, "validation_mae": mae})
    validation_mae, selected_index, estimator, validation_predictions = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    threshold_candidates: list[tuple[float, float, dict[str, object]]] = []
    for threshold in (0.0, 0.00025, 0.0005, 0.001, 0.0015, 0.002):
        result = _proxy(validation, validation_predictions, threshold=threshold, cost=0.0002)
        if int(result["trades"]) >= 30:
            threshold_candidates.append((float(result["net_return_proxy"]), threshold, result))
    if threshold_candidates:
        _, threshold, validation_proxy = max(
            threshold_candidates, key=lambda item: (item[0], -item[1])
        )
    else:
        threshold = 0.0
        validation_proxy = _proxy(
            validation, validation_predictions, threshold=threshold, cost=0.0002
        )
    test_predictions = estimator.predict(x_test).tolist()
    test_mae = fmean(
        abs(item.target - prediction)
        for item, prediction in zip(test, test_predictions, strict=True)
    )
    test_proxy = _proxy(test, test_predictions, threshold=threshold, cost=0.0002)
    validation_baseline_mae = fmean(abs(item.target - baseline) for item in validation)
    test_baseline_mae = fmean(abs(item.target - baseline) for item in test)
    reasons: list[str] = []
    if validation_mae >= validation_baseline_mae:
        reasons.append("validation_did_not_beat_baseline")
    if test_mae >= test_baseline_mae:
        reasons.append("test_did_not_beat_baseline")
    if float(validation_proxy["net_return_proxy"]) <= 0:
        reasons.append("validation_net_return_proxy_not_positive")
    if float(test_proxy["net_return_proxy"]) <= 0:
        reasons.append("test_net_return_proxy_not_positive")
    if min(int(validation_proxy["trades"]), int(test_proxy["trades"])) < 30:
        reasons.append("insufficient_non_overlapping_trades")
    if not reasons:
        reasons.append("predictive_and_proxy_economic_gates_passed")
    model_path = output_root / "random-forest.joblib"
    joblib.dump(estimator, model_path)
    model_id = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return {
        "model_id": model_id,
        "model_path": str(model_path.resolve()),
        "family": "RANDOM_FOREST_REGRESSOR",
        "selected_configuration": configurations[selected_index],
        "selected_signal_threshold": threshold,
        "candidate_report": candidate_report,
        "validation": {
            "mae": validation_mae,
            "baseline_mae": validation_baseline_mae,
            **validation_proxy,
        },
        "test": {"mae": test_mae, "baseline_mae": test_baseline_mae, **test_proxy},
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "paper_eligible": reasons == ["predictive_and_proxy_economic_gates_passed"],
        "reason_codes": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("models-private/contest-iex"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    bars = _fetch()
    canonical_bars = json.dumps(bars, sort_keys=True, separators=(",", ":"))
    source_sha256 = hashlib.sha256(canonical_bars.encode("utf-8")).hexdigest()
    raw_path = args.output_root / "spy-iex-m15-2021-2024.json"
    raw_path.write_text(canonical_bars, encoding="utf-8")

    examples = build_examples(parse_alpaca_bars(bars))
    ridge = train_temporal_ridge(examples)
    forest = _train_random_forest(examples, args.output_root)
    trained_at = datetime.now(timezone.utc)
    ridge_artifact = ridge.as_dict(trained_at=trained_at, source_sha256=source_sha256)
    selected_family = min(
        (
            (float(ridge_artifact["validation"]["mae"]), "RIDGE_LINEAR_REGRESSION"),
            (float(forest["validation"]["mae"]), "RANDOM_FOREST_REGRESSOR"),
        ),
        key=lambda item: (item[0], item[1]),
    )[1]
    if selected_family == "RANDOM_FOREST_REGRESSOR":
        artifact = {
            "schema_version": "aurora.contest-iex-model.v1",
            "track": "CONTEST_EXPERIMENTAL_IEX",
            "core_model_evidence": False,
            "paper_only": True,
            "live_trading_allowed": False,
            "model_id": forest["model_id"],
            "trained_at": trained_at.isoformat(),
            "source_sha256": source_sha256,
            "source_feed": "iex",
            "source_window": {"start": "2021-01-01", "end_exclusive": "2025-01-01"},
            "core_holdout_2025_blackout": True,
            "feature_set": "MARKET_CORE_M15_V1",
            "target": "forward_close_return_4",
            "target_horizon_bars": 4,
            "split": {
                "train": "2021-01-01/2023-01-01",
                "validation": "2023-01-01/2024-01-01",
                "test": "2024-01-01/2025-01-01",
                "purge_boundary_targets": True,
            },
            "model": {
                "family": forest["family"],
                "selected_configuration": forest["selected_configuration"],
                "selected_signal_threshold": forest["selected_signal_threshold"],
                "private_artifact": forest["model_path"],
            },
            "counts": forest["counts"],
            "validation": forest["validation"],
            "test": forest["test"],
            "economic_proxy": {
                "round_trip_cost_bps": 2.0,
                "instrument": "SPY_UNDERLYING_PROXY",
                "overlap": "NON_OVERLAPPING_4_BAR_SIGNALS",
                "status": "UNVERIFIED_PROXY_NOT_OPTIONS_PNL",
            },
            "paper_eligible": forest["paper_eligible"],
            "reason_codes": forest["reason_codes"],
        }
    else:
        artifact = ridge_artifact
    artifact["candidate_families"] = {
        "ridge": {
            "validation_mae": ridge.validation.mae,
            "paper_eligible": ridge.paper_eligible,
            "reason_codes": ridge.reason_codes,
        },
        "random_forest": {
            "candidates": forest["candidate_report"],
            "validation_mae": forest["validation"]["mae"],
            "paper_eligible": forest["paper_eligible"],
            "reason_codes": forest["reason_codes"],
        },
        "selection_rule": "minimum_validation_mae_then_family_name",
        "selected": selected_family,
    }
    artifact_path = args.output_root / "model-artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "artifact_path": str(artifact_path.resolve()),
        "bar_count": len(bars),
        "example_count": len(examples),
        "model_id": artifact["model_id"],
        "selected_family": selected_family,
        "paper_eligible": artifact["paper_eligible"],
        "reason_codes": artifact["reason_codes"],
        "validation": artifact["validation"],
        "test": artifact["test"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if bool(artifact["paper_eligible"]) else 2


if __name__ == "__main__":
    sys.exit(main())

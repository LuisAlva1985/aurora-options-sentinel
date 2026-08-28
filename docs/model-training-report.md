# Contest model training report — generation 1

Status: `TRAINED_REJECTED`  
Model track: `CONTEST_EXPERIMENTAL_IEX`  
Core evidence: `false`  
Paper eligible: `false`

## Protocol

- Source: SPY 15-minute IEX bars, 2021-01-01 through 2024-12-31.
- Raw bars: 29,613; usable point-in-time examples: 17,996.
- Train: 2021-2022; validation/model selection: 2023; test: 2024.
- AURORA Core's 2025 holdout was blacked out and never requested.
- Features: the six `MARKET_CORE_M15_V1` definitions.
- Target: `forward_close_return_4`.
- Candidates: train-mean baseline, five ridge configurations, five random-forest configurations.
- Economic diagnostic: non-overlapping four-bar SPY-direction proxy with 2 bps round-trip cost. It is not options P&L and its friction status is unverified.

## Frozen result

The random forest won model selection on validation MAE.

| Period | Candidate MAE | Baseline MAE | Net return proxy | Decisions |
|---|---:|---:|---:|---:|
| Validation 2023 | 0.0017424552 | 0.0017432553 | +0.0038293449 | 234 |
| Test 2024 | 0.0015049804 | 0.0015024890 | -0.0055166948 | 160 |

Rejection codes:

- `test_did_not_beat_baseline`
- `test_net_return_proxy_not_positive`

No order submission flag was changed. The private bars, estimator, credentials,
and full artifact remain outside Git.

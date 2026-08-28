# AURORA Core and competition alignment

The competition is an acceleration track, not a shortcut around AURORA's
scientific controls.

| Boundary | AURORA Core | Options Sentinel contest track |
|---|---|---|
| Purpose | Certified research path toward limited deployment | Fast, isolated Paper experiment |
| Feed | SIP research dataset | Authorized IEX historical bars |
| 2025 | Sealed holdout, zero openings | Hard blackout; never requested by the trainer |
| Training gate | False until written SIP scope confirmation | Explicitly allowed only for the separate IEX scope |
| Broker | Out of scope until S4.1 | Dedicated $100,000 Alpaca Paper account |
| Live money | False | False; no live endpoint exists |
| Evidence transfer | None from contest into Core automatically | Aggregate Paper costs, failures, and slippage may become future inputs after review |

Both tracks preserve `MARKET_CORE_M15_V1`, `forward_close_return_4`, temporal
splits, four-bar horizon purging, deterministic lineage, baseline comparison,
economic friction, and fail-closed decisions. Private rows and model artifacts
are excluded from the public repository.

The economic objective is not raw prediction accuracy. A candidate must:

1. beat the train-mean baseline on validation and untouched test MAE;
2. produce positive net return in both periods under the declared proxy cost;
3. contain at least 30 non-overlapping decisions per evaluation period;
4. remain Paper-only, recent, traceable, and inside all broker risk limits.

Failure of any condition means `NO_ACTION`. Competition P&L is simulated and
cannot establish that a model will make money with real capital.

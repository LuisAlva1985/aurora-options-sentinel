# Contest model training report — generation 2

Status: `EXECUTED_REJECTED`  
Sealed-test openings used: `1/1`  
Paper eligible: `false`

Generation 2 was committed publicly before acquiring its sealed 2026 test. It
trained cost-aware three-class models (`CALL`, `PUT`, `NO_ACTION`) while keeping
the complete 2025 AURORA Core holdout blacked out.

| Period | Macro-F1 | Majority baseline | Net return proxy | Decisions | Max drawdown proxy |
|---|---:|---:|---:|---:|---:|
| Validation 2024 | 0.1745577133 | 0.1897623983 | +0.0326616624 | 133 | 0.0223516894 |
| Sealed test 2026 through Aug 26 | 0.1763874116 | 0.1852489950 | -0.0519736556 | 128 | 0.0545460763 |

The selected random-forest classifier failed three frozen gates:

- `sealed_test_net_return_proxy_not_positive`
- `validation_macro_f1_did_not_beat_majority`
- `sealed_test_macro_f1_did_not_beat_majority`

The sealed test cannot be reused for tuning. The safe next experiment is forward
shadow observation during the competition, with options quotes and realized
Paper fills used to replace the underlying-only cost proxy. No order flag was
enabled by this run.

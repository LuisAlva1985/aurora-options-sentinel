# Competition architecture

## Golden path

1. Read SPY and option-chain observations from Alpaca.
2. Ask the AI reasoning layer for a bounded bullish, bearish, or neutral thesis.
3. Validate the thesis against freshness, confidence, and traceability rules.
4. Select one liquid long call or long put with bounded premium risk.
5. Reject the proposal if any hard gate fails.
6. Convert an approved proposal into a Paper-only Alpaca CLI/MCP request.
7. Persist a redacted audit event and display the decision rationale.

## Deliberate exclusions

- Live accounts and live endpoints.
- Naked short options.
- Unlimited-loss positions.
- Market orders.
- Secret storage in Git.
- Claims of profitability before observed competition Paper results.
- Reuse or publication of the private AURORA SIP research dataset.

## MVP acceptance

- A complete offline demonstration is deterministic.
- Risk rejection is fail-closed and has stable reason codes.
- No adapter can emit an order for any environment other than `paper`.
- The live gate is a literal false invariant.
- Tests cover both approved and rejected proposals.

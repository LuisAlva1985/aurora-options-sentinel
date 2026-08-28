# AURORA Options Sentinel

Paper-only autonomous options agent prepared for the 2026 Alpaca AI Trading
Agents Hackathon.

## Safety invariant

This repository cannot route live orders. Every order intent must pass a
deterministic risk gate and must target a dedicated Alpaca Paper account.

```text
MARKET OBSERVATION
  -> AI THESIS
  -> OPTION CONTRACT SELECTION
  -> HARD RISK GATES
  -> PAPER ORDER INTENT
  -> ALPACA CLI/MCP BOUNDARY
  -> AUDIT EVENT
```

Current state: `PREPARATION_ONLY`. No account credentials, API keys, orders,
positions, P&L claims, or private AURORA datasets are stored here.

Official LabLab team: [AURORA Options Sentinel](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon/aurora-options-sentinel).
The team was created closed, with the Costa Rica timezone (UTC-6).

Every decision is recorded in a SHA-256 hash chain. The audit layer rejects
fields whose names could contain credentials, tokens, passwords, or secrets.

The repository also contains a dated Alpaca market-data fixture and an explicit
spread/regulatory-fee model. It never assumes that commission-free means
friction-free.

## Local checks

```powershell
python -m unittest discover -s tests -v
python -m aurora_sentinel.demo
```

## Competition constraints

- Alpaca Paper only; no real capital.
- Options are mandatory.
- Alpaca MCP or CLI integration is mandatory.
- Final judging account must be fresh and start at $100,000.
- Maximum loss and order frequency are capped before routing.
- Public release uses the MIT license.

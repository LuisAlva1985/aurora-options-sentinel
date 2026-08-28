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

Current state: `PAPER_READY`. The dedicated competition account is
`PA3HAW9279NN`, starts at $100,000 simulated cash, and has been authenticated
against Alpaca's Paper endpoint. No credential values, orders, positions, P&L
claims, or private AURORA datasets are stored here.

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
python -m aurora_sentinel.paper_account
```

The final command reads the API key and secret from Windows Credential Manager
under account-scoped `AURORA/Alpaca/Paper/...` targets. It fails closed if the
account is not active, is blocked, does not match `PA3HAW9279NN`, does not start
at exactly $100,000, or if any endpoint other than Alpaca Paper is requested.

## Competition constraints

- Alpaca Paper only; no real capital.
- Options are mandatory.
- Alpaca MCP or CLI integration is mandatory.
- Final judging account must be fresh and start at $100,000.
- Maximum loss and order frequency are capped before routing.
- Public release uses the MIT license.

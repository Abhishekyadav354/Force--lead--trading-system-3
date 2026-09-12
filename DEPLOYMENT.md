# Force-Lead Trading System Deployment Guide

This document describes how to deploy and operate the Force-Lead Trading System in a safe, simulation-first manner. It is intended for paper trading and evaluation only. It does not provide a guarantee of profitability or a recommendation to trade live with real capital.

## 1. Prerequisites

Before deployment, confirm the following:

- Python 3.10 or newer is installed.
- A working virtual environment is available.
- A supported Linux or macOS environment is used for local deployment.
- Git is available for repository management.
- The project has been cloned locally.
- Network access is available only for the dependencies and data sources you explicitly configure.
- You understand that this project is a research and trading simulation system, not a live brokerage client by default.

Note: This project does not require any real broker credentials for paper trading.

## 2. Project Setup

### GitHub repository / Codespaces

1. Open the repository in GitHub Codespaces or clone it locally.
2. Open the project root in the terminal.
3. Create and activate a Python virtual environment if you are working outside Codespaces.
4. Install the project requirements.
5. Verify the repository structure before continuing.

Example project-relative commands:

```bash
cd /workspaces/Force--lead--trading-system-
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the project is run from a subdirectory, ensure the package root is included in `PYTHONPATH` or run commands from the repo root.

## 3. Dependencies

Install the Python dependencies defined in the repository requirements file:

```bash
pip install -r requirements.txt
```

For GitHub Codespaces or a fresh local environment, use the project-relative command from the repo root:

```bash
cd /workspaces/Force--lead--trading-system-
pip install -r requirements.txt
```

Important dependency notes:

- The project may rely on common scientific and data utility packages such as pandas, numpy, and pytest.
- Do not install or enable live trading APIs unless you are intentionally setting up a separate live-trading environment.
- Keep dependencies minimal and version-pinned where possible for reproducibility.

## 4. Environment Configuration

Create environment variables only when they are required for the specific mode you are running.

### Create `.env` from `.env.example`

If an `.env.example` file exists in the repository, copy it to `.env` before configuring values.

```bash
cp .env.example .env
```

If no `.env.example` exists, create a minimal `.env` with only the variables needed for the current run.

Recommended practices:

- Use a local `.env` file for development-only configuration.
- Never commit real credentials or secret tokens to source control.
- Keep broker-related environment variables separate from paper-trading configuration.
- Prefer minimal configuration for testing and paper trading.
- Do not add real API keys or credentials to this file.

Example environment pattern:

```bash
# Example only. Replace with non-sensitive local values as needed.
APP_ENV=development
LOG_LEVEL=INFO
```

Required environment variables for this system should be reviewed from the application configuration and added only if explicitly needed for the selected environment.

For this project, paper trading is the default safe mode. Live trading should only be configured in a clearly isolated environment after a separate approval and validation flow.

## 5. Testing

Run the integration validation from the repository root before any paper trading workflow:

```bash
python tests/test_system.py
```

This must exit successfully before continuing with paper trading or any further deployment steps.

After the core system checks pass, run the full suite if needed:

```bash
pytest
```

For a focused check on paper-trading logic:

```bash
pytest -q tests/test_paper_trader.py
```

For the CLI-only paper report flow:

```bash
python paper_trading/paper_trader.py --report
```

Testing goals:

- validate simulation logic
- confirm paper-trading rules without live market data
- verify no real orders are triggered in the paper mode
- catch misconfiguration before any live deployment work

## 6. Paper Trading

Paper trading is the recommended default mode for evaluating the system.

### Paper-trading workflow

1. Verify the project is healthy before starting any simulation run:

```bash
python tests/test_system.py
```

The system test command must exit successfully before continuing. If it fails, stop and investigate before starting any paper-trading workflow.

2. Start the simulation in paper mode from the repository root:

```bash
python force-lead-trading-system/main.py --mode paper --symbol NIFTY --capital 10000 --port 5000
```

3. Optional: run the lightweight virtual-trade report workflow directly:

```bash
python paper_trading/paper_trader.py --report
```

### Operational rules

- Paper mode uses virtual capital, not real cash.
- Market data may be live, but the execution path remains simulation-only.
- No real orders should be submitted in this mode. The project is not a broker client by default.
- Monitor the logs and the dashboard while the system is running.
- Keep the paper-trading workflow running for multiple weeks before evaluating the system's longer-term performance.
- Evaluate enough completed trades before drawing conclusions. Do not rely on a tiny sample or a short period of favorable outcomes.
- Review the generated log files under `logs/paper_trading/` and the dashboard output for errors, stale signals, and execution anomalies.

### What to monitor

- Log output from the trading loop and any paper-trade entries/exits.
- CSV trade and signal logs in `logs/paper_trading/`.
- The dashboard at `http://localhost:5000` when the system is started in paper mode.
- Final report output:

```text
logs/paper_trading/final_report.json
```

### Important distinction

- Paper trading is for simulation, evaluation, and process validation.
- Live trading is a separate operational mode that may involve real market risk, broker APIs, and real capital.

Paper results may be informative, but they do not automatically qualify the system for live trading. A high or low win rate alone is not a sufficient pass/fail criterion. The system should be evaluated on a broader set of evidence, including sample size, drawdown, consistency across time, and the quality of the trade process.

## 7. Results Review

After running paper-trading simulations or reports, review the generated artifacts in the logs directory:

- `logs/paper_trading/trade_log.csv`
- `logs/paper_trading/signal_log.csv`
- `logs/paper_trading/final_report.json`

Review questions:

- How many trades were executed in paper mode?
- What was the net paper-trading return?
- Did the system produce consistent outcomes across samples?
- Did win rate, profit factor, and drawdown indicate a valid simulation or merely a favorable sample?
- Was the evaluation based on enough completed trades over a long enough period to be meaningful?

Remember: a paper-trading result does not automatically justify live trading. Strategy validation should also consider sample size, time horizon, out-of-sample performance, transaction costs, and drawdown.

## 8. Monitoring

Monitoring in this project is primarily log-based and simulation-focused.

Recommended checks:

- review the paper-trading logs after each run
- confirm the system is not attempting a live order path
- check for errors in the log files
- validate that the report is being written to the correct output directory

Common monitoring artifacts:

- stdout logs from the Python process
- CSV logs in `logs/paper_trading/`
- JSON report output for final analysis

## 9. Troubleshooting

If deployment or execution fails, review the following:

1. Dependency installation problems
   - reinstall dependencies from `requirements.txt`
   - verify the Python version matches the project requirement

2. Import path errors
   - run the command from the repository root
   - confirm the project root is in `PYTHONPATH`

3. Missing log directory
   - the paper-trading path is created automatically by the logger when needed

4. Report generation issues
   - ensure the project has completed at least some paper trades before generating a final summary
   - confirm the log directory is writable

5. Unexpected live-trading behavior
   - verify the script is running in paper mode only
   - check that no live order functions are invoked
   - confirm no real broker credentials or API tokens are configured

## 10. Security Notes

Security is essential, even in a simulation-only system.

- Never commit real API keys, passwords, secrets, or broker credentials.
- Keep environment variables in local-only files and do not track them in Git.
- Do not add real brokerage credentials to shared notebooks, scripts, or logs.
- Keep paper trading and live trading separated by configuration and operational discipline.
- Use a separate environment for any future live-trading work if it is ever approved.
- Treat logs as potentially sensitive operational artifacts.

This project should remain a paper-trading and research tool unless a separate, explicit live-trading setup is authorized and reviewed.

## Final Note

The Force-Lead Trading System is a research and simulation tool. Paper-trading results may be informative, but they are not evidence of future profitability. Real-money trading requires a separate, carefully reviewed risk framework and independent validation.

# File Inventory

## Root modules

| File | Function |
|---|---|
| `core.py` | Shared data and portfolio utilities |
| `llm.py` | Ollama request, response parsing, and repair logic |
| `llm_portfolio.py` | LLM-driven portfolio pipeline |
| `portfolios.py` | Portfolio construction helpers |
| `paper_canonical.py` | Canonical experiment orchestration |
| `gui.py` | Desktop experiment interface |
| `supplementary_gui.py` | Supplementary-experiment interface |
| `nlpi_monitor.py` | Progress monitoring |
| `q1_checkpoint_status.py` | Resume/checkpoint status inspection |

## Packages

| Package | Function |
|---|---|
| `engine` | Backtest, metrics, covariance, strategy, and significance code |
| `q1_experiments` | Reliability prompts, reference policies, runners, and summaries |
| `supplementary_experiments` | Bridge, repeatability, baseline, projection, and audit code |
| `p2_reconciliation` | Controlled P2 protocol comparison and counterfactual target test |
| `paper_figures` | Reproducible manuscript figures |

## Launchers

| Script | Function |
|---|---|
| `setup_venv_current.sh` | Creates the main Python environment and checks imports |
| `setup_supplementary_mac.sh` | Creates the supplementary environment on macOS/Linux |
| `run_reliability_resume.sh` | Resume-safe full reliability audit |
| `run_corrected_baseline_mac.sh` | Corrected non-LLM baseline run |
| `run_supplementary_smoke_test.sh` | Supplementary unit and smoke checks |
| `run_p2_reconciliation.sh` | Production A/B/C P2 diagnostic |
| `run_p2_counterfactual.sh` | Production D/E P2 diagnostic |
| `run_p2_reconciliation_smoke_test.sh` | Small reconciliation test |
| `run_p2_counterfactual_smoke_test.sh` | Small counterfactual test |

The Springer LaTeX source and compiled manuscript are distributed as separate submission artifacts and are intentionally excluded from this GitHub package.

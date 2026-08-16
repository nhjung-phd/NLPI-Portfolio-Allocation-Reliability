#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x .venv_p2_reconciliation/bin/python ]]; then
  python3 -m venv .venv_p2_reconciliation
  .venv_p2_reconciliation/bin/python -m pip install --upgrade pip
  .venv_p2_reconciliation/bin/python -m pip install -r requirements.txt
fi

source .venv_p2_reconciliation/bin/activate

python -m p2_reconciliation.counterfactual_runner \
  --experiment-id p2_counterfactual_smoke \
  --models gemma3:270m gemma3:1b llama3.1:8b \
  --n-dates 7 \
  --seed 42 \
  --temperature 0.0 \
  --top-p 0.9 \
  --dry-run

echo "P2 counterfactual smoke test complete."

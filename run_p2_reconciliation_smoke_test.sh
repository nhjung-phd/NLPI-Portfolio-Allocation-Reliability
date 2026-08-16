#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [[ ! -x .venv_p2_reconciliation/bin/python ]]; then
  python3 -m venv .venv_p2_reconciliation
  .venv_p2_reconciliation/bin/python -m pip install --upgrade pip
  .venv_p2_reconciliation/bin/python -m pip install -r requirements.txt
fi
source .venv_p2_reconciliation/bin/activate

python -m p2_reconciliation.runner \
  --experiment-id p2_reconciliation_smoke \
  --models gemma3:270m \
  --n-dates 1 \
  --dry-run

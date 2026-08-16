#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv_supplementary/bin/python
if [ ! -x "$PY" ]; then PY=python3; fi
"$PY" -m pytest -q tests_supplementary
"$PY" -m supplementary_experiments.runner \
  --experiments bridge baseline repeatability projection_ablation p5_audit \
  --models smoke:model --policies P1 P2 P3 P4 P5 P6 \
  --max-dates 2 --repeats 2 --dry-run --synthetic-data \
  --experiment-id smoke_test --out-root validation_results
echo "Smoke test complete: validation_results/smoke_test"

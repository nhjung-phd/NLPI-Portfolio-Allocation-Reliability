#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" audit_cluster_bootstrap.py \
  --reps 10000 \
  --seed 42 \
  --outdir results/statistical_enhancement

echo "[DONE] results/statistical_enhancement"

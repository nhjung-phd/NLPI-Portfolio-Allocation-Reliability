#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
python "$ROOT/analysis/run_all_analysis.py" \
  --project-root "$ROOT" \
  --output "$ROOT/analysis/reproduced_outputs"

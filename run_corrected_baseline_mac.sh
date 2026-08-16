#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
EXPERIMENT_ID="${EXPERIMENT_ID:-baseline_corrected_rerun_v1}"

if [ ! -x .venv_supplementary/bin/python ]; then
  echo "[SETUP] Creating .venv_supplementary"
  ./setup_supplementary_mac.sh
fi

PYTHON=.venv_supplementary/bin/python

echo "[RUN] Corrected non-LLM baseline: 7 dates x 6 policies x 2 methods = 84"
echo "[RUN] experiment_id=$EXPERIMENT_ID"
echo "[RUN] Existing sources/results remain unchanged; Ollama is not called."

"$PYTHON" -m supplementary_experiments.runner \
  --experiments baseline \
  --max-dates 7 \
  --seed 42 \
  --temperature 0.0 \
  --top-p 0.9 \
  --experiment-id "$EXPERIMENT_ID"

"$PYTHON" - "$EXPERIMENT_ID" <<'PY'
import sys
from pathlib import Path
import pandas as pd

experiment_id = sys.argv[1]
result_dir = Path("results") / "supplementary" / experiment_id
result_file = result_dir / "supplementary_calls.csv"
if not result_file.exists():
    raise SystemExit(f"[ERROR] Missing result file: {result_file}")

df = pd.read_csv(result_file)
if len(df) != 84:
    raise SystemExit(f"[ERROR] Expected 84 rows, found {len(df)}")

summary = (
    df.groupby("method", as_index=False)
      .agg(n=("policy_accuracy", "size"),
           accuracy=("policy_accuracy", "mean"),
           mean_allocation_l1=("allocation_l1_to_reference", "mean"),
           max_allocation_l1=("allocation_l1_to_reference", "max"))
)
summary.to_csv(result_dir / "baseline_summary.csv", index=False)
print("\n[VALIDATION]")
print(summary.to_string(index=False))
print(f"\n[DONE] {result_file}")
PY


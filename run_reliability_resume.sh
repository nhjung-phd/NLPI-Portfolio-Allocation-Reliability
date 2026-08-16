#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
OUTDIR="${OUTDIR:-outputs/NLPI_PAPER_CANONICAL_V1_20260717_152949/reliability_resumable_20260727}"
RUN_LOG="${RUN_LOG:-reliability_resumable_20260727.log}"

MODELS=(
  "gemma3:270m"
  "gemma3:1b"
  "llama3.1:8b"
  "qwen3.5:4b"
)

EXPERIMENTS=(
  "prompt_robustness"
  "ticker_masking"
  "policy_complexity"
  "constraint_stress"
  "model_generalization"
  "sensitivity_template"
)

TICKERS=(
  SPY QQQ IWM DIA EFA EEM VEA VWO
  AGG BND SHY IEF TLT TIP LQD HYG
  GLD SLV DBC VNQ UUP BIL
)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found: $PYTHON_BIN"
  echo "Set PYTHON_BIN to the project virtual-environment Python."
  exit 1
fi

mkdir -p "$OUTDIR"

for model in "${MODELS[@]}"; do
  echo "[MODEL START] $model $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$RUN_LOG"
  "$PYTHON_BIN" -u -m q1_experiments.runner \
    --start 2010-01-01 \
    --end 2025-12-29 \
    --rebalance 42 \
    --tcost 0.001 \
    --maxw 0.60 \
    --turncap 0.25 \
    --prompt-cap 60 \
    --tickers "${TICKERS[@]}" \
    --models "$model" \
    --experiments "${EXPERIMENTS[@]}" \
    --decision-sample stratified \
    --n-per-regime 6 \
    --policies P1 P2 P3 P4 P5 P6 \
    --seed 42 \
    --ollama-connect-timeout 30 \
    --ollama-read-timeout 1200 \
    --max-retries 2 \
    --parse-retries 1 \
    --timeout-retries 1 \
    --num-predict 512 \
    --ollama-keep-alive 30m \
    --outdir "$OUTDIR" 2>&1 | tee -a "$RUN_LOG"
  echo "[MODEL DONE] $model $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$RUN_LOG"
done

echo "[ALL DONE] Reliability experiments completed: $OUTDIR" | tee -a "$RUN_LOG"

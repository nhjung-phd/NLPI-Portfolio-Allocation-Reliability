#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-venv}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python not found: ${PYTHON_BIN}"
  exit 1
fi

if [ "${FORCE_RECREATE}" = "1" ]; then
  rm -rf "${VENV_DIR}"
fi

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
import importlib
modules = [
    "numpy", "pandas", "scipy", "sklearn", "matplotlib",
    "yfinance", "requests", "pydantic", "tqdm", "tkinter",
]
failed = []
for name in modules:
    try:
        importlib.import_module(name)
        print(f"[OK] {name}")
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        failed.append(name)
if failed:
    raise SystemExit("Import check failed: " + ", ".join(failed))
PY

python -m py_compile \
  gui.py core.py llm.py llm_portfolio.py portfolios.py paper_canonical.py \
  engine/*.py q1_experiments/*.py paper_figures/*.py

echo "[DONE] Environment ready."
echo "Activate: source ./${VENV_DIR}/bin/activate"
echo "Run GUI: python gui.py"

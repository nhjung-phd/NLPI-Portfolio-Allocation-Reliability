# Reviewer-requested CSV reanalysis

Run this analysis from the repository root, the directory containing `reviewer_reanalysis.py` and `results/`.

## macOS/Linux

```bash
cd /path/to/NLPI-Portfolio-Allocation-Reliability
source venv/bin/activate
./run_reviewer_reanalysis.sh
```

Alternatively:

```bash
python reviewer_reanalysis.py
```

No Ollama process is required. The program does not call an LLM, does not download market data, and does not overwrite retained source CSV files.

Outputs are written to:

```text
results/reviewer_reanalysis/
```

The generated files cover:

1. policy/model fidelity decomposition and P4-excluded/macro averages;
2. policy-level cluster intervals and decision-date block-bootstrap sensitivity;
3. exact-zero feature values contained in retained prompts and their relationship to policy targets;
4. multidimensional P5 semantic diagnostics.

The compact retained prompts do not include an upstream imputation flag. Consequently, the zero audit can identify values of exactly zero that were supplied to the LLM, but cannot determine whether each zero was observed in the market-derived feature or inserted by residual zero filling.

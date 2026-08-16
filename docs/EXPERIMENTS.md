# Experiment Guide

## 1. Walk-forward out-of-sample evaluation

The canonical configuration is `configs/paper_protocol.json`. The WFCV output retained in `results/wfcv/` includes the main performance table, fold metrics, prompt-fidelity diagnostics, run diagnostics, and the tidy out-of-sample panel.

An exact end-to-end rerun also requires compatible adjusted-close market data, which is not redistributed. The retained WFCV files are sufficient to audit the numerical claims in the manuscript.

## 2. Primary repeated-measures reliability audit

The archived primary audit uses crossed prompt, policy, model, date, and stress conditions. Its resume-safe launcher is:

```bash
PYTHON_BIN=venv/bin/python \
OUTDIR=outputs/reliability_reproduction_01 \
RUN_LOG=reliability_reproduction_01.log \
./run_reliability_resume.sh
```

The original launcher includes the auxiliary P6 stress-test policy because it is part of the retained archive. Manuscript policy-specific conclusions and the bridge comparison are restricted to P1--P5. The 5,670 primary records are clustered repeated measurements rather than independent observations.

Always select a new `OUTDIR` to preserve archived results.

## 3. Supplementary bridge, repeatability, and deterministic baseline

Environment and smoke test:

```bash
./setup_supplementary_mac.sh
source .venv_supplementary/bin/activate
./run_supplementary_smoke_test.sh
```

The retained production outputs are:

| Result directory | Records | Purpose |
|---|---:|---|
| `results/supplementary/bridge_baseline_504part/` | 252 | Canonical bridge calls plus baseline-related records |
| `results/supplementary/repeatability_504part/` | 168 | Repeated calls under fixed settings |
| `results/supplementary/baseline_corrected_rerun_v1/` | 84 | Corrected non-LLM deterministic baseline |

To rerun the corrected baseline without overwriting the retained directory, use a new identifier:

```bash
python -m supplementary_experiments.runner \
  --experiments baseline \
  --experiment-id baseline_corrected_reproduction_01
```

Inspect `python -m supplementary_experiments.runner --help` for version-specific options.

## 4. Controlled P2 reconciliation

Conditions:

- A: WFCV-formatted prompt with few-shot demonstrations.
- B: the same prompt with only the few-shot demonstrations removed.
- C: canonical bridge rendering.

Production command:

```bash
python -m p2_reconciliation.runner \
  --experiment-id p2_reconciliation_reproduction_01 \
  --models gemma3:270m gemma3:1b llama3.1:8b \
  --n-dates 7 --seed 42 --temperature 0.0 --top-p 0.9 --timeout 900
```

This produces 3 conditions x 3 models x 7 dates = 63 calls.

## 5. Controlled P2 counterfactual test

Conditions:

- D: WFCV-formatted few-shot prompt with the minimum-volatility target reassigned to a non-BIL asset.
- E: the same counterfactual target without demonstrations.

Production command:

```bash
python -m p2_reconciliation.counterfactual_runner \
  --experiment-id p2_counterfactual_reproduction_01 \
  --models gemma3:270m gemma3:1b llama3.1:8b \
  --n-dates 7 --seed 42 --temperature 0.0 --top-p 0.9 --timeout 900
```

This produces 2 conditions x 3 models x 7 dates = 42 calls. Together with the reconciliation experiment, the controlled P2 diagnostic contains 105 calls.

## 6. Result regeneration

From the repository root, validate the retained results:

```bash
python validate_release.py
```

All reruns should use new experiment identifiers. The runners reject an existing identifier when settings differ, which protects retained results from accidental overwrite.

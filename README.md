# NLPI Portfolio Allocation Reliability

Reproducibility repository for **Natural-Language Policy Interfaces for Constraint-Projected Portfolio Allocation: Performance, Reliability, and Semantic Fidelity**.

**Journal:** Computational Economics  
**Authors:** Nak Hyun Jung and Taeyeon Oh (corresponding author)  
**Release:** v1.0.0 (frozen submission snapshot)

> Scope: this repository supports audit-framework validation. Portfolio performance is a downstream consequence, not the primary validation objective.

Reproducibility repository for **Natural-Language Policy Interfaces for Constraint-Projected Portfolio Allocation: Performance, Reliability, and Semantic Fidelity**.

## Research objective

This study does **not** ask whether an LLM can outperform portfolio optimizers. It evaluates whether a natural-language policy interface (NLPI) can translate an explicitly stated investment policy into a semantically faithful, feasible, repeatable, and auditable portfolio action. Portfolio performance is examined as a downstream economic consequence of policy translation, not as the primary validation criterion.

The repository contains the research code, retained call-level outputs, walk-forward results, reliability audit, controlled P2 experiments, deterministic references, projection diagnostics, and post-hoc statistical programs required to audit the reported findings. Manuscript files are distributed separately.

## Verified findings

- The primary audit contains 5,670 calls. Final JSON validity is 100%, whereas projected semantic fidelity is 14.44%.
- A 10,000-replicate bootstrap of whole model-policy-decision-date clusters gives a 95% percentile interval of 11.95%--17.07% (756 clusters; seed 42).
- The wider repeated-measures audit contains 6,174 retained records. These are clustered repeated measurements, not independent market observations.
- The controlled P2 diagnostic contains 105 additional calls and is reported separately.
- P2 fidelity is 21/21 with the WFCV prompt and BIL-centered few-shot demonstrations, but 0/21 when those demonstrations are removed.
- After counterfactual reassignment of the minimum-volatility target, fidelity is 0/21 and BIL is still returned in 21/21 few-shot calls. Paired raw responses and allocations are unchanged in all 21 WFCV-versus-counterfactual comparisons.
- Constraint projection guarantees numerical admissibility but does not repair semantic policy errors.
- WFCV performance shows downstream economic consequences; it is not evidence of semantic understanding or LLM dominance.

## Repository map

| Path | Contents |
|---|---|
| `engine/` | Backtesting, covariance, portfolio metrics, strategies, and statistical utilities |
| `q1_experiments/` | Prompt library, reference policies, reliability runners, and result tables |
| `supplementary_experiments/` | Bridge, repeatability, corrected baseline, projection, and audit analyses |
| `p2_reconciliation/` | Controlled P2 reconciliation and counterfactual runners |
| `paper_figures/` | Reproducible figure-generation utilities |
| `configs/` | Canonical experimental protocol |
| `results/wfcv/` | Retained walk-forward performance and diagnostics |
| `results/reliability_primary/` | 5,562 primary call-level records and tables |
| `results/reliability_qwen/` | 108 Qwen model-family records and tables |
| `results/supplementary/` | Bridge, repeatability, and corrected baseline results |
| `results/p2_controlled/` | 105 controlled P2 calls, prompts, manifests, and checkpoints |
| `results/statistical_enhancement/` | Cluster-bootstrap estimates and manifest |
| `analysis/` | Post-hoc statistical program, retained tables, and figures |
| `docs/` | Experiment, data, result, and file documentation |
| `validate_release.py` | Machine-checks headline record counts and numerical claims |

## Installation

Python 3.10 or newer is required; Python 3.11 is recommended.

### Option A: `venv` and `pip`

```bash
./setup_venv_current.sh
source venv/bin/activate
pip install -r analysis/requirements_analysis.txt
```

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate nlpi-reliability
```

Ollama is required only when generating new LLM outputs. Verification and post-hoc analysis of retained results do not call an LLM.

Retained experiments use `gemma3:270m`, `gemma3:1b`, `llama3.1:8b`, and a bounded `qwen3.5:4b` model-family stress test.

## Reproduce reported results without an LLM

### 1. Verify the release

```bash
python validate_release.py
```

Expected final line: `RELEASE_VALIDATION_OK`.

### 2. Recompute cluster-bootstrap intervals

```bash
./run_audit_cluster_bootstrap.sh
```

Expected output:

```text
CLUSTER_BOOTSTRAP_OK records=5670 clusters=756 fidelity=0.1444 ci95=[0.1195,0.1707]
```

### 3. Reproduce the post-hoc analysis

```bash
./run_posthoc_analysis.sh
```

Fresh outputs are written to `analysis/reproduced_outputs/`; retained results are read only.

### 4. Run reviewer-requested CSV reanalysis

```bash
./run_reviewer_reanalysis.sh
```

This no-LLM program produces fidelity aggregation sensitivity, decision-date block-bootstrap intervals, an exact-zero prompt-feature audit, and multidimensional P5 semantic diagnostics under `results/reviewer_reanalysis/`. See [`REVIEWER_REANALYSIS_README.md`](REVIEWER_REANALYSIS_README.md).

### 5. Run tests

```bash
pytest -q tests_supplementary
```

## Generate new controlled P2 calls with Ollama

```bash
ollama list
./run_p2_reconciliation_smoke_test.sh
./run_p2_counterfactual_smoke_test.sh
```

Full controlled diagnostics:

```bash
./run_p2_reconciliation.sh
./run_p2_counterfactual.sh
```

Use a new experiment identifier to preserve retained production results. Exact commands and scopes are documented in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Data provenance and redistribution boundary

Market prices were obtained from Yahoo Finance through `yfinance` with adjusted prices enabled. The canonical collection date was 2026-07-17. Call-level outputs, derived WFCV results, configurations, manifests, prompts, and validation scripts required to audit the published results are included.

Third-party market-price observations are not redistributed. An end-to-end rerun is subject to the provider's current terms and may not reproduce the archived snapshot exactly because external data can be revised. See [`docs/DATA.md`](docs/DATA.md).

## Reproducibility scope

1. **Computational verification:** recompute headline statistics from retained outputs without an LLM. This is deterministic and recommended first.
2. **Generative replication:** call the same local model families with archived prompts and settings. Exact byte-for-byte responses are not guaranteed across Ollama, model builds, hardware, or runtime versions, even with temperature 0 and a fixed seed.

The controlled scripts preserve prompts, model identifiers, seeds, temperature, top-p, decision dates, raw responses, parsed weights, projected weights, and targets.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). No software license is asserted in this release. Before public release, the authors should add the license they intend to grant; absent a license, default copyright applies.
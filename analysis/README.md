# Post-hoc statistical analysis

This directory contains the no-LLM post-hoc analysis used to audit reliability and portfolio results. The retained outputs were generated from the call-level files under `results/`.

From the repository root, run:

```bash
./run_posthoc_analysis.sh
```

The command reads the retained data without modifying it and writes a fresh set of tables and figures to `analysis/reproduced_outputs/`. The large master reliability table is intentionally not duplicated here because it is exactly reconstructed by concatenating:

- `results/reliability_primary/q1_decision_log.csv` (5,562 records)
- `results/reliability_qwen/q1_decision_log.csv` (108 records)

The cluster-bootstrap analysis reported in the manuscript is separately reproduced with:

```bash
./run_audit_cluster_bootstrap.sh
```

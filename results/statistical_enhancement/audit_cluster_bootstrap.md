# Cluster-bootstrap reliability intervals

Whole model-policy-decision-date clusters are resampled with replacement.
Retained call records are not treated as independent observations.

| Scope | Records | Clusters | Fidelity | 95% cluster-bootstrap CI |
|---|---:|---:|---:|---:|
| Primary audit: overall | 5,670 | 756 | 0.1444 | [0.1195, 0.1707] |
| constraint_conflict_stress | 1,188 | 54 | 0.1481 | [0.0943, 0.2071] |
| model_family_generalization | 432 | 432 | 0.2153 | [0.1759, 0.2546] |
| policy_complexity | 324 | 324 | 0.0432 | [0.0216, 0.0679] |
| prompt_robustness | 2,754 | 324 | 0.1151 | [0.0912, 0.1403] |
| ticker_masking | 972 | 324 | 0.2253 | [0.1852, 0.2675] |

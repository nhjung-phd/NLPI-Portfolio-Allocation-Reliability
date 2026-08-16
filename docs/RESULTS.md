# Verified Results

## Record accounting

| Component | Retained records |
|---|---:|
| Primary reliability audit | 5,670 |
| Bridge/baseline run | 252 |
| Repeatability run | 168 |
| Corrected deterministic baseline | 84 |
| Repeated-measures audit total | 6,174 |
| Controlled P2 reconciliation | 63 |
| Controlled P2 counterfactual | 42 |
| Controlled P2 total | 105 |

The 6,174-record audit and 105-call controlled diagnostic are reported separately. A simple archive count is 6,279, but it must not be interpreted as 6,279 independent observations.

## WFCV performance

| Strategy | Sharpe ratio |
|---|---:|
| Minimum-variance portfolio (MVP) | 0.979107 |
| Best NLPI configuration (P2) | 0.951624 |
| CODED P2 | 0.951624 |

The WFCV evidence does not support performance dominance by NLPI. P2 remains a realized pipeline result, not proof that the language model independently inferred the low-volatility instruction.

## Primary semantic-fidelity audit

- Final JSON validity: 100%.
- Primary projected policy fidelity: 14.44%.
- These rates summarize repeated call-level measurements with shared models, prompts, dates, and market states.

## Controlled P2 results

| Condition | P2 fidelity | Dominant output |
|---|---:|---|
| A: WFCV format + demonstrations | 21/21 | BIL, 21/21 |
| B: WFCV format, no demonstrations | 0/21 | SPY, 21/21 |
| C: canonical bridge | 0/21 | SPY, 18/21 |
| D: counterfactual target + demonstrations | 0/21 | BIL, 21/21 |
| E: counterfactual target, no demonstrations | 0/21 | SPY, 21/21 |

For the 21 matched A-versus-D pairs, raw response text, raw weights, projected weights, and top asset are identical in 21/21 comparisons. The prescribed counterfactual targets are AGG, BND, SHY, IEF, TIP, LQD, and UUP across the seven decision dates.

## Interpretation

The apparent conflict between WFCV P2 fidelity and bridge P2 fidelity is protocol dependence, not a reason to discard either experiment. The controlled tests identify the mechanism: the WFCV-formatted few-shot block raises measured fidelity to 21/21, but changing the target while leaving the BIL-centered demonstrations in place leaves every output unchanged. Therefore, the observed WFCV P2 allocation is demonstration-conditioned template reproduction rather than feature-grounded semantic execution.

The optimization projection remains important for feasibility, but feasibility, syntactic validity, repeatability, and semantic fidelity are distinct properties and must be evaluated separately.


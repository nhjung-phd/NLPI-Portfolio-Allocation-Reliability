#!/usr/bin/env python3
"""Validate the retained release files against the manuscript's key claims."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def read_csv(relative: str) -> pd.DataFrame:
    path = ROOT / relative
    require(path.is_file(), f"file exists: {relative}")
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    primary = read_csv("results/reliability_primary/q1_decision_log.csv")
    qwen = read_csv("results/reliability_qwen/q1_decision_log.csv")
    audit_primary = pd.concat([primary, qwen], ignore_index=True)
    require(len(primary) == 5_562 and len(qwen) == 108, "primary archive split is 5,562 + 108")
    require(len(audit_primary) == 5_670, "primary reliability audit contains 5,670 records")
    require(float(audit_primary["json_valid"].mean()) == 1.0, "primary final JSON validity is 100%")
    require(round(float(audit_primary["projected_fidelity"].mean()) * 100, 2) == 14.44,
            "primary projected fidelity is 14.44%")

    bridge = read_csv("results/supplementary/bridge_baseline_504part/supplementary_calls.csv")
    repeat = read_csv("results/supplementary/repeatability_504part/supplementary_calls.csv")
    baseline = read_csv("results/supplementary/baseline_corrected_rerun_v1/supplementary_calls.csv")
    require((len(bridge), len(repeat), len(baseline)) == (252, 168, 84),
            "supplementary counts are 252, 168, and 84")
    require(len(audit_primary) + len(bridge) + len(repeat) + len(baseline) == 6_174,
            "repeated-measures audit total is 6,174")

    performance = read_csv("results/wfcv/performance_main.csv").set_index("name")
    require(abs(float(performance.loc["MVP", "Sharpe"]) - 0.979107) < 5e-7,
            "MVP Sharpe is 0.979107")
    require(abs(float(performance.loc["CODED_P2", "Sharpe"]) - 0.951624) < 5e-7,
            "CODED P2 Sharpe is 0.951624")
    nlpi_p2 = performance.loc[performance.index.str.endswith("|P2]"), "Sharpe"]
    require(abs(float(nlpi_p2.max()) - 0.951624) < 5e-7, "best NLPI P2 Sharpe is 0.951624")

    reconciliation = read_csv(
        "results/p2_controlled/p2_reconciliation_v1/p2_reconciliation_calls.csv"
    )
    counterfactual = read_csv(
        "results/p2_controlled/p2_counterfactual_v1/p2_counterfactual_calls.csv"
    )
    require((len(reconciliation), len(counterfactual)) == (63, 42),
            "controlled P2 counts are 63 + 42 = 105")

    expected_fidelity = {
        "A_wfcv_exact": (reconciliation, 21),
        "B_wfcv_no_fewshot": (reconciliation, 0),
        "C_bridge_canonical": (reconciliation, 0),
        "D_counterfactual_fewshot": (counterfactual, 0),
        "E_counterfactual_no_fewshot": (counterfactual, 0),
    }
    for condition, (frame, expected) in expected_fidelity.items():
        subset = frame.loc[frame["condition_id"] == condition]
        require(len(subset) == 21, f"{condition} contains 21 calls")
        require(int(subset["projected_fidelity"].sum()) == expected,
                f"{condition} fidelity count is {expected}/21")

    condition_a = reconciliation.loc[reconciliation["condition_id"] == "A_wfcv_exact"].copy()
    condition_d = counterfactual.loc[
        counterfactual["condition_id"] == "D_counterfactual_fewshot"
    ].copy()
    keys = ["model_id", "decision_date"]
    paired = condition_a.merge(condition_d, on=keys, suffixes=("_a", "_d"), validate="one_to_one")
    require(len(paired) == 21, "A-versus-D comparison contains 21 matched pairs")
    for column in ["raw_output", "raw_weights", "projected_weights", "top_asset_projected"]:
        require(bool((paired[f"{column}_a"] == paired[f"{column}_d"]).all()),
                f"A-versus-D {column} is identical in 21/21 pairs")
    require(int(condition_d["bil_copy_projected"].sum()) == 21,
            "counterfactual few-shot condition copies BIL in 21/21 calls")
    require(set(condition_d["assigned_counterfactual_target"]) ==
            {"AGG", "BND", "SHY", "IEF", "TIP", "LQD", "UUP"},
            "counterfactual target set matches the seven prescribed assets")

    with (ROOT / "results/wfcv/paper_protocol.json").open(encoding="utf-8") as handle:
        json.load(handle)
    print("RELEASE_VALIDATION_OK")


if __name__ == "__main__":
    main()

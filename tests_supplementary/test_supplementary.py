import json
import pandas as pd
from supplementary_experiments.baselines import rule_parse, tfidf_match
from supplementary_experiments.analysis import projection_ablation, p5_audit
from supplementary_experiments.checkpoint import TaskLedger

def test_rule_and_tfidf():
    assert rule_parse("choose the highest r12m momentum winner") == "P1"
    assert rule_parse("lowest volatility defensive allocation") == "P2"
    assert tfidf_match("contrarian weakest r3m underperformer")[0] == "P3"
    assert rule_parse("equal-weight control; ignore all feature data") == "P4"
    assert rule_parse("risk-adjusted return with lower volatility and smaller drawdowns") == "P5"
    assert rule_parse("hybrid defensive-momentum with an above-median r12m screen") == "P6"
    assert tfidf_match("equal weight control ignore features uniform allocation")[0] == "P4"
    assert tfidf_match("risk adjusted return lower volatility smaller drawdowns")[0] == "P5"
    assert tfidf_match("hybrid defensive momentum with a bond cash sleeve")[0] == "P6"

def test_projection_and_p5_audit(tmp_path):
    src=tmp_path/"calls.csv"
    pd.DataFrame([{"experiment_id":"x","condition_id":"c","model_id":"m","policy_id":"P5","decision_date":"2025-01-01",
      "raw_weights":json.dumps({"A":.9,"B":.1}),"projected_weights":json.dumps({"A":.5,"B":.5}),"reference_weights":json.dumps({"A":.6,"B":.4}),
      "raw_output":"{}","parse_fail":0,"repair_used":0}]).to_csv(src,index=False)
    assert len(projection_ablation(src,tmp_path/"a")) == 1
    a=p5_audit(src,tmp_path/"a"); assert int(a.iloc[0].collapse)==1

def test_crash_safe_resume(tmp_path):
    ledger=TaskLedger(tmp_path)
    ledger.add("done","bridge",{"x":1}); ledger.add("interrupted","bridge",{"x":2}); ledger.commit()
    ledger.start("done"); ledger.complete("done",{"ok":True})
    ledger.start("interrupted"); ledger.close()
    ledger=TaskLedger(tmp_path)
    assert ledger.recover_interrupted()==1
    assert [x[0] for x in ledger.pending()]==["interrupted"]
    assert ledger.counts()["completed"]==1
    ledger.close()

import json
import pandas as pd
import pytest
from structural_admission import HistoricalAdmission
from structural_data import ROOT


def test_admission_has_no_target_truth_interface_and_cold_fallback():
    thresholds=json.loads((ROOT/"structural_protocol.json").read_text())["selection"]
    guard=HistoricalAdmission(thresholds)
    f=pd.DataFrame({"date":[pd.Timestamp("2025-08-05")],"食堂名称":["A"],"餐次":["午餐"],"calendar_state":["普通工作日"]})
    decision,audit=guard.decide(f)
    assert not decision and "insufficient_previous_prediction_evidence" in audit["reasons"]
    guard.max_date=pd.Timestamp("2025-08-05")
    with pytest.raises(ValueError,match="target-day"):
        guard.decide(f)


def test_admission_observes_only_completed_day():
    thresholds=json.loads((ROOT/"structural_protocol.json").read_text())["selection"]
    guard=HistoricalAdmission(thresholds)
    actual=pd.DataFrame({"date":pd.to_datetime(["2025-08-05","2025-08-06"])})
    with pytest.raises(ValueError,match="completed day"):
        guard.observe_day([],actual)

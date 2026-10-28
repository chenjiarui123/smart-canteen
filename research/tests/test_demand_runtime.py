import json
import numpy as np
import pandas as pd
import pytest
import demand_data
from demand_data import OUT
from demand_experiments import load
from demand_runtime import LockedMenuPredictor
from provenance import verify_registered_snapshot


@pytest.fixture(autouse=True)
def use_frozen_demand_snapshot(monkeypatch):
    snapshot = OUT / "snapshot"
    if not snapshot.is_dir():
        pytest.skip("frozen demand snapshot unavailable")

    def verify_snapshot():
        return verify_registered_snapshot(
            OUT / "preregistration.json", snapshot
        )

    monkeypatch.setattr(demand_data, "verify", verify_snapshot)


@pytest.mark.parametrize("trial",["linear_all_log","conditional_all_log","direct_relative"])
def test_runtime_matches_cached_complete_day_predictions(trial):
    p=OUT/"experiments"/trial
    if not (p/"a_hupan/menu.pkl.gz").exists():pytest.skip("Experiment fixture unavailable")
    a,f,t=load("a_hupan");config=json.loads((p/"record.json").read_text())["config"]
    expected=pd.read_pickle(p/"a_hupan/menu.pkl.gz")
    runtime=LockedMenuPredictor("a_hupan",config)
    for date in sorted(f.date.unique())[:2]:
        menu=a["baseline_menu"].loc[a["baseline_menu"].date.eq(date)]
        old=a["features"].loc[a["features"].date.eq(date)]
        features=f.loc[f.date.eq(date)];past=t.loc[t.date<date]
        output,audit=runtime.predict_day(menu,old,features,past)
        pd.testing.assert_frame_equal(output.reset_index(drop=True),expected.loc[expected.date.eq(date)].reset_index(drop=True),check_exact=False,rtol=1e-12,atol=1e-8)
        assert audit["experimental"]
        with pytest.raises(ValueError,match="chronological"):
            runtime.predict_day(menu,old,features,past)


def test_retained_forecast_only_adds_separate_preparation_and_guards_inputs():
    a,f,t=load("f_jichuyuan");date=pd.Timestamp("2025-01-31")
    menu=a["baseline_menu"].loc[a["baseline_menu"].date.eq(date)]
    old=a["features"].loc[a["features"].date.eq(date)];features=f.loc[f.date.eq(date)]
    runtime=LockedMenuPredictor("f_jichuyuan",{"family":"baseline"})
    with pytest.raises(ValueError,match="actual outcomes"):
        runtime.predict_day(menu,old,features.assign(actual_qty=0),t.loc[t.date<date])
    with pytest.raises(ValueError,match="future"):
        runtime.predict_day(menu,old,features,t)
    output,audit=runtime.predict_day(menu,old,features,t.loc[t.date<date])
    pd.testing.assert_frame_equal(output[menu.columns],menu,check_exact=True)
    assert (output.suggested_preparation_qty>=output.min_batch_qty).all()
    assert output.minimum_batch_surplus_risk.sum()>0
    assert audit["intervals"] is None and not audit["forecast_improvement_claimed"]

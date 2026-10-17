import numpy as np
import pandas as pd
import pytest
from phase_models import Partition, RelativeModel
from phase_features import RELATIVE_FEATURES
from phase_sessions import SessionModel
from phase_calendar import PhaseCalendar


def test_partition_never_extrapolates_outside_observed_support():
    x=np.arange(240).reshape(-1,1);y=np.r_[np.zeros(120),np.ones(120)]
    p=Partition().fit(x,y);pred=p.predict(np.array([[-10000],[10000]]))
    assert (pred>=0).all() and (pred<=1).all()


def test_no_absolute_size_features_in_relative_design():
    assert not any(c in RELATIVE_FEATURES for c in ["historical_scale","log_mean","log_median","old_final_qty"])


def test_relative_model_rejects_target_day_fit():
    f=pd.DataFrame({c:np.ones(200) for c in RELATIVE_FEATURES})
    f["date"]=pd.Timestamp("2025-02-01");f["target_y"]=np.arange(200);f["historical_scale"]=100.
    f["own_drop_threshold"]=.5;f["sample_count"]=5
    model=RelativeModel().fit(f)
    with pytest.raises(AssertionError):model.predict(f)
    f["date"]=pd.Timestamp("2025-02-02")
    samples,_,risk=model.predict(f)
    assert np.isfinite(samples).all() and (samples>=0).all()


def test_ordinary_final_demand_is_pinned():
    calendar=PhaseCalendar(pd.DataFrame([{"日期":"2025-09-01","是否校历寒暑假":0,"校历假期类型":"非寒暑假","是否法定节假日假期":0,"法定节假日名称":"非节假日"}]))
    result=SessionModel(calendar).predict(pd.Timestamp("2025-09-01"),"A","午餐",{},123.456,17)
    assert result["predicted_total"]==123.456 and result["predicted_count"]==17

import numpy as np
import pandas as pd
import pytest
from demand_models import IndependentGroup,IndependentForecaster,decision
from demand_features import FEATURES,RecordedHistory
from demand_coordination import marginal_allocate


def training_fixture(n=100):
    rng=np.random.default_rng(32)
    f=pd.DataFrame(rng.normal(size=(n,len(FEATURES))),columns=FEATURES)
    f["date"]=pd.Timestamp("2025-01-01")+pd.to_timedelta(np.arange(n)%10,unit="D")
    f["餐次"]="午餐";f["sale_unit"]="份"
    f["historical_scale"]=20.;f["low_threshold"]=5.
    f["target_y"]=np.maximum(20+5*f.log_mean,0)
    f.loc[0,"target_y"]=0.
    f["old_date_conditioned"]=1.
    f["warmup"]=False;f["was_old_hit"]=np.arange(n)%2==0
    f["log_median"]=np.log1p(20.)
    return f


def test_independent_model_is_not_bounded_to_old_forecast():
    train=training_fixture();model=IndependentGroup().fit(train)
    f=train.iloc[:4].copy();f["date"]=pd.Timestamp("2025-01-20")
    first=model.predict(f)[0]
    f["old_date_conditioned"]=1e9;f["target_y"]=1e12
    second=model.predict(f)[0]
    np.testing.assert_array_equal(first,second)
    assert np.median(first)>1.5
    assert model.training_rows==len(train) # Explicit zero was not removed.


@pytest.mark.parametrize("family",["linear","tree","hurdle","conditional","conditional_hurdle","relative_linear","mixed_linear"])
def test_dates_intervals_and_finite_recorded_zero(family):
    train=training_fixture();model=IndependentGroup(family).fit(train)
    with pytest.raises(ValueError,match="current/future"):model.predict(train.iloc[:3])
    f=train.iloc[:3].copy();f["date"]=pd.Timestamp("2025-01-20")
    s,r,p=model.predict(f)
    assert s.shape==(3,41) and np.isfinite(s).all() and (s>=0).all()
    assert ((p>=0)&(p<=1)).all()
    assert (np.quantile(s,.1,axis=1)<=np.quantile(s,.9,axis=1)).all()


def test_only_hit_training_control_and_unit_isolation():
    train=training_fixture(200);f=train.iloc[:2].copy();f["date"]=pd.Timestamp("2025-01-20")
    all_model=IndependentForecaster({});hits=IndependentForecaster({"hits_only":True})
    a=all_model.predict(f,train);h=hits.predict(f,train)
    assert (a[4]==200).all() and (h[4]==100).all()
    other=train.copy();other["sale_unit"]="克";other["target_y"]*=1e6
    combined=IndependentForecaster({}).predict(f,pd.concat([train,other]))
    np.testing.assert_allclose(a[0],combined[0])
    with pytest.raises(ValueError,match="precede"):all_model.predict(train.iloc[:2].assign(date=pd.Timestamp("2025-01-10")),train)


def test_relative_objective_and_zeros_are_finite():
    s=np.array([[1.,10.,100.],[0.,0.,0.]])
    assert decision(s,"relative",[10.,10.]).tolist()==[1.,0.]
    assert np.isfinite(decision(s,"mixed",[10.,10.])).all()
    assert decision(s,"mean",[10.,10.])[0]==37.


def test_expected_risk_allocation_matches_exhaustive_small_grid():
    samples=np.array([[1.,2.,3.],[3.,4.,5.]])
    scale=np.array([2.,4.]);upper=np.array([6.,6.]);total=6.
    q=marginal_allocate(samples,upper,total,scale,"mixed")
    def loss(x):return float((np.abs(x[:,None]-samples)*(.5/samples+.5/scale[:,None])).mean(axis=1).sum())
    grid=[loss(np.array([v,total-v])) for v in np.linspace(0,total,601)]
    assert loss(q)<=min(grid)+1e-12
    assert abs(q.sum()-total)<1e-8
    assert marginal_allocate(samples,np.ones(2),total,scale) is None


def test_history_missing_supply_is_not_zero_and_target_quantity_ignored():
    attrs=pd.DataFrame({"_dish_norm":["a","b"],"菜品类别":["荤菜","荤菜"]})
    h=RecordedHistory(attrs)
    past=pd.DataFrame({"date":[pd.Timestamp("2025-01-01")],"餐次":["午餐"],"_dish_norm":["a"],"菜品售卖单位":["份"],"菜品总销量":[10.],"_calendar_state":["普通工作日"],"_temperature_band":["温和"],"_rain_band":["无雨"],"是否校历寒暑假":[0]})
    h.append(past)
    assert ("午餐","b","份") not in h.items
    menu=pd.DataFrame({"date":[pd.Timestamp("2025-01-03")],"食堂名称":["A"],"餐次":["午餐"],"菜品名称":["a"]})
    f=h.features(menu,{},["份"])
    menu["菜品总销量"]=1e10
    g=h.features(menu,{},["份"])
    pd.testing.assert_frame_equal(f,g)
    assert f.sample_count.iloc[0]==1 and f.low_threshold.iloc[0]==10.
    assert h.features(menu.assign(date=pd.Timestamp("2025-01-05")),{},["份"]).sample_count.iloc[0]==1

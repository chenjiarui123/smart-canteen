import numpy as np
import pandas as pd
import pytest

from structural_policy import allocate,ForecastReliability
from structural_features import StructuralHistory
from structural_borrowing import PeerResidualModel


def fixture():
    date=pd.Timestamp("2025-08-05")
    menu=pd.DataFrame({"date":[date]*4,"食堂名称":["A"]*4,"餐次":["午餐"]*4,
        "菜品名称":["a","b","c","d"],"菜品类别":["荤菜","荤菜","主食","主食"],
        "pred_qty":[20.,80.,10.,10.],"pred_orders":[10.,40.,5.,5.],"pred_rev":[40.,160.,20.,20.],
        "expected_orders_per_qty":[.5]*4,"expected_rev_per_qty":[2.]*4,"min_batch_qty":[19.,70.,8.,8.],"pred_total_qty":[120.]*4})
    f=menu[["date","食堂名称","餐次"]].copy()
    f["_dish_norm"]=["a","b","c","d"]
    f["baseline_qty"]=menu.pred_qty
    f["structural_history_max_date"]=date-pd.Timedelta(days=1)
    f["date_conditioned"]=[20.,80.,10.,10.]
    f["q99"]=[40.,150.,20.,20.]
    f["sale_unit"]=["克","克","份","份"]
    f["dish_category"]=menu["菜品类别"]
    f["calendar_state"]="普通工作日"
    f["unit_share"]=[.01,.99,.5,.5]
    f["within_category_share"]=[.01,.99,.5,.5]
    f["category_share"]=1.
    f["competition_factor"]=1.
    f["sample_count"]=10;f["n14"]=3;f["q25"]=menu.min_batch_qty
    f["median"]=menu.pred_qty;f["mean"]=menu.pred_qty;f["std"]=0.
    f["generated_category_count"]=2;f["state_share_n"]=5
    f["temperature_band"]="温和";f["rain_band"]="无雨"
    menu["主要食材"]=["猪肉","鸡肉","面粉","面粉"]
    menu["制作方法"]=["炒","炖","蒸","蒸"]
    return menu,f


def test_fixed_session_unit_totals_and_names():
    menu,f=fixture()
    new,a,_=allocate(menu,f,{"shares":"flat","separate_preparation":True})
    assert abs(new.pred_qty.sum()-120)<1e-8
    assert abs(new.pred_qty.iloc[:2].sum()-100)<1e-8
    assert abs(new.pred_qty.iloc[2:].sum()-20)<1e-8
    assert new["菜品名称"].tolist()==menu["菜品名称"].tolist()
    np.testing.assert_allclose(new.pred_orders,new.pred_qty*.5)
    np.testing.assert_allclose(new.pred_rev,new.pred_qty*2)
    assert a["new_hhi"]<=a["baseline_hhi"]*1.15+1e-10
    assert a["top_share_increase"]<=.03+1e-10


def test_demand_is_not_forced_to_operating_minimum():
    menu,f=fixture()
    new,a,_=allocate(menu,f,{"shares":"flat","separate_preparation":True})
    assert a["below_minimum_demand_items"]>0
    assert a["below_minimum_preparation_items"]==0
    assert new.minimum_batch_surplus_risk.sum()>0
    assert (new.suggested_preparation_qty>=new.pred_qty).all()


def test_separation_alone_does_not_claim_sales_improvement():
    menu,f=fixture()
    new,a,_=allocate(menu,f,{"identity":True,"separate_preparation":True})
    np.testing.assert_array_equal(new.pred_qty,menu.pred_qty)


def test_conflict_is_explicit_no_cross_unit_surplus():
    menu,f=fixture()
    f.loc[0:1,"q99"]=.1
    new,a,_=allocate(menu,f,{"shares":"flat","separate_preparation":True})
    assert "conflict" in a["status"] and a["conflicts"]
    np.testing.assert_array_equal(new.pred_qty,menu.pred_qty)


def test_target_actual_columns_do_not_affect_forecast():
    menu,f=fixture()
    one,_,_=allocate(menu,f,{"shares":"flat"})
    f["actual_qty"]=[0,1e9,23,10]
    f["hit"]=[True,False,True,False]
    two,_,_=allocate(menu,f,{"shares":"flat"})
    pd.testing.assert_frame_equal(one,two)


def test_history_and_reliability_date_barriers():
    menu,f=fixture()
    f["structural_history_max_date"]=menu.date
    with pytest.raises(ValueError,match="nonhistorical"):
        allocate(menu,f,{})
    f["structural_history_max_date"]-=pd.Timedelta(days=1)
    r=ForecastReliability()
    np.testing.assert_array_equal(r.factors(f),np.ones(4))
    actual=menu[["date","食堂名称","餐次","菜品名称"]].assign(菜品总销量=1.,订单数量=1.,销售额=1.)
    r.observe_day(f,[20.,80.,10.,10.],actual)
    with pytest.raises(ValueError,match="previous-day"):
        r.factors(f)
    np.testing.assert_array_equal(ForecastReliability().factors(f),np.ones(4))


def test_holiday_only_share_not_applied_to_normal_day():
    hist=[(pd.Timestamp("2025-01-01"),"元旦","寒冷|无雨",.9)]*30
    value,_,n,_=StructuralHistory.conditional(hist,"普通工作日","寒冷|无雨",3,.1)
    assert value==.1 and n==0


def test_competition_uses_generated_structure():
    attrs=pd.DataFrame({"_dish_norm":["a","b","c","d"],"菜品类别":["荤菜","荤菜","主食","主食"]})
    history=StructuralHistory(attrs)
    # Four past observations: category competition differs while own share falls.
    actual=[]
    for i in range(1,7):
        date=pd.Timestamp("2025-07-20")+pd.Timedelta(days=i)
        names=["a","b"] if i%2 else ["a"]
        for name in names:
            actual.append({"date":date,"食堂名称":"A","餐次":"午餐","_dish_norm":name,"菜品总销量":10.,
                "菜品售卖单位":"克","_calendar_state":"普通工作日","_temperature_band":"温和","_rain_band":"无雨"})
    history.append(pd.DataFrame(actual))
    menu,f=fixture()
    f["temperature_band"]="温和";f["rain_band"]="无雨"
    full=history.augment(menu,f)
    reduced=history.augment(menu.iloc[[0,2,3]],f.iloc[[0,2,3]])
    assert full.loc[0,"generated_category_count"]==2
    assert reduced.loc[0,"generated_category_count"]==1
    assert full.loc[0,"competition_factor"]!=reduced.loc[0,"competition_factor"]


def test_peer_learning_is_past_only_and_unit_separated():
    menu,f=fixture()
    model=PeerResidualModel()
    prediction,x=model.predict(menu,f)
    np.testing.assert_array_equal(prediction,menu.pred_qty)
    actual=menu.iloc[:2][["date","食堂名称","餐次","菜品名称"]].assign(菜品总销量=1.,订单数量=1.,销售额=1.)
    model.observe_day(f,x,prediction,actual)
    with pytest.raises(ValueError,match="target/future"):
        model.predict(menu,f)
    f["date"]+=pd.Timedelta(days=1);menu["date"]+=pd.Timedelta(days=1)
    prediction,_=model.predict(menu,f)
    np.testing.assert_array_equal(prediction[2:],menu.pred_qty.to_numpy()[2:])
    assert (prediction[:2]<menu.pred_qty.to_numpy()[:2]).all()


def test_peer_unknown_ordinary_state_does_not_get_holiday_only_uplift():
    menu,f=fixture()
    model=PeerResidualModel()
    f["calendar_state"]="春节"
    estimate,x=model.predict(menu,f)
    actual=menu[["date","食堂名称","餐次","菜品名称"]].assign(菜品总销量=1000.,订单数量=1.,销售额=1.)
    model.observe_day(f,x,estimate,actual)
    menu["date"]+=pd.Timedelta(days=1);f["date"]+=pd.Timedelta(days=1)
    f["calendar_state"]="普通工作日";f["state_share_n"]=0
    prediction,_=model.predict(menu,f)
    assert (prediction<=menu.pred_qty.to_numpy()).all()

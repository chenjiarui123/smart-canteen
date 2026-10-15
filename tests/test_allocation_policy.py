import numpy as np
import pandas as pd
import pytest

from allocation_features import OfferedHistory
from allocation_policy import allocate_session, bounded_projection, HistoricalBias


def fixture():
    date=pd.Timestamp("2025-08-05")
    menu=pd.DataFrame({"date":[date]*3,"食堂名称":["A"]*3,"餐次":["午餐"]*3,
        "菜品名称":["a","b","c"],"pred_qty":[20.,30.,50.],"pred_orders":[10.,15.,25.],
        "pred_rev":[40.,60.,100.],"expected_orders_per_qty":[.5]*3,
        "expected_rev_per_qty":[2.]*3,"min_batch_qty":[5.,10.,20.],"pred_total_qty":[100.]*3})
    f=menu[["date","食堂名称","餐次"]].copy()
    f["_dish_norm"]=["a","b","c"]
    f["baseline_qty"]=menu.pred_qty
    f["history_max_date"]=date-pd.Timedelta(days=1)
    f["mean"]=[10.,40.,40.]
    f["q99"]=[40.,60.,70.]
    f["sample_count"]=[8,20,30]
    f["sale_unit"]="份"
    f["calendar_state"]="普通工作日"
    return menu,f


def test_same_menu_total_metadata_minima_and_ratios():
    m,f=fixture()
    new,a=allocate_session(m,f,{})
    assert abs(new.pred_qty.sum()-m.pred_qty.sum())<1e-8
    assert (new.pred_qty>=new.min_batch_qty-1e-8).all()
    pd.testing.assert_frame_equal(m.drop(columns=["pred_qty","pred_orders","pred_rev"]),new.drop(columns=["pred_qty","pred_orders","pred_rev"]))
    np.testing.assert_allclose(new.pred_orders,new.pred_qty*.5)
    np.testing.assert_allclose(new.pred_rev,new.pred_qty*2)


def test_infeasible_box_explicit_fallback():
    m,f=fixture()
    f["q99"]=1.
    new,a=allocate_session(m,f,{})
    assert a["status"]=="infeasible_bounds_baseline_fallback"
    assert a["upper_shortfall"]>0
    np.testing.assert_array_equal(new.pred_qty,m.pred_qty)


@pytest.mark.parametrize("offset",[0,1])
def test_no_target_or_future_sales(offset):
    m,f=fixture()
    f["history_max_date"]=m.date.iloc[0]+pd.Timedelta(days=offset)
    with pytest.raises(ValueError,match="history"):
        allocate_session(m,f,{})


def test_no_surplus_dumping():
    m,f=fixture()
    f["mean"]=[1.,1.,1000.]
    f["q99"]=10000.
    new,a=allocate_session(m,f,{"change_cap":20})
    assert a["new_hhi"]<=a["baseline_hhi"]*1.15+1e-10
    assert a["new_top_share"]<=a["baseline_top_share"]+.03+1e-10
    assert abs(new.pred_qty.sum()-100)<1e-8


def test_uncertain_unit_frozen_not_converted():
    m,f=fixture()
    f.loc[0,"sale_unit"]="��"
    n,a=allocate_session(m,f,{})
    assert n.pred_qty.iloc[0]==m.pred_qty.iloc[0]
    assert a["uncertain_unit_frozen_items"]==1


def test_missing_supply_not_zero():
    h=OfferedHistory()
    h.append(pd.DataFrame({"date":pd.to_datetime(["2025-08-01","2025-08-02"]),"餐次":["午餐"]*2,
        "_dish_norm":["a","b"],"菜品总销量":[10,90],"菜品售卖单位":["份"]*2}))
    m,_=fixture()
    f=h.features(m,{})
    assert f.loc[0,"mean"]==10
    assert f.loc[0,"sample_count"]==1
    assert f.loc[0,"explicit_zero_count"]==0
    assert f.loc[0,"offer_frequency"]==.5


def test_unit_history_not_pooled():
    h=OfferedHistory()
    h.append(pd.DataFrame({"date":pd.to_datetime(["2025-08-01","2025-08-02"]),"餐次":["午餐"]*2,
        "_dish_norm":["a","a"],"菜品总销量":[500,10],"菜品售卖单位":["克","份"]}))
    m,_=fixture()
    f=h.features(m,{})
    assert f.loc[0,"mean"]==10
    assert f.loc[0,"incompatible_unit_records"]==1


def test_bias_requires_previous_day_and_resets():
    m,f=fixture()
    b=HistoricalBias()
    np.testing.assert_array_equal(b.factors(f),np.ones(3))
    truth=f[["date","食堂名称","餐次","_dish_norm"]].assign(actual_qty=10.)
    b.observe(f,np.ones(3)*20,truth)
    with pytest.raises(ValueError,match="target date"):
        b.factors(f)
    f["date"]+=pd.Timedelta(days=1)
    assert (b.factors(f)<1).all()
    np.testing.assert_array_equal(HistoricalBias().factors(f),np.ones(3))


def test_projection_minimal_example():
    np.testing.assert_allclose(bounded_projection([2,2],[1,1],[1,1],[3,8],10),[3,7])
    assert bounded_projection([2,2],[1,1],[1,1],[3,8],12) is None


def test_prediction_ignores_evaluation_columns():
    m,f=fixture()
    n,_=allocate_session(m,f,{})
    f["actual_qty"]=[1e9,0,500]
    f["is_hit"]=[True,False,True]
    other,_=allocate_session(m,f,{})
    pd.testing.assert_frame_equal(n,other)


def test_inherited_minimum_conflict_never_claimed_feasible():
    m,f=fixture()
    m["min_batch_qty"]=100.
    n,a=allocate_session(m,f,{})
    np.testing.assert_array_equal(n.pred_qty,m.pred_qty)
    assert a["status"]=="infeasible_baseline_minimum"
    assert not a["allocation_feasible"]
    assert a["minimum_total_shortfall"]==200.

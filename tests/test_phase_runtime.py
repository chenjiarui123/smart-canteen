"""Current rice-merged, fingerprint-protected fixtures (never old cache)."""
import json
import numpy as np
import pandas as pd
import pytest
import policy
from phase_data import OUT,verify
from phase_features import PhaseHistory,assets
from phase_calendar import PhaseCalendar
from phase_models import walk
from phase_item_experiments import load
from demand_coordination import coordinate


def test_target_outcome_poison_does_not_change_context():
    a=assets("a_hupan");day=a["actual"].loc[a["actual"].date.eq(a["actual"].date.min())].copy()
    original=policy.build_contexts(a["warmup"],day,a["attributes"])
    poisoned=day.copy();poisoned["菜品名称"]="target_name_must_never_be_used";poisoned["_dish_norm"]="poison"
    for col in ["菜品总销量","菜品销售额","菜品订单量"]:
        if col in poisoned:poisoned[col]=99999999
    changed=policy.build_contexts(a["warmup"],poisoned,a["attributes"])
    for left,right in zip(original,changed):
        pd.testing.assert_frame_equal(left["features"],right["features"],check_exact=True)
        assert left["target_context"]==right["target_context"]
        assert left["max_history_date"]<day.date.iloc[0]


def test_current_cached_drop_first_day_reproduces():
    path=OUT/"experiments/own_drop_a/a_hupan/distribution.npz"
    if not path.exists():pytest.skip("fresh drop fixture not ready")
    a,f,t=load("a_hupan");day=f.loc[f.date.eq(f.date.min())].reset_index(drop=True)
    point,samples,_,_,_,_=walk(day,t,{"mode":"drop"})
    cache=np.load(path)
    np.testing.assert_allclose(samples,cache["samples"][:len(day)],rtol=1e-12,atol=1e-8)
    # Poison current/future labels. date<t filtering must keep predictions same.
    poisoned=t.copy();poisoned.loc[poisoned.date.ge(day.date.iloc[0]),"target_y"]=1e15
    actual=walk(day,poisoned,{"mode":"drop"})[0]
    np.testing.assert_allclose(point,actual,rtol=1e-12,atol=1e-8)


def test_new_coordinator_preserves_metadata_units_and_demand_preparation():
    a,f,t=load("a_hupan");key,index=next(iter(a["baseline_menu"].groupby(policy.GROUP_KEYS,sort=False).indices.items()))
    menu=a["baseline_menu"].iloc[index];nf=f.iloc[index];old=a["base_features"].iloc[index].reset_index(drop=True)
    point=nf.historical_scale.to_numpy(float);samples=np.tile(point[:,None],(1,41))
    new,audit=coordinate(menu,old,nf,point,samples,"risk")
    keep=[c for c in menu if c not in {"pred_qty","pred_orders","pred_rev"}]
    pd.testing.assert_frame_equal(menu[keep],new[keep],check_exact=True)
    assert abs(menu.pred_qty.sum()-new.pred_qty.sum())<1e-8
    assert audit["max_unit_total_error"]<1e-8
    np.testing.assert_allclose(new.pred_orders,new.pred_qty*new.expected_orders_per_qty)
    np.testing.assert_allclose(new.pred_rev,new.pred_qty*new.expected_rev_per_qty)
    np.testing.assert_allclose(new.suggested_preparation_qty,np.maximum(new.pred_qty,new.min_batch_qty))


def test_original_protected_live_and_snapshot_files_unchanged():
    assert verify()["model_version"]=="targeted_repeat_holiday_20260911"


def test_retained_runtime_is_exact_and_rejects_unapproved_models():
    from phase_runtime import predict_day
    a,f,t=load("a_hupan");day=f.date.min();m=a["baseline_menu"].loc[a["baseline_menu"].date.eq(day)]
    args=(m,f.loc[f.date.eq(day)],a["base_features"].loc[a["base_features"].date.eq(day)],t.loc[t.date<day])
    out,audit=predict_day(*args)
    pd.testing.assert_frame_equal(out[m.columns],m,check_exact=True)
    assert audit["status"]=="retained_baseline"
    with pytest.raises(ValueError):predict_day(*args,config={"mode":"drop"})

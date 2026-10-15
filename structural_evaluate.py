"""Unchanged primary evaluator, separate preparation and stage diagnostics."""
import json
import numpy as np
import pandas as pd
import policy
from diagnostics import hit_quantity_diagnostics,prediction_consistency_diagnostics
from evaluator import prepare_actual
from targeted_search import date_type,dump
from structural_data import OUT
from structural_trace import STAGES

KEYS=policy.GROUP_KEYS+["_dish_norm"]


def hit_stats(h):
    if h.empty:return {"hit_item_count":0,"hit_qty_mape":None}
    e=h.pred_qty-h.actual_qty
    ape=e.abs()/h.actual_qty
    return {"hit_item_count":len(h),"hit_qty_mape":float(100*ape.mean()),"hit_qty_wmape":float(100*e.abs().sum()/h.actual_qty.sum()),
        "hit_qty_median_ape":float(100*ape.median()),"ape_p95":float(100*ape.quantile(.95)),"ape_p99":float(100*ape.quantile(.99)),
        "ape_max":float(100*ape.max()),"hit_qty_mae":float(e.abs().mean()),"weighted_qty_bias":float(100*e.sum()/h.actual_qty.sum()),
        "overprediction_qty":float(e.clip(lower=0).sum()),"underprediction_qty":float((-e).clip(lower=0).sum())}


def evaluate(menu,baseline,features,actual,audits,edges):
    metrics=policy.evaluate_policy_metrics(menu,actual,policy.BASE_CONSTRAINTS)
    hmetrics,hits=hit_quantity_diagnostics(menu,actual)
    metrics.update(hmetrics)
    metrics.update(hit_stats(hits))
    metrics.update(prediction_consistency_diagnostics(menu))
    keep=[c for c in baseline if c not in {"pred_qty","pred_orders","pred_rev"}]
    pd.testing.assert_frame_equal(menu[keep],baseline[keep],check_exact=True)
    bt=baseline.groupby(policy.GROUP_KEYS).pred_qty.sum()
    nt=menu.groupby(policy.GROUP_KEYS).pred_qty.sum()
    np.testing.assert_allclose(bt,nt,rtol=1e-12,atol=1e-8)
    oldhits=hit_quantity_diagnostics(baseline,actual)[1]
    assert set(map(tuple,hits[KEYS].to_numpy()))==set(map(tuple,oldhits[KEYS].to_numpy()))
    metrics.update(fixed_names_counts_metadata=True,same_hit_set=True,max_session_total_error=float((bt-nt).abs().max()),
        max_unit_total_error=float(audits.max_unit_total_error.max()),conflict_sessions=int(audits.status.str.contains("conflict").sum()),
        below_minimum_demand_items=int(audits.below_minimum_demand_items.sum()),below_minimum_preparation_items=int(audits.below_minimum_preparation_items.sum()),
        baseline_below_minimum_items=int(audits.baseline_below_minimum_items.sum()),surplus_risk_qty=float(audits.surplus_risk_qty.sum()),
        maximum_hhi_ratio=float((audits.new_hhi/audits.baseline_hhi).max()),maximum_top_share_increase=float(audits.top_share_increase.max()))
    source=actual.copy()
    source["date_type"]=source.apply(date_type,axis=1)
    source["month"]=source.date.dt.strftime("%Y-%m")
    source["regular_holiday"]=np.where(source.date_type.isin(["普通工作日","普通周末","调休工作日"]),"普通日期含调休","假期")
    sessions=source.groupby(policy.GROUP_KEYS).agg(actual_total=("菜品总销量","sum"),actual_dishes=("_dish_norm","nunique"),date_type=("date_type","first"))
    sessions=sessions.join(bt.rename("baseline_total")).join(nt.rename("pred_total")).fillna({"pred_total":0,"baseline_total":0}).reset_index()
    sessions["qty_error"]=sessions.pred_total-sessions.actual_total
    metrics.update(total_wape=float(100*sessions.qty_error.abs().sum()/sessions.actual_total.sum()),total_mae=float(sessions.qty_error.abs().mean()),
        total_overprediction_qty=float(sessions.qty_error.clip(lower=0).sum()),total_underprediction_qty=float((-sessions.qty_error).clip(lower=0).sum()))
    prep=menu.copy()
    prep["pred_qty"]=prep.suggested_preparation_qty
    prep["pred_orders"]=prep.pred_qty*prep.expected_orders_per_qty
    prep["pred_rev"]=prep.pred_qty*prep.expected_rev_per_qty
    prepmetrics=policy.evaluate_policy_metrics(prep,actual,policy.BASE_CONSTRAINTS)
    prepmetrics.update(hit_quantity_diagnostics(prep,actual)[0])
    metrics["preparation_metrics"]=prepmetrics
    hits=hits.merge(features.drop(columns=["菜品名称"],errors="ignore"),on=KEYS,validate="one_to_one")
    hits["month"]=hits.date.dt.strftime("%Y-%m")
    hits["historical_volume"]=np.where(hits["mean"]<=edges["volume"],"历史低销量","历史高销量")
    hits["frequency"]=np.where(hits.offer_frequency<=edges["frequency"],"低频","高频")
    hits["samples"]=pd.cut(hits.sample_count,[-1,3,10,30,np.inf],labels=["0—3","4—10","11—30","31+"])
    segments=[]
    for dim in ["month","餐次","date_type","historical_volume","frequency","samples"]:
        for name,h in hits.groupby(dim,observed=True):
            segments.append(dict(dimension=dim,segment=str(name),**hit_stats(h)))
    full=[]
    for dim in ["month","餐次","date_type","regular_holiday"]:
        for name,a in source.groupby(dim):
            p=menu.merge(a[policy.GROUP_KEYS].drop_duplicates(),on=policy.GROUP_KEYS,how="inner")
            m=policy.evaluate_policy_metrics(p,a,policy.BASE_CONSTRAINTS)
            m.update(hit_quantity_diagnostics(p,a)[0])
            full.append(dict(dimension=dim,segment=str(name),**m))
    return metrics,hits,pd.DataFrame(segments),pd.DataFrame(full),sessions


def stage_diagnostics(dataset):
    folder=OUT/("development_"+dataset)
    trace=pd.read_pickle(folder/"baseline_stages.pkl.gz")
    actual=pd.read_pickle(folder/"actual.pkl")
    truth=prepare_actual(actual)
    features=pd.read_pickle(folder/"features.pkl")
    edges=json.loads((folder/"group_edges.json").read_text())
    joined=trace.merge(truth[KEYS+["actual_qty"]],on=KEYS,how="left")
    joined=joined.merge(features[KEYS+["date_type","mean","offer_frequency","sample_count"]],on=KEYS,how="left")
    rows=[]
    # The stability stage is an unnormalised weight; its sales interpretation
    # is examined by a controlled same-total/no-stability allocation below.
    allocated=joined.total_allocated.notna()
    def unstabilized(g):
        out=g.date_conditioned/g.date_conditioned.sum()*g.total_allocated.sum() if g.date_conditioned.sum()>0 else g.date_conditioned*0
        return out
    joined["same_total_without_stability"]=np.nan
    for _,g in joined.loc[allocated].groupby(policy.GROUP_KEYS):
        joined.loc[g.index,"same_total_without_stability"]=unstabilized(g)
    positive=joined.actual_qty.gt(0)
    common=positive & joined.repeat_final.notna()
    for stage in STAGES+["same_total_without_stability"]:
        for scope,mask in [("stage_own_hits",positive),("final_menu_common_hits",common)]:
            h=joined.loc[mask & joined[stage].notna()].copy()
            h["pred_qty"]=h[stage]
            rows.append(dict(stage=stage,scope=scope,dimension="overall",segment="all",unnormalized_weight=stage=="stability_weight",**hit_stats(h)))
            if scope=="final_menu_common_hits":
                h["historical_volume"]=np.where(h["mean"]<=edges["volume"],"历史低销量","历史高销量")
                h["frequency"]=np.where(h.offer_frequency<=edges["frequency"],"低频","高频")
                h["samples"]=pd.cut(h.sample_count,[-1,3,10,30,np.inf],labels=["0—3","4—10","11—30","31+"])
                for dimension in ["餐次","date_type","historical_volume","frequency","samples"]:
                    for name,g in h.groupby(dimension,observed=True):
                        rows.append(dict(stage=stage,scope=scope,dimension=dimension,segment=str(name),unnormalized_weight=stage=="stability_weight",**hit_stats(g)))
    pd.DataFrame(rows).to_csv(folder/"stage_error_segments.csv",index=False)
    joined.to_pickle(folder/"stage_actual_diagnostics.pkl.gz",compression="gzip")
    dump(folder/"stage_change_summary.json",{"trace_rows":len(joined),"min_batch_removed_items":int(joined.removed_at_minimum.sum()),
        "repeat_added_items":int(joined.added_at_repeat.sum()),"repeat_removed_items":int(joined.removed_at_repeat.sum()),
        "note":"Missing stage means not selected at that stage, not zero demand. Stage3 is an unnormalised allocation weight, not an independently calibrated sales forecast."})
    print(dataset,pd.DataFrame(rows).query("dimension=='overall' and scope=='final_menu_common_hits'")[["stage","hit_item_count","hit_qty_mape"]].to_string(index=False),flush=True)

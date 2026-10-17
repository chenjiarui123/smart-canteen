"""Separate diagnostic reporting around the unchanged original evaluator."""
import json
import numpy as np
import pandas as pd
import policy
from evaluator import prepare_actual
from diagnostics import hit_quantity_diagnostics
from targeted_search import evaluate as session_evaluate, dump
from structural_evaluate import hit_stats
from demand_evaluate import auc
from phase_calendar import PhaseCalendar
from phase_data import OUT

KEYS=policy.GROUP_KEYS+["_dish_norm"]


def fold_names(dates,dataset):
    p=json.loads((OUT/"preregistered_protocol.json").read_text())
    result=pd.Series("outside_development",index=dates.index)
    for fold in p["development_folds"][dataset]:result.loc[dates.between(fold["start"],fold["end"])]=fold["name"]
    return result


def evaluate(dataset,menu,actual,audit,features=None):
    audit=[dict(a,date=pd.Timestamp(a["date"])) for a in audit]
    metrics,sessions,segments=session_evaluate(menu,actual,audit)
    _,hits=hit_quantity_diagnostics(menu,actual);metrics.update(hit_stats(hits))
    sessions["month"]=sessions.date.dt.strftime("%Y-%m");sessions["fold"]=fold_names(sessions.date,dataset)
    calendar=PhaseCalendar()
    phases=pd.DataFrame([calendar.features(pd.Timestamp(d)) for d in sessions.date])
    sessions=pd.concat([sessions.reset_index(drop=True),phases],axis=1)
    for dim in ["month","fold","餐次","phase_family"]:
        for name,g in sessions.groupby(dim):
            keys=g[policy.GROUP_KEYS];a=actual.merge(keys,on=policy.GROUP_KEYS);p=menu.merge(keys,on=policy.GROUP_KEYS)
            m=policy.evaluate_policy_metrics(p,a,policy.BASE_CONSTRAINTS);m.update(hit_quantity_diagnostics(p,a)[0])
            m.update(dimension=dim,segment=name,session_count=len(g),actual_count_mean=float(g.actual_count.mean()),pred_count_mean=float(g.pred_count.mean()),
                actual_qty_sum=float(g.actual_total.sum()),pred_qty_sum=float(g.pred_total.sum()),total_wape=float(100*g.absolute_error.sum()/g.actual_total.sum()),
                overprediction_qty=float(g.overprediction_qty.sum()),underprediction_qty=float(g.underprediction_qty.sum()))
            segments.append(m)
    if features is not None:
        hits=hits.merge(features[[c for c in features if c not in hits or c in KEYS]],on=KEYS,how="left",validate="one_to_one")
        hits["volume_segment"]=np.where(hits.actual_qty<=hits.extreme_threshold,"extreme_low",np.where(hits.actual_qty<=hits.low_threshold,"low","normal"))
        hits["frequency_group"]=np.where(hits.offer_frequency>=.5,"high","low")
        hits["sample_group"]=np.where(hits.sample_count>=10,"sufficient","sparse")
        for dim in ["volume_segment","frequency_group","sample_group"]:
            for name,g in hits.groupby(dim):segments.append(dict(dimension=dim,segment=name,**hit_stats(g)))
    return metrics,sessions,pd.DataFrame(segments),hits


def raw_diagnostics(dataset,f,points,samples,risk,actual):
    truth=prepare_actual(actual)
    detail=f.merge(truth[KEYS+["actual_qty"]],on=KEYS,how="left",validate="one_to_one")
    detail["raw_prediction"]=points;detail["drop_risk"]=risk
    detail["lower80"],detail["upper80"]=np.quantile(samples,[.1,.9],axis=1)
    detail["own_drop_observed"]=detail.actual_qty/detail.historical_scale<=detail.own_drop_threshold
    detail["month"]=detail.date.dt.strftime("%Y-%m");detail["fold"]=fold_names(detail.date,dataset)
    detail["volume_segment"]=np.where(detail.actual_qty<=detail.extreme_threshold,"extreme_low",np.where(detail.actual_qty<=detail.low_threshold,"low","normal"))
    detail["frequency_group"]=np.where(detail.offer_frequency>=.5,"high","low")
    detail["sample_group"]=np.where(detail.sample_count>=10,"sufficient","sparse")
    rows=[];positive=detail.loc[detail.actual_qty.gt(0)]
    for dim in ["overall","month","fold","餐次","date_type","phase_family","volume_segment","frequency_group","sample_group","own_drop_observed"]:
        for name,g in ([("all",positive)] if dim=="overall" else positive.groupby(dim)):
            for stage in ["old_initial_history","old_date_conditioned","old_final_qty","raw_prediction"]:
                rows.append(dict(dimension=dim,segment=str(name),stage=stage,**hit_stats(g.assign(pred_qty=g[stage]))))
    segments=pd.DataFrame(rows);overall=segments.loc[segments.dimension.eq("overall")].set_index("stage")
    old,new=overall.loc["old_date_conditioned"],overall.loc["raw_prediction"]
    reasons=[]
    if new.hit_qty_mape>old.hit_qty_mape-.25:reasons.append("raw gain <0.25pp")
    if new.hit_qty_wmape>old.hit_qty_wmape*1.05:reasons.append("raw WMAPE regression >5%")
    normal=segments.loc[segments.dimension.eq("volume_segment") & segments.segment.eq("normal")].set_index("stage")
    if len(normal) and normal.loc["raw_prediction","hit_qty_mape"]>normal.loc["old_date_conditioned","hit_qty_mape"]+max(1.,normal.loc["old_date_conditioned","hit_qty_mape"]*.05):reasons.append("normal raw regression")
    obs=detail.loc[detail.actual_qty.notna()];labelled=obs.loc[obs.own_drop_threshold.notna() & obs.sample_count.ge(3)]
    label=labelled.own_drop_observed.to_numpy(bool);prob=labelled.drop_risk.to_numpy(float)
    width=obs.upper80-obs.lower80
    metrics=dict(raw_mape=float(new.hit_qty_mape),raw_wmape=float(new.hit_qty_wmape),raw_bias=float(new.weighted_qty_bias),
        precoordination_gate=dict(eligible=not reasons,reasons=reasons),own_drop_auc=auc(label,prob),
        prior_own_drop_auc=auc(label,labelled.prior_own_drop_rate),drop_labelled_samples=len(labelled),drop_rate=float(label.mean()) if len(label) else None,
        drop_brier=float(np.mean((prob-label)**2)) if len(label) else None,prior_drop_brier=float(np.mean((labelled.prior_own_drop_rate-label)**2)) if len(label) else None,
        interval_coverage=float(((obs.actual_qty>=obs.lower80)&(obs.actual_qty<=obs.upper80)).mean()),interval_width=float(width.mean()),
        interval_score=float((width+10*(obs.lower80-obs.actual_qty).clip(lower=0)+10*(obs.actual_qty-obs.upper80).clip(lower=0)).mean()),
        explicit_zero_hits=int(obs.actual_qty.eq(0).sum()),zero_mae=float(obs.loc[obs.actual_qty.eq(0),"raw_prediction"].mean()) if obs.actual_qty.eq(0).any() else None)
    return metrics,detail,segments


def gate(metrics,segments,baseline,base_segments,item_only=False,raw=None):
    reasons=[]
    total_gain=metrics["total_mape"]<=baseline["total_mape"]*(1-.0005)
    if item_only:
        if not np.isclose(metrics["total_mape"],baseline["total_mape"],rtol=1e-12,atol=1e-8):reasons.append("fixed total metric mismatch")
        if metrics["hit_qty_mape"]>baseline["hit_qty_mape"]-.25:reasons.append("hit MAPE gain <0.25pp")
    elif not total_gain:reasons.append("total MAPE material gain absent")
    if not item_only and metrics["hit_qty_mape"]>baseline["hit_qty_mape"]+max(1.,baseline["hit_qty_mape"]*.05):reasons.append("hit MAPE regression")
    for name,limit in [("hit_qty_wmape",1.05),("total_wape",1.05),("overpred_rev",1.01)]:
        if metrics[name]>baseline[name]*limit+1e-8:reasons.append(name+" regression")
    if metrics["net_benefit"]<baseline["net_benefit"]-abs(baseline["net_benefit"])*.01:reasons.append("net loss >1%")
    if metrics["nonhit_pred_qty_share"]>baseline["nonhit_pred_qty_share"]+.005:reasons.append("nonhit share +>0.5pp")
    if abs(metrics["weighted_qty_bias"])>abs(baseline["weighted_qty_bias"])+2:reasons.append("absolute weighted bias +>2pp")
    for name in ["dish_hit_rate","revenue_coverage","menu_fill_rate"]:
        if name in metrics and metrics[name]<baseline[name]-.01:reasons.append(name+" drop >1pp")
    if metrics["repeat_over_80_sessions"]>0:reasons.append("automatic adjacent repeat >80%")
    merged=segments.merge(base_segments,on=["dimension","segment"],suffixes=("_new","_old"))
    for r in merged.to_dict("records"):
        n=r.get("hit_item_count_old",r.get("hit_qty_count_old",0))
        if pd.isna(n):n=r.get("hit_qty_count_old",0)
        if n>=100 and r["hit_qty_mape_new"]>r["hit_qty_mape_old"]+max(1.,r["hit_qty_mape_old"]*.05):reasons.append("subgroup: "+r["dimension"]+"/"+str(r["segment"]))
    if raw is not None and not raw["precoordination_gate"]["eligible"]:reasons.append("raw entry gate failed")
    return dict(eligible=not reasons,reasons=reasons,total_material_gain=bool(total_gain))

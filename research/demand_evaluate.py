"""Separate raw, interval and risk diagnostics; original evaluator unchanged."""
import numpy as np
import pandas as pd
import policy
from evaluator import prepare_actual
from structural_evaluate import hit_stats

KEYS=policy.GROUP_KEYS+["_dish_norm"]


def auc(label,score):
    label=np.asarray(label,bool);n=label.sum();m=len(label)-n
    if min(n,m)==0:return None
    rank=pd.Series(score).rank(method="average").to_numpy()
    return float((rank[label].sum()-n*(n+1)/2)/(n*m))


def raw_diagnostics(features,point,samples,reference,risk,actual):
    truth=prepare_actual(actual)
    frame=features.copy()
    frame["raw_prediction"]=point;frame["low_risk_probability"]=risk
    frame["lower80"],frame["upper80"]=np.quantile(samples,[.1,.9],axis=1)
    frame["reference_lower80"],frame["reference_upper80"]=np.quantile(reference,[.1,.9],axis=1)
    frame=frame.merge(truth[KEYS+["actual_qty"]],on=KEYS,how="left",validate="one_to_one")
    frame["volume_segment"]=np.where(frame.actual_qty<=frame.extreme_threshold,"极低记录量",np.where(frame.actual_qty<=frame.low_threshold,"低记录量","正常记录量"))
    frame["month"]=frame.date.dt.strftime("%Y-%m")
    frame["frequency_group"]=np.where(frame.offer_frequency>=.5,"高频","低频")
    frame["sample_group"]=np.where(frame.sample_count>=10,"至少10次历史","不足10次历史")
    observed=frame.loc[frame.actual_qty.notna()].copy()
    positive=observed.loc[observed.actual_qty>0]
    rows=[]
    for dim in ["overall","month","餐次","date_type","volume_segment","frequency_group","sample_group"]:
        groups=[("全部",positive)] if dim=="overall" else positive.groupby(dim,observed=True)
        for name,g in groups:
            for stage in ["old_date_conditioned","old_final_qty","raw_prediction"]:
                rows.append(dict(dimension=dim,segment=str(name),stage=stage,**hit_stats(g.assign(pred_qty=g[stage]))))
    y=observed.actual_qty.to_numpy(float);p=observed.low_risk_probability.to_numpy(float)
    low=(y<=observed.low_threshold.to_numpy(float))
    width=observed.upper80-observed.lower80
    score=width+10*(observed.lower80-observed.actual_qty).clip(lower=0)+10*(observed.actual_qty-observed.upper80).clip(lower=0)
    reference_width=observed.reference_upper80-observed.reference_lower80
    probability_prior=float(low.mean())
    m=dict(observed_menu_items=len(observed),positive_hits=len(positive),explicit_zero_hits=int((y==0).sum()),
        zero_prediction_mae=float(observed.loc[observed.actual_qty.eq(0),"raw_prediction"].abs().mean()) if (y==0).any() else None,
        low_event_rate=probability_prior,low_risk_auc=auc(low,p),low_risk_brier=float(np.mean((p-low)**2)),
        retrospective_constant_brier=float(probability_prior*(1-probability_prior)),
        interval_coverage=float(((observed.actual_qty>=observed.lower80)&(observed.actual_qty<=observed.upper80)).mean()),
        interval_width_mean=float(width.mean()),interval_width_relative_to_actual=float(width.sum()/max(y.sum(),1e-12)),
        reference_interval_coverage=float(((observed.actual_qty>=observed.reference_lower80)&(observed.actual_qty<=observed.reference_upper80)).mean()),
        interval_width_ratio=float(width.sum()/max(reference_width.sum(),1e-12)),interval_score_mean=float(score.mean()))
    comparisons=pd.DataFrame(rows)
    old=comparisons.query("dimension=='overall' and stage=='old_date_conditioned'").iloc[0]
    new=comparisons.query("dimension=='overall' and stage=='raw_prediction'").iloc[0]
    normal=comparisons.query("dimension=='volume_segment' and segment=='正常记录量'").set_index("stage")
    reasons=[]
    if new.hit_qty_mape>old.hit_qty_mape-.25:reasons.append("raw MAPE improvement below 0.25pp")
    if new.hit_qty_wmape>old.hit_qty_wmape*1.05:reasons.append("raw WMAPE worsens >5%")
    if "raw_prediction" in normal.index and normal.loc["raw_prediction","hit_qty_mape"]>normal.loc["old_date_conditioned","hit_qty_mape"]+max(1.,normal.loc["old_date_conditioned","hit_qty_mape"]*.05):reasons.append("normal-volume raw error regression")
    m["precoordination_gate"]={"eligible":not reasons,"reasons":reasons}
    m["raw_mape"]=float(new.hit_qty_mape);m["raw_wmape"]=float(new.hit_qty_wmape)
    return m,frame,comparisons

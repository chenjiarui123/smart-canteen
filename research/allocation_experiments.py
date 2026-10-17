"""Development-only allocation experiments; locked-period paths are absent."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import policy
from evaluator import prepare_actual
from diagnostics import hit_quantity_diagnostics, prediction_consistency_diagnostics
from allocation_policy import allocate_session, demand_estimate, HistoricalBias
from allocation_prepare import OUT, ROOT, load_baseline
from targeted_search import date_type, dump

KEYS=policy.GROUP_KEYS+["_dish_norm"]
TRIALS={
 "baseline":("冻结基线，不调整分配",{"identity":True}),
 "offered_mean":("只估计实际供应条件下均值，取消供菜频率对数量的稀释",{}),
 "recent_mean":("近期均值按样本量向长期收缩",{"recent_days":30,"recent_tau":8}),
 "offered_median":("中位数降低异常大销量影响",{"statistic":"median"}),
 "offered_logmean":("几何均值抑制右尾，不丢弃低销量观测",{"statistic":"logmean"}),
 "winsor_mean":("历史10—90分位缩尾均值",{"statistic":"trimmed_mean"}),
 "joint_context":("餐次内类别到联合假期状态分层收缩，非重复乘因子",{"recent_days":30,"context":["category","state"],"context_tau":8}),
 "weather_weekday":("星期及天气条件估计的顺序收缩",{"recent_days":30,"context":["weekday","weather"],"context_tau":12}),
 "relative_projection":("相对误差二次投影，对高销量给予较大调整空间",{"recent_days":30,"projection_power":2}),
 "additive_projection":("等绝对调整成本，检验统一比例缩放的影响",{"recent_days":30,"projection_power":0}),
 "reliable_projection":("样本可靠性参与固定总量投影",{"recent_days":30,"sample_weight":True}),
 "historical_mape_median":("逆销量加权中位数；保留固定总量及防集中约束",{"statistic":"mape_median"}),
 "rolling_dish_bias":("仅使用先前日期的单菜滚动残差修正估计",{"recent_days":30,"bias":"dish","bias_tau":10}),
 "rolling_state_bias":("仅使用先前日期同餐次联合状态残差修正",{"recent_days":30,"bias":"state","bias_tau":20}),
}


def load_assets(dataset):
    folder=OUT/("development_"+dataset)
    coverage=json.loads((folder/"coverage.json").read_text())
    assert coverage["state"]=="complete" and not coverage["locked_2026_data_read"]
    assert policy.sha256_file(folder/"baseline_menu.csv")==coverage["baseline_sha256"]
    assert policy.sha256_file(folder/"features.pkl")==coverage["features_sha256"]
    menu=pd.read_csv(folder/"baseline_menu.csv",parse_dates=["date"])
    features=pd.read_pickle(folder/"features.pkl")
    actual=pd.read_pickle(folder/"actual.pkl")
    assert actual.date.max()<=pd.Timestamp("2025-12-31")
    return menu,features,actual


def walk(menu,features,actual,config):
    fg={k:g for k,g in features.groupby(policy.GROUP_KEYS,sort=False)}
    truth=prepare_actual(actual)
    daily_truth={d:g for d,g in truth.groupby("date",sort=False)}
    bias=HistoricalBias(config.get("bias_tau",10),config.get("bias","dish"),config.get("bias_strength",1))
    result,audits=[],[]
    for day,dm in menu.groupby("date",sort=True):
        observations=[]
        for key,group in dm.groupby(policy.GROUP_KEYS,sort=False):
            f=fg[key].set_index("_dish_norm").loc[group["菜品名称"].map(policy.normalize_name)].reset_index()
            factors=bias.factors(f) if config.get("bias") else None
            output,audit=allocate_session(group,f,config,factors)
            result.append(output)
            audits.append(audit)
            observations.append((f,demand_estimate(f,config)))
        # All meals use the same beginning-of-day residual history.
        if config.get("bias"):
            for f,estimate in observations:
                bias.observe(f,estimate,daily_truth[day])
    output=pd.concat(result).sort_index()
    return output,pd.DataFrame(audits)


def hit_summary(g):
    if g.empty:
        return {"hit_item_count":0,"hit_qty_mape":None}
    error=g.pred_qty-g.actual_qty
    ape=error.abs()/g.actual_qty
    return {"hit_item_count":len(g),"hit_qty_mape":float(100*ape.mean()),
            "hit_qty_wmape":float(100*error.abs().sum()/g.actual_qty.sum()),
            "hit_qty_median_ape":float(100*ape.median()),"ape_p95":float(100*ape.quantile(.95)),
            "ape_p99":float(100*ape.quantile(.99)),"ape_max":float(100*ape.max()),
            "hit_qty_bias":float(error.sum()),"hit_qty_mae":float(error.abs().mean()),
            "overprediction_qty":float(error.clip(lower=0).sum()),"underprediction_qty":float((-error).clip(lower=0).sum()),
            "overprediction_rate":float((error>0).mean())}


def diagnose(menu,features,actual,audits,baseline,edges):
    metrics=policy.evaluate_policy_metrics(menu,actual,policy.BASE_CONSTRAINTS)
    hm,hits=hit_quantity_diagnostics(menu,actual)
    metrics.update(hm)
    metrics.update(hit_summary(hits))
    metrics.update(prediction_consistency_diagnostics(menu))
    keep=[c for c in baseline.columns if c not in {"pred_qty","pred_orders","pred_rev"}]
    pd.testing.assert_frame_equal(menu[keep].reset_index(drop=True),baseline[keep].reset_index(drop=True))
    bt=baseline.groupby(policy.GROUP_KEYS).pred_qty.sum()
    nt=menu.groupby(policy.GROUP_KEYS).pred_qty.sum()
    assert np.allclose(bt,nt,rtol=1e-12,atol=1e-8)
    metrics.update(menu_names_counts_metadata_identical=True,max_session_sum_error=float((bt-nt).abs().max()),
                   bounds_fallback_sessions=int(audits.status.str.contains("fallback").sum()),
                   baseline_empty_sessions=int(actual.groupby(policy.GROUP_KEYS).ngroups-menu.groupby(policy.GROUP_KEYS).ngroups),
                   historical_upper_violations=int(audits.historical_upper_violations.sum()),
                   min_batch_violations=int(audits.min_batch_violations.sum()),
                   inherited_minimum_infeasible_sessions=int(audits.status.eq("infeasible_baseline_minimum").sum()),
                   new_min_batch_violations=int((audits.min_batch_violations-audits.baseline_minimum_violations).clip(lower=0).sum()),
                   max_hhi_ratio=float((audits.new_hhi/audits.baseline_hhi).max()),
                   max_top_share_increase=float((audits.new_top_share-audits.baseline_top_share).max()))
    assert metrics["new_min_batch_violations"]==0
    extra=features.drop(columns=["菜品名称"])
    hits=hits.merge(extra,on=KEYS,validate="one_to_one")
    hits["month"]=hits.date.dt.strftime("%Y-%m")
    hits["historical_volume"]=np.where(hits["mean"]<=edges["volume"],"历史低销量","历史高销量")
    hits["frequency"]=np.where(hits.offer_frequency<=edges["frequency"],"低出菜频率","高出菜频率")
    hits["samples"]=pd.cut(hits.sample_count,[-1,3,10,30,np.inf],labels=["0—3","4—10","11—30","31+"])
    hits["recent_samples"]=pd.cut(hits.n14,[-1,0,3,10,np.inf],labels=["0","1—3","4—10","11+"])
    hits["error_direction"]=np.where(hits.pred_qty>hits.actual_qty,"高估","低估或相等")
    segments=[]
    for dimension in ["month","餐次","date_type","historical_volume","frequency","samples","recent_samples","error_direction"]:
        for name,g in hits.groupby(dimension,observed=True):
            segments.append(dict(dimension=dimension,segment=str(name),**hit_summary(g)))
    # Full-session date/month/meal metrics include empty menus and all actuals.
    source=actual.copy()
    source["date_type"]=source.apply(date_type,axis=1)
    source["month"]=source.date.dt.strftime("%Y-%m")
    source["regular_holiday"]=np.where(source.date_type.isin(["普通工作日","普通周末","调休工作日"]),"普通日期含调休","假期")
    sessions=source.groupby(policy.GROUP_KEYS).agg(actual_total=("菜品总销量","sum"),date_type=("date_type","first")).join(nt.rename("pred_total")).fillna({"pred_total":0}).reset_index()
    sessions["error"]=sessions.pred_total-sessions.actual_total
    metrics.update(total_wape=float(100*sessions.error.abs().sum()/sessions.actual_total.sum()),
                   total_mae=float(sessions.error.abs().mean()),total_qty_bias=float(sessions.error.sum()))
    full=[]
    for dimension in ["month","餐次","date_type","regular_holiday"]:
        for name,a in source.groupby(dimension):
            m=menu.merge(a[policy.GROUP_KEYS].drop_duplicates(),on=policy.GROUP_KEYS,how="inner")
            values=policy.evaluate_policy_metrics(m,a,policy.BASE_CONSTRAINTS)
            values.update(hit_quantity_diagnostics(m,a)[0])
            full.append(dict(dimension=dimension,segment=str(name),**values))
    metrics["raw_before_total_calibration_hit_mape"]=float(100*((hits.uncalibrated_qty-hits.actual_qty).abs()/hits.actual_qty).mean())
    metrics["baseline_calibration_ratio_median"]=float((hits.baseline_qty/hits.uncalibrated_qty).median())
    metrics["incompatible_unit_records"]=int(features.incompatible_unit_records.sum())
    metrics["no_positive_history_rows"]=int(features.sample_count.eq(0).sum())
    metrics["no_recent14_history_rows"]=int(features.n14.eq(0).sum())
    metrics["explicit_zero_history_references"]=int(features.explicit_zero_count.sum())
    return metrics,hits,pd.DataFrame(segments),pd.DataFrame(full),sessions


def run_trial(name,config,hypothesis,datasets=("a","f")):
    load_baseline() # verify web default/source unchanged before every trial
    folder=OUT/"experiments"/name
    folder.mkdir(parents=True,exist_ok=False)
    start=time.time()
    hashes={n:policy.sha256_file(ROOT/n) for n in ["allocation_policy.py","allocation_features.py","allocation_experiments.py","policy.py","evaluator.py","diagnostics.py"]}
    record={"name":name,"config":config,"hypothesis":hypothesis,"source_hashes":hashes,"status":"running","locked_data_read":False,"started_at":pd.Timestamp.now().isoformat()}
    dump(folder/"record.json",record)
    for n in ["allocation_policy.py","allocation_features.py","allocation_experiments.py"]:
        shutil.copy2(ROOT/n,folder/n)
    try:
        for dataset in datasets:
            baseline,features,actual=load_assets(dataset)
            edgepath=OUT/("development_"+dataset)/"group_edges.json"
            if not edgepath.exists():
                dump(edgepath,{"volume":float(features["mean"].median()),"frequency":float(features.offer_frequency.median()),"source":"2025 development historical features only; reporting bins, not prediction features"})
            edges=json.loads(edgepath.read_text())
            menu,audits=walk(baseline,features,actual,config)
            metrics,hits,segments,full,sessions=diagnose(menu,features,actual,audits,baseline,edges)
            dump(folder/(dataset+"_metrics.json"),metrics)
            audits.to_csv(folder/(dataset+"_allocation_audit.csv"),index=False)
            segments.to_csv(folder/(dataset+"_hit_segments.csv"),index=False)
            full.to_csv(folder/(dataset+"_session_segments.csv"),index=False)
            sessions.to_csv(folder/(dataset+"_sessions.csv"),index=False)
            menu.to_pickle(folder/(dataset+"_menu.pkl"))
            hits.to_pickle(folder/(dataset+"_hits.pkl"))
            record[dataset]=metrics
            dump(folder/"record.json",record)
            print(name,dataset,"hit",round(metrics["hit_qty_mape"],4),"total",round(metrics["total_mape"],4),"net",round(metrics["net_benefit"],2),"fallback",metrics["bounds_fallback_sessions"],flush=True)
        record.update(status="complete",elapsed_seconds=time.time()-start)
    except Exception as exc:
        record.update(status="failed",failure=repr(exc),elapsed_seconds=time.time()-start)
        dump(folder/"record.json",record)
        raise
    dump(folder/"record.json",record)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("names",nargs="*",default=list(TRIALS))
    parser.add_argument("--datasets",nargs="+",default=["a","f"])
    parser.add_argument("--spec",type=Path)
    args=parser.parse_args()
    if args.spec:
        specs=json.loads(args.spec.read_text())
        for spec in specs:
            run_trial(spec["name"],spec["config"],spec["hypothesis"],args.datasets)
    else:
        for name in args.names:
            hypothesis,config=TRIALS[name]
            run_trial(name,config,hypothesis,args.datasets)

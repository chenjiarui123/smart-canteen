"""Development-only complete rolling experiments, with immutable pre-set gates."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path
import pandas as pd
import policy
from structural_data import OUT,ROOT,DATASETS,verify_protected
from structural_policy import allocate,ForecastReliability,initial_estimate
from structural_borrowing import PeerResidualModel
from structural_evaluate import evaluate
from targeted_search import dump

SOURCES=["structural_data.py","structural_policy.py","structural_borrowing.py","structural_features.py","structural_evaluate.py","structural_experiments.py","structural_protocol.json","evaluator.py","diagnostics.py","allocation_policy.py"]
IMPORT_HASHES={name:policy.sha256_file(ROOT/name) for name in SOURCES}
TRIALS={
    "baseline":("Current cleaned deployed baseline with final menu and totals saved",{"identity":True}),
    "baseline_f":("F current cleaned deployed baseline; separate reference because F feature preparation finished after A",{"identity":True}),
    "preparation_only":("A: separate operating minimum advice, keep the original sales forecast; no redefinition-based improvement claim",{"identity":True,"separate_preparation":True}),
    "shares_flat":("B1: conditional item share within exact sale-unit group, learned only on observed historical supply",{"shares":"flat"}),
    "shares_hierarchical":("B2: learned unit-group/category/item hierarchy; no fixed category percentages",{"shares":"hierarchical"}),
    "competition_only":("E: past unit-share association with historical competing category count; target uses baseline generated menu only",{"competition":True}),
    "reliability_only":("F: retain date-conditioned demand instead of volatility discount, reconcile using prior out-of-sample residual reliability",{"reliability":True,"remove_volatility_discount":True}),
}


def load_assets(dataset):
    folder=OUT/("development_"+dataset)
    coverage=json.loads((folder/"coverage.json").read_text())
    fm=json.loads((folder/"feature_manifest.json").read_text())
    assert coverage["status"]=="complete" and not coverage["locked_2026_sales_loaded"]
    assert policy.sha256_file(folder/"baseline_menu.pkl")==coverage["baseline_sha256"]
    assert policy.sha256_file(folder/"features.pkl")==fm["sha256"]
    actual=pd.read_pickle(folder/"actual.pkl")
    assert actual.date.max()<=pd.Timestamp("2025-12-31")
    return pd.read_pickle(folder/"baseline_menu.pkl"),pd.read_pickle(folder/"features.pkl"),actual,json.loads((folder/"group_edges.json").read_text())


def walk(baseline,features,actual,config):
    fg={k:g for k,g in features.groupby(policy.GROUP_KEYS,sort=False)}
    days={d:g for d,g in actual.groupby("date",sort=False)}
    reliability=ForecastReliability()
    peer=PeerResidualModel()
    results,audits=[],[]
    for date,today in baseline.groupby("date",sort=True):
        observation_features=[];estimates=[]
        peer_features=[];peer_design=[]
        for key,menu in today.groupby(policy.GROUP_KEYS,sort=False):
            f=fg[key].set_index("_dish_norm").loc[menu["菜品名称"].map(policy.normalize_name)].reset_index()
            override=None
            if config.get("peer_borrowing"):
                pf=f.copy()
                pf["baseline_qty"]=initial_estimate(f,config)
                override,x=peer.predict(menu,pf)
                peer_features.append(pf);peer_design.append(pd.DataFrame(x))
            new,audit,estimate=allocate(menu,f,config,reliability,override)
            results.append(new);audits.append(audit)
            observation_features.append(f);estimates.extend(estimate)
        if config.get("reliability"):
            reliability.observe_day(pd.concat(observation_features,ignore_index=True),estimates,days[date])
        if config.get("peer_borrowing"):
            peer.observe_day(pd.concat(peer_features,ignore_index=True),pd.concat(peer_design,ignore_index=True).to_numpy(),estimates,days[date])
    return pd.concat(results).sort_index(),pd.DataFrame(audits)


def gates(dataset,metrics,segments):
    reference="baseline" if dataset=="a_hupan" else "baseline_f"
    basefolder=OUT/"experiments"/reference/dataset
    if not (basefolder/"metrics.json").exists():
        return {"status":"baseline_reference_not_yet_available"}
    b=json.loads((basefolder/"metrics.json").read_text())
    old=pd.read_csv(basefolder/"session_segments.csv")
    threshold=json.loads((OUT/"preregistered_protocol.json").read_text())["selection"]
    reasons=[]
    if abs(metrics["total_mape"]-b["total_mape"])>1e-8:reasons.append("total MAPE changed")
    if metrics["max_session_total_error"]>1e-8 or metrics["max_unit_total_error"]>1e-8:reasons.append("fixed totals violated")
    if metrics["hit_qty_mape"]>b["hit_qty_mape"]-threshold["minimum_hit_mape_improvement_pp"]:reasons.append("hit MAPE improvement below 0.25pp")
    if metrics["net_benefit"]<b["net_benefit"]*(1-threshold["net_relative_loss_max"]):reasons.append("net loss exceeds 1%")
    if metrics["hit_qty_wmape"]>b["hit_qty_wmape"]*(1+threshold["hit_wmape_relative_increase_max"]):reasons.append("hit WMAPE increase exceeds 5%")
    if metrics["nonhit_pred_qty_share"]>b["nonhit_pred_qty_share"]+threshold["nonhit_share_absolute_increase_max"]:reasons.append("nonhit demand share increase exceeds 0.5pp")
    if abs(metrics["weighted_qty_bias"])>abs(b["weighted_qty_bias"])+threshold["absolute_weighted_bias_increase_pp_max"]:reasons.append("absolute quantity bias increase exceeds 2pp")
    merged=old.merge(segments,on=["dimension","segment"],suffixes=("_old","_new"))
    failures=[]
    for r in merged.to_dict("records"):
        if r["dimension"] not in {"month","餐次","date_type"} or r["hit_item_count_old"]<threshold["subgroup_min_hits_for_gate"]:continue
        limit=max(threshold["subgroup_hit_mape_increase_pp_max"],r["hit_qty_mape_old"]*threshold["subgroup_hit_mape_relative_increase_max"])
        if r["hit_qty_mape_new"]>r["hit_qty_mape_old"]+limit:
            failures.append({"dimension":r["dimension"],"segment":r["segment"],"old":r["hit_qty_mape_old"],"new":r["hit_qty_mape_new"],"permitted_increase_pp":limit})
    if failures:reasons.append("pre-set subgroup stability gate failed")
    return {"eligible":not reasons,"reasons":reasons,"subgroup_failures":failures,"thresholds_fixed_before_baseline":True}


def run(name,config,hypothesis,datasets):
    verify_protected()
    assert all(policy.sha256_file(ROOT/n)==sha for n,sha in IMPORT_HASHES.items()),"Source changed after this process started"
    folder=OUT/"experiments"/name
    folder.mkdir(parents=True,exist_ok=False)
    record={"name":name,"config":config,"hypothesis":hypothesis,"status":"running","started_at":pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "source_hashes":IMPORT_HASHES,"protocol_sha256":policy.sha256_file(ROOT/"structural_protocol.json"),"test_sales_used":False,"datasets":{}}
    dump(folder/"record.json",record)
    snapshot=folder/"source";snapshot.mkdir()
    for n in SOURCES:shutil.copy2(ROOT/n,snapshot/n)
    started=time.time()
    try:
        for dataset in datasets:
            sub=folder/dataset;sub.mkdir()
            baseline,features,actual,edges=load_assets(dataset)
            menu,audits=walk(baseline,features,actual,config)
            metrics,hits,hitsegments,full,sessions=evaluate(menu,baseline,features,actual,audits,edges)
            review=gates(dataset,metrics,full) if name not in {"baseline","baseline_f"} else {"status":"reference"}
            menu.to_pickle(sub/"menu.pkl.gz",compression="gzip")
            hits.to_pickle(sub/"hits.pkl.gz",compression="gzip")
            audits.to_csv(sub/"allocation_audit.csv",index=False)
            hitsegments.to_csv(sub/"hit_segments.csv",index=False)
            full.to_csv(sub/"session_segments.csv",index=False)
            sessions.to_csv(sub/"sessions.csv",index=False)
            dump(sub/"metrics.json",metrics)
            dump(sub/"gate_review.json",review)
            record["datasets"][dataset]={"metrics":metrics,"gate":review}
            dump(folder/"record.json",record)
            print(name,dataset,"hit",round(metrics["hit_qty_mape"],4),"total",round(metrics["total_mape"],4),"net",round(metrics["net_benefit"],2),"gate",review.get("eligible"),"conflicts",metrics["conflict_sessions"],flush=True)
        record.update(status="complete",seconds=time.time()-started)
    except Exception as exc:
        record.update(status="failed",failure=repr(exc),seconds=time.time()-started)
        dump(folder/"record.json",record)
        raise
    dump(folder/"record.json",record)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("names",nargs="*",default=list(TRIALS))
    parser.add_argument("--datasets",nargs="+",default=list(DATASETS))
    parser.add_argument("--spec",type=Path)
    args=parser.parse_args()
    specs=json.loads(args.spec.read_text()) if args.spec else [{"name":n,"hypothesis":TRIALS[n][0],"config":TRIALS[n][1]} for n in args.names]
    for s in specs:run(s["name"],s["config"],s["hypothesis"],args.datasets)

"""Freeze an optional allocator, then retest existing locked periods once.

Does not install or alter the web model. The only default is the frozen baseline.
"""
from __future__ import annotations

import argparse
import json
import shutil
import types
import pandas as pd

import policy
from allocation_prepare import ROOT,OUT,HISTORY,LIBRARY,load_baseline
from allocation_features import OfferedHistory
from allocation_experiments import walk,diagnose
from targeted_search import date_type,dump


def freeze(a_name,f_name):
    load_baseline()
    folder=OUT/"frozen"
    folder.mkdir(exist_ok=False)
    config={}
    for dataset,name in [("a",a_name),("f",f_name)]:
        record=json.loads((OUT/"experiments"/name/"record.json").read_text())
        assert record["status"]=="complete"
        for n,h in record["source_hashes"].items():
            assert policy.sha256_file(ROOT/n)==h, "Reverify final sources before freeze: "+n
        config[dataset]=record["config"]
    dump(folder/"allocation_config.json",config)
    names=["allocation_policy.py","allocation_features.py","allocation_runtime.py","allocation_experiments.py","allocation_prepare.py","allocation_release.py","policy.py","evaluator.py","diagnostics.py"]
    for name in names:
        shutil.copy2(ROOT/name,folder/name)
    manifest={"version":"fixed_menu_total_allocation_20260911","frozen_at":pd.Timestamp.now().isoformat(),
              "development_runs":{"a":a_name,"f":f_name},"status":"frozen","web_default_switched":False,
              "files":{p.name:policy.sha256_file(p) for p in folder.iterdir() if p.is_file()},
              "web_manifest_sha256":policy.sha256_file(ROOT/"smart_canteen_web/model/targeted_manifest.json"),
              "evaluation_note":"既有2026区间复测，冻结后一次运行；禁止依据复测调参"}
    dump(folder/"allocation_manifest.json",manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


def retest(dataset):
    frozen=OUT/"frozen"
    manifest=json.loads((frozen/"allocation_manifest.json").read_text())
    for name,sha in manifest["files"].items():
        assert policy.sha256_file(frozen/name)==sha
        if name.endswith(".py"):
            assert policy.sha256_file(ROOT/name)==sha
    module,baseline_config,_=load_baseline()
    config=json.loads((frozen/"allocation_config.json").read_text())[dataset]
    folder=OUT/("retest_"+dataset)
    folder.mkdir(exist_ok=False)
    lock={"status":"started","label":"既有锁定区间冻结后复测","started_at":pd.Timestamp.now().isoformat(),
          "freeze_sha256":policy.sha256_file(frozen/"allocation_manifest.json")}
    dump(folder/"retest_lock.json",lock)
    # Locked sales first read AFTER freeze and lock creation.
    path=ROOT/("数据/菜品销量_吉大湖畔智慧餐厅_菜名修正后.csv" if dataset=="a" else "数据/菜品销量_吉大南校基础园智慧餐厅_菜名修正后.csv")
    actual=policy.read_sales(path)
    actual=actual.loc[actual.date.between("2026-01-01","2026-06-10")].copy()
    history=policy.read_sales(HISTORY[dataset])
    assert history.date.max()<actual.date.min()
    attributes,_=policy.build_library_attributes(LIBRARY[dataset])
    offered=OfferedHistory()
    offered.append(history)
    menus,feature_rows,baseline_audits,raw_rows=[],[],[],[]
    menu_history={}
    original_enforce=module.enforce_repeat
    def capture(*args):
        result=original_enforce(*args)
        c=args[-1]
        for r in result.to_dict("records"):
            raw_rows.append({"date":pd.Timestamp(c["date"]),"食堂名称":c["食堂名称"],"餐次":c["餐次"],
                             "_dish_norm":r["_dish_norm"],"uncalibrated_qty":float(r["_allocation_weight"]),
                             "baseline_qty_stability":float(r["_qty_stability"]),"baseline_recent_weight":float(r["_adaptive_recent_weight"])})
        return result
    core=module._integrated_function()
    function=types.FunctionType(core.__code__,dict(core.__globals__,enforce_repeat=capture),core.__name__,core.__defaults__)
    function.__kwdefaults__=core.__kwdefaults__
    for i,(day,day_actual) in enumerate(actual.groupby("date",sort=True)):
        contexts=policy.build_contexts(history,day_actual,attributes)
        for c in contexts:
            c["day_group_count"]=len(contexts)
        audit=[]
        recs=function(contexts,baseline_config,policy.BASE_CONSTRAINTS,selection_audit=audit,initial_menu_history=menu_history)
        for c,a in zip(contexts,audit):
            a.update(c.get("_targeted_audit",{}))
            meal=c["餐次"]
            selected=recs.loc[recs["餐次"].eq(meal)].copy()
            if not selected.empty:
                f=offered.features(selected,c["target_context"])
                f["date_type"]=date_type(day_actual.loc[day_actual["餐次"].eq(meal)].iloc[0])
                feature_rows.append(f)
            menu_history.setdefault((c["食堂名称"],meal),[]).append((pd.Timestamp(day),set(selected["菜品名称"].map(policy.normalize_name))))
        menus.append(recs)
        baseline_audits.extend(audit)
        offered.append(day_actual)
        history=pd.concat([history,day_actual],ignore_index=True)
        if i%20==0:
            print("frozen retest",dataset,i+1,"/",actual.date.nunique(),flush=True)
    baseline=pd.concat(menus,ignore_index=True)
    features=pd.concat(feature_rows,ignore_index=True).merge(pd.DataFrame(raw_rows),on=policy.GROUP_KEYS+["_dish_norm"],validate="one_to_one")
    oldname="hupan" if dataset=="a" else "jichuyuan"
    old=pd.read_csv(ROOT/"runs/targeted_20260911"/("retest_"+oldname)/"new_recommended_menu.csv",parse_dates=["date"])
    pd.testing.assert_frame_equal(baseline.reset_index(drop=True),old.reset_index(drop=True),check_exact=False,rtol=1e-12,atol=1e-8)
    baseline.to_csv(folder/"baseline_menu.csv",index=False)
    features.to_pickle(folder/"features.pkl")
    actual.to_pickle(folder/"actual.pkl")
    dump(folder/"baseline_audit.json",baseline_audits)
    edges=json.loads((OUT/("development_"+dataset)/"group_edges.json").read_text())
    results={}
    for label,cfg in [("baseline",{"identity":True}),("new",config)]:
        menu,audits=walk(baseline,features,actual,cfg)
        metrics,hits,segments,full,sessions=diagnose(menu,features,actual,audits,baseline,edges)
        dump(folder/(label+"_metrics.json"),metrics)
        menu.to_pickle(folder/(label+"_menu.pkl"))
        menu.to_csv(folder/(label+"_menu.csv"),index=False)
        hits.to_pickle(folder/(label+"_hits.pkl"))
        segments.to_csv(folder/(label+"_hit_segments.csv"),index=False)
        full.to_csv(folder/(label+"_session_segments.csv"),index=False)
        sessions.to_csv(folder/(label+"_sessions.csv"),index=False)
        audits.to_csv(folder/(label+"_allocation_audit.csv"),index=False)
        results[label]=metrics
        if dataset=="a":
            for split,start,end in [("validation_retest","2026-01-01","2026-03-31"),("test_retest","2026-04-01","2026-06-10")]:
                keep=menu.date.between(start,end)
                m,h,s,full_sub,ss=diagnose(menu.loc[keep],features.loc[features.date.between(start,end)],actual.loc[actual.date.between(start,end)],
                    audits.loc[audits.date.between(start,end)],baseline.loc[baseline.date.between(start,end)],edges)
                dump(folder/(label+"_"+split+"_metrics.json"),m)
                full_sub.to_csv(folder/(label+"_"+split+"_session_segments.csv"),index=False)
        print("retest result",dataset,label,metrics["hit_qty_mape"],metrics["total_mape"],metrics["net_benefit"],flush=True)
    lock.update(status="complete",finished_at=pd.Timestamp.now().isoformat(),results=results,
                baseline_reproduced=True,actual_sha256=policy.sha256_file(path),history_sha256=policy.sha256_file(HISTORY[dataset]))
    dump(folder/"retest_lock.json",lock)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["freeze","retest"])
    parser.add_argument("names",nargs="+")
    args=parser.parse_args()
    freeze(*args.names) if args.action=="freeze" else retest(*args.names)

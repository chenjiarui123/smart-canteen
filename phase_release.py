"""Freeze development decisions, then once-only existing-interval retest.

No web default switch. Retest is explicitly reused data, not independent test.
"""
import argparse
import json
import shutil
import time
import pandas as pd
import policy
from phase_data import ROOT,OUT,DATA,DATASETS,verify
from structural_data import sales_subset
from allocation_prepare import load_baseline
from structural_trace import BaselineTrace
from allocation_features import OfferedHistory
from phase_evaluate import evaluate
from targeted_search import dump


def freeze():
    verify();records=[]
    for path in sorted((OUT/"experiments").glob("*/record.json")):
        r=json.loads(path.read_text());assert r["status"]!="running",path
        for name,sha in r["source_hashes"].items():assert policy.sha256_file(path.parent/"source"/name)==sha,path
        records.append((path,r))
    chosen={};decisions=[]
    for dataset in DATASETS:
        eligible=[];directions=set();names=[]
        for path,r in records:
            if dataset not in r["datasets"]:continue
            v=r["datasets"][dataset];name=r["name"];names.append(name)
            if name.startswith("phase_"):directions.add("holiday_phase")
            if name.startswith("own_drop"):directions.add("own_relative_drop")
            if "hierarchy" in name:directions.add("category_hierarchy")
            if v.get("gate",{}).get("eligible"):eligible.append((name,v))
            decisions.append(dict(dataset=dataset,name=name,status=v.get("status",r["status"]),eligible=v.get("gate",{}).get("eligible",False),reasons=v.get("gate",{}).get("reasons",v.get("raw",{}).get("precoordination_gate",{}).get("reasons",[]))))
        assert directions=={"holiday_phase","own_relative_drop","category_hierarchy"},(dataset,directions)
        assert any("joint" in n for n in names) and any("no_phase" in n for n in names) and any("drop_hierarchy" in n for n in names),dataset
        if eligible:
            # Never quietly replace an accepted model with baseline merely to
            # simplify delivery. Its opt-in streaming adapter must be verified.
            raise RuntimeError("Eligible candidates require final simplification and streaming equivalence verification before freeze: "+str(eligible))
        chosen[dataset]="baseline"
    frozen=OUT/"frozen";frozen.mkdir(exist_ok=False)
    paths=list(ROOT.glob("phase_*.py"))+[ROOT/n for n in ["phase_protocol.json","phase_trial_design.json","demand_models.py","demand_features.py","demand_coordination.py","structural_policy.py","allocation_policy.py","allocation_features.py","structural_trace.py"]]
    hashes={p.name:policy.sha256_file(p) for p in paths}
    for p in paths:shutil.copy2(p,frozen/p.name)
    dump(frozen/"selection.json",dict(selected=chosen,frozen_at=str(pd.Timestamp.now(tz="Asia/Shanghai")),
        stop="All three distinct directions plus combinations and ablations failed registered full-development/risk gates; no equivalent parameter recycling; baseline retained",decisions=decisions,
        test_used_for_selection=False,web_default_changed=False,source_hashes=hashes,
        record_hashes={str(p.relative_to(OUT)):policy.sha256_file(p) for p,r in records}))
    print("FROZEN",chosen,flush=True)


def verify_freeze():
    verify();frozen=OUT/"frozen";record=json.loads((frozen/"selection.json").read_text())
    for n,h in record["source_hashes"].items():assert policy.sha256_file(ROOT/n)==h and policy.sha256_file(frozen/n)==h,n
    for n,h in record["record_hashes"].items():assert policy.sha256_file(OUT/n)==h,n
    return record


def retest(dataset):
    frozen=verify_freeze();assert frozen["selected"][dataset]=="baseline"
    folder=OUT/"retest"/dataset;folder.mkdir(parents=True,exist_ok=False)
    config=json.loads((OUT/"effective_baseline_config.json").read_text());module,_,_=load_baseline();trace=BaselineTrace(module)
    history=sales_subset(dataset,end="2025-12-31")
    # FIRST post-freeze load of locked 2026 sales and real menus.
    actual=sales_subset(dataset,start="2026-01-01",end="2026-06-10")
    attributes,_=policy.build_library_attributes(DATA/"datasets"/dataset/"dish_library.csv")
    old=pd.read_pickle(OUT/("development_"+dataset)/"baseline_menu.pkl");menu_history={}
    for key,g in old.groupby(policy.GROUP_KEYS,sort=True):menu_history.setdefault((key[1],key[2]),[]).append((key[0],set(g["菜品名称"].map(policy.normalize_name))))
    offered=OfferedHistory();offered.append(history);menus=[];traces=[];audits=[];features=[];schedules=[];start=time.time()
    record=dict(status="running",dataset=dataset,label="既有区间复测",selected="baseline",tuning_after_retest_forbidden=True,first_2026_outcome_load_after_freeze=True)
    dump(folder/"record.json",record)
    try:
        for i,(day,today) in enumerate(actual.groupby("date",sort=True)):
            contexts=policy.build_contexts(history,today,attributes)
            menu,audit,stages=trace.generate(contexts,config,menu_history)
            if i==0:
                direct=module.make_recommendations(contexts,config,policy.BASE_CONSTRAINTS,initial_menu_history=menu_history)
                pd.testing.assert_frame_equal(menu,direct,check_exact=True)
            for c in contexts:
                chosen=menu.loc[menu["餐次"].eq(c["餐次"])]
                if not chosen.empty:
                    features.append(offered.features(chosen,c["target_context"]))
                    menu_history.setdefault((c["食堂名称"],c["餐次"]),[]).append((day,set(chosen["菜品名称"].map(policy.normalize_name))))
            menus.append(menu);traces.append(stages);audits.extend(audit);schedules.append(policy.context_schedule_frame(contexts))
            offered.append(today);history=pd.concat([history,today],ignore_index=True)
            if i%20==0:print(dataset,"frozen existing-interval retest",i+1,"/",actual.date.nunique(),flush=True)
        menu=pd.concat(menus,ignore_index=True);stages=pd.concat(traces,ignore_index=True)
        menu.to_pickle(folder/"menu.pkl.gz",compression="gzip");menu.to_csv(folder/"menu.csv.gz",index=False);actual.to_pickle(folder/"actual.pkl.gz",compression="gzip")
        stages.to_csv(folder/"stages.csv.gz",index=False);pd.concat(features).to_csv(folder/"features.csv.gz",index=False);pd.concat(schedules).to_csv(folder/"schedule.csv",index=False);dump(folder/"menu_audit.json",audits)
        results={}
        for name,start_day,end in [("full_existing_interval","2026-01-01","2026-06-10"),("old_validation" if dataset=="a_hupan" else "q1_diagnostic","2026-01-01","2026-03-31"),("old_test" if dataset=="a_hupan" else "q2_diagnostic","2026-04-01","2026-06-10")]:
            truth=actual.loc[actual.date.between(start_day,end)];p=menu.loc[menu.date.between(start_day,end)]
            m,s,seg,h=evaluate(dataset,p,truth,audits);results[name]=m
            s.to_csv(folder/(name+"_sessions.csv"),index=False);seg.to_csv(folder/(name+"_segments.csv"),index=False);h.to_csv(folder/(name+"_hits.csv.gz"),index=False)
        dump(folder/"metrics.json",results);record.update(status="complete",seconds=time.time()-start,results=results,sha256={p.name:policy.sha256_file(p) for p in folder.glob("*.gz")})
        verify_freeze();dump(folder/"record.json",record);print(dataset,"RETEST COMPLETE",results,flush=True)
    except Exception as exc:record.update(status="failed",failure=repr(exc));dump(folder/"record.json",record);raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["freeze","retest"]);p.add_argument("--dataset",choices=DATASETS);a=p.parse_args()
    freeze() if a.mode=="freeze" else retest(a.dataset)

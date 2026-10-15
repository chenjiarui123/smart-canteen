"""Replay frozen counterfactual experts through a fresh past-only guard."""
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import policy
from structural_data import OUT,ROOT,DATASETS,verify_protected
from structural_experiments import load_assets,gates,IMPORT_HASHES
from structural_evaluate import evaluate
from structural_admission import HistoricalAdmission
from structural_policy import allocate
from targeted_search import dump


def run(expert_name,name):
    verify_protected()
    expert_record=json.loads((OUT/"experiments"/expert_name/"record.json").read_text())
    assert expert_record["status"]=="complete"
    folder=OUT/"experiments"/name;folder.mkdir(exist_ok=False)
    config=dict(expert_record["config"],admission_guard=True)
    hashes=dict(IMPORT_HASHES)
    for n in ["structural_admission.py","structural_guard_experiment.py"]:hashes[n]=policy.sha256_file(ROOT/n)
    snapshot=folder/"source";snapshot.mkdir()
    for n in hashes:shutil.copy2(ROOT/n,snapshot/n)
    record={"name":name,"config":config,"hypothesis":"Use only prior same-meal/calendar forecast performance to admit the expert; otherwise forecast the complete baseline menu, never exclude the date",
        "status":"running","source_hashes":hashes,"expert_record_sha256":policy.sha256_file(OUT/"experiments"/expert_name/"record.json"),
        "expert":expert_name,"datasets":{},"test_sales_used":False,"started_at":pd.Timestamp.now(tz="Asia/Shanghai").isoformat()}
    dump(folder/"record.json",record)
    thresholds=json.loads((OUT/"preregistered_protocol.json").read_text())["selection"]
    try:
        for dataset in DATASETS:
            baseline,features,actual,edges=load_assets(dataset)
            expert_folder=OUT/"experiments"/expert_name/dataset
            candidate=pd.read_pickle(expert_folder/"menu.pkl.gz")
            ca=pd.read_csv(expert_folder/"allocation_audit.csv",parse_dates=["date"])
            audit_map={tuple(r[k] for k in policy.GROUP_KEYS):r for r in ca.to_dict("records")}
            fg={k:g for k,g in features.groupby(policy.GROUP_KEYS,sort=False)}
            cg={k:g for k,g in candidate.groupby(policy.GROUP_KEYS,sort=False)}
            days={d:g for d,g in actual.groupby("date",sort=False)}
            guard=HistoricalAdmission(thresholds)
            outputs,audits,decisions=[],[],[]
            for date,day in baseline.groupby("date",sort=True):
                observations=[]
                for key,base in day.groupby(policy.GROUP_KEYS,sort=False):
                    f=fg[key];expert=cg[key]
                    admitted,decision=guard.decide(f)
                    if admitted:
                        selected=expert.copy();audit=audit_map[key].copy()
                    else:
                        selected,audit,_=allocate(base,f,{"identity":True,"separate_preparation":config.get("separate_preparation",False)})
                    outputs.append(selected);audits.append(audit)
                    decisions.append(dict(zip(policy.GROUP_KEYS,key))|decision)
                    observations.append((f,base,expert))
                guard.observe_day(observations,days[date])
            menu=pd.concat(outputs).sort_index();audits=pd.DataFrame(audits)
            m,h,hs,full,sessions=evaluate(menu,baseline,features,actual,audits,edges)
            review=gates(dataset,m,full)
            sub=folder/dataset;sub.mkdir()
            menu.to_pickle(sub/"menu.pkl.gz",compression="gzip");h.to_pickle(sub/"hits.pkl.gz",compression="gzip")
            audits.to_csv(sub/"allocation_audit.csv",index=False);hs.to_csv(sub/"hit_segments.csv",index=False)
            full.to_csv(sub/"session_segments.csv",index=False);sessions.to_csv(sub/"sessions.csv",index=False)
            dump(sub/"admission_decisions.json",decisions);dump(sub/"metrics.json",m);dump(sub/"gate_review.json",review)
            record["datasets"][dataset]={"metrics":m,"gate":review,"admitted_sessions":sum(d["admitted"] for d in decisions)}
            dump(folder/"record.json",record)
            print(name,dataset,"hit",m["hit_qty_mape"],"gate",review["eligible"],"admitted",record["datasets"][dataset]["admitted_sessions"],flush=True)
        record["status"]="complete"
    except Exception as exc:
        record.update(status="failed",failure=repr(exc));dump(folder/"record.json",record);raise
    dump(folder/"record.json",record)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("expert");parser.add_argument("name")
    args=parser.parse_args();run(args.expert,args.name)

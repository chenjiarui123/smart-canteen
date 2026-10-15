"""Stage demand/count interventions with direct final-total locks."""
import argparse
import gzip
import json
import pickle
import shutil
import time
import types
import numpy as np
import pandas as pd
import policy
from phase_data import ROOT,OUT,DATASETS,verify
from phase_features import assets,KEYS
from phase_sessions import forecasts
from phase_evaluate import evaluate,gate
from allocation_prepare import load_baseline
from allocation_features import OfferedHistory
from targeted_search import dump


def rescale(menu,f,total):
    old=menu.pred_qty.to_numpy(float);q=old*total/old.sum() if old.sum()>0 else old.copy()
    conflict=bool((q>f.q99.to_numpy(float)*2+1e-8).any())
    if conflict:q=old.copy()
    out=menu.copy();out["pred_qty"]=q;out["pred_orders"]=q*out.expected_orders_per_qty;out["pred_rev"]=q*out.expected_rev_per_qty
    out["pred_total_qty"]=float(q.sum())
    out["expected_demand_qty"]=q;out["suggested_preparation_qty"]=np.maximum(q,out.min_batch_qty)
    return out,conflict


def run(name,config,datasets):
    verify()
    if (OUT/"frozen").exists():raise RuntimeError("Development frozen")
    folder=OUT/"experiments"/name;folder.mkdir(parents=True,exist_ok=False);src=folder/"source";src.mkdir()
    paths=list(ROOT.glob("phase_*.py"))+[ROOT/"phase_trial_design.json"]
    hashes={p.name:policy.sha256_file(p) for p in paths}
    for p in paths:shutil.copy2(p,src/p.name)
    record=dict(name=name,kind="session_phase",config=config,status="running",source_hashes=hashes,datasets={},test_used=False);dump(folder/"record.json",record);start=time.time()
    try:
        for dataset in datasets:
            a=assets(dataset);basefolder=OUT/("development_"+dataset);sub=folder/dataset;sub.mkdir()
            forecast=forecasts(a,config.get("window",7),config.get("trend",False));forecast.to_csv(sub/"session_forecasts.csv",index=False)
            ff=forecast.set_index(policy.GROUP_KEYS).to_dict("index");b=a["baseline_menu"];old=a["base_features"].set_index(KEYS)
            features=pd.read_pickle(basefolder/"phase_prediction.pkl.gz") if (basefolder/"phase_prediction.pkl.gz").exists() else None
            ba=json.loads((basefolder/"baseline_audit.json").read_text());outputs=[];audits=[];paired=[];allocation=[]
            if config.get("count"):
                module,_,_=load_baseline();core=module._integrated_function();original_group=core.__globals__["_group_total_forecast"];original_capacity=core.__globals__["_capacity_forecast"]
                def group(meta,metric,cfg,target_context=None,return_components=False):
                    result=original_group(meta,metric,cfg,target_context,True)
                    if metric=="qty":result.update(group_total_qty=target_context["_locked_total"],factor_combination_mode="joint_absolute_history")
                    return result if return_components else result["group_total_qty"]
                def capacity(meta,cfg,target,total):
                    result=original_capacity(meta,cfg,target,total);result.update(planned_capacity=target["_phase_count"]);return result
                fn=types.FunctionType(core.__code__,dict(core.__globals__,_group_total_forecast=group,_capacity_forecast=capacity),core.__name__,core.__defaults__);fn.__kwdefaults__=core.__kwdefaults__
                cfg=json.loads((OUT/"effective_baseline_config.json").read_text());menu_history={};offered=OfferedHistory();offered.append(a["warmup"])
                manifest=json.loads((basefolder/"coverage.json").read_text())
                for i,(day,today) in enumerate(a["actual"].groupby("date",sort=True)):
                    p=basefolder/"contexts"/(str(day.date())+".pkl.gz");assert policy.sha256_file(p)==manifest["context_sha256"][p.name]
                    with gzip.open(p,"rb") as handle:contexts=pickle.load(handle)
                    for c in contexts:
                        r=ff[(day,c["食堂名称"],c["餐次"])];c["target_context"]=dict(c["target_context"],_locked_total=r["predicted_total"] if config.get("demand") else r["baseline_total"],_phase_count=r["predicted_count"] if r["phase_special"] else r["baseline_count"])
                    current_audit=[];menu=fn(contexts,cfg,policy.BASE_CONSTRAINTS,selection_audit=current_audit,initial_menu_history=menu_history)
                    for c in contexts:
                        key=(day,c["食堂名称"],c["餐次"]);r=ff[key];base=b.loc[b.date.eq(day)&b["餐次"].eq(c["餐次"])];new=menu.loc[menu["餐次"].eq(c["餐次"])].copy()
                        if not new.empty:
                            nf=offered.features(new,c["target_context"]);bf=offered.features(base,c["target_context"])
                            target_total=r["predicted_total"] if config.get("demand") and r["phase_special"] else r["baseline_total"]
                            unit_totals=base.assign(unit=bf.sale_unit.to_numpy()).groupby("unit").pred_qty.sum()*target_total/max(r["baseline_total"],1e-8)
                            conflict=set(nf.sale_unit)!=set(bf.sale_unit)
                            q=new.pred_qty.to_numpy(float)
                            for unit,total in unit_totals.items():
                                mask=nf.sale_unit.eq(unit).to_numpy()
                                if not mask.any() or q[mask].sum()<=0:conflict=True
                                else:q[mask]*=float(total)/q[mask].sum()
                            total=q.sum();reference=base.pred_qty.to_numpy(float);bt=reference.sum()
                            if (q>2*nf.q99.to_numpy()+1e-8).any():conflict=True
                            if total and bt and (((q/total)**2).sum()>1.15*((reference/bt)**2).sum()+1e-12 or q.max()/total>reference.max()/bt+.03+1e-12):conflict=True
                            if conflict:new=base.copy();q=new.pred_qty.to_numpy(float)
                            new["pred_qty"]=q;new["pred_orders"]=q*new.expected_orders_per_qty;new["pred_rev"]=q*new.expected_rev_per_qty
                            new["pred_total_qty"]=float(q.sum())
                            new["expected_demand_qty"]=q;new["suggested_preparation_qty"]=np.maximum(q,new.min_batch_qty)
                            previous=next((names for d,names in menu_history.get((c["食堂名称"],c["餐次"]),[]) if d==day-pd.Timedelta(days=1)),None)
                            names=set(new["菜品名称"].map(policy.normalize_name));rate=len(names&previous)/len(names) if previous and names else None
                            if rate is not None and rate>.8+1e-12:
                                c["_targeted_audit"]={"adjacent_status":"infeasible","adjacent_repeat_rate":None,"replacement_shortage":len(names&previous)-(4*len(names))//5,"infeasible_reason":"unit/concentration fallback incompatible with prior generated menu","unreleased_total_qty":float(q.sum())};new=new.iloc[:0]
                            else:c.setdefault("_targeted_audit",{}).update(adjacent_status="missing_previous" if previous is None else "feasible",adjacent_repeat_rate=rate)
                            allocation.append(dict(date=day,餐次=c["餐次"],conflict=conflict,target_total=target_total,final_total=float(new.pred_qty.sum())))
                        if not new.empty:menu_history.setdefault((c["食堂名称"],c["餐次"]),[]).append((day,set(new["菜品名称"].map(policy.normalize_name))))
                        outputs.append(new)
                        if not base.empty:
                            idx=pd.MultiIndex.from_frame(base[policy.GROUP_KEYS].assign(_dish_norm=base["菜品名称"].map(policy.normalize_name)))
                            fixed,_=rescale(base,old.loc[idx],r["predicted_total"] if config.get("demand") and r["phase_special"] else r["baseline_total"]);paired.append(fixed)
                    cm={(c["date"],c["食堂名称"],c["餐次"]):c for c in contexts}
                    for audit in current_audit:audit.update(cm[(audit["date"],audit["食堂名称"],audit["餐次"])].get("_targeted_audit",{}))
                    audits.extend(current_audit);offered.append(today)
                    if i%40==0:print(name,dataset,"count replay",i+1,flush=True)
            else:
                audits=ba
                for key,g in b.groupby(policy.GROUP_KEYS,sort=False):
                    r=ff[key];idx=pd.MultiIndex.from_frame(g[policy.GROUP_KEYS].assign(_dish_norm=g["菜品名称"].map(policy.normalize_name)))
                    total=r["predicted_total"] if config.get("demand") and r["phase_special"] else r["baseline_total"]
                    if not r["phase_special"]:new=g.copy();conflict=False
                    else:new,conflict=rescale(g,old.loc[idx],total)
                    outputs.append(new);allocation.append(dict(date=key[0],餐次=key[2],conflict=conflict,target_total=total,final_total=float(new.pred_qty.sum())))
            menu=pd.concat(outputs,ignore_index=True)
            bm,_,bs,_=evaluate(dataset,b,a["actual"],ba,features)
            metrics,sessions,segments,hits=evaluate(dataset,menu,a["actual"],audits,features if not config.get("count") else None)
            review=gate(metrics,segments,bm,bs,item_only=False)
            ordinary=forecast.loc[~forecast.phase_special,policy.GROUP_KEYS]
            bt=b.merge(ordinary,on=policy.GROUP_KEYS).groupby(policy.GROUP_KEYS).pred_qty.sum();nt=menu.merge(ordinary,on=policy.GROUP_KEYS).groupby(policy.GROUP_KEYS).pred_qty.sum().reindex(bt.index,fill_value=0)
            error=float((bt-nt).abs().max());metrics["ordinary_max_total_error"]=error
            if error>1e-8:review["eligible"]=False;review["reasons"].append("ordinary final total changed")
            if paired:
                pm,ps,pseg,_=evaluate(dataset,pd.concat(paired,ignore_index=True),a["actual"],ba,features)
                dump(sub/"fixed_menu_paired_metrics.json",pm);pseg.to_csv(sub/"fixed_menu_paired_segments.csv",index=False)
                if pm["hit_qty_mape"]>bm["hit_qty_mape"]+max(1,bm["hit_qty_mape"]*.05):review["eligible"]=False;review["reasons"].append("fixed menu paired hit regression")
            metrics["conflict_sessions"]=sum(r["conflict"] for r in allocation)
            menu.to_pickle(sub/"menu.pkl.gz",compression="gzip");sessions.to_csv(sub/"sessions.csv",index=False);segments.to_csv(sub/"segments.csv",index=False);hits.to_csv(sub/"item_outcomes.csv.gz",index=False)
            pd.DataFrame(allocation).to_csv(sub/"allocation_audit.csv",index=False);dump(sub/"menu_audit.json",audits);dump(sub/"metrics.json",metrics);dump(sub/"gate.json",review)
            record["datasets"][dataset]=dict(metrics=metrics,gate=review);dump(folder/"record.json",record);print(name,dataset,metrics["total_mape"],metrics["hit_qty_mape"],review,flush=True)
        record.update(status="complete",seconds=time.time()-start)
    except Exception as e:record.update(status="failed",failure=repr(e));dump(folder/"record.json",record);raise
    assert all(policy.sha256_file(ROOT/n)==h for n,h in hashes.items());dump(folder/"record.json",record);verify()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("name");p.add_argument("--config",default="{}");p.add_argument("--datasets",nargs="+",default=list(DATASETS));args=p.parse_args()
    run(args.name,json.loads(args.config),args.datasets)

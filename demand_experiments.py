"""Complete rolling development experiments with locked menus/totals."""
import argparse
import json
import shutil
import time
import numpy as np
import pandas as pd
import policy
from demand_data import ROOT,OUT,PREVIOUS,DATASETS,assets,verify
from demand_models import IndependentForecaster,decision
from demand_evaluate import raw_diagnostics,KEYS
from demand_coordination import coordinate
from structural_evaluate import evaluate,hit_stats
from structural_experiments import gates as old_gates
from targeted_search import dump


def load(dataset):
    a=assets(dataset);folder=OUT/dataset
    fm=json.loads((folder/"feature_manifest.json").read_text())
    for name,sha in fm["sha256"].items():assert policy.sha256_file(folder/name)==sha
    f=pd.read_pickle(folder/"prediction_features.pkl.gz")
    index=pd.MultiIndex.from_frame(a["baseline_menu"][policy.GROUP_KEYS].assign(_dish_norm=a["baseline_menu"]["菜品名称"].map(policy.normalize_name)))
    f=f.set_index(KEYS).loc[index].reset_index()
    return a,f,pd.read_pickle(folder/"supplied_training.pkl.gz")


def train_walk(f,training,config):
    model=IndependentForecaster(config)
    points=np.empty(len(f));samples=np.empty((len(f),41));reference=np.empty_like(samples);risk=np.empty(len(f));counts=np.empty(len(f),int);dates=np.empty(len(f),object)
    for i,(date,index) in enumerate(f.groupby("date",sort=True).indices.items()):
        day=f.iloc[index].reset_index(drop=True)
        # The learner is never given the target-day outcome or its menu.
        before=training.loc[training.date<date]
        points[index],samples[index],reference[index],risk[index],counts[index],dates[index]=model.predict(day,before)
        if i%40==0:print("prediction day",i+1,"raw mean",round(float(points[index].mean()),3),flush=True)
    return points,samples,reference,risk,counts,dates


def final_gate(dataset,metrics,segments,raw_frame,menu,pre):
    review=old_gates(dataset,metrics,segments)
    reasons=list(review["reasons"])
    if not pre["precoordination_gate"]["eligible"]:reasons.append("independent raw forecast failed entry gate")
    if pre["interval_coverage"]<.7:reasons.append("80% interval coverage below 70%")
    if pre["interval_width_ratio"]>1.2:reasons.append("interval width >120% of unconditional reference")
    diagnosed=raw_frame.assign(final_prediction=menu.pred_qty.to_numpy())
    extra=[]
    for dimension in ["frequency_group","sample_group","volume_segment"]:
        for name,g in diagnosed.loc[diagnosed.actual_qty>0].groupby(dimension):
            if len(g)<100:continue
            old=hit_stats(g.assign(pred_qty=g.old_final_qty));new=hit_stats(g.assign(pred_qty=g.final_prediction))
            limit=max(1.,old["hit_qty_mape"]*.05)
            extra.append(dict(dimension=dimension,segment=name,old=old["hit_qty_mape"],new=new["hit_qty_mape"],hits=len(g)))
            if new["hit_qty_mape"]>old["hit_qty_mape"]+limit:reasons.append("historical-frequency/sample/volume group regression: "+str(name))
    review.update(eligible=not reasons,reasons=reasons,additional_segments=extra,precoordination_gate=pre["precoordination_gate"])
    return review


def run(name,config,datasets,source=None):
    verify()
    if (OUT/"frozen").exists():raise RuntimeError("This development round is frozen; preserve it and register a new run root before new experiments")
    folder=OUT/"experiments"/name;folder.mkdir(parents=True,exist_ok=False)
    paths=list(ROOT.glob("demand_*.py"))+[ROOT/n for n in ["demand_protocol.json","structural_policy.py","structural_evaluate.py","structural_experiments.py","evaluator.py","policy.py"]]
    hashes={p.name:policy.sha256_file(p) for p in paths}
    snap=folder/"source";snap.mkdir()
    for p in paths:shutil.copy2(p,snap/p.name)
    record=dict(name=name,config=config,status="running",source_model=source,source_hashes=hashes,started_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),datasets={},test_sales_used=False)
    dump(folder/"record.json",record);start=time.time()
    try:
        for dataset in datasets:
            a,f,training=load(dataset)
            sub=folder/dataset;sub.mkdir()
            if source:
                parent=json.loads((OUT/"experiments"/source/"record.json").read_text())
                path=OUT/parent["datasets"][dataset]["distribution_cache"]
                assert policy.sha256_file(path)==parent["datasets"][dataset]["distribution_sha256"]
                cache=np.load(path,allow_pickle=False)
                samples=cache["samples"];reference=cache["reference"];risk=cache["risk"];counts=cache["training_rows"];dates=cache["fit_max_date"]
                if config.get("unconditional"):
                    samples=reference.copy()
                    risk=(samples<=f.low_threshold.to_numpy(float)[:,None]).mean(axis=1)
                point=decision(samples,config.get("objective","log"),f.historical_scale.to_numpy(float))
                record["parent_record_sha256"]=policy.sha256_file(OUT/"experiments"/source/"record.json")
            else:
                point,samples,reference,risk,counts,dates=train_walk(f,training,config)
                path=sub/"distributions.npz"
                np.savez_compressed(path,samples=samples,reference=reference,risk=risk,training_rows=counts,fit_max_date=pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]"))
            f["model_training_rows"]=counts;f["model_fit_max_date"]=pd.to_datetime(dates)
            assert (f.model_fit_max_date.isna()|f.model_fit_max_date.lt(f.date)).all()
            raw,detail,raw_segments=raw_diagnostics(f,point,samples,reference,risk,a["actual"])
            method=config.get("coordination","projection")
            if method!="projection" and not raw["precoordination_gate"]["eligible"]:
                dump(sub/"raw_metrics.json",raw)
                record["datasets"][dataset]=dict(status="not_run_precoordination_gate_failed",raw=raw,distribution_cache=str(path.relative_to(OUT)),distribution_sha256=policy.sha256_file(path))
                print(name,dataset,"SKIP NEW COORDINATION",raw["precoordination_gate"],flush=True);continue
            outputs=[];audits=[]
            old_features=a["features"].set_index(KEYS).loc[pd.MultiIndex.from_frame(f[KEYS])].reset_index()
            for key,index in a["baseline_menu"].groupby(policy.GROUP_KEYS,sort=False).indices.items():
                menu=a["baseline_menu"].iloc[index]
                new,audit=coordinate(menu,old_features.iloc[index].reset_index(drop=True),f.iloc[index],point[index],samples[index],method,config.get("risk_objective","mixed"),config.get("narrow_box",False))
                outputs.append(new);audits.append(audit)
            menu=pd.concat(outputs).sort_index();audits=pd.DataFrame(audits)
            unchanged=[c for c in a["baseline_menu"] if c not in {"pred_qty","pred_orders","pred_rev"}]
            pd.testing.assert_frame_equal(menu[unchanged],a["baseline_menu"][unchanged],check_exact=True)
            edges=json.loads((PREVIOUS/("development_"+dataset)/"group_edges.json").read_text())
            metrics,hits,hs,segments,sessions=evaluate(menu,a["baseline_menu"],a["features"],a["actual"],audits,edges)
            review=final_gate(dataset,metrics,segments,detail,menu,raw)
            detail["final_prediction"]=menu.pred_qty.to_numpy();detail["preparation_prediction"]=menu.suggested_preparation_qty.to_numpy()
            detail.to_pickle(sub/"pre_post_detail.pkl.gz",compression="gzip")
            detail.to_csv(sub/"pre_post_detail.csv.gz",index=False,compression="gzip")
            menu.to_pickle(sub/"menu.pkl.gz",compression="gzip");hits.to_pickle(sub/"hits.pkl.gz",compression="gzip")
            hs.to_csv(sub/"hit_segments.csv",index=False);segments.to_csv(sub/"session_segments.csv",index=False);sessions.to_csv(sub/"sessions.csv",index=False)
            raw_segments.to_csv(sub/"raw_segments.csv",index=False);audits.to_csv(sub/"allocation_audit.csv",index=False)
            dump(sub/"raw_metrics.json",raw);dump(sub/"metrics.json",metrics);dump(sub/"gate_review.json",review)
            record["datasets"][dataset]=dict(status="complete",metrics=metrics,raw=raw,gate=review,distribution_cache=str(path.relative_to(OUT)),distribution_sha256=policy.sha256_file(path))
            dump(folder/"record.json",record)
            print(name,dataset,"raw",round(raw["raw_mape"],4),"raw_gate",raw["precoordination_gate"]["eligible"],"final",round(metrics["hit_qty_mape"],4),"net",round(metrics["net_benefit"],2),"GATE",review["eligible"],flush=True)
        record.update(status="complete",seconds=time.time()-start)
    except Exception as exc:
        record.update(status="failed",failure=repr(exc));dump(folder/"record.json",record);raise
    assert all(policy.sha256_file(ROOT/n)==h for n,h in hashes.items()),"Sources changed while experiment running"
    dump(folder/"record.json",record);verify()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("name");p.add_argument("--config",default="{}");p.add_argument("--source");p.add_argument("--datasets",nargs="+",default=list(DATASETS));args=p.parse_args()
    run(args.name,json.loads(args.config),args.datasets,args.source)

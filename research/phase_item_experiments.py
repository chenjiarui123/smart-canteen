"""Fresh-data fixed-final-menu paired item experiments and their failures."""
import argparse
import json
import shutil
import time
import numpy as np
import pandas as pd
import policy
from phase_data import ROOT, OUT, DATASETS, verify
from phase_features import assets, KEYS
from phase_models import walk,hierarchy
from demand_models import decision
from phase_evaluate import evaluate,raw_diagnostics,gate
from demand_coordination import coordinate
from targeted_search import dump


def load(dataset):
    a=assets(dataset);folder=OUT/("development_"+dataset)
    manifest=json.loads((folder/"phase_feature_manifest.json").read_text())
    for n,h in manifest["hash"].items():assert policy.sha256_file(folder/n)==h
    f=pd.read_pickle(folder/"phase_prediction.pkl.gz")
    idx=pd.MultiIndex.from_frame(a["baseline_menu"][policy.GROUP_KEYS].assign(_dish_norm=a["baseline_menu"]["菜品名称"].map(policy.normalize_name)))
    f=f.set_index(KEYS).loc[idx].reset_index()
    old=a["base_features"].set_index(KEYS).loc[idx].reset_index();old["structural_history_max_date"]=old.history_max_date
    a["base_features"]=old
    return a,f,pd.read_pickle(folder/"phase_training.pkl.gz")


def run(name,config,datasets):
    verify()
    if (OUT/"frozen").exists():raise RuntimeError("Development frozen")
    folder=OUT/"experiments"/name;folder.mkdir(parents=True,exist_ok=False);src=folder/"source";src.mkdir()
    paths=list(ROOT.glob("phase_*.py"))+[ROOT/n for n in ["phase_trial_design.json","demand_coordination.py","demand_models.py","structural_policy.py","allocation_policy.py"]]
    hashes={p.name:policy.sha256_file(p) for p in paths}
    for p in paths:shutil.copy2(p,src/p.name)
    record=dict(name=name,config=config,kind="fixed_menu_item",status="running",source_hashes=hashes,datasets={},started=str(pd.Timestamp.now()),test_used=False)
    dump(folder/"record.json",record);start=time.time()
    try:
        for dataset in datasets:
            a,f,training=load(dataset);sub=folder/dataset;sub.mkdir()
            if config.get("source"):
                parent=OUT/"experiments"/config["source"]/dataset
                manifest=json.loads((parent/"distribution_manifest.json").read_text())
                assert policy.sha256_file(parent/"distribution.npz")==manifest["sha256"]
                assert __import__("hashlib").sha256(pd.util.hash_pandas_object(f[KEYS],index=False).values.tobytes()).hexdigest()==manifest["row_keys_sha256"],"Parent distribution row order mismatch"
                cache=np.load(parent/"distribution.npz",allow_pickle=False)
                samples=cache["samples"].copy();reference=cache["reference"].copy();risk=cache["risk"].copy()
                fit_dates=pd.to_datetime(cache["fit_dates"]);counts=cache["counts"]
                point=decision(samples,config.get("objective","log"),f.historical_scale.to_numpy(float))
                record["parent_sha256"]=policy.sha256_file(parent.parent/"record.json")
            else:point,samples,reference,risk,fit_dates,counts=walk(f,training,config)
            if config.get("hierarchy"):
                original=point.copy();point,cats=hierarchy(f,training,point,config.get("phase",True))
                ratio=point/np.maximum(original,1e-8);samples*=ratio[:,None];cats.to_csv(sub/"category_predictions.csv",index=False)
            f["model_fit_max_date"]=fit_dates;f["model_training_rows"]=counts
            raw,detail,raw_segments=raw_diagnostics(dataset,f,point,samples,risk,a["actual"])
            raw_segments.to_csv(sub/"raw_segments.csv",index=False);dump(sub/"raw_metrics.json",raw)
            np.savez_compressed(sub/"distribution.npz",samples=samples,reference=reference,risk=risk,point=point,fit_dates=fit_dates.to_numpy(dtype="datetime64[ns]"),counts=counts)
            dump(sub/"distribution_manifest.json",dict(sha256=policy.sha256_file(sub/"distribution.npz"),row_keys_sha256=__import__("hashlib").sha256(pd.util.hash_pandas_object(f[KEYS],index=False).values.tobytes()).hexdigest()))
            method=config.get("coordination","projection")
            if method!="projection" and not raw["precoordination_gate"]["eligible"]:
                detail.to_csv(sub/"pre_post_detail.csv.gz",index=False)
                record["datasets"][dataset]=dict(status="skipped_raw_gate",raw=raw);dump(folder/"record.json",record);continue
            outputs=[];audits=[]
            for key,index in a["baseline_menu"].groupby(policy.GROUP_KEYS,sort=False).indices.items():
                menu=a["baseline_menu"].iloc[index];new,audit=coordinate(menu,a["base_features"].iloc[index].reset_index(drop=True),f.iloc[index],point[index],samples[index],method,config.get("risk_objective","mixed"))
                outputs.append(new);audits.append(audit)
            menu=pd.concat(outputs).sort_index();audit=pd.DataFrame(audits)
            baseline=a["baseline_menu"];unchanged=[c for c in baseline if c not in {"pred_qty","pred_orders","pred_rev"}]
            pd.testing.assert_frame_equal(menu[unchanged],baseline[unchanged],check_exact=True)
            bt=baseline.groupby(policy.GROUP_KEYS).pred_qty.sum();nt=menu.groupby(policy.GROUP_KEYS).pred_qty.sum()
            np.testing.assert_allclose(bt,nt,rtol=1e-12,atol=1e-8);assert audit.max_unit_total_error.max()<1e-8
            ba=json.loads((OUT/("development_"+dataset)/"baseline_audit.json").read_text())
            bm,_,bs,_=evaluate(dataset,baseline,a["actual"],ba,f)
            metrics,sessions,segments,hits=evaluate(dataset,menu,a["actual"],ba,f)
            metrics.update(max_session_total_error=float((nt-bt).abs().max()),max_unit_total_error=float(audit.max_unit_total_error.max()),
                fixed_menu=True,conflict_sessions=int(audit.status.str.contains("conflict").sum()),maximum_hhi_ratio=float((audit.new_hhi/audit.baseline_hhi).max()))
            review=gate(metrics,segments,bm,bs,item_only=True,raw=raw)
            detail["final_prediction"]=menu.pred_qty.to_numpy();detail["preparation_prediction"]=menu.suggested_preparation_qty.to_numpy()
            detail["final_ape"]=abs(detail.final_prediction-detail.actual_qty)/detail.actual_qty.replace(0,np.nan)
            detail.to_pickle(sub/"pre_post_detail.pkl.gz",compression="gzip");detail.to_csv(sub/"pre_post_detail.csv.gz",index=False)
            detail.nlargest(100,"final_ape").to_csv(sub/"failure_samples.csv",index=False)
            menu.to_pickle(sub/"menu.pkl.gz",compression="gzip");sessions.to_csv(sub/"sessions.csv",index=False);segments.to_csv(sub/"segments.csv",index=False);audit.to_csv(sub/"allocation_audit.csv",index=False)
            dump(sub/"metrics.json",metrics);dump(sub/"gate.json",review)
            record["datasets"][dataset]=dict(status="complete",metrics=metrics,raw=raw,gate=review)
            dump(folder/"record.json",record);print(name,dataset,"RAW",raw["raw_mape"],"FINAL",metrics["hit_qty_mape"],"GATE",review,flush=True)
        record.update(status="complete",seconds=time.time()-start)
    except Exception as exc:record.update(status="failed",failure=repr(exc));dump(folder/"record.json",record);raise
    assert all(policy.sha256_file(ROOT/n)==h for n,h in hashes.items()),"Source changed during experiment"
    dump(folder/"record.json",record);verify()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("name");p.add_argument("--config",default="{}");p.add_argument("--datasets",nargs="+",default=list(DATASETS));args=p.parse_args()
    run(args.name,json.loads(args.config),args.datasets)

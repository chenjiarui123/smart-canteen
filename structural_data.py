"""Current cleaned web data; calendar-only row selection isolates development."""
from __future__ import annotations
import inspect
import json
import shutil
from pathlib import Path
import pandas as pd
import policy
from allocation_prepare import load_baseline
from targeted_search import dump

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs/structural_20260912"
WEB=ROOT/"smart_canteen_web/data"
DATASETS=("a_hupan","f_jichuyuan")


def sales_subset(dataset,start=None,end="2025-12-31"):
    path=WEB/"datasets"/dataset/"sales.csv"
    # No target/test sales quantities, menus, or outcomes are loaded to select rows.
    calendar=pd.read_csv(path,usecols=["年","月","日"])
    dates=pd.to_datetime(dict(year=calendar["年"],month=calendar["月"],day=calendar["日"]))
    keep=dates.le(pd.Timestamp(end))
    if start is not None:
        keep &= dates.ge(pd.Timestamp(start))
    excluded=set(calendar.index[~keep]+1)
    raw=pd.read_csv(path,skiprows=lambda line:line in excluded)
    source=inspect.getsource(policy.read_sales)
    assert source.count("    frame = pd.read_csv(path)")==1
    source=source.replace("def read_sales(path: Path)","def read_sales(path: Path, frame)").replace("    frame = pd.read_csv(path)","    frame = frame.copy()")
    namespace=dict(policy.__dict__)
    exec(compile(source,"structural_data::unchanged_sales_normalization","exec"),namespace)
    decoded=namespace["read_sales"](path,raw)
    assert decoded.date.max()<=pd.Timestamp(end)
    return decoded


def preregister():
    OUT.mkdir(parents=True,exist_ok=False)
    module,config,manifest=load_baseline()
    config=dict(config,runtime_history_end_inactivity=True)
    data_manifest=json.loads((WEB/"manifest.json").read_text(encoding="utf-8-sig"))
    protected={}
    for ds in DATASETS:
        d=data_manifest["datasets"][ds]
        for name,key in [("sales.csv","sales_sha256"),("dish_library.csv","dish_library_sha256")]:
            p=WEB/"datasets"/ds/name
            actual=policy.sha256_file(p)
            assert actual==d[key],(ds,name)
            protected[str(p.relative_to(ROOT))]=actual
    for name in manifest["files"]:
        protected["smart_canteen_web/model/"+name]=policy.sha256_file(ROOT/"smart_canteen_web/model"/name)
    for name in ["policy.py","evaluator.py","diagnostics.py"]:
        protected[name]=policy.sha256_file(ROOT/name)
    snapshot=OUT/"baseline_source"
    snapshot.mkdir()
    for name in manifest["files"]:
        shutil.copy2(ROOT/"smart_canteen_web/model"/name,snapshot/name)
    shutil.copy2(ROOT/"structural_protocol.json",OUT/"preregistered_protocol.json")
    dump(OUT/"effective_baseline_config.json",config)
    dump(OUT/"preregistration.json",{"created_at":pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "before_new_baseline_metrics":True,"protocol_sha256":policy.sha256_file(ROOT/"structural_protocol.json"),
        "protected":protected,"deployed_version":manifest["model_version"],"runtime_flag":True,
        "prior_experiments":"runs/allocation_20260911; prior parameter/median/date-smoothing variants are not novel mechanisms",
        "cleaning_updates":data_manifest.get("updates",[]),"test_sales_loaded":False})
    print("preregistered",OUT,flush=True)


def verify_protected():
    record=json.loads((OUT/"preregistration.json").read_text())
    assert policy.sha256_file(ROOT/"structural_protocol.json")==record["protocol_sha256"]
    for name,sha in record["protected"].items():
        assert policy.sha256_file(ROOT/name)==sha,name
    return record


if __name__=="__main__": preregister()

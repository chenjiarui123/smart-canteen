"""New fingerprint-checked rice-merged data snapshot, no old baseline reuse."""
import json
import shutil
from pathlib import Path
import pandas as pd
import policy
from allocation_prepare import load_baseline
from structural_data import sales_subset
from targeted_search import dump

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs/phase_20260913"
DATA=ROOT/"smart_canteen_web/data"
DATASETS=("a_hupan","f_jichuyuan")


def register():
    OUT.mkdir(parents=True,exist_ok=False)
    data_manifest=json.loads((DATA/"manifest.json").read_text())
    _,config,model_manifest=load_baseline()
    protected={};snapshot=OUT/"snapshot";snapshot.mkdir()
    files=[DATA/"manifest.json",DATA/"features/calendar.csv",DATA/"features/weather.csv"]
    for ds in DATASETS:
        for filename,key in [("sales.csv","sales_sha256"),("dish_library.csv","dish_library_sha256")]:
            p=DATA/"datasets"/ds/filename
            assert policy.sha256_file(p)==data_manifest["datasets"][ds][key],(ds,filename)
            files.append(p)
            # Read names only to verify the authorized alias identity; no
            # locked-period quantities enter registration or selection.
            names=pd.read_csv(p,usecols=["菜品名称"])["菜品名称"]
            assert not names.eq("米饭🍚").any(),(ds,filename)
    files += [ROOT/"smart_canteen_web/model"/n for n in model_manifest["files"]]
    files += [ROOT/n for n in ["policy.py","evaluator.py","diagnostics.py","targeted_policy.py","targeted_search.py","structural_trace.py","phase_protocol.json","phase_data.py"]]
    files += [ROOT/"smart_canteen_web/backend/app/active_strategy.py"]
    for p in files:
        relative=p.relative_to(ROOT);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);protected[str(relative)]=policy.sha256_file(p)
        assert policy.sha256_file(target)==protected[str(relative)]
    dump(OUT/"effective_baseline_config.json",dict(config,runtime_history_end_inactivity=True))
    shutil.copy2(ROOT/"phase_protocol.json",OUT/"preregistered_protocol.json")
    dump(OUT/"registration.json",dict(registered_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),protected=protected,
        snapshot_hashes=protected,model_version=model_manifest["model_version"],data_version=data_manifest["updated_at"],
        prediction_cache_used=False,prior_failures="structural_20260912 and demand_20260913 only as design history, never baseline cache",
        calendar_limitation="2025-07-06 through 2025-07-20 absent; summer start unknown pending calendar evidence",
        development_start={"a_hupan":"2025-08-05","f_jichuyuan":"2025-01-15"},development_end="2025-12-31"))


def verify():
    r=json.loads((OUT/"registration.json").read_text())
    for name,sha in r["protected"].items():
        assert policy.sha256_file(ROOT/name)==sha,name
        assert policy.sha256_file(OUT/"snapshot"/name)==sha,"snapshot: "+name
    return r


def read_development(dataset):
    verify()
    # The date-filtered normalization helper reads current files, not any old
    # cache. The NEW fingerprint above is mandatory before using it.
    frame=sales_subset(dataset,end="2025-12-31")
    assert frame.date.max()<=pd.Timestamp("2025-12-31")
    assert not frame["菜品名称"].eq("米饭🍚").any()
    return frame


if __name__=="__main__":register()

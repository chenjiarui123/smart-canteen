"""New development run, protected inputs and reused immutable baseline evidence."""
import json
import shutil
from pathlib import Path
import policy
from structural_data import ROOT, DATASETS, sales_subset, verify_protected
from targeted_search import dump

OUT=ROOT/"runs/demand_20260913"
PREVIOUS=ROOT/"runs/structural_20260912"


def register():
    old=verify_protected()
    OUT.mkdir(exist_ok=False)
    shutil.copy2(ROOT/"demand_protocol.json",OUT/"preregistered_protocol.json")
    dump(OUT/"preregistration.json",dict(protocol_sha256=policy.sha256_file(ROOT/"demand_protocol.json"),
        protected=old["protected"], previous_protocol_sha256=old["protocol_sha256"],
        deployed_version=old["deployed_version"],registered_before_new_experiment_results=True,
        previous_failures="structural_20260912: all candidates failed, notably F July regressions; no source/data changes since freeze",
        independent_scope="new all-offered absolute-record-sales models, not old residual or allocation retuning"))


def verify():
    verify_protected()
    r=json.loads((OUT/"preregistration.json").read_text())
    assert policy.sha256_file(ROOT/"demand_protocol.json")==r["protocol_sha256"]
    for name,sha in r["protected"].items():assert policy.sha256_file(ROOT/name)==sha,name


def assets(dataset):
    import pandas as pd
    verify()
    p=PREVIOUS/("development_"+dataset)
    coverage=json.loads((p/"coverage.json").read_text())
    assert policy.sha256_file(p/"baseline_menu.pkl")==coverage["baseline_sha256"]
    return {name:pd.read_pickle(p/(name+".pkl")) for name in ["baseline_menu","features","actual","warmup","attributes"]}


if __name__=="__main__":register()

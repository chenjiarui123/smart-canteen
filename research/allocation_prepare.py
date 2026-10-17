"""Snapshot and reproduce current baseline using development/history files only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import shutil
import types
from pathlib import Path

import pandas as pd

import policy
from allocation_features import OfferedHistory
from targeted_search import dump, date_type

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = PROJECT / "runs/allocation_20260911"
HISTORY = {"a":ROOT/"数据/长春A食堂_菜品销量_菜名修正后.csv", "f":ROOT/"数据/长春F食堂_菜品销量_菜名修正后.csv"}
LIBRARY = {"a":ROOT/"数据/长春A食堂_菜品库.csv", "f":ROOT/"数据/长春F食堂_菜品库.csv"}


def load_baseline():
    path = PROJECT/"smart_canteen_web/model"
    manifest = json.loads((path/"targeted_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        assert policy.sha256_file(path/name) == sha, name
    assert policy.sha256_file(ROOT/"policy.py") == manifest["files"]["policy.py"]
    spec = importlib.util.spec_from_file_location("allocation_frozen_baseline", path/"targeted_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, json.loads((path/"targeted_config.json").read_text()), manifest


def prepare(dataset):
    OUT.mkdir(exist_ok=True,parents=True)
    folder = OUT/("development_"+dataset)
    folder.mkdir(exist_ok=False)
    module, config, manifest = load_baseline()
    snapshot = folder/"baseline_source"
    snapshot.mkdir()
    for name in manifest["files"]:
        shutil.copy2(PROJECT/"smart_canteen_web/model"/name, snapshot/name)
    dump(snapshot/"manifest.json",manifest)
    sales = policy.read_sales(HISTORY[dataset])
    assert sales.date.max() <= pd.Timestamp("2025-12-31")
    warmup, actual, first = policy.training_backtest_frames(sales)
    coverage = {"dataset":dataset,"data_path":str(HISTORY[dataset].relative_to(ROOT)),
                "data_sha256":policy.sha256_file(HISTORY[dataset]),"library_sha256":policy.sha256_file(LIBRARY[dataset]),
                "development_start":str(first.date()),"development_end":str(actual.date.max().date()),
                "warmup_days":14,"date_type_sessions":actual.assign(date_type=actual.apply(date_type,axis=1)).drop_duplicates(policy.GROUP_KEYS).groupby("date_type").size().to_dict(),
                "locked_2026_data_read":False,"state":"running"}
    dump(folder/"coverage.json",coverage)
    history = warmup.copy()
    attributes,_ = policy.build_library_attributes(LIBRARY[dataset])
    menus, feature_rows, audits = [], [], []
    offered = OfferedHistory()
    offered.append(warmup)
    menu_history = {}
    raw_rows = []
    original_enforce = module.enforce_repeat

    def capture(*args):
        result = original_enforce(*args)
        c = args[-1]
        for r in result.to_dict("records"):
            raw_rows.append({"date":pd.Timestamp(c["date"]),"食堂名称":c["食堂名称"],"餐次":c["餐次"],
                             "_dish_norm":r["_dish_norm"],"uncalibrated_qty":float(r["_allocation_weight"]),
                             "baseline_qty_stability":float(r["_qty_stability"]),
                             "baseline_recent_weight":float(r["_adaptive_recent_weight"])})
        return result
    core = module._integrated_function()
    function = types.FunctionType(core.__code__, dict(core.__globals__,enforce_repeat=capture), core.__name__, core.__defaults__)
    function.__kwdefaults__ = core.__kwdefaults__
    for i, (day, day_actual) in enumerate(actual.groupby("date",sort=True)):
        contexts = policy.build_contexts(history,day_actual,attributes)
        for c in contexts:
            c["day_group_count"] = len(contexts)
        audit = []
        recs = function(contexts,config,policy.BASE_CONSTRAINTS,selection_audit=audit,initial_menu_history=menu_history)
        for c, a in zip(contexts,audit):
            a.update(c.get("_targeted_audit",{}))
            meal = c["餐次"]
            selected = recs.loc[recs["餐次"].eq(meal)].copy()
            if not selected.empty:
                f = offered.features(selected,c["target_context"])
                f["date_type"] = date_type(day_actual.loc[day_actual["餐次"].eq(meal)].iloc[0])
                feature_rows.append(f)
            menu_history.setdefault((c["食堂名称"],meal),[]).append((pd.Timestamp(day),set(selected["菜品名称"].map(policy.normalize_name))))
        menus.append(recs)
        audits.extend(audit)
        offered.append(day_actual)
        history = pd.concat([history,day_actual],ignore_index=True)
        if i%20 == 0:
            print(dataset,"baseline days",i+1,"/",actual.date.nunique(),flush=True)
    recs = pd.concat(menus,ignore_index=True)
    features = pd.concat(feature_rows,ignore_index=True)
    raw = pd.DataFrame(raw_rows)
    features = features.merge(raw,on=policy.GROUP_KEYS+["_dish_norm"],validate="one_to_one")
    recs.to_csv(folder/"baseline_menu.csv",index=False)
    features.to_pickle(folder/"features.pkl")
    actual.to_pickle(folder/"actual.pkl")
    features.to_csv(folder/"features.csv",index=False)
    dump(folder/"baseline_audit.json",audits)
    if dataset == "a":
        old = pd.read_csv(PROJECT/"runs/targeted_20260911/final_verified/recommended_menu.csv",parse_dates=["date"])
        pd.testing.assert_frame_equal(recs.reset_index(drop=True),old.reset_index(drop=True),check_exact=False,rtol=1e-12,atol=1e-10)
        coverage["latest_development_baseline_reproduced"] = True
    coverage.update(state="complete",menu_rows=len(recs),session_count=actual.groupby(policy.GROUP_KEYS).ngroups,
                    baseline_sha256=policy.sha256_file(folder/"baseline_menu.csv"),features_sha256=policy.sha256_file(folder/"features.pkl"))
    dump(folder/"coverage.json",coverage)
    print(dataset,"baseline ready",len(recs),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",choices=["a","f"])
    prepare(parser.parse_args().dataset)

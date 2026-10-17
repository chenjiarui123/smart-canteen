"""Freeze the selected development result, then run explicitly labelled retests."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

import policy
import targeted_policy
from targeted_search import OUT, ROOT, PROJECT, dump, evaluate


def freeze(name):
    folder = OUT / name
    record = json.loads((folder / "record.json").read_text())
    if record["status"] != "complete":
        raise RuntimeError("Cannot freeze an incomplete experiment")
    for filename in ["targeted_policy.py", "policy.py", "evaluator.py"]:
        if policy.sha256_file(ROOT / filename) != record["source_hashes"][filename]:
            raise RuntimeError("Repeat selected configuration against final source before freeze: " + filename)
    if record["metrics"]["repeat_over_80_sessions"] or record["metrics"]["infeasible_sessions"]:
        raise RuntimeError("Development repeat constraint or feasibility failed")
    destination = OUT / "frozen"
    destination.mkdir(exist_ok=False)
    for filename in ["targeted_policy.py", "policy.py", "diagnostics.py", "evaluator.py"]:
        shutil.copy2(ROOT / filename, destination / filename)
    shutil.copy2(folder / "config.json", destination / "targeted_config.json")
    manifest = {"model_version": "targeted_repeat_holiday_20260911", "status": "frozen",
                "development_run": name, "frozen_at": pd.Timestamp.now().isoformat(),
                "development_schedule_sha256": record["schedule_sha256"],
                "files": {p.name: policy.sha256_file(p) for p in destination.iterdir() if p.is_file()},
                "evaluation_note": "既有湖畔验证/测试区间与基础园测试区间均为冻结后复测，不用于调参"}
    dump(destination / "targeted_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def retest(dataset):
    frozen = OUT / "frozen"
    manifest = json.loads((frozen / "targeted_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if policy.sha256_file(frozen / name) != sha:
            raise RuntimeError("Frozen artifact changed: " + name)
    if policy.sha256_file(ROOT / "targeted_policy.py") != manifest["files"]["targeted_policy.py"]:
        raise RuntimeError("Active source differs from frozen source")
    config = json.loads((frozen / "targeted_config.json").read_text())
    baseline = json.loads((ROOT / "config.json").read_text())["best_policy"]
    if dataset == "hupan":
        history_path = policy.TRAIN_PATH
        library_path = policy.LIBRARY_PATH
        external_path = ROOT / "数据/菜品销量_吉大湖畔智慧餐厅_菜名修正后.csv"
    else:
        history_path = ROOT / "数据/长春F食堂_菜品销量_菜名修正后.csv"
        library_path = ROOT / "数据/长春F食堂_菜品库.csv"
        external_path = ROOT / "数据/菜品销量_吉大南校基础园智慧餐厅_菜名修正后.csv"
    folder = OUT / ("retest_" + dataset)
    folder.mkdir(exist_ok=False)
    dump(folder / "retest_lock.json", {"status": "started", "freeze_sha256": policy.sha256_file(frozen / "targeted_manifest.json"),
                                       "started_at": pd.Timestamp.now().isoformat(), "label": "既有测试区间复测"})
    history = policy.read_sales(history_path)
    actual = policy.read_sales(external_path)
    actual = actual.loc[actual.date.between("2026-01-01", "2026-06-10")].copy()
    assert history.date.max() < actual.date.min()
    attributes, _ = policy.build_library_attributes(library_path)
    print("building frozen retest contexts", dataset, flush=True)
    contexts = policy.build_contexts(history, actual, attributes)
    results = {}
    for label, cfg in [("baseline", baseline), ("new", config)]:
        audit = []
        print("predicting frozen retest", dataset, label, len(contexts), flush=True)
        # Disabled extension was verified byte-for-byte on the full development
        # menu; use its identical array selector to make frozen retests faster.
        recs = targeted_policy.make_recommendations(contexts, cfg, policy.BASE_CONSTRAINTS, selection_audit=audit)
        recs.to_csv(folder / (label + "_recommended_menu.csv"), index=False)
        dump(folder / (label + "_audit.json"), audit)
        metrics, sessions, segments = evaluate(recs, actual, audit)
        results[label] = metrics
        sessions.to_csv(folder / (label + "_sessions.csv"), index=False)
        pd.DataFrame(segments).to_csv(folder / (label + "_segments.csv"), index=False)
        dump(folder / (label + "_metrics.json"), metrics)
        if dataset == "hupan":
            for split, start, end in [("validation_retest", "2026-01-01", "2026-03-31"),
                                      ("test_retest", "2026-04-01", "2026-06-10")]:
                suba = actual.loc[actual.date.between(start, end)]
                subr = recs.loc[recs.date.between(start, end)]
                subaudit = [a for a in audit if pd.Timestamp(start) <= a["date"] <= pd.Timestamp(end)]
                m, s, seg = evaluate(subr, suba, subaudit)
                # Preserve the March 31 generated menu as April 1 comparator.
                s = sessions.loc[sessions.date.between(start, end)].copy()
                m["missing_previous_sessions"] = int(s.missing_previous.sum())
                m["repeat_over_80_sessions"] = int(s.repeat_over_80.sum())
                m["adjacent_repeat_rate"] = float(s.adjacent_repeat_rate.mean())
                dump(folder / (label + "_" + split + "_metrics.json"), m)
                s.to_csv(folder / (label + "_" + split + "_sessions.csv"), index=False)
                pd.DataFrame(seg).to_csv(folder / (label + "_" + split + "_segments.csv"), index=False)
        print(dataset, label, json.dumps({k:metrics[k] for k in ["total_mape", "hit_qty_mape", "net_benefit", "infeasible_sessions", "repeat_over_80_sessions"]}), flush=True)
    dump(folder / "retest_lock.json", {"status": "complete", "freeze_sha256": policy.sha256_file(frozen / "targeted_manifest.json"),
                                       "finished_at": pd.Timestamp.now().isoformat(), "label": "既有测试区间复测",
                                       "history_sha256": policy.sha256_file(history_path), "actual_sha256": policy.sha256_file(external_path),
                                       "library_sha256": policy.sha256_file(library_path), "results": results})


def install():
    frozen = OUT / "frozen"
    web = PROJECT / "smart_canteen_web/model"
    manifest = json.loads((frozen / "targeted_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if policy.sha256_file(frozen / name) != sha:
            raise RuntimeError("Frozen artifact changed")
    for name in ["targeted_policy.py", "targeted_config.json"]:
        if (web / name).exists():
            raise RuntimeError("Targeted release already installed")
        shutil.copy2(frozen / name, web / name)
    dump(web / "targeted_manifest.json", manifest)
    print("installed", manifest["model_version"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "retest", "install"])
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.name)
    elif args.action == "retest":
        retest(args.name)
    else:
        install()

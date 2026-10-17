"""Auditable, training-only continuation of Trial 18. One explicit hypothesis per run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import policy
from diagnostics import hit_quantity_diagnostics, menu_capacity_diagnostics, prediction_consistency_diagnostics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/targeted_20260911"
KEYS = policy.GROUP_KEYS


def dump(path, value):
    def default(v):
        if isinstance(v, set):
            return sorted(v)
        return policy._json_default(v)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default) + "\n", encoding="utf-8")


def date_type(row):
    school = policy._flag(row.get("是否校历寒暑假", False))
    legal = policy._flag(row.get("是否法定节假日假期", False))
    if school and legal:
        return "寒暑假与法定节假日重叠"
    if school:
        return "寒假" if "寒" in str(row.get("校历假期类型", "")) else "暑假"
    if legal:
        return "法定节假日"
    if policy._flag(row.get("是否调休工作日", False)):
        return "调休工作日"
    return "普通周末" if pd.Timestamp(row["date"]).weekday() >= 5 else "普通工作日"


def evaluate(recs, actual, audit):
    metrics = policy.evaluate_policy_metrics(recs, actual, policy.BASE_CONSTRAINTS)
    metrics.update(hit_quantity_diagnostics(recs, actual)[0])
    metrics.update(prediction_consistency_diagnostics(recs))
    capacity, _ = menu_capacity_diagnostics(recs, actual, audit)
    metrics.update(capacity)
    source = actual.copy()
    source["date_type"] = source.apply(date_type, axis=1)
    sessions = source.groupby(KEYS, as_index=False).agg(
        actual_total=("菜品总销量", "sum"), actual_count=("_dish_norm", "nunique"),
        date_type=("date_type", "first"))
    pred = recs.assign(_norm=recs["菜品名称"].map(policy.normalize_name))
    totals = pred.groupby(KEYS, as_index=False).agg(pred_total=("pred_qty", "sum"), pred_count=("_norm", "nunique"))
    sessions = sessions.merge(totals, on=KEYS, how="left").fillna({"pred_total": 0, "pred_count": 0})
    menus = {tuple(k): set(g["_norm"]) for k, g in pred.groupby(KEYS)}
    audit_map = {tuple(a[k] for k in KEYS): a for a in audit}
    checks = []
    for r in sessions.to_dict("records"):
        key = tuple(r[k] for k in KEYS)
        previous = menus.get((pd.Timestamp(r["date"]) - pd.Timedelta(days=1), r["食堂名称"], r["餐次"]))
        current = menus.get(key, set())
        check = audit_map.get(key, {})
        rate = len(current & previous) / len(current) if previous and current else None
        missing_previous = not bool(previous)
        if "adjacent_status" in check:
            # A sliced reporting window may still have a valid previous-day
            # menu in the full walk. Use the recorded run-time comparison.
            rate = check.get("adjacent_repeat_rate")
            missing_previous = check["adjacent_status"] == "missing_previous"
        checks.append({"adjacent_repeat_rate": rate, "missing_previous": missing_previous,
                       "repeat_over_80": rate is not None and rate > .8 + 1e-12,
                       "repeat_status": check.get("adjacent_status", "baseline"),
                       "replacement_shortage": check.get("replacement_shortage", 0),
                       "replacement_count": check.get("replacement_count", 0),
                       "planned_count": check.get("planned_capacity", check.get("capacity", len(current))),
                       "unreleased_total_qty": check.get("unreleased_total_qty", 0),
                       "infeasible_reason": check.get("infeasible_reason", "")})
    sessions = pd.concat([sessions.reset_index(drop=True), pd.DataFrame(checks)], axis=1)
    sessions["qty_bias"] = sessions.pred_total - sessions.actual_total
    sessions["count_bias"] = sessions.pred_count - sessions.actual_count
    sessions["absolute_error"] = sessions.qty_bias.abs()
    sessions["overprediction_qty"] = sessions.qty_bias.clip(lower=0)
    sessions["underprediction_qty"] = (-sessions.qty_bias).clip(lower=0)

    def extras(frame):
        actual_sum = float(frame.actual_total.sum())
        return {"session_count": len(frame), "actual_count_mean": float(frame.actual_count.mean()),
                "pred_count_mean": float(frame.pred_count.mean()), "count_bias_mean": float(frame.count_bias.mean()),
                "actual_qty_sum": actual_sum, "pred_qty_sum": float(frame.pred_total.sum()),
                "qty_bias": float(frame.qty_bias.sum()), "overprediction_qty": float(frame.overprediction_qty.sum()),
                "underprediction_qty": float(frame.underprediction_qty.sum()),
                "total_wape": float(frame.absolute_error.sum() / actual_sum * 100) if actual_sum else None,
                "total_mae": float(frame.absolute_error.mean()),
                "adjacent_repeat_rate": float(frame.adjacent_repeat_rate.mean()) if frame.adjacent_repeat_rate.notna().any() else None,
                "adjacent_repeat_max": float(frame.adjacent_repeat_rate.max()) if frame.adjacent_repeat_rate.notna().any() else None,
                "menu_fill_rate": float((frame.pred_count / frame.planned_count.replace(0, np.nan)).fillna(0).mean()),
                "unreleased_total_qty": float(frame.unreleased_total_qty.sum()),
                "replacement_count_sum": int(frame.replacement_count.sum()),
                "repeat_over_80_sessions": int(frame.repeat_over_80.sum()),
                "missing_previous_sessions": int(frame.missing_previous.sum()),
                "infeasible_sessions": int(frame.repeat_status.eq("infeasible").sum()),
                "replacement_shortage_sum": int(frame.replacement_shortage.sum())}
    metrics.update(extras(sessions))
    segments = []
    groupings = [("overall", "全部", sessions)]
    groupings += [("date_type", name, g) for name, g in sessions.groupby("date_type")]
    groupings += [("date_type_canteen_meal", "|".join(name), g) for name, g in sessions.groupby(["date_type", "食堂名称", "餐次"])]
    regular = sessions.date_type.isin(["普通工作日", "普通周末", "调休工作日"])
    groupings += [("regular_holiday", "普通日期含调休", sessions.loc[regular]), ("regular_holiday", "假期", sessions.loc[~regular])]
    for dimension, name, group in groupings:
        if group.empty:
            continue
        keys = group[KEYS]
        a = actual.merge(keys, on=KEYS, how="inner")
        p = recs.merge(keys, on=KEYS, how="inner")
        m = policy.evaluate_policy_metrics(p, a, policy.BASE_CONSTRAINTS)
        m.update(hit_quantity_diagnostics(p, a)[0])
        m.update(extras(group))
        m.update(dimension=dimension, segment=name)
        segments.append(m)
    return metrics, sessions, segments


def assets():
    # Cache is strictly training-only and content-addressed by source and inputs.
    fingerprint = hashlib.sha256((policy.sha256_file(policy.TRAIN_PATH) + policy.sha256_file(policy.LIBRARY_PATH)
                                  + policy.sha256_file(ROOT / "policy.py")).encode()).hexdigest()
    path = OUT / ("contexts_" + fingerprint[:16] + ".pkl")
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    train = policy.read_sales(policy.TRAIN_PATH)
    attributes, _ = policy.build_library_attributes()
    warmup, actual, _ = policy.training_backtest_frames(train)
    print("building training contexts", len(actual), flush=True)
    contexts = policy.build_contexts(warmup, actual, attributes)
    value = (actual, contexts)
    with path.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--patch", default="{}")
    parser.add_argument("--hypothesis", required=True)
    args = parser.parse_args()
    if (OUT / "frozen/targeted_manifest.json").exists():
        raise RuntimeError("策略已冻结，开发搜索关闭；使用targeted_release.py retest执行复测")
    OUT.mkdir(parents=True, exist_ok=True)
    folder = OUT / args.name
    folder.mkdir(exist_ok=False)
    original = json.loads((ROOT / "config.json").read_text())
    config = dict(original["best_policy"])
    config.update(json.loads(args.patch))
    for filename in ["policy.py", "diagnostics.py", "evaluator.py", "targeted_search.py", "targeted_policy.py"]:
        if (ROOT / filename).exists():
            shutil.copy2(ROOT / filename, folder / filename)
    dump(folder / "config.json", config)
    record = {"name": args.name, "hypothesis": args.hypothesis, "patch": json.loads(args.patch),
              "started_at": pd.Timestamp.now().isoformat(), "status": "running",
              "source_hashes": {p.name: policy.sha256_file(p) for p in folder.glob("*.py")},
              "train_sha256": policy.sha256_file(policy.TRAIN_PATH), "library_sha256": policy.sha256_file(policy.LIBRARY_PATH)}
    dump(folder / "record.json", record)
    start = time.monotonic()
    actual, contexts = assets()
    record["schedule_sha256"] = policy.context_schedule_sha256(contexts)
    audit = []
    print("running", args.name, len(contexts), flush=True)
    if config.get("targeted_enabled"):
        spec = importlib.util.spec_from_file_location("trial_targeted_policy", folder / "targeted_policy.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        recs = module.make_recommendations(contexts, config, policy.BASE_CONSTRAINTS, selection_audit=audit)
    else:
        recs = policy.make_recommendations(contexts, config, policy.BASE_CONSTRAINTS, selection_audit=audit)
    print("evaluating", args.name, flush=True)
    metrics, sessions, segments = evaluate(recs, actual, audit)
    recs.to_csv(folder / "recommended_menu.csv", index=False)
    sessions.to_csv(folder / "sessions.csv", index=False)
    pd.DataFrame(segments).to_csv(folder / "segments.csv", index=False)
    dump(folder / "audit.json", audit)
    record.update(status="complete", metrics=metrics, elapsed_seconds=time.monotonic() - start)
    dump(folder / "record.json", record)
    with (OUT / "experiments.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=policy._json_default) + "\n")
    print(json.dumps({"name": args.name, "seconds": record["elapsed_seconds"], "metrics": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

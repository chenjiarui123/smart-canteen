"""Read-only delivery audit. Writes aggregate evidence next to this script.

No application import, inference, fitting, model/data mutation, or sales detail
export occurs. Run from any working directory using Python with pandas.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "smart_canteen_web/data"
MODEL = ROOT / "smart_canteen_web/model"
SDK = ROOT / "sdk/src/smart_canteen_sdk"
AS_OF = date(2026, 10, 19)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def normalise(value):
    value = unicodedata.normalize("NFKC", str(value)).strip().lower()
    value = value.translate(str.maketrans({"（": "(", "）": ")", "【": "[", "】": "]"}))
    return re.sub(r"\s+", " ", value)


def csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def missing(series):
    return series.astype(str).str.strip().isin(["", "nan", "None", "null"])


def protected(entries, base):
    checks = []
    for name, expected in entries.items():
        path = base / name
        actual = sha(path)
        checks.append({"file": name, "expected": expected, "actual": actual, "ok": actual == expected})
    return {"count": len(checks), "passed": sum(x["ok"] for x in checks), "checks": checks}


def source_path(name):
    direct = ROOT / name
    return direct if direct.exists() else ROOT / "research" / name


def old_registration(path):
    obj = read_json(path)
    checks = []
    for name, expected in obj["protected"].items():
        actual = sha(source_path(name))
        checks.append({"file": name, "expected": expected, "actual": actual, "ok": actual == expected})
    return {"file": str(path.relative_to(ROOT)), "count": len(checks), "passed": sum(x["ok"] for x in checks), "mismatches": [x for x in checks if not x["ok"]]}


def main():
    manifest = read_json(DATA / "manifest.json")
    registry = read_json(DATA / "dataset_registry.json")
    features, feature_dates = {}, {}
    for name in ("weather", "calendar"):
        path = DATA / "features" / (name + ".csv")
        frame = csv(path)
        dates = pd.to_datetime(frame["日期"], errors="coerce")
        unique = set(dates.dropna().dt.date)
        feature_dates[name] = unique
        all_dates = pd.date_range(dates.min(), dates.max()).date
        gaps = sorted(set(all_dates) - unique)
        features[name] = {"rows": len(frame), "schema": list(frame.columns), "duplicate_dates": int(dates.duplicated().sum()), "invalid_dates": int(dates.isna().sum()), "date_range": [str(dates.min().date()), str(dates.max().date())], "missing_calendar_dates_count": len(gaps), "missing_calendar_dates": list(map(str, gaps)), "missing_cells": {col: int(missing(frame[col]).sum()) for col in frame}, "manifest_rows_ok": len(frame) == manifest["features"][name + "_rows"], "sha256": sha(path)}
    datasets = {}
    numeric_cols = ["菜品总销量", "订单数量", "销售额", "平均取菜量", "气温（℃）", "相对湿度（%）", "降水量（毫米）", "10米风速（公里/小时）"]
    context_cols = ["星期", "是否校历寒暑假", "校历假期类型", "是否法定节假日假期", "法定节假日名称", "是否调休工作日", "气温（℃）", "相对湿度（%）", "降水量（毫米）", "10米风速（公里/小时）"]
    forbidden = {"历史平均销售额", "历史平均销量", "平均单价", "总销售额", "总销量_原始", "总销量_处理后", "数据天数", "订单数量", "记录数", "售卖天数", "总销量"}
    dataset_paths = []
    for dataset_id, info in registry["datasets"].items():
        folder = DATA / info["directory"]
        dataset_paths.append(folder.resolve())
        sales = csv(folder / "sales.csv")
        runtime = csv(folder / "runtime_actual_sales.csv")
        library = csv(folder / "dish_library.csv")
        history = csv(folder / "recommendation_history.csv")
        dates = pd.to_datetime({"year": sales["年"], "month": sales["月"], "day": sales["日"]}, errors="coerce")
        names = sales["菜品名称"].map(normalise)
        library_names = library["菜品名称"].map(normalise)
        keys = pd.DataFrame({"date": dates, "meal": sales["餐次"], "dish": names})
        unit_groups = pd.DataFrame({"dish": names, "meal": sales["餐次"], "unit": sales["菜品售卖单位"]})
        unit_sets = unit_groups.groupby("dish")["unit"].agg(lambda x: tuple(sorted(set(x))))
        changed = unit_sets.loc[unit_sets.map(len) > 1]
        changed_meal = unit_groups.groupby(["dish", "meal"])["unit"].nunique()
        not_in_library = ~names.isin(set(library_names))
        unique_dates = set(dates.dropna().dt.date)
        invalid_numbers = {}
        for col in numeric_cols:
            values = pd.to_numeric(sales[col], errors="coerce")
            invalid_numbers[col] = {"not_finite": int((values.isna() | values.isin([float("inf"), -float("inf")])).sum()), "negative": int(values.lt(0).sum()), "zero": int(values.eq(0).sum())}
        wanted = manifest["datasets"][dataset_id]
        canteens = sorted(set(sales["食堂名称"]))
        allowed = set(info["aliases"]) | {info["display_name"]}
        lib_ids = library["菜品id"].astype(str)
        datasets[dataset_id] = {
            "display_name": info["display_name"], "sales_rows": len(sales), "schema": list(sales.columns),
            "runtime_sales_rows": len(runtime), "runtime_schema_matches_sales": list(runtime.columns) == list(sales.columns),
            "recommendation_history_rows": len(history), "date_range": [str(dates.min().date()), str(dates.max().date())],
            "days_since_history_end": (AS_OF - dates.max().date()).days, "distinct_sales_days": len(unique_dates),
            "meal_counts": {str(k): int(v) for k, v in sales["餐次"].value_counts().items()},
            "invalid_dates": int(dates.isna().sum()), "future_sales_rows": int(dates.dt.date.gt(AS_OF).sum()),
            "duplicate_business_key_rows": int(keys.duplicated().sum()), "duplicate_complete_rows": int(sales.duplicated().sum()),
            "empty_dish_names": int(missing(sales["菜品名称"]).sum()), "sales_dish_count": names.nunique(),
            "unmatched_sales_rows": int(not_in_library.sum()), "unmatched_distinct_dishes": names.loc[not_in_library].nunique(),
            "canteen_values": canteens, "cross_dataset_canteen_rows": int((~sales["食堂名称"].isin(allowed)).sum()),
            "library_rows": len(library), "library_schema": list(library.columns),
            "library_duplicate_normalised_names": int(library_names.duplicated().sum()), "library_duplicate_ids": int(lib_ids.duplicated().sum()),
            "library_empty_ids": int(missing(lib_ids).sum()), "library_missing_cells": {col: int(missing(library[col]).sum()) for col in library},
            "library_dynamic_statistics_columns": sorted(forbidden & set(library.columns)),
            "unit_row_counts": {str(k): int(v) for k, v in sales["菜品售卖单位"].value_counts().items()},
            "historically_changed_unit_dishes": len(changed), "changed_unit_dish_meal_pairs": int(changed_meal.gt(1).sum()),
            "changed_unit_rows": int(names.isin(set(changed.index)).sum()), "changed_unit_pairs": {" + ".join(k): v for k, v in Counter(changed).items()},
            "missing_context_cells": {col: int(missing(sales[col]).sum()) for col in context_cols},
            "missing_feature_dates_for_sales": {name: len(unique_dates - values) for name, values in feature_dates.items()},
            "numeric_checks": invalid_numbers,
            "manifest_checks": {"sales_rows": len(sales) == wanted["sales_rows"], "date_range": [str(dates.min().date()), str(dates.max().date())] == wanted["sales_date_range"], "sales_sha256": sha(folder / "sales.csv") == wanted["sales_sha256"], "dish_library_sha256": sha(folder / "dish_library.csv") == wanted["dish_library_sha256"]},
        }
    base = read_json(MODEL / "model_manifest.json")
    target = read_json(MODEL / "targeted_manifest.json")
    asset_checks = {
        "web_trial18_code": protected(base["frozen_source_hashes"], MODEL),
        "web_active_targeted": protected(target["files"], MODEL),
        "research_frozen_baseline": protected(base["source_project_integrity_baseline"], ROOT / "research"),
        "sdk_bundled_files": protected(read_json(SDK / "BUNDLE_MANIFEST.json"), SDK),
        "sdk_original_source_references": protected(read_json(SDK / "SOURCE_MANIFEST.json")["source_sha256"], ROOT),
    }
    # Manifest parameter digests represent JSON content, not the pretty-printed file bytes.
    canonical_checks = {}
    for name, field in (("best_policy.json", "best_policy_sha256"), ("constraints.json", "constraints_sha256")):
        obj = read_json(MODEL / name)
        options = {"sorted_default": json.dumps(obj, ensure_ascii=False, sort_keys=True), "sorted_compact": json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}
        digests = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in options.items()}
        canonical_checks[name] = {"manifest_digest": base[field], "byte_digest": sha(MODEL / name), "content_digests": digests, "content_match": base[field] in digests.values()}
    result = {"audit_date": str(AS_OF), "scope": "read-only aggregate data and frozen-asset audit; no training or application test run", "data_manifest_updated_at": manifest["updated_at"], "active_strategy": target["model_version"], "features": features, "datasets": datasets, "registry_directories_distinct": len(set(dataset_paths)) == len(dataset_paths), "assets": asset_checks, "base_parameter_content_hashes": canonical_checks, "historical_research_registration": [old_registration(ROOT / "runs/demand_20260913/preregistration.json"), old_registration(ROOT / "runs/phase_20260913/registration.json")]}
    (OUT / "data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"datasets": datasets, "features": features, "asset_summary": {k: {"passed": v["passed"], "count": v["count"], "mismatch_files": [x["file"] for x in v["checks"] if not x["ok"]]} for k, v in asset_checks.items()}, "base_parameter_content_hashes": canonical_checks, "historical_research_registration": result["historical_research_registration"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

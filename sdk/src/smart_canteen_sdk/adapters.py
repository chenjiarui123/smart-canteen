"""Optional adapters for the existing CSV layout; the engine never reads files."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ._vendor.dataset_repository import normalize_dish_name
from .data import unit_scale
from .errors import CanteenError
from .schemas import DataSnapshot


def from_legacy_frames(sales, dishes, *, canteen_id, snapshot_id,
                       recommendation_history=None, automatic_menus=None,
                       allow_legacy_unit_changes=False):
    """Import legacy MODEL quantities: gram rows are converted to business grams.

    For new databases use DataSnapshot directly with business quantities.
    recommendation_history is the original formal row-level history dataframe.
    automatic_menus are MenuHistory-compatible objects, supplied explicitly.
    """
    sales, dishes = sales.copy(), dishes.copy()
    if sales.empty:
        raise CanteenError("INSUFFICIENT_HISTORY", "旧销售表为空")
    canteens = sales["食堂名称"].dropna().astype(str).unique()
    if len(canteens) != 1:
        raise CanteenError("CANTEEN_MISMATCH", "旧销售表必须只包含一个逻辑食堂")
    dishes["_norm"] = dishes["菜品名称"].map(normalize_dish_name)
    if dishes["_norm"].duplicated().any():
        raise CanteenError("AMBIGUOUS_DISH", "旧菜品库标准化菜名重复")
    catalog, identities = [], {}
    def register(row, attributes_available=True):
        name = str(row["菜品名称"])
        norm = normalize_dish_name(name)
        raw_id = row.get("菜品id", row.get("菜品ID", row.get("dish_id")))
        dish_id = str(raw_id) if raw_id is not None and pd.notna(raw_id) else "legacy_" + hashlib.sha256(norm.encode()).hexdigest()[:20]
        item = {"dish_id": dish_id, "dish_name": name,
                "attributes_available": attributes_available,
                "category": str(row.get("类别", row.get("菜品类别", "其他"))),
                "flavor": str(row.get("口味", "unknown"))}
        for dest, choices in {"main_ingredient": ["主要食材", "主料"], "ingredient_family": ["食材大类", "主要食材大类"],
                              "cooking_method": ["制作方法", "烹饪方法"], "similarity_group": ["相似菜分组", "相似菜品分组"]}.items():
            for key in choices:
                if key in row and pd.notna(row[key]):
                    item[dest] = str(row[key]); break
        for key in ["主食标记", "是否主食"]:
            if key in row and pd.notna(row[key]):
                item["staple"] = str(row[key]).strip().lower() in {"1", "true", "是", "yes", "主食"}; break
        catalog.append(item); identities[norm] = dish_id
    for row in dishes.to_dict("records"):
        register(row)
    for name in sales["菜品名称"].astype(str).unique():
        if normalize_dish_name(name) not in identities:
            register({"菜品名称": name}, attributes_available=False)
    if recommendation_history is not None and not recommendation_history.empty:
        for name in recommendation_history["菜品名称"].astype(str).unique():
            if normalize_dish_name(name) not in identities:
                register({"菜品名称": name}, attributes_available=False)
    if "date" not in sales:
        sales["date"] = pd.to_datetime(dict(year=sales["年"], month=sales["月"], day=sales["日"]))
    sales["date"] = pd.to_datetime(sales["date"])
    contexts, records = {}, []
    for row in sales.to_dict("records"):
        day = pd.Timestamp(row["date"]).date().isoformat()
        meal = str(row["餐次"])
        context = {"date": day, "canteen_id": canteen_id, "meal": meal,
                   "weather": {"temperature_c": float(row["气温（℃）"]), "humidity_pct": float(row["相对湿度（%）"]),
                               "precipitation_mm": float(row["降水量（毫米）"]), "wind_speed_kmh": float(row["10米风速（公里/小时）"])},
                   "calendar": {"school_break": bool(int(row["是否校历寒暑假"])), "school_break_type": str(row["校历假期类型"]),
                                "statutory_holiday": bool(int(row["是否法定节假日假期"])), "statutory_holiday_name": str(row["法定节假日名称"]),
                                "adjusted_workday": bool(int(row["是否调休工作日"]))}}
        key = (day, meal)
        if key in contexts and context != contexts[key]:
            raise CanteenError("INCONSISTENT_CONTEXT", "旧销售表同餐次天气或校历不一致")
        contexts[key] = context
        unit = str(row["菜品售卖单位"])
        records.append({"date": day, "canteen_id": canteen_id, "meal": meal,
                        "dish_id": identities[normalize_dish_name(row["菜品名称"])],
                        "quantity": float(row["菜品总销量"]) * unit_scale(unit), "unit": unit,
                        "order_count": float(row["订单数量"]), "revenue": float(row["销售额"]),
                        "unit_price": str(row.get("菜品单价", "")), "average_portion": float(row.get("平均取菜量", 1))})
    menus = list(automatic_menus or [])
    if recommendation_history is not None and not recommendation_history.empty:
        formal = recommendation_history.copy()
        if set(formal["食堂名称"].astype(str)) != {str(canteens[0])}:
            raise CanteenError("CANTEEN_MISMATCH", "旧推荐历史包含其他食堂")
        for (day, meal, menu_id), group in formal.groupby(["date", "餐次", "request_id"], sort=False):
            stamp = pd.Timestamp(group["created_at"].max())
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("Asia/Shanghai")
            low = group.loc[group.get("recommendation_source", pd.Series("", index=group.index)).eq("low_frequency_opportunity")]
            menus.append({"date": pd.Timestamp(day).date().isoformat(), "canteen_id": canteen_id, "meal": meal,
                          "menu_id": str(menu_id), "created_at": stamp.isoformat(), "source": "finalized",
                          "dish_ids": [identities[normalize_dish_name(n)] for n in group["菜品名称"]],
                          "low_frequency_dish_ids": [identities[normalize_dish_name(n)] for n in low["菜品名称"]]})
    return DataSnapshot.model_validate({"snapshot_id": snapshot_id, "canteen_id": canteen_id, "canteen_name": str(canteens[0]),
                                        "unit_policy": "legacy_compatibility" if allow_legacy_unit_changes else "strict",
                                        "sales_history": records, "dish_catalog": catalog,
                                        "historical_context": list(contexts.values()), "menu_history": menus})


def load_legacy_csv(sales_path, dish_library_path, *, canteen_id, snapshot_id,
                    runtime_sales_path=None, recommendation_history_path=None,
                    allow_legacy_unit_changes=False):
    """Read explicit caller paths only. Original files are never changed."""
    sales = pd.read_csv(sales_path, encoding="utf-8-sig")
    if runtime_sales_path is not None:
        runtime = pd.read_csv(runtime_sales_path, encoding="utf-8-sig")
        if not runtime.empty:
            sales = pd.concat([sales, runtime], ignore_index=True)
            sales["_norm"] = sales["菜品名称"].map(normalize_dish_name)
            sales = sales.drop_duplicates(["年", "月", "日", "食堂名称", "餐次", "_norm"], keep="last").drop(columns="_norm")
    formal = pd.read_csv(recommendation_history_path, encoding="utf-8-sig") if recommendation_history_path is not None else None
    return from_legacy_frames(sales, pd.read_csv(dish_library_path, encoding="utf-8-sig"),
                              canteen_id=canteen_id, snapshot_id=snapshot_id, recommendation_history=formal,
                              allow_legacy_unit_changes=allow_legacy_unit_changes)

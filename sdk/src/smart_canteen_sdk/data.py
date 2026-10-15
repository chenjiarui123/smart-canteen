"""Dataframe preparation without filesystem access or global mutable state."""
from __future__ import annotations

from collections import Counter
from io import StringIO

import pandas as pd

from ._vendor.context_builder import build_target_context, weekday_name
from ._vendor.dataset_repository import normalize_dish_name
from ._vendor.model_loader import frozen_policy
from .errors import CanteenError
from .schemas import DataSnapshot, RecommendationRequest


def unit_scale(unit: str) -> float:
    # Compatibility with the deployed 50 g model quantity convention.
    return 50.0 if unit.strip() == "克" else 1.0


def context_columns(context):
    return {
        "是否校历寒暑假": int(context.calendar.school_break),
        "校历假期类型": context.calendar.school_break_type,
        "是否法定节假日假期": int(context.calendar.statutory_holiday),
        "法定节假日名称": context.calendar.statutory_holiday_name,
        "是否调休工作日": int(context.calendar.adjusted_workday),
        "气温（℃）": context.weather.temperature_c,
        "相对湿度（%）": context.weather.humidity_pct,
        "降水量（毫米）": context.weather.precipitation_mm,
        "10米风速（公里/小时）": context.weather.wind_speed_kmh,
    }


def validate_snapshot(data: DataSnapshot):
    ids = [d.dish_id for d in data.dish_catalog]
    norms = [normalize_dish_name(d.dish_name) for d in data.dish_catalog]
    if len(ids) != len(set(ids)) or len(norms) != len(set(norms)) or "" in norms:
        raise CanteenError("AMBIGUOUS_DISH", "菜品 ID 和标准化菜名必须分别唯一且非空")
    known = set(ids)
    contexts = {}
    for c in data.historical_context:
        if c.canteen_id != data.canteen_id:
            raise CanteenError("CANTEEN_MISMATCH", "历史日期信息包含其他食堂")
        key = (c.date, c.meal)
        if key in contexts:
            raise CanteenError("DUPLICATE_CONTEXT", "历史日期上下文重复")
        contexts[key] = c
    keys, units = set(), {}
    for s in data.sales_history:
        if s.canteen_id != data.canteen_id:
            raise CanteenError("CANTEEN_MISMATCH", "销售历史包含其他食堂")
        if s.dish_id not in known:
            raise CanteenError("UNKNOWN_DISH", f"销售菜品未在菜品库登记: {s.dish_id}")
        key = (s.date, s.meal, s.dish_id)
        if key in keys:
            raise CanteenError("DUPLICATE_SALE", "请先按日期、餐次、菜品汇总并去重")
        keys.add(key)
        if s.dish_id in units and units[s.dish_id] != s.unit and data.unit_policy == "strict":
            raise CanteenError("UNIT_CHANGED", "同一菜品的历史单位不一致，请先统一单位")
        units[s.dish_id] = s.unit
        if (s.date, s.meal) not in contexts and (s.date, None) not in contexts:
            raise CanteenError("MISSING_CONTEXT", f"缺少 {s.date} {s.meal} 的历史天气和校历")
    menu_keys = set()
    for m in data.menu_history:
        if m.canteen_id != data.canteen_id:
            raise CanteenError("CANTEEN_MISMATCH", "菜单历史包含其他食堂")
        if not set(m.dish_ids) <= known:
            raise CanteenError("UNKNOWN_DISH", "历史菜单存在未登记菜品 ID")
        key = (m.date, m.meal, m.source, m.menu_id)
        if key in menu_keys:
            raise CanteenError("DUPLICATE_MENU", "历史菜单版本重复")
        menu_keys.add(key)
    return contexts


def prepare(data: DataSnapshot, request: RecommendationRequest):
    contexts = validate_snapshot(data)
    if request.canteen_id != data.canteen_id:
        raise CanteenError("CANTEEN_MISMATCH", "请求与快照食堂不一致")
    dishes = {d.dish_id: d for d in data.dish_catalog}
    sales = []
    for s in data.sales_history:
        if s.date >= request.prediction_date:
            continue
        context = contexts.get((s.date, s.meal), contexts.get((s.date, None)))
        sales.append({
            "年": s.date.year, "月": s.date.month, "日": s.date.day,
            "星期": weekday_name(s.date), "餐次": s.meal,
            "食堂名称": data.canteen_name or data.canteen_id, "菜品名称": dishes[s.dish_id].dish_name,
            "菜品总销量": s.quantity / unit_scale(s.unit), "菜品售卖单位": s.unit,
            "订单数量": s.order_count, "销售额": s.revenue,
            "菜品单价": s.unit_price, "平均取菜量": s.average_portion,
            **context_columns(context),
        })
    if not sales:
        raise CanteenError("INSUFFICIENT_HISTORY", "目标日期之前没有销售历史")
    if not any(s["餐次"] == request.meal for s in sales):
        raise CanteenError("INSUFFICIENT_HISTORY", "目标餐次没有销售历史")
    # Frozen parsers accept file-like objects: preserve parsing without temp files.
    history = frozen_policy.read_sales(StringIO(pd.DataFrame(sales).to_csv(index=False)))
    rows = []
    mappings = {"main_ingredient": "主要食材", "ingredient_family": "食材大类",
                "cooking_method": "制作方法", "similarity_group": "相似菜分组"}
    for d in data.dish_catalog:
        row = {"菜品名称": d.dish_name, "类别": d.category, "口味": d.flavor}
        for key, dest in mappings.items():
            if getattr(d, key) is not None:
                row[dest] = getattr(d, key)
        # A mixed optional staple column must still retain category defaults.
        row["主食标记"] = int(d.staple if d.staple is not None else d.category == "主食")
        rows.append(row)
    attrs, _ = frozen_policy.build_library_attributes(StringIO(pd.DataFrame(rows).to_csv(index=False)))
    available = {normalize_dish_name(d.dish_name) for d in data.dish_catalog if d.attributes_available}
    attrs = attrs.loc[attrs["_dish_norm"].isin(available)].copy()
    attrs["_library_dish_category"] = attrs["菜品类别"].where(
        attrs["菜品类别"].isin(["荤菜", "素菜", "主食", "甜品", "水果"]), "其他")
    target = build_target_context(pd.Timestamp(request.prediction_date), request.meal,
                                  **request.calendar.model_dump(), **request.weather.model_dump())
    return history, attrs, target


def menu_context(data: DataSnapshot, request: RecommendationRequest):
    canteen = data.canteen_name or data.canteen_id
    names = {d.dish_id: normalize_dish_name(d.dish_name) for d in data.dish_catalog}
    prior = [m for m in data.menu_history if m.date < request.prediction_date and m.meal == request.meal]
    confirmed = [m for m in prior if m.source == "finalized"]
    by_day = {}
    for m in sorted(confirmed, key=lambda x: (x.created_at, x.menu_id)):
        by_day[m.date] = {names[i] for i in m.dish_ids}
    yesterday = (pd.Timestamp(request.prediction_date) - pd.Timedelta(days=1)).date()
    reference = by_day.get(yesterday)
    source = "confirmed" if reference else None
    if not reference:
        auto = [m for m in prior if m.source == "automatic" and m.date == yesterday]
        if auto:
            chosen = max(auto, key=lambda x: (x.created_at, x.menu_id))
            reference = {names[i] for i in chosen.dish_ids}
            if reference:
                by_day[yesterday] = reference
                source = "automatic_high_frequency"
    values = [(pd.Timestamp(day), dishes) for day, dishes in sorted(by_day.items())]
    initial = {(canteen, request.meal): values} if values else {}
    recent7, recent14 = Counter(), Counter()
    target = pd.Timestamp(request.prediction_date)
    for day, dishes in values:
        if day >= target - pd.Timedelta(days=7):
            recent7.update(dishes)
        if day >= target - pd.Timedelta(days=14):
            recent14.update(dishes)
    records = []
    catalog = {d.dish_id: d.dish_name for d in data.dish_catalog}
    # Preserve deployed opportunity penalties across all formal selections.
    for m in confirmed:
        for dish_id in m.dish_ids:
            records.append({"date": pd.Timestamp(m.date), "食堂名称": canteen,
                            "餐次": m.meal, "菜品名称": catalog[dish_id],
                            "recommendation_source": "low_frequency_opportunity" if dish_id in m.low_frequency_dish_ids else "high_frequency"})
    return initial, values[-1][1] if values else set(), dict(recent7), dict(recent14), pd.DataFrame(records), reference, source

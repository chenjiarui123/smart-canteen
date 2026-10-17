#!/usr/bin/env python3
"""Leakage-safe expanding-window menu recommendation core.

Every trial walks through the same complete set of eligible training dates.
The adaptive training runner and frozen external evaluator live in separate
modules so this core cannot accidentally open external sales data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from diagnostics import (
    factor_stack_diagnostics,
    hit_quantity_diagnostics,
    menu_capacity_diagnostics,
    prediction_consistency_diagnostics,
    recommendation_repeat_diagnostics,
    selection_diagnostics,
    soft_diversity_diagnostics,
)
from evaluator import evaluate_recommendations, normalize_name


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "数据/长春A食堂_菜品销量_菜名修正后.csv"
LIBRARY_PATH = ROOT / "数据/长春A食堂_菜品库.csv"
EVALUATOR_PATH = ROOT / "evaluator.py"

CONFIG_PATH = ROOT / "config.json"

SALES_REQUIRED = {
    "年", "月", "日", "餐次", "菜品名称", "菜品总销量", "菜品售卖单位",
    "订单数量", "菜品单价", "销售额", "平均取菜量", "食堂名称",
}
KNOWN_CONTEXT_COLUMNS = [
    "星期",
    "是否校历寒暑假",
    "校历假期类型",
    "是否法定节假日假期",
    "法定节假日名称",
    "是否调休工作日",
    "气温（℃）",
    "相对湿度（%）",
    "降水量（毫米）",
    "10米风速（公里/小时）",
]
GROUP_KEYS = ["date", "食堂名称", "餐次"]
MIN_HISTORY_DAYS = 14
FEATURE_WINDOWS = [7, 14, 21, 30, 45, 60, 90]
FEATURE_HALF_LIVES = [3, 7]

BASE_CONSTRAINTS: dict[str, Any] = {
    "max_same_main_ingredient": 1,
    "max_same_ingredient_family": 2,
    "max_same_cooking_method": 2,
    "max_same_flavor": 3,
    "max_same_similarity_group": 1,
    "min_confidence_for_hard_constraint": "high",
    "medium_confidence_penalty": 0.8,
    "avoid_consecutive_non_staple": True,
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"无法序列化：{type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _flag(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "是", "工作日"}


def _calendar_state(row: pd.Series) -> str:
    school_break = _flag(row.get("是否校历寒暑假", False))
    statutory = _flag(row.get("是否法定节假日假期", False))
    adjusted = _flag(row.get("是否调休工作日", False))
    school_type = str(row.get("校历假期类型", "校历假期")).strip() or "校历假期"
    statutory_name = str(row.get("法定节假日名称", "法定节假日")).strip() or "法定节假日"
    if school_break and statutory:
        return f"校历:{school_type}|法定:{statutory_name}"
    if school_break:
        return f"校历:{school_type}"
    if statutory:
        return f"法定:{statutory_name}"
    if adjusted:
        return "调休工作日"
    return "普通周末" if pd.Timestamp(row["date"]).weekday() >= 5 else "普通工作日"


def _temperature_band(value: Any) -> str:
    temperature = float(value) if pd.notna(value) else math.nan
    if not math.isfinite(temperature):
        return "温度未知"
    if temperature <= -15:
        return "极寒"
    if temperature < 0:
        return "寒冷"
    if temperature < 10:
        return "偏凉"
    if temperature < 25:
        return "温和"
    if temperature < 30:
        return "偏热"
    if temperature < 35:
        return "高温"
    return "极端高温"


def _rain_band(value: Any) -> str:
    rain = float(value) if pd.notna(value) else math.nan
    if not math.isfinite(rain):
        return "降雨未知"
    if rain <= 0:
        return "无雨"
    if rain <= 2:
        return "小雨"
    if rain <= 10:
        return "中雨"
    if rain <= 25:
        return "大雨"
    return "强降雨"


def _derive_context_states(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_calendar_state"] = result.apply(_calendar_state, axis=1)
    school_break = result["是否校历寒暑假"].map(_flag)
    statutory = result["是否法定节假日假期"].map(_flag)
    adjusted = result["是否调休工作日"].map(_flag)
    result["_calendar_category"] = np.select(
        [school_break & statutory, school_break, statutory, adjusted],
        ["校历法定重叠", "校历假期", "法定节假日", "调休工作日"],
        default="普通日期",
    )
    result["_day_type"] = np.where(
        result["是否调休工作日"].map(_flag),
        "工作日",
        np.where(result["date"].dt.weekday.ge(5), "周末", "工作日"),
    )
    result["_temperature_band"] = result["气温（℃）"].map(_temperature_band)
    result["_rain_band"] = result["降水量（毫米）"].map(_rain_band)
    humidity = pd.to_numeric(result["相对湿度（%）"], errors="coerce")
    wind = pd.to_numeric(result["10米风速（公里/小时）"], errors="coerce")
    temperature = pd.to_numeric(result["气温（℃）"], errors="coerce")
    rain = pd.to_numeric(result["降水量（毫米）"], errors="coerce")
    result["_humidity_state"] = np.where(humidity.ge(85), "高湿", "非高湿")
    result["_wind_state"] = np.where(wind.ge(25), "大风", "非大风")
    result["_extreme_weather"] = (
        temperature.le(-15)
        | temperature.ge(35)
        | rain.ge(25)
        | wind.ge(35)
    )
    result["_weather_state"] = (
        result["_temperature_band"].astype(str)
        + "|"
        + result["_rain_band"].astype(str)
        + "|"
        + result["_humidity_state"].astype(str)
        + "|"
        + result["_wind_state"].astype(str)
    )
    result["_weather_coarse"] = (
        result["_temperature_band"].astype(str)
        + "|"
        + result["_rain_band"].astype(str)
    )
    return result


def read_sales(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(SALES_REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"{path} 缺少销售字段：{missing}")
    frame["date"] = pd.to_datetime(
        {"year": frame["年"], "month": frame["月"], "day": frame["日"]},
        errors="raise",
    )
    frame["菜品名称"] = frame["菜品名称"].astype(str)
    frame["_dish_norm"] = frame["菜品名称"].map(normalize_name)
    for column in ["菜品总销量", "订单数量", "销售额"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(frame[column])).any() or (frame[column] < 0).any():
            raise ValueError(f"{path} 的 {column} 存在负数或非有限值")
    missing_context = sorted(set(KNOWN_CONTEXT_COLUMNS) - set(frame.columns))
    if missing_context:
        raise ValueError(f"{path} 缺少已声明的日期/天气字段：{missing_context}")
    for column in ["气温（℃）", "相对湿度（%）", "降水量（毫米）", "10米风速（公里/小时）"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in KNOWN_CONTEXT_COLUMNS:
        variants = frame.groupby(GROUP_KEYS, sort=False)[column].nunique(dropna=False)
        if variants.gt(1).any():
            raise ValueError(f"{path} 同一餐次的上下文字段 {column} 不一致")
    aggregate: dict[str, tuple[str, str]] = {
        "菜品名称": ("菜品名称", "first"),
        "菜品总销量": ("菜品总销量", "sum"),
        "订单数量": ("订单数量", "sum"),
        "销售额": ("销售额", "sum"),
        "菜品售卖单位": ("菜品售卖单位", "first"),
        "菜品单价": ("菜品单价", "first"),
        "平均取菜量": ("平均取菜量", "mean"),
    }
    aggregate.update({column: (column, "first") for column in KNOWN_CONTEXT_COLUMNS})
    result = (
        frame.groupby(GROUP_KEYS + ["_dish_norm"], as_index=False, sort=False)
        .agg(**aggregate)
        .sort_values(GROUP_KEYS + ["菜品名称"])
        .reset_index(drop=True)
    )
    result["_orders_per_qty"] = (
        result["订单数量"] / result["菜品总销量"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    result["_rev_per_qty"] = (
        result["销售额"] / result["菜品总销量"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    return _derive_context_states(result)


def semantic_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", normalize_name(name))
    text = re.sub(r"^[a-z]{1,2}(?=[\u4e00-\u9fff])", "", text)
    return re.sub(r"[\s()（）\[\]【】/\\·._-]+", "", text)


ANIMAL_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("牛肉", ("牛肉", "牛腩", "牛排", "肥牛", "牛柳", "牛肚", "牛杂", "牛筋")),
    ("羊肉", ("羊肉", "羊排", "羊蝎", "羊杂", "羊肚")),
    ("鸡肉", ("鸡肉", "鸡腿", "鸡翅", "鸡丁", "鸡块", "鸡胸", "鸡柳", "鸡排", "鸡胗", "鸡心", "鸡爪", "鸡架", "白切鸡", "手撕鸡", "宫保鸡")),
    ("鸭肉", ("鸭肉", "鸭腿", "鸭翅", "鸭脖", "鸭胗", "烤鸭")),
    ("鱼肉", ("鱼", "鲈", "鲤", "鲫", "鳕", "龙利", "巴沙", "带鱼", "鲅", "草鱼", "黑鱼", "黄花", "鱼丸")),
    ("虾", ("虾", "虾仁", "大虾")),
    ("海鲜", ("蟹", "贝", "鱿鱼", "章鱼", "蛤", "扇贝", "海鲜")),
    ("猪肉", ("猪", "排骨", "五花", "里脊", "肉丝", "肉片", "肉末", "肉丸", "肉饼", "扣肉", "红烧肉", "回锅肉", "锅包肉", "肘子", "猪蹄", "肥肠", "培根", "火腿", "香肠", "午餐肉")),
]

VEGETABLE_RULES: list[tuple[str, tuple[str, ...], str]] = [
    ("西兰花", ("西兰花",), "花菜类"), ("菜花", ("菜花", "花菜"), "花菜类"),
    ("黄瓜", ("黄瓜",), "瓜茄类"), ("茄子", ("茄子", "茄条", "茄盒"), "瓜茄类"),
    ("番茄", ("番茄", "西红柿"), "瓜茄类"), ("土豆", ("土豆", "薯条"), "根茎类"),
    ("萝卜", ("萝卜",), "根茎类"), ("山药", ("山药",), "根茎类"),
    ("莲藕", ("莲藕", "藕片", "藕丁"), "根茎类"), ("南瓜", ("南瓜",), "瓜茄类"),
    ("冬瓜", ("冬瓜",), "瓜茄类"), ("苦瓜", ("苦瓜",), "瓜茄类"),
    ("丝瓜", ("丝瓜",), "瓜茄类"), ("豆角", ("豆角", "四季豆"), "豆类"),
    ("豆芽", ("豆芽",), "豆类"), ("豆苗", ("豆苗",), "叶菜类"),
    ("豆腐", ("豆腐", "豆干", "干豆腐", "千页豆腐", "腐竹"), "豆制品"),
    ("白菜", ("白菜", "娃娃菜"), "叶菜类"), ("菠菜", ("菠菜",), "叶菜类"),
    ("油菜", ("油菜",), "叶菜类"), ("生菜", ("生菜",), "叶菜类"),
    ("芹菜", ("芹菜",), "叶菜类"), ("韭菜", ("韭菜",), "叶菜类"),
    ("圆白菜", ("圆白菜", "甘蓝", "卷心菜"), "叶菜类"),
    ("青椒", ("青椒", "尖椒", "彩椒"), "椒类"), ("木耳", ("木耳",), "菌藻类"),
    ("菌菇", ("蘑菇", "香菇", "金针菇", "杏鲍菇", "平菇", "口蘑", "菌菇"), "菌藻类"),
    ("海带", ("海带",), "菌藻类"), ("玉米", ("玉米",), "谷薯类"),
]

STAPLE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("米", ("米饭", "炒饭", "盖饭", "饭团", "粥")),
    ("面", ("面条", "拌面", "汤面", "炒面", "冷面", "馄饨", "饺子")),
    ("面粉", ("馒头", "花卷", "包子", "烧麦", "面包", "蛋糕", "饼")),
]

METHOD_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("红烧", ("红烧",)), ("凉拌", ("凉拌", "拌", "拍黄瓜", "沙拉")),
    ("清炒", ("清炒", "素炒",)), ("炒", ("炒", "爆",)),
    ("蒸", ("蒸", "清蒸",)), ("炖", ("炖", "煲",)), ("煮", ("煮", "水煮", "涮")),
    ("炸", ("炸", "酥", "锅包",)), ("煎", ("煎",)), ("烤", ("烤",)),
    ("卤", ("卤",)), ("焖", ("焖",)), ("酱", ("酱",)), ("熘", ("熘", "溜")),
    ("汤", ("汤", "羹",)), ("腌制", ("泡菜", "腌",)),
]


def _method_from_name(text: str) -> tuple[str, bool]:
    for method, keywords in METHOD_RULES:
        if any(keyword in text for keyword in keywords):
            return method, True
    return "unknown", False


def infer_name_attributes(name: str, category: str = "") -> dict[str, Any]:
    """Conservative, static rules built without looking at validation/test outcomes."""

    text = semantic_name(name)
    method, method_hard = _method_from_name(text)
    ingredient = "unknown"
    broad_family = "unknown"
    confidence = "low"
    main_hard = False

    animal_hits: list[str] = []
    for label, keywords in ANIMAL_RULES:
        if any(keyword in text for keyword in keywords):
            animal_hits.append(label)
    if len(set(animal_hits)) == 1:
        ingredient = animal_hits[0]
        confidence = "high"
        main_hard = True
    elif len(set(animal_hits)) > 1:
        ingredient = "+".join(dict.fromkeys(animal_hits))
        confidence = "medium"

    if ingredient == "unknown" and any(word in text for word in ["鸡蛋", "蛋羹", "蛋花", "炒蛋"]):
        ingredient, broad_family, confidence, main_hard = "鸡蛋", "蛋类", "high", True

    vegetable_hits: list[tuple[str, str]] = []
    for label, keywords, family in VEGETABLE_RULES:
        if any(keyword in text for keyword in keywords):
            vegetable_hits.append((label, family))
    if ingredient == "unknown" and len({item[0] for item in vegetable_hits}) == 1:
        ingredient, broad_family = vegetable_hits[0]
        confidence, main_hard = "high", True
    elif ingredient == "unknown" and len({item[0] for item in vegetable_hits}) > 1:
        ingredient = "+".join(dict.fromkeys(item[0] for item in vegetable_hits))
        broad_family, confidence = "复合蔬菜", "medium"

    if ingredient == "unknown" and str(category) == "主食":
        staple_hits = [label for label, words in STAPLE_RULES if any(word in text for word in words)]
        if len(set(staple_hits)) == 1:
            ingredient, broad_family, confidence, main_hard = staple_hits[0], "主食", "high", True
        else:
            broad_family, confidence = "主食", "medium"

    if ingredient == "unknown" and "肉" in text:
        ingredient, broad_family, confidence = "肉类", "肉类-来源不明", "medium"

    if broad_family == "unknown" and ingredient != "unknown":
        if ingredient in {"猪肉", "牛肉", "羊肉", "鸡肉", "鸭肉"}:
            function = (
                "浓味" if method in {"红烧", "酱", "卤", "焖"}
                else "炸煎" if method in {"炸", "煎"}
                else "清淡" if method in {"蒸", "煮", "炖", "汤"}
                else "常规"
            )
            broad_family = f"{function}{ingredient}"
        elif ingredient in {"鱼肉", "虾", "海鲜"}:
            broad_family = f"水产-{method if method != 'unknown' else '常规'}"
        elif ingredient == "鸡蛋":
            broad_family = "蛋类"
        else:
            broad_family = ingredient

    similarity = "unknown"
    similarity_hard = False
    if ingredient == "黄瓜" and method == "凉拌":
        similarity, similarity_hard = "凉拌黄瓜", True
    elif ingredient == "茄子" and method == "凉拌":
        similarity, similarity_hard = "凉拌茄子", True
    elif ingredient == "番茄" and "蛋" in text:
        similarity, similarity_hard = "番茄鸡蛋", True
    elif main_hard and method_hard:
        similarity, similarity_hard = f"{ingredient}-{method}", True

    return {
        "inferred_main_ingredient": ingredient,
        "ingredient_family": broad_family,
        "制作方法": method,
        "相似菜分组": similarity,
        "inference_confidence": confidence,
        "attribute_source": "dish_name" if ingredient != "unknown" else "unknown",
        "_main_hard": main_hard,
        "_family_hard": main_hard and broad_family not in {"unknown", "肉类-来源不明"},
        "_method_hard": method_hard,
        "_similarity_hard": similarity_hard,
    }


def _first_existing(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def build_library_attributes(library_path: Path = LIBRARY_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    library = pd.read_csv(library_path)
    if not {"菜品名称", "类别", "口味"}.issubset(library.columns):
        raise ValueError("菜品库至少必须包含 菜品名称/类别/口味")
    library = library.copy()
    library["_dish_norm"] = library["菜品名称"].map(normalize_name)
    if library["_dish_norm"].duplicated().any():
        collisions = library.loc[library["_dish_norm"].duplicated(False), "菜品名称"].tolist()
        raise ValueError(f"名称标准化后菜品库存在歧义：{collisions[:10]}")

    main_col = _first_existing(library, ["主要食材", "主料"])
    family_col = _first_existing(library, ["食材大类", "主要食材大类"])
    method_col = _first_existing(library, ["制作方法", "烹饪方法"])
    staple_col = _first_existing(library, ["主食标记", "是否主食"])
    similarity_col = _first_existing(library, ["相似菜分组", "相似菜品分组"])
    rows: list[dict[str, Any]] = []
    for row in library.to_dict("records"):
        inferred = infer_name_attributes(str(row["菜品名称"]), str(row.get("类别", "")))
        lib_main = str(row.get(main_col, "")).strip() if main_col else ""
        lib_family = str(row.get(family_col, "")).strip() if family_col else ""
        lib_method = str(row.get(method_col, "")).strip() if method_col else ""
        lib_similarity = str(row.get(similarity_col, "")).strip() if similarity_col else ""
        has_library_main = bool(lib_main and lib_main.lower() not in {"nan", "none", "unknown", "未知"})
        main = lib_main if has_library_main else inferred["inferred_main_ingredient"]
        family = lib_family if lib_family and lib_family.lower() != "nan" else inferred["ingredient_family"]
        method = lib_method if lib_method and lib_method.lower() != "nan" else inferred["制作方法"]
        similarity = (
            lib_similarity
            if lib_similarity and lib_similarity.lower() != "nan"
            else inferred["相似菜分组"]
        )
        if staple_col:
            raw_staple = row.get(staple_col)
            staple = str(raw_staple).strip().lower() in {"1", "true", "是", "yes", "主食"}
        else:
            staple = str(row.get("类别", "")) == "主食"
        source = "library" if has_library_main else inferred["attribute_source"]
        confidence = "high" if has_library_main else inferred["inference_confidence"]
        rows.append(
            {
                "_dish_norm": row["_dish_norm"],
                "菜品名称_库": str(row["菜品名称"]),
                "菜品类别": str(row.get("类别", "unknown")),
                "口味": str(row.get("口味", "unknown")),
                "主要食材": main,
                "食材大类": family,
                "制作方法": method,
                "相似菜分组": similarity,
                "主食标记": bool(staple),
                "inferred_main_ingredient": inferred["inferred_main_ingredient"],
                "ingredient_family": inferred["ingredient_family"],
                "attribute_source": source,
                "inference_confidence": confidence,
                "library_matched": True,
                "_main_hard": True if has_library_main else bool(inferred["_main_hard"]),
                "_family_hard": True if lib_family else bool(inferred["_family_hard"]),
                "_method_hard": True if lib_method else bool(inferred["_method_hard"]),
                "_flavor_hard": str(row.get("口味", "")).strip() not in {"", "unknown", "未知"},
                "_similarity_hard": True if lib_similarity else bool(inferred["_similarity_hard"]),
                "特征来源": str(row.get("特征来源", "")),
            }
        )
    return pd.DataFrame(rows), library


def _dish_aggregate(
    history: pd.DataFrame, target_date: pd.Timestamp, weekday: int, prefix: str
) -> pd.DataFrame:
    """Aggregate only rows strictly before target_date."""

    columns = [
        f"{prefix}offer_count", f"{prefix}last_date", f"{prefix}qty_mean",
        f"{prefix}qty_log_mean", f"{prefix}qty_std",
        f"{prefix}qty_positive_count", f"{prefix}qty_min_batch",
        f"{prefix}orders_mean", f"{prefix}rev_mean", f"{prefix}rev_std",
        f"{prefix}orders_per_qty_mean", f"{prefix}rev_per_qty_mean",
        f"{prefix}weekday_count", f"{prefix}weekday_qty_mean",
        f"{prefix}weekday_qty_log_mean",
        f"{prefix}weekday_orders_mean", f"{prefix}weekday_rev_mean",
        f"{prefix}recent7_count",
    ]
    columns += [
        f"{prefix}w{window}_{metric}_mean"
        for window in FEATURE_WINDOWS
        for metric in ["qty", "orders", "rev", "orders_per_qty", "rev_per_qty"]
    ]
    columns += [f"{prefix}w{window}_count" for window in FEATURE_WINDOWS]
    columns += [f"{prefix}w{window}_qty_log_mean" for window in FEATURE_WINDOWS]
    if history.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="_dish_norm"))
    if not {"_orders_per_qty", "_rev_per_qty"}.issubset(history.columns):
        history = history.copy()
        history["_orders_per_qty"] = (
            history["订单数量"] / history["菜品总销量"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        history["_rev_per_qty"] = (
            history["销售额"] / history["菜品总销量"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
    if history["date"].max() >= target_date:
        raise AssertionError("检测到未来数据进入菜品特征")
    history = history.copy()
    history["_qty_log1p"] = np.log1p(history["菜品总销量"].clip(lower=0))
    history["_positive_qty"] = history["菜品总销量"].where(
        history["菜品总销量"].gt(0)
    )

    base = history.groupby("_dish_norm", sort=False).agg(
        offer_count=("date", "nunique"),
        last_date=("date", "max"),
        qty_mean=("菜品总销量", "mean"),
        qty_log_mean=("_qty_log1p", "mean"),
        qty_std=("菜品总销量", "std"),
        qty_positive_count=("_positive_qty", "count"),
        qty_min_batch=("_positive_qty", lambda values: values.quantile(0.25)),
        orders_mean=("订单数量", "mean"),
        rev_mean=("销售额", "mean"),
        rev_std=("销售额", "std"),
        orders_per_qty_mean=("_orders_per_qty", "mean"),
        rev_per_qty_mean=("_rev_per_qty", "mean"),
    )
    same_weekday = history.loc[history["date"].dt.weekday == weekday]
    weekday_stats = same_weekday.groupby("_dish_norm", sort=False).agg(
        weekday_count=("date", "nunique"),
        weekday_qty_mean=("菜品总销量", "mean"),
        weekday_qty_log_mean=("_qty_log1p", "mean"),
        weekday_orders_mean=("订单数量", "mean"),
        weekday_rev_mean=("销售额", "mean"),
    )
    base["qty_log_mean"] = np.expm1(base["qty_log_mean"])
    weekday_stats["weekday_qty_log_mean"] = np.expm1(
        weekday_stats["weekday_qty_log_mean"]
    )
    base = base.join(weekday_stats, how="left")
    recent7 = history.loc[history["date"] >= target_date - pd.Timedelta(days=7)]
    base = base.join(
        recent7.groupby("_dish_norm")["date"].nunique().rename("recent7_count"), how="left"
    )
    for window in FEATURE_WINDOWS:
        recent = history.loc[history["date"] >= target_date - pd.Timedelta(days=window)]
        stats = recent.groupby("_dish_norm", sort=False).agg(
            **{
                f"w{window}_count": ("date", "nunique"),
                f"w{window}_qty_mean": ("菜品总销量", "mean"),
                f"w{window}_qty_log_mean": ("_qty_log1p", "mean"),
                f"w{window}_orders_mean": ("订单数量", "mean"),
                f"w{window}_rev_mean": ("销售额", "mean"),
                f"w{window}_orders_per_qty_mean": ("_orders_per_qty", "mean"),
                f"w{window}_rev_per_qty_mean": ("_rev_per_qty", "mean"),
            }
        )
        stats[f"w{window}_qty_log_mean"] = np.expm1(
            stats[f"w{window}_qty_log_mean"]
        )
        base = base.join(stats, how="left")
    base = base.rename(columns={column: f"{prefix}{column}" for column in base.columns})
    return base


def _state_total_stats(
    sessions: pd.DataFrame,
    column: str,
    eligible: pd.Series | None = None,
) -> dict[str, dict[str, float]]:
    source = sessions if eligible is None else sessions.loc[eligible]
    if source.empty or column not in source:
        return {}
    grouped = source.groupby(column, dropna=False).agg(
        mean=("total_qty", "mean"),
        median=("total_qty", "median"),
        count=("total_qty", "count"),
        dish_count_mean=("dish_count", "mean"),
        dish_count_median=("dish_count", "median"),
        qty_per_dish_mean=("qty_per_dish", "mean"),
        qty_per_dish_median=("qty_per_dish", "median"),
    )
    return {
        str(index): {
            "mean": float(row["mean"]),
            "median": float(row["median"]),
            "count": float(row["count"]),
            "dish_count_mean": float(row["dish_count_mean"]),
            "dish_count_median": float(row["dish_count_median"]),
            "qty_per_dish_mean": float(row["qty_per_dish_mean"]),
            "qty_per_dish_median": float(row["qty_per_dish_median"]),
        }
        for index, row in grouped.iterrows()
    }


def _session_aggregate(
    history: pd.DataFrame, target_date: pd.Timestamp, weekday: int, prefix: str
) -> dict[str, Any]:
    """Summarise historical menu sessions for capacity and group totals."""

    result: dict[str, Any] = {f"{prefix}session_count": 0.0}
    if history.empty:
        return result
    if history["date"].max() >= target_date:
        raise AssertionError("检测到未来数据进入餐次特征")
    aggregate: dict[str, tuple[str, str]] = {
        "dish_count": ("_dish_norm", "nunique"),
        "total_qty": ("菜品总销量", "sum"),
        "total_orders": ("订单数量", "sum"),
        "total_rev": ("销售额", "sum"),
    }
    for column in [
        "_calendar_state",
        "_calendar_category",
        "_day_type",
        "_temperature_band",
        "_rain_band",
        "_humidity_state",
        "_wind_state",
        "_extreme_weather",
        "_weather_state",
        "_weather_coarse",
    ]:
        if column in history:
            aggregate[column] = (column, "first")
    sessions = history.groupby(["date", "食堂名称"], as_index=False).agg(**aggregate)
    sessions["qty_per_dish"] = (
        sessions["total_qty"] / sessions["dish_count"].replace(0, np.nan)
    )
    result[f"{prefix}session_count"] = float(len(sessions))
    result[f"{prefix}recent7_session_count"] = float(
        (sessions["date"] >= target_date - pd.Timedelta(days=7)).sum()
    )
    same_weekday = sessions.loc[sessions["date"].dt.weekday == weekday]
    regular_mask = sessions.get(
        "_calendar_category", pd.Series("普通日期", index=sessions.index)
    ).eq("普通日期")
    regular_sessions = sessions.loc[regular_mask]
    if "_day_type" in regular_sessions:
        regular_day_types = regular_sessions["_day_type"].astype(str)
    else:
        regular_day_types = pd.Series(
            np.where(regular_sessions["date"].dt.weekday.ge(5), "周末", "工作日"),
            index=regular_sessions.index,
        )
    regular_workdays = regular_sessions.loc[regular_day_types.eq("工作日")]
    regular_weekends = regular_sessions.loc[regular_day_types.eq("周末")]
    regular_same_weekday = regular_sessions.loc[
        regular_sessions["date"].dt.weekday == weekday
    ]

    def exponentially_weighted_mean(
        source_frame: pd.DataFrame, value_column: str, half_life: int
    ) -> float:
        if source_frame.empty:
            return math.nan
        ages = (target_date - source_frame["date"]).dt.days.clip(lower=1)
        weights = np.power(0.5, ages / float(half_life))
        values = pd.to_numeric(source_frame[value_column], errors="coerce")
        valid = values.notna() & np.isfinite(weights)
        if not valid.any() or float(weights[valid].sum()) <= 0:
            return math.nan
        return float(np.average(values[valid], weights=weights[valid]))

    result[f"{prefix}weekday_session_count"] = float(len(same_weekday))
    result[f"{prefix}capacity_median"] = float(
        (same_weekday if not same_weekday.empty else sessions)["dish_count"].median()
    )
    result[f"{prefix}capacity_mean"] = float(sessions["dish_count"].mean())
    result[f"{prefix}qty_per_dish_median"] = float(
        sessions["qty_per_dish"].median()
    )
    for metric in ["qty", "orders", "rev"]:
        source = f"total_{metric}"
        result[f"{prefix}{metric}_mean"] = float(sessions[source].mean())
        result[f"{prefix}{metric}_last"] = float(
            sessions.sort_values("date")[source].iloc[-1]
        )
        ordered_sessions = sessions.sort_values("date")
        result[f"{prefix}{metric}_previous"] = float(
            ordered_sessions[source].iloc[-2]
            if len(ordered_sessions) >= 2
            else ordered_sessions[source].iloc[-1]
        )
        result[f"{prefix}weekday_{metric}_mean"] = float(
            same_weekday[source].mean() if not same_weekday.empty else sessions[source].mean()
        )
        for window in FEATURE_WINDOWS:
            recent = sessions.loc[sessions["date"] >= target_date - pd.Timedelta(days=window)]
            regular_recent = regular_sessions.loc[
                regular_sessions["date"] >= target_date - pd.Timedelta(days=window)
            ]
            result[f"{prefix}w{window}_session_count"] = float(len(recent))
            result[f"{prefix}w{window}_{metric}_mean"] = float(
                recent[source].mean() if not recent.empty else sessions[source].mean()
            )
            result[f"{prefix}regular_w{window}_session_count"] = float(len(regular_recent))
            result[f"{prefix}regular_w{window}_{metric}_mean"] = float(
                regular_recent[source].mean()
                if not regular_recent.empty
                else (
                    regular_sessions[source].mean()
                    if not regular_sessions.empty
                    else sessions[source].mean()
                )
            )
        for half_life in FEATURE_HALF_LIVES:
            result[f"{prefix}ewm{half_life}_{metric}_mean"] = (
                exponentially_weighted_mean(sessions, source, half_life)
            )
            regular_ewm_source = (
                regular_sessions if not regular_sessions.empty else sessions
            )
            result[f"{prefix}regular_ewm{half_life}_{metric}_mean"] = (
                exponentially_weighted_mean(
                    regular_ewm_source, source, half_life
                )
            )
        regular_source = regular_sessions if not regular_sessions.empty else sessions
        regular_weekday_source = (
            regular_same_weekday
            if not regular_same_weekday.empty
            else regular_source
        )
        result[f"{prefix}regular_{metric}_mean"] = float(regular_source[source].mean())
        result[f"{prefix}regular_{metric}_last"] = float(
            regular_source.sort_values("date")[source].iloc[-1]
        )
        ordered_regular = regular_source.sort_values("date")
        result[f"{prefix}regular_{metric}_previous"] = float(
            ordered_regular[source].iloc[-2]
            if len(ordered_regular) >= 2
            else ordered_regular[source].iloc[-1]
        )
        result[f"{prefix}regular_weekday_{metric}_mean"] = float(
            regular_weekday_source[source].mean()
        )
        result[f"{prefix}regular_workday_{metric}_mean"] = float(
            (regular_workdays if not regular_workdays.empty else regular_source)[source].mean()
        )
        result[f"{prefix}regular_weekend_{metric}_mean"] = float(
            (regular_weekends if not regular_weekends.empty else regular_source)[source].mean()
        )
    result[f"{prefix}regular_session_count"] = float(len(regular_sessions))
    regular_capacity_source = (
        regular_sessions if not regular_sessions.empty else sessions
    )
    result[f"{prefix}regular_capacity_median"] = float(
        regular_capacity_source["dish_count"].median()
    )
    result[f"{prefix}regular_qty_per_dish_median"] = float(
        regular_capacity_source["qty_per_dish"].median()
    )
    result[f"{prefix}regular_workday_capacity_median"] = float(
        (regular_workdays if not regular_workdays.empty else regular_capacity_source)[
            "dish_count"
        ].median()
    )
    result[f"{prefix}regular_weekend_capacity_median"] = float(
        (regular_weekends if not regular_weekends.empty else regular_capacity_source)[
            "dish_count"
        ].median()
    )
    result[f"{prefix}regular_workday_qty_per_dish_median"] = float(
        (regular_workdays if not regular_workdays.empty else regular_capacity_source)[
            "qty_per_dish"
        ].median()
    )
    result[f"{prefix}regular_weekend_qty_per_dish_median"] = float(
        (regular_weekends if not regular_weekends.empty else regular_capacity_source)[
            "qty_per_dish"
        ].median()
    )
    result[f"{prefix}calendar_state_stats"] = _state_total_stats(
        sessions, "_calendar_state"
    )
    result[f"{prefix}calendar_category_stats"] = _state_total_stats(
        sessions, "_calendar_category"
    )
    special_sessions = sessions.loc[~regular_mask]
    if not special_sessions.empty:
        result[f"{prefix}special_calendar_stats"] = {
            "special": {
                "mean": float(special_sessions["total_qty"].mean()),
                "median": float(special_sessions["total_qty"].median()),
                "count": float(len(special_sessions)),
                "dish_count_mean": float(special_sessions["dish_count"].mean()),
                "dish_count_median": float(
                    special_sessions["dish_count"].median()
                ),
                "qty_per_dish_mean": float(
                    special_sessions["qty_per_dish"].mean()
                ),
                "qty_per_dish_median": float(
                    special_sessions["qty_per_dish"].median()
                ),
            }
        }
    else:
        result[f"{prefix}special_calendar_stats"] = {}
    for column, name in [
        ("_weather_state", "weather_state_stats"),
        ("_weather_coarse", "weather_coarse_stats"),
        ("_temperature_band", "temperature_stats"),
        ("_rain_band", "rain_stats"),
    ]:
        result[f"{prefix}{name}"] = _state_total_stats(
            sessions, column, regular_mask
        )
    return result


def build_contexts(
    prior_history: pd.DataFrame,
    evaluation_actual: pd.DataFrame,
    attributes: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build rolling contexts, appending a day only after all its predictions."""

    if evaluation_actual.empty:
        return []
    first_date = evaluation_actual["date"].min()
    if not prior_history.empty and prior_history["date"].max() >= first_date:
        raise ValueError("prior_history 必须严格早于评估区间")
    history = prior_history.copy()
    contexts: list[dict[str, Any]] = []
    for target_date in sorted(evaluation_actual["date"].unique()):
        target_date = pd.Timestamp(target_date)
        if not history.empty and history["date"].max() >= target_date:
            raise AssertionError("滚动历史越过预测日期")
        day_actual = evaluation_actual.loc[evaluation_actual["date"] == target_date]
        target_groups = day_actual[["食堂名称", "餐次"]].drop_duplicates()
        day_context_start = len(contexts)
        history_rows_at_day_start = int(len(history))
        history_dates_at_day_start = int(history["date"].nunique()) if not history.empty else 0
        for canteen, meal in target_groups.itertuples(index=False, name=None):
            weekday = int(target_date.weekday())
            target_rows = day_actual.loc[
                day_actual["食堂名称"].eq(canteen) & day_actual["餐次"].eq(meal)
            ]
            context_columns = KNOWN_CONTEXT_COLUMNS + [
                "_calendar_state",
                "_calendar_category",
                "_day_type",
                "_temperature_band",
                "_rain_band",
                "_humidity_state",
                "_wind_state",
                "_extreme_weather",
                "_weather_state",
                "_weather_coarse",
            ]
            target_context = {
                column: target_rows[column].iloc[0]
                for column in context_columns
                if column in target_rows
            }
            target_context["_meal"] = str(meal)
            global_history = history.loc[history["餐次"] == meal]
            local_history = global_history.loc[global_history["食堂名称"] == canteen]
            local_features = _dish_aggregate(local_history, target_date, weekday, "l_")
            global_features = _dish_aggregate(global_history, target_date, weekday, "g_")
            local_regular_history = local_history.loc[
                local_history.get(
                    "_calendar_category",
                    pd.Series("普通日期", index=local_history.index),
                ).eq("普通日期")
            ]
            global_regular_history = global_history.loc[
                global_history.get(
                    "_calendar_category",
                    pd.Series("普通日期", index=global_history.index),
                ).eq("普通日期")
            ]
            local_regular_features = _dish_aggregate(
                local_regular_history, target_date, weekday, "lr_"
            )
            global_regular_features = _dish_aggregate(
                global_regular_history, target_date, weekday, "gr_"
            )
            features = (
                local_features.join(global_features, how="outer")
                .join(local_regular_features, how="outer")
                .join(global_regular_features, how="outer")
                .reset_index()
            )
            features = attributes.merge(features, on="_dish_norm", how="outer")
            unmatched_mask = features["library_matched"].isna()
            if unmatched_mask.any():
                name_lookup = (
                    global_history.drop_duplicates("_dish_norm")
                    .set_index("_dish_norm")["菜品名称"]
                    .to_dict()
                )
                for index in features.index[unmatched_mask]:
                    dish_norm = str(features.at[index, "_dish_norm"])
                    dish_name = str(name_lookup.get(dish_norm, dish_norm))
                    inferred = infer_name_attributes(dish_name)
                    fallback = {
                        "菜品名称_库": dish_name, "菜品类别": "unknown", "口味": "unknown",
                        "主要食材": inferred["inferred_main_ingredient"],
                        "食材大类": inferred["ingredient_family"], "制作方法": inferred["制作方法"],
                        "相似菜分组": inferred["相似菜分组"], "主食标记": False,
                        "inferred_main_ingredient": inferred["inferred_main_ingredient"],
                        "ingredient_family": inferred["ingredient_family"],
                        "attribute_source": inferred["attribute_source"],
                        "inference_confidence": inferred["inference_confidence"],
                        "library_matched": False, "_main_hard": inferred["_main_hard"],
                        "_family_hard": inferred["_family_hard"], "_method_hard": inferred["_method_hard"],
                        "_flavor_hard": False, "_similarity_hard": inferred["_similarity_hard"],
                        "特征来源": "未匹配-仅菜名推断",
                    }
                    for key, value in fallback.items():
                        features.at[index, key] = value
            local_meta = _session_aggregate(local_history, target_date, weekday, "l_")
            global_meta = _session_aggregate(global_history, target_date, weekday, "g_")
            contexts.append(
                {
                    "date": target_date,
                    "食堂名称": str(canteen),
                    "餐次": str(meal),
                    "weekday": weekday,
                    "target_context": target_context,
                    "features": features,
                    "meta": {**global_meta, **local_meta},
                    "max_history_date": None if history.empty else history["date"].max(),
                    "history_rows_at_day_start": history_rows_at_day_start,
                    "history_dates_at_day_start": history_dates_at_day_start,
                }
            )
        day_group_count = len(contexts) - day_context_start
        for context in contexts[day_context_start:]:
            context["day_group_count"] = day_group_count
            context["actual_appended_after_day_batch"] = True
        # This ordering is the central no-future-data guarantee.
        history = pd.concat([history, day_actual], ignore_index=True)
    return contexts


def training_backtest_frames(
    train: pd.DataFrame, min_history_days: int = MIN_HISTORY_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split only the minimal warm-up prefix from all eligible walk-forward dates."""

    if min_history_days < 1:
        raise ValueError("min_history_days 必须为正整数")
    train_start = pd.Timestamp(train["date"].min())
    first_eval_date = train_start + pd.Timedelta(days=int(min_history_days))
    warmup = train.loc[train["date"] < first_eval_date].copy()
    evaluation = train.loc[train["date"] >= first_eval_date].copy()
    if evaluation.empty:
        raise ValueError("训练集在最小预热期后没有可评分日期")
    if pd.Timestamp(evaluation["date"].min()) < first_eval_date:
        raise AssertionError("回测日期早于最小预热边界")
    if not warmup.empty and warmup["date"].max() >= evaluation["date"].min():
        raise AssertionError("预热数据与评分日期重叠")
    return warmup, evaluation, first_eval_date


def context_schedule_frame(contexts: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(context["date"]),
                "食堂名称": context["食堂名称"],
                "餐次": context["餐次"],
                "max_history_date": context["max_history_date"],
                "history_rows_at_day_start": context["history_rows_at_day_start"],
                "history_dates_at_day_start": context["history_dates_at_day_start"],
                "day_group_count": context["day_group_count"],
            }
            for context in contexts
        ]
    ).sort_values(GROUP_KEYS).reset_index(drop=True)


def context_schedule_sha256(contexts: list[dict[str, Any]]) -> str:
    schedule = context_schedule_frame(contexts).copy()
    for column in ["date", "max_history_date"]:
        schedule[column] = pd.to_datetime(schedule[column]).dt.strftime("%Y-%m-%d")
    payload = schedule.to_json(orient="records", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _series(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _effective_series(
    frame: pd.DataFrame, suffix: str, tau: float, local_count: pd.Series
) -> pd.Series:
    local = _series(frame, f"l_{suffix}")
    global_value = _series(frame, f"g_{suffix}")
    if tau <= 0:
        return local.where(local.notna(), global_value)
    weight = local_count / (local_count + float(tau))
    blended = weight * local.fillna(global_value) + (1 - weight) * global_value.fillna(local)
    return blended.where(blended.notna(), local.where(local.notna(), global_value))


def _effective_probability(
    local_count: pd.Series,
    global_count: pd.Series,
    local_sessions: float,
    global_sessions: float,
    tau: float,
) -> pd.Series:
    local_probability = local_count / max(local_sessions, 1.0)
    global_probability = global_count / max(global_sessions, 1.0)
    if local_sessions <= 0:
        return global_probability.clip(0, 1)
    weight = local_count / (local_count + tau) if tau > 0 else pd.Series(1.0, index=local_count.index)
    return (weight * local_probability + (1 - weight) * global_probability).clip(0, 1)


def _weekday_factor(
    weekday_mean: pd.Series,
    long_mean: pd.Series,
    weekday_count: pd.Series,
    policy: dict[str, Any],
) -> pd.Series:
    raw = (weekday_mean / long_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    low, high = policy["weekday_clip"]
    raw = raw.clip(float(low), float(high))
    shrink = float(policy.get("weekday_shrink", 0.0))
    if shrink > 0:
        reliability = weekday_count / (weekday_count + shrink)
        raw = 1.0 + (raw - 1.0) * reliability
    return raw.pow(float(policy["weekday_strength"]))


def _is_known(value: Any) -> bool:
    return str(value).strip().lower() not in {"", "nan", "none", "unknown", "未知"}


def _adaptive_recent_weight(
    base_weight: float | pd.Series,
    recent_count: float | pd.Series,
    policy: dict[str, Any],
) -> float | pd.Series:
    """Reduce recent influence when its window contains few observations."""

    maximum = float(policy.get("max_recent_weight", 1.0))
    tau = float(policy.get("adaptive_blend_tau", 0.0))
    weight = np.minimum(base_weight, maximum)
    if tau > 0:
        reliability = recent_count / (recent_count + tau)
        weight = weight * reliability
    if isinstance(weight, pd.Series):
        return weight.clip(lower=0.0, upper=1.0)
    return float(min(max(float(weight), 0.0), 1.0))


def _add_all_counters(row: pd.Series, counters: dict[str, Counter]) -> None:
    similarity = row.get("相似菜分组", "unknown")
    if bool(row.get("_similarity_hard", False)) and _is_known(similarity):
        counters["similarity"][str(similarity)] += 1
    for value_col, counter_key in [
        ("主要食材", "main_all"),
        ("食材大类", "family_all"),
        ("制作方法", "method_all"),
        ("口味", "flavor_all"),
    ]:
        value = row.get(value_col, "unknown")
        if _is_known(value):
            counters[counter_key][str(value)] += 1


def _state_adjustment(
    meta: dict[str, Any],
    target_context: dict[str, Any],
    policy: dict[str, Any],
    kind: str,
) -> tuple[float, float]:
    """Return a shrunk local/global state factor and its effective sample count."""

    if kind == "calendar":
        state = str(target_context.get("_calendar_state", "普通工作日"))
        category = str(target_context.get("_calendar_category", "普通日期"))
        if category == "普通日期":
            return 1.0, math.inf
        candidates = [
            ("calendar_state_stats", state),
            ("calendar_category_stats", category),
            ("special_calendar_stats", "special"),
        ]
        minimum = float(policy.get("calendar_exact_min_count", 3.0))
        shrinkage = float(policy.get("calendar_shrinkage_tau", 5.0))
        clip_low, clip_high = policy.get("calendar_factor_clip", [0.5, 1.3])
        strength = float(policy.get("calendar_factor_strength", 1.0))
        strengths_by_category = policy.get(
            "calendar_factor_strength_by_category", {}
        )
        if (
            isinstance(strengths_by_category, dict)
            and category in strengths_by_category
        ):
            strength = float(strengths_by_category[category])
    elif kind == "weather":
        weather_candidates = {
            "joint": (
                "weather_state_stats",
                str(target_context.get("_weather_state", "")),
            ),
            "coarse": (
                "weather_coarse_stats",
                str(target_context.get("_weather_coarse", "")),
            ),
            "temperature": (
                "temperature_stats",
                str(target_context.get("_temperature_band", "")),
            ),
            "rain": (
                "rain_stats",
                str(target_context.get("_rain_band", "")),
            ),
        }
        components = policy.get(
            "weather_components", ["joint", "coarse", "temperature", "rain"]
        )
        candidates = [
            weather_candidates[name]
            for name in components
            if name in weather_candidates
        ]
        minimum = float(policy.get("weather_exact_min_count", 5.0))
        shrinkage = float(policy.get("weather_shrinkage_tau", 8.0))
        clip_low, clip_high = policy.get("weather_factor_clip", [0.85, 1.15])
        strength = float(policy.get("weather_factor_strength", 1.0))
    else:
        raise ValueError(f"未知状态修正类型：{kind}")

    day_type = str(target_context.get("_day_type", "工作日"))
    reference_suffix = (
        "regular_workday_qty_mean" if day_type == "工作日" else "regular_weekend_qty_mean"
    )

    def one(prefix: str) -> tuple[float, float]:
        reference = float(meta.get(f"{prefix}{reference_suffix}", math.nan))
        if not math.isfinite(reference) or reference <= 0:
            reference = float(meta.get(f"{prefix}regular_qty_mean", math.nan))
        available: list[tuple[dict[str, float], int]] = []
        for order, (stats_key, state_key) in enumerate(candidates):
            stats = meta.get(f"{prefix}{stats_key}", {})
            entry = stats.get(state_key) if isinstance(stats, dict) else None
            if entry and float(entry.get("count", 0.0)) > 0:
                available.append((entry, order))
                if float(entry["count"]) >= minimum:
                    break
        if not available or not math.isfinite(reference) or reference <= 0:
            return 1.0, 0.0
        entry, _ = available[-1]
        statistic = str(policy.get(f"{kind}_statistic", "mean"))
        state_value = float(entry.get(statistic, entry.get("mean", math.nan)))
        count = float(entry.get("count", 0.0))
        if not math.isfinite(state_value) or state_value <= 0:
            return 1.0, 0.0
        raw = state_value / reference
        reliability = count / (count + shrinkage) if shrinkage > 0 else 1.0
        factor = 1.0 + (raw - 1.0) * reliability * strength
        return float(min(max(factor, float(clip_low)), float(clip_high))), count

    local_factor, local_count = one("l_")
    global_factor, global_count = one("g_")
    if local_count <= 0:
        return global_factor, global_count
    if global_count <= 0:
        return local_factor, local_count
    transfer_tau = float(policy.get("transfer_shrinkage_tau", 0.0))
    weight = local_count / (local_count + transfer_tau) if transfer_tau > 0 else 1.0
    return weight * local_factor + (1 - weight) * global_factor, local_count


def _group_total_forecast(
    meta: dict[str, Any],
    metric: str,
    policy: dict[str, Any],
    target_context: dict[str, Any] | None = None,
    return_components: bool = False,
) -> float | dict[str, float | bool | str]:
    window = int(policy["recent_window"])
    recent_weight = float(policy["recent_weight"])
    tau = float(policy["transfer_shrinkage_tau"])
    local_sessions = float(meta.get("l_session_count", 0.0))
    global_sessions = float(meta.get("g_session_count", 0.0))

    def blended(key: str) -> float:
        local = float(meta.get(f"l_{key}", math.nan))
        global_value = float(meta.get(f"g_{key}", math.nan))
        if local_sessions <= 0 or not math.isfinite(local):
            return global_value if math.isfinite(global_value) else 0.0
        if not math.isfinite(global_value) or tau <= 0:
            return local
        weight = local_sessions / (local_sessions + tau)
        return weight * local + (1 - weight) * global_value

    target_context = target_context or {}
    calendar_mode = str(policy.get("calendar_mode", "none"))
    target_category = str(target_context.get("_calendar_category", "普通日期"))
    allowed_categories = policy.get("calendar_apply_categories")
    calendar_enabled = calendar_mode != "none" and (
        not allowed_categories or target_category in set(map(str, allowed_categories))
    )
    use_regular_history = (
        calendar_enabled
        and (
            calendar_mode == "joint_shrinkage"
            or (calendar_mode == "special_only" and target_category != "普通日期")
        )
    )
    # A holiday may be a valid factor-training observation for another holiday,
    # but it must not depress the recent/weekday baseline of the first ordinary
    # days after that holiday.  This switch is deliberately independent of
    # calendar_apply_categories: ordinary dates receive no holiday factor, only
    # a clean ordinary-date baseline.
    if (
        bool(policy.get("exclude_special_from_regular_history", False))
        and target_category == "普通日期"
    ):
        use_regular_history = True
    key_prefix = "regular_" if use_regular_history else ""
    long_mean = blended(f"{key_prefix}{metric}_mean")
    recent_mean = blended(f"{key_prefix}w{window}_{metric}_mean")
    ewm_categories = policy.get("group_ewm_apply_categories")
    ewm_enabled = (
        str(policy.get("group_recent_estimator", "window_mean")) == "ewm"
        and (
            not ewm_categories
            or target_category in set(map(str, ewm_categories))
        )
    )
    if ewm_enabled:
        half_life = int(policy.get("group_ewm_half_life", 3))
        if half_life not in FEATURE_HALF_LIVES:
            raise ValueError(
                f"group_ewm_half_life必须是{FEATURE_HALF_LIVES}之一"
            )
        recent_mean = blended(
            f"{key_prefix}ewm{half_life}_{metric}_mean"
        )
    if use_regular_history:
        category = target_category
        day_type = str(target_context.get("_day_type", "工作日"))
        if category == "普通日期":
            weekday_mean = blended(f"regular_weekday_{metric}_mean")
        else:
            day_key = "workday" if day_type == "工作日" else "weekend"
            weekday_mean = blended(f"regular_{day_key}_{metric}_mean")
    else:
        weekday_mean = blended(f"weekday_{metric}_mean")
    count_prefix = "regular_" if use_regular_history else ""
    local_recent_count = float(meta.get(f"l_{count_prefix}w{window}_session_count", 0.0))
    global_recent_count = float(meta.get(f"g_{count_prefix}w{window}_session_count", 0.0))
    recent_count = local_recent_count if local_sessions > 0 else global_recent_count
    recent_weight = float(
        _adaptive_recent_weight(recent_weight, recent_count, policy)
    )
    base = recent_weight * recent_mean + (1 - recent_weight) * long_mean
    raw_factor = weekday_mean / long_mean if long_mean > 0 else 1.0
    low, high = policy["weekday_clip"]
    factor = min(max(raw_factor, float(low)), float(high)) ** float(policy["weekday_strength"])
    group_trend_factor = 1.0
    group_trend_strength = float(policy.get("group_trend_strength", 0.0))
    meal_strengths = policy.get("group_trend_strength_by_meal", {})
    if isinstance(meal_strengths, dict):
        meal = str(target_context.get("_meal", ""))
        if meal in meal_strengths:
            group_trend_strength = float(meal_strengths[meal])
    if group_trend_strength != 0.0 and metric == "qty":
        trend_window = int(policy.get("group_trend_reference_window", 30))
        trend_reference = blended(f"{key_prefix}w{trend_window}_{metric}_mean")
        raw_trend = recent_mean / trend_reference if trend_reference > 0 else 1.0
        trend_low, trend_high = policy.get("group_trend_clip", [0.85, 1.15])
        group_trend_factor = min(
            max(raw_trend, float(trend_low)), float(trend_high)
        ) ** group_trend_strength
    group_momentum_factor = 1.0
    momentum_strength = float(policy.get("group_momentum_strength", 0.0))
    momentum_categories = policy.get(
        "group_momentum_apply_categories", ["普通日期"]
    )
    momentum_enabled = (
        momentum_strength != 0.0
        and metric == "qty"
        and (
            not momentum_categories
            or target_category in set(map(str, momentum_categories))
        )
    )
    if momentum_enabled:
        last_total = blended(f"{key_prefix}{metric}_last")
        momentum_mode = str(policy.get("group_momentum_mode", "level"))
        if momentum_mode != "last_ratio":
            raise ValueError("冻结策略仅支持group_momentum_mode=last_ratio")
        previous_total = blended(f"{key_prefix}{metric}_previous")
        raw_momentum = (
            last_total / previous_total if previous_total > 0 else 1.0
        )
        momentum_low, momentum_high = policy.get(
            "group_momentum_clip", [0.85, 1.15]
        )
        group_momentum_factor = min(
            max(raw_momentum, float(momentum_low)), float(momentum_high)
        ) ** momentum_strength
    calendar_factor = 1.0
    calendar_sample_count = 0.0
    if calendar_enabled and metric == "qty":
        calendar_factor, calendar_sample_count = _state_adjustment(
            meta, target_context, policy, "calendar"
        )
    weather_factor = 1.0
    weather_sample_count = 0.0
    weather_mode = str(policy.get("weather_mode", "none"))
    weather_enabled = weather_mode != "none"
    if weather_mode == "rain_only":
        weather_enabled = str(target_context.get("_rain_band", "无雨")) != "无雨"
    if weather_enabled and metric == "qty":
        weather_factor, weather_sample_count = _state_adjustment(
            meta, target_context, policy, "weather"
        )
    weather_residual_strength = 1.0
    if target_category != "普通日期":
        weather_residual_strength = float(
            policy.get("holiday_weather_residual_strength", 1.0)
        )
    weather_residual_factor = 1.0 + (
        weather_factor - 1.0
    ) * weather_residual_strength
    raw_combined_factor = (
        factor
        * group_trend_factor
        * group_momentum_factor
        * calendar_factor
        * weather_residual_factor
    )
    combination_mode = str(
        policy.get("group_factor_combination_mode", "multiplicative")
    )
    final_combined_factor = raw_combined_factor
    clip_applied = False
    if combination_mode == "clipped_multiplicative":
        combined_low, combined_high = policy.get(
            "group_combined_factor_clip", [0.65, 1.45]
        )
        final_combined_factor = min(
            max(raw_combined_factor, float(combined_low)),
            float(combined_high),
        )
        clip_applied = not math.isclose(
            final_combined_factor,
            raw_combined_factor,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    elif combination_mode != "multiplicative":
        raise ValueError(
            f"未知 group_factor_combination_mode：{combination_mode}"
        )
    prediction = max(0.0, base * final_combined_factor)
    if not return_components:
        return prediction
    return {
        "base_total_qty": float(base),
        "weekday_factor": float(factor),
        "group_trend_factor": float(group_trend_factor),
        "group_momentum_factor": float(group_momentum_factor),
        "calendar_factor": float(calendar_factor),
        "weather_factor": float(weather_factor),
        "weather_residual_factor": float(weather_residual_factor),
        "raw_combined_factor": float(raw_combined_factor),
        "final_combined_factor": float(final_combined_factor),
        "factor_clip_applied": bool(clip_applied),
        "calendar_sample_count": float(calendar_sample_count),
        "weather_sample_count": float(weather_sample_count),
        "factor_combination_mode": combination_mode,
        "group_total_qty": float(prediction),
    }


def _legacy_capacity(meta: dict[str, Any], policy: dict[str, Any]) -> int:
    local_sessions = float(meta.get("l_session_count", 0.0))
    local = float(meta.get("l_capacity_median", math.nan))
    global_value = float(meta.get("g_capacity_median", math.nan))
    if local_sessions <= 0 or not math.isfinite(local):
        value = global_value
    else:
        tau = float(policy["transfer_shrinkage_tau"])
        if math.isfinite(global_value) and tau > 0:
            weight = local_sessions / (local_sessions + tau)
            value = weight * local + (1 - weight) * global_value
        else:
            value = local
    if not math.isfinite(value):
        return 0
    return max(1, int(round(value * float(policy["capacity_scale"]))))


def _capacity_forecast(
    meta: dict[str, Any],
    policy: dict[str, Any],
    target_context: dict[str, Any],
    pred_total_qty: float,
) -> dict[str, float | int | str]:
    """Predict menu size without reusing the total-demand factor stack."""

    mode = str(policy.get("capacity_mode", "weekday_median"))
    legacy = _legacy_capacity(meta, policy)
    if mode == "weekday_median":
        return {
            "planned_capacity": int(legacy),
            "capacity_state_estimate": float(legacy),
            "capacity_demand_estimate": float(legacy),
            "capacity_qty_per_dish": (
                float(pred_total_qty) / legacy if legacy > 0 else 0.0
            ),
            "capacity_state_sample_count": 0.0,
            "capacity_weather_factor": 1.0,
            "capacity_mode": mode,
        }
    if mode != "calendar_hierarchical":
        raise ValueError(f"未知 capacity_mode：{mode}")

    state = str(target_context.get("_calendar_state", "普通工作日"))
    category = str(target_context.get("_calendar_category", "普通日期"))
    day_type = str(target_context.get("_day_type", "工作日"))
    day_key = "workday" if day_type == "工作日" else "weekend"
    shrinkage = float(policy.get("capacity_state_shrinkage_tau", 5.0))
    transfer_tau = float(policy.get("transfer_shrinkage_tau", 5.0))

    def estimate(prefix: str) -> tuple[float, float, float]:
        parent_capacity = float(
            meta.get(f"{prefix}regular_{day_key}_capacity_median", math.nan)
        )
        parent_qpd = float(
            meta.get(
                f"{prefix}regular_{day_key}_qty_per_dish_median", math.nan
            )
        )
        if not math.isfinite(parent_capacity) or parent_capacity <= 0:
            parent_capacity = float(
                meta.get(f"{prefix}capacity_median", math.nan)
            )
        if not math.isfinite(parent_qpd) or parent_qpd <= 0:
            parent_qpd = float(
                meta.get(f"{prefix}qty_per_dish_median", math.nan)
            )
        current_capacity = parent_capacity
        current_qpd = parent_qpd
        effective_count = 0.0
        candidates: list[tuple[str, str]] = []
        if category not in {"普通日期", "调休工作日"}:
            candidates.append(("special_calendar_stats", "special"))
        if category != "普通日期":
            candidates.append(("calendar_category_stats", category))
        candidates.append(("calendar_state_stats", state))
        for stats_key, state_key in candidates:
            stats = meta.get(f"{prefix}{stats_key}", {})
            entry = stats.get(state_key) if isinstance(stats, dict) else None
            if not entry:
                continue
            count = float(entry.get("count", 0.0))
            state_capacity = float(
                entry.get("dish_count_median", math.nan)
            )
            state_qpd = float(entry.get("qty_per_dish_median", math.nan))
            if count <= 0:
                continue
            reliability = (
                count / (count + shrinkage) if shrinkage > 0 else 1.0
            )
            if math.isfinite(state_capacity) and state_capacity > 0:
                if math.isfinite(current_capacity) and current_capacity > 0:
                    current_capacity = (
                        reliability * state_capacity
                        + (1 - reliability) * current_capacity
                    )
                else:
                    current_capacity = state_capacity
            if math.isfinite(state_qpd) and state_qpd > 0:
                if math.isfinite(current_qpd) and current_qpd > 0:
                    current_qpd = (
                        reliability * state_qpd
                        + (1 - reliability) * current_qpd
                    )
                else:
                    current_qpd = state_qpd
            effective_count = max(effective_count, count)
        return current_capacity, current_qpd, effective_count

    local_capacity, local_qpd, local_count = estimate("l_")
    global_capacity, global_qpd, global_count = estimate("g_")
    if local_count <= 0 or not math.isfinite(local_capacity):
        state_capacity = global_capacity
        qpd = global_qpd
        state_count = global_count
    elif global_count <= 0 or not math.isfinite(global_capacity):
        state_capacity = local_capacity
        qpd = local_qpd
        state_count = local_count
    else:
        local_weight = (
            local_count / (local_count + transfer_tau)
            if transfer_tau > 0
            else 1.0
        )
        state_capacity = (
            local_weight * local_capacity
            + (1 - local_weight) * global_capacity
        )
        if math.isfinite(local_qpd) and math.isfinite(global_qpd):
            qpd = local_weight * local_qpd + (1 - local_weight) * global_qpd
        else:
            qpd = local_qpd if math.isfinite(local_qpd) else global_qpd
        state_count = local_count

    if not math.isfinite(state_capacity) or state_capacity <= 0:
        state_capacity = float(max(legacy, 1))
    if not math.isfinite(qpd) or qpd <= 0:
        qpd = pred_total_qty / state_capacity if state_capacity > 0 else 1.0
    demand_capacity = (
        float(pred_total_qty) / qpd if pred_total_qty > 0 and qpd > 0 else 1.0
    )
    demand_weight = float(policy.get("capacity_demand_weight", 0.5))
    combined = (
        (1 - demand_weight) * state_capacity
        + demand_weight * demand_capacity
    )
    hierarchical_strength = float(
        policy.get("capacity_hierarchical_strength", 1.0)
    )
    combined = float(legacy) + hierarchical_strength * (
        combined - float(legacy)
    )

    weather_factor = 1.0
    weather_strength = float(policy.get("capacity_weather_strength", 0.0))
    if weather_strength != 0.0:
        rain_state = str(target_context.get("_rain_band", "无雨"))
        stats = meta.get("l_rain_stats", {})
        entry = stats.get(rain_state) if isinstance(stats, dict) else None
        if not entry:
            stats = meta.get("g_rain_stats", {})
            entry = stats.get(rain_state) if isinstance(stats, dict) else None
        if entry:
            count = float(entry.get("count", 0.0))
            observed = float(entry.get("dish_count_median", math.nan))
            if count > 0 and math.isfinite(observed) and state_capacity > 0:
                reliability = count / (
                    count + float(policy.get("capacity_weather_shrinkage_tau", 10.0))
                )
                raw = observed / state_capacity
                weather_factor = 1.0 + (
                    raw - 1.0
                ) * reliability * weather_strength
                weather_low, weather_high = policy.get(
                    "capacity_weather_factor_clip", [0.95, 1.05]
                )
                weather_factor = min(
                    max(weather_factor, float(weather_low)),
                    float(weather_high),
                )
    combined *= weather_factor
    min_capacity = int(policy.get("capacity_min", 1))
    max_capacity = int(policy.get("capacity_max", 10000))
    planned = min(
        max(int(round(combined * float(policy.get("capacity_scale", 1.0)))), min_capacity),
        max_capacity,
    )
    return {
        "planned_capacity": int(planned),
        "capacity_state_estimate": float(state_capacity),
        "capacity_demand_estimate": float(demand_capacity),
        "capacity_qty_per_dish": float(qpd),
        "capacity_state_sample_count": float(state_count),
        "capacity_weather_factor": float(weather_factor),
        "capacity_mode": mode,
    }


def _attach_min_batch_estimates(
    frame: pd.DataFrame,
    policy: dict[str, Any],
    local_count: pd.Series,
) -> pd.DataFrame:
    result = frame.copy()
    mode = str(policy.get("min_batch_mode", "none"))
    if mode == "none":
        result["_min_batch_qty"] = 0.0
        result["_min_batch_source"] = "disabled"
        return result
    if mode not in {"estimate_only", "dish_q25_prune"}:
        raise ValueError(f"未知 min_batch_mode：{mode}")
    tau = float(policy.get("transfer_shrinkage_tau", 5.0))
    positive_count = _effective_series(
        result, "qty_positive_count", tau, local_count
    ).fillna(0.0)
    dish_batch = _effective_series(
        result, "qty_min_batch", tau, local_count
    )
    minimum_count = float(policy.get("min_batch_dish_min_count", 3.0))
    reliable = positive_count.ge(minimum_count) & dish_batch.gt(0)
    reliable_values = dish_batch.where(reliable)
    category_fallback = reliable_values.groupby(
        result["菜品类别"].astype(str)
    ).transform("median")
    meal_fallback = float(reliable_values.median())
    if not math.isfinite(meal_fallback) or meal_fallback <= 0:
        all_values = dish_batch.loc[dish_batch.gt(0)]
        meal_fallback = float(all_values.median()) if not all_values.empty else 0.0
    estimate = dish_batch.where(reliable, category_fallback)
    estimate = estimate.fillna(meal_fallback).clip(lower=0.0)
    scale = float(policy.get("min_batch_scale", 1.0))
    result["_min_batch_qty"] = estimate * scale
    result["_min_batch_source"] = np.where(
        reliable,
        "dish",
        np.where(category_fallback.notna(), "category_meal", "meal_global"),
    )
    return result


def _allocate_and_prune_min_batch(
    selected_frame: pd.DataFrame,
    target_total: float,
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    result = selected_frame.copy()
    removed = 0
    uncertainty_power = float(
        policy.get("allocation_uncertainty_power", 0.0)
    )
    min_batch_mode = str(policy.get("min_batch_mode", "none"))

    def allocate(frame: pd.DataFrame) -> pd.DataFrame:
        allocated = frame.copy()
        weights = allocated["_allocation_weight"].clip(lower=0.0)
        if uncertainty_power != 0:
            weights = weights * allocated["_qty_stability"].clip(
                lower=1e-12
            ).pow(uncertainty_power)
        weight_total = float(weights.sum())
        if target_total > 0 and weight_total > 0:
            allocated["pred_qty"] = target_total * weights / weight_total
        else:
            allocated["pred_qty"] = 0.0
        return allocated

    result = allocate(result)
    if min_batch_mode == "dish_q25_prune":
        minimum_capacity = int(policy.get("capacity_min", 1))
        while len(result) > minimum_capacity:
            below = result["pred_qty"] + 1e-9 < result["_min_batch_qty"]
            if not below.any():
                break
            candidates = result.loc[below].copy()
            candidates["_batch_ratio"] = candidates["pred_qty"] / candidates[
                "_min_batch_qty"
            ].replace(0, np.nan)
            remove_index = candidates.sort_values(
                ["_batch_ratio", "score", "菜品名称_库"],
                ascending=[True, True, False],
            ).index[0]
            result = result.drop(index=remove_index)
            removed += 1
            result = allocate(result)
    elif min_batch_mode not in {"none", "estimate_only"}:
        raise ValueError(f"未知 min_batch_mode：{min_batch_mode}")
    return result, removed


def _recommendation_columns() -> list[str]:
    return [
        "date", "食堂名称", "餐次", "菜品名称", "pred_qty", "pred_orders", "pred_rev",
        "score", "pred_total_qty", "menu_capacity", "planned_menu_capacity",
        "expected_orders_per_qty", "expected_rev_per_qty", "min_batch_qty",
        "min_batch_source", "base_total_qty", "weekday_total_factor",
        "group_trend_factor", "group_momentum_factor", "calendar_factor",
        "weather_factor", "weather_residual_factor", "raw_combined_factor",
        "final_combined_factor", "factor_clip_applied", "calendar_sample_count",
        "weather_sample_count", "capacity_state_estimate",
        "capacity_demand_estimate", "capacity_qty_per_dish",
        "capacity_state_sample_count", "capacity_weather_factor", "capacity_mode",
        "group_total_target_mode", "shadow_calibration_capacity",
        "shadow_bottom_up_total",
        "calendar_category", "calendar_state", "day_type", "weather_state",
        "temperature_band", "rain_band",
        "主要食材", "食材大类", "制作方法", "口味", "菜品类别", "相似菜分组",
        "主食标记", "inferred_main_ingredient", "ingredient_family", "attribute_source",
        "inference_confidence", "菜品库属性", "推断属性", "library_matched", "_main_hard",
        "_family_hard", "_method_hard", "_flavor_hard", "_similarity_hard", "reason",
    ]


def make_recommendations(
    contexts: list[dict[str, Any]],
    policy: dict[str, Any],
    constraints: dict[str, Any],
    selection_audit: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    menu_history: dict[tuple[str, str], list[tuple[pd.Timestamp, set[str]]]] = {}
    for context in contexts:
        target_date = pd.Timestamp(context["date"])
        if context["max_history_date"] is not None and context["max_history_date"] >= target_date:
            raise AssertionError("推荐时发现未来信息")
        frame = context["features"].copy()
        meta = context["meta"]
        target_context = context.get("target_context", {})
        total_components = _group_total_forecast(
            meta,
            "qty",
            policy,
            target_context,
            return_components=True,
        )
        if not isinstance(total_components, dict):
            raise AssertionError("餐次总量分解输出无效")
        group_total = float(total_components["group_total_qty"])
        capacity_components = _capacity_forecast(
            meta, policy, target_context, group_total
        )
        target_capacity = int(capacity_components["planned_capacity"])
        history_key = (str(context["食堂名称"]), str(context["餐次"]))
        prior_history = menu_history.get(history_key, [])
        prior_norms = prior_history[-1][1] if prior_history else set()
        recent7_counts: Counter = Counter(
            dish
            for old_date, dishes in prior_history
            if old_date >= target_date - pd.Timedelta(days=7)
            for dish in dishes
        )
        recent14_counts: Counter = Counter(
            dish
            for old_date, dishes in prior_history
            if old_date >= target_date - pd.Timedelta(days=14)
            for dish in dishes
        )
        local_count = _series(frame, "l_offer_count", 0.0).fillna(0.0)
        global_count = _series(frame, "g_offer_count", 0.0).fillna(0.0)
        local_weekday_count = _series(frame, "l_weekday_count", 0.0).fillna(0.0)
        global_weekday_count = _series(frame, "g_weekday_count", 0.0).fillna(0.0)
        local_sessions = float(meta.get("l_session_count", 0.0))
        global_sessions = float(meta.get("g_session_count", 0.0))
        minimum = int(policy["candidate_min_appearances"])
        local_eligible = (local_count >= minimum) | (local_weekday_count > 0)
        global_eligible = (global_count >= minimum) | (global_weekday_count > 0)
        warmup = local_sessions < float(policy["candidate_warmup_sessions"])
        staple = frame["主食标记"].fillna(False).astype(bool)
        eligible = local_eligible | staple
        if warmup:
            eligible |= global_eligible
        frame = frame.loc[eligible].copy()
        if frame.empty:
            menu_history.setdefault(history_key, []).append((target_date, set()))
            if selection_audit is not None:
                selection_audit.append(
                    {
                        "date": target_date,
                        "食堂名称": history_key[0],
                        "餐次": history_key[1],
                        "capacity": target_capacity,
                        "planned_capacity": target_capacity,
                        "selected_count": 0,
                        "candidate_pool": set(),
                        "hard_excluded": set(),
                        "below_min_batch_count": 0,
                        "min_batch_removed_count": 0,
                        **total_components,
                        **capacity_components,
                    }
                )
            continue

        if (
            bool(policy.get("item_exclude_special_from_regular_history", False))
            and str(
                context.get("target_context", {}).get(
                    "_calendar_category", "普通日期"
                )
            )
            == "普通日期"
        ):
            # Candidate eligibility and recency remain based on all history.
            # Only item quantity/value statistics switch to their ordinary-date
            # counterparts, with per-dish fallback when no regular observation
            # exists yet.
            for regular_prefix, target_prefix in [("lr_", "l_"), ("gr_", "g_")]:
                for regular_column in [
                    column
                    for column in frame.columns
                    if column.startswith(regular_prefix)
                ]:
                    suffix = regular_column[len(regular_prefix):]
                    if suffix in {"offer_count", "last_date", "recent7_count"}:
                        continue
                    target_column = f"{target_prefix}{suffix}"
                    if target_column in frame:
                        frame[target_column] = frame[regular_column].where(
                            frame[regular_column].notna(), frame[target_column]
                        )
            # Consolidate the columns replaced above before adding prediction
            # diagnostics; this avoids pandas fragmentation without changing
            # values or index order.
            frame = frame.copy()

        local_count = _series(frame, "l_offer_count", 0.0).fillna(0.0)
        global_count = _series(frame, "g_offer_count", 0.0).fillna(0.0)
        tau = float(policy["transfer_shrinkage_tau"])
        window = int(policy["recent_window"])
        nominal_recent_weight = float(policy["recent_weight"])
        effective: dict[str, pd.Series] = {}
        recent_observation_count = _effective_series(
            frame, f"w{window}_count", tau, local_count
        ).fillna(0.0)
        adaptive_recent_weight = _adaptive_recent_weight(
            nominal_recent_weight, recent_observation_count, policy
        )
        for metric in ["qty", "orders", "rev"]:
            long_mean = _effective_series(frame, f"{metric}_mean", tau, local_count).fillna(0.0)
            recent_mean = _effective_series(
                frame, f"w{window}_{metric}_mean", tau, local_count
            ).fillna(long_mean)
            weekday_mean = _effective_series(
                frame, f"weekday_{metric}_mean", tau, local_count
            ).fillna(long_mean)
            weekday_count = _effective_series(
                frame, "weekday_count", tau, local_count
            ).fillna(0.0)
            base = (
                adaptive_recent_weight * recent_mean
                + (1 - adaptive_recent_weight) * long_mean
            )
            factor = _weekday_factor(weekday_mean, long_mean, weekday_count, policy)
            short_mean = _effective_series(frame, f"w14_{metric}_mean", tau, local_count).fillna(long_mean)
            trend_ratio = (short_mean / long_mean.replace(0, np.nan)).fillna(1.0)
            trend_low, trend_high = policy["trend_clip"]
            trend_factor = trend_ratio.clip(float(trend_low), float(trend_high)).pow(
                float(policy["trend_strength"])
            )
            effective[f"long_{metric}"] = long_mean
            effective[f"recent_{metric}"] = recent_mean
            effective[f"weekday_factor_{metric}"] = factor
            effective[f"pred_{metric}"] = (base * factor * trend_factor).clip(lower=0)

        item_quantity_mode = str(policy.get("item_quantity_mode", ""))
        if item_quantity_mode != "robust_conditional":
            raise ValueError("冻结策略仅支持item_quantity_mode=robust_conditional")

        long_robust = _effective_series(
            frame, "qty_log_mean", tau, local_count
        ).fillna(effective["long_qty"])
        recent_robust = _effective_series(
            frame, f"w{window}_qty_log_mean", tau, local_count
        ).fillna(long_robust)
        weekday_robust = _effective_series(
            frame, "weekday_qty_log_mean", tau, local_count
        ).fillna(long_robust)
        robust_factor = _weekday_factor(
            weekday_robust,
            long_robust,
            _effective_series(
                frame, "weekday_count", tau, local_count
            ).fillna(0.0),
            policy,
        )
        short_robust = _effective_series(
            frame, "w14_qty_log_mean", tau, local_count
        ).fillna(long_robust)
        robust_trend = (
            short_robust / long_robust.replace(0, np.nan)
        ).fillna(1.0)
        trend_low, trend_high = policy["trend_clip"]
        robust_trend = robust_trend.clip(
            float(trend_low), float(trend_high)
        ).pow(float(policy["trend_strength"]))
        robust_quantity = (
            adaptive_recent_weight * recent_robust
            + (1 - adaptive_recent_weight) * long_robust
        ) * robust_factor * robust_trend
        robust_quantity = robust_quantity.clip(lower=0.0)

        ratio_weight = float(policy.get("unit_ratio_recent_weight", 0.5))
        long_orders_ratio = _effective_series(
            frame, "orders_per_qty_mean", tau, local_count
        ).fillna(0.0)
        recent_orders_ratio = _effective_series(
            frame, f"w{window}_orders_per_qty_mean", tau, local_count
        ).fillna(long_orders_ratio)
        long_revenue_ratio = _effective_series(
            frame, "rev_per_qty_mean", tau, local_count
        ).fillna(0.0)
        recent_revenue_ratio = _effective_series(
            frame, f"w{window}_rev_per_qty_mean", tau, local_count
        ).fillna(long_revenue_ratio)
        unit_orders_ratio = (
            ratio_weight * recent_orders_ratio
            + (1 - ratio_weight) * long_orders_ratio
        ).clip(lower=0.0)
        unit_revenue_ratio = (
            ratio_weight * recent_revenue_ratio
            + (1 - ratio_weight) * long_revenue_ratio
        ).clip(lower=0.0)
        effective["pred_qty"] = robust_quantity
        effective["pred_orders"] = robust_quantity * unit_orders_ratio
        effective["pred_rev"] = robust_quantity * unit_revenue_ratio
        effective["unit_orders_ratio"] = unit_orders_ratio
        effective["unit_revenue_ratio"] = unit_revenue_ratio
        offer_probability = _effective_probability(
            local_count, global_count, local_sessions, global_sessions, tau
        )
        score_revenue = effective["pred_rev"] * offer_probability.pow(
            float(policy["revenue_offer_power"])
        )
        rev_std = _effective_series(frame, "rev_std", tau, local_count).fillna(0.0)
        revenue_cv = (rev_std / effective["long_rev"].replace(0, np.nan)).fillna(0.0).clip(lower=0)
        stability = 1.0 / (1.0 + revenue_cv)
        qty_std = _effective_series(frame, "qty_std", tau, local_count).fillna(0.0)
        qty_cv = (
            qty_std / effective["long_qty"].replace(0, np.nan)
        ).fillna(0.0).clip(lower=0.0)
        qty_stability = 1.0 / (1.0 + qty_cv)
        recent7_count = _effective_series(frame, "recent7_count", tau, local_count).fillna(0.0)
        recent_sessions = (
            float(meta.get("l_recent7_session_count", 0.0))
            if local_sessions > 0
            else float(meta.get("g_recent7_session_count", 0.0))
        )
        recent_rate = recent7_count / max(recent_sessions, 1.0)
        recent_offer_tau = float(policy.get("recent_offer_shrinkage_tau", 3.0))
        recent_offer_probability = (
            recent7_count + recent_offer_tau * offer_probability
        ) / max(recent_sessions + recent_offer_tau, 1.0)
        recent_offer_blend = float(policy.get("score_recent_offer_blend", 0.0))
        score_offer_probability = (
            (1 - recent_offer_blend) * offer_probability
            + recent_offer_blend * recent_offer_probability
        ).clip(lower=0.0, upper=1.0)
        repeat_penalty = 1.0 / (
            1.0 + float(policy.get("repeat_weight", 0.0)) * recent_rate
        )
        generated_repeat_load = frame["_dish_norm"].astype(str).map(
            lambda dish: (
                float(policy.get("previous_business_day_repeat_weight", 0.0))
                * int(dish in prior_norms)
                + float(policy.get("recommendation_repeat_7d_weight", 0.0))
                * recent7_counts[dish]
                + float(policy.get("recommendation_repeat_14d_weight", 0.0))
                * recent14_counts[dish]
            )
        )
        repeat_penalty_scale = float(
            policy.get("recommendation_repeat_penalty_scale", 1.0)
        )
        recommendation_repeat_penalty = 1.0 / (
            1.0 + repeat_penalty_scale * generated_repeat_load
        )
        score = (
            score_revenue
            * stability.pow(float(policy["stability_power"]))
            * score_offer_probability.pow(float(policy["score_offer_power"]))
            * repeat_penalty
            * recommendation_repeat_penalty
        )
        confidence_penalty = frame["inference_confidence"].astype(str).map(
            lambda value: float(constraints["medium_confidence_penalty"])
            if value == "medium" else 1.0
        )
        frame["pred_qty"] = effective["pred_qty"]
        frame["pred_orders"] = effective["pred_orders"]
        frame["pred_rev"] = effective["pred_rev"]
        frame["score"] = score * confidence_penalty
        frame["_offer_probability"] = offer_probability
        frame["_stability"] = stability
        frame["_qty_stability"] = qty_stability
        frame["_recent7_count"] = recent7_count
        frame["_recent_offer_probability"] = recent_offer_probability
        frame["_adaptive_recent_weight"] = adaptive_recent_weight
        frame["_generated_repeat_load"] = generated_repeat_load
        frame["_long_qty"] = effective["long_qty"]
        frame["_recent_qty"] = effective["recent_qty"]
        frame["_weekday_factor"] = effective["weekday_factor_qty"]
        frame["_allocation_weight"] = effective["pred_qty"]
        frame["_unit_orders_ratio"] = effective["unit_orders_ratio"]
        frame["_unit_revenue_ratio"] = effective["unit_revenue_ratio"]
        frame = _attach_min_batch_estimates(frame, policy, local_count)

        local_last = pd.to_datetime(frame.get("l_last_date"), errors="coerce")
        global_last = pd.to_datetime(frame.get("g_last_date"), errors="coerce")
        last_date = local_last.where(local_last.notna(), global_last)
        days_inactive = (target_date - last_date).dt.days
        active = days_inactive.le(int(policy["max_inactive_days"])) | staple.loc[frame.index]
        frame = frame.loc[active & frame["score"].gt(0)].copy()
        frame = frame.sort_values(["score", "菜品名称_库"], ascending=[False, True])
        candidate_pool = set(frame["_dish_norm"].astype(str))
        counters = {
            key: Counter()
            for key in [
                "similarity",
                "main_all",
                "family_all",
                "method_all",
                "flavor_all",
            ]
        }
        selected: list[pd.Series] = []
        selected_norms: set[str] = set()
        hard_excluded: set[str] = set()
        total_target_mode = str(
            policy.get("group_total_target_mode", "calibrated_bottom_up")
        )
        shadow_calibration_capacity = (
            _legacy_capacity(meta, policy)
            if total_target_mode == "independent_shadow_calibrated"
            else target_capacity
        )
        selection_capacity = max(
            target_capacity, shadow_calibration_capacity
        )
        constraint_mode = str(policy.get("constraint_mode", ""))
        if constraint_mode != "penalized_diversity_greedy":
            raise ValueError(
                "冻结策略仅支持constraint_mode=penalized_diversity_greedy"
            )
        if constraint_mode == "penalized_diversity_greedy":
            penalty_specs = [
                (
                    "主要食材",
                    "main_all",
                    int(
                        constraints.get(
                            "soft_max_same_main_ingredient",
                            constraints["max_same_main_ingredient"],
                        )
                    ),
                    float(policy.get("selection_same_main_penalty", 0.85)),
                ),
                (
                    "食材大类",
                    "family_all",
                    int(constraints["max_same_ingredient_family"]),
                    float(policy.get("selection_same_family_penalty", 0.85)),
                ),
                (
                    "制作方法",
                    "method_all",
                    int(constraints["max_same_cooking_method"]),
                    float(policy.get("selection_same_method_penalty", 0.90)),
                ),
                (
                    "口味",
                    "flavor_all",
                    int(constraints["max_same_flavor"]),
                    float(policy.get("selection_same_flavor_penalty", 0.95)),
                ),
            ]
            while len(selected) < selection_capacity:
                available = frame.loc[
                    ~frame["_dish_norm"].astype(str).isin(selected_norms)
                ].copy()
                if available.empty:
                    break
                similarity_values = available["相似菜分组"].astype(str)
                similarity_known = similarity_values.map(_is_known)
                hard_conflict = (
                    available["_similarity_hard"].fillna(False).astype(bool)
                    & similarity_known
                    & similarity_values.map(counters["similarity"]).ge(
                        int(constraints["max_same_similarity_group"])
                    )
                )
                hard_excluded.update(
                    available.loc[hard_conflict, "_dish_norm"].astype(str)
                )
                available = available.loc[~hard_conflict].copy()
                if available.empty:
                    break
                available["_dynamic_utility"] = available["score"].astype(float)
                for value_column, counter_key, limit, penalty in penalty_specs:
                    values = available[value_column].astype(str)
                    counts = values.map(counters[counter_key]).astype(float)
                    excess = (counts + 1.0 - float(limit)).clip(lower=0.0)
                    excess = excess.where(values.map(_is_known), 0.0)
                    available["_dynamic_utility"] *= np.power(penalty, excess)
                chosen = available.sort_values(
                    ["_dynamic_utility", "菜品名称_库"],
                    ascending=[False, True],
                ).iloc[0]
                selected.append(chosen)
                selected_norms.add(str(chosen["_dish_norm"]))
                _add_all_counters(chosen, counters)
        else:
            raise ValueError(f"未知 constraint_mode：{constraint_mode}")
        shadow_bottom_up_total = float(
            sum(
                float(row.get("pred_qty", 0.0))
                for row in selected[:shadow_calibration_capacity]
            )
        )
        selected = selected[:target_capacity]
        selected_norms = {
            str(row["_dish_norm"]) for row in selected
        }
        if not selected:
            menu_history.setdefault(history_key, []).append((target_date, set()))
            if selection_audit is not None:
                selection_audit.append(
                    {
                        "date": target_date,
                        "食堂名称": history_key[0],
                        "餐次": history_key[1],
                        "capacity": target_capacity,
                        "planned_capacity": target_capacity,
                        "selected_count": 0,
                        "candidate_pool": candidate_pool,
                        "hard_excluded": hard_excluded,
                        "below_min_batch_count": 0,
                        "min_batch_removed_count": 0,
                        "group_total_target_mode": total_target_mode,
                        "shadow_calibration_capacity": shadow_calibration_capacity,
                        "shadow_bottom_up_total": shadow_bottom_up_total,
                        **total_components,
                        **capacity_components,
                    }
                )
            continue

        selected_frame = pd.DataFrame(selected).copy()
        selected_frame["expected_orders_per_qty"] = selected_frame[
            "_unit_orders_ratio"
        ].fillna(0.0)
        selected_frame["expected_rev_per_qty"] = selected_frame[
            "_unit_revenue_ratio"
        ].fillna(0.0)
        if not bool(policy.get("group_total_calibration", False)):
            raise ValueError("冻结策略要求启用餐次总量校准")
        bottom_up_total = float(selected_frame["pred_qty"].sum())
        if total_target_mode == "independent_group_total":
            target_total = max(group_total, 0.0)
        elif total_target_mode == "independent_shadow_calibrated":
            strength = float(
                policy.get("group_calibration_strength", 1.0)
            )
            day_type_strengths = policy.get(
                "group_calibration_strength_by_day_type", {}
            )
            if isinstance(day_type_strengths, dict):
                day_type = str(
                    target_context.get("_day_type", "工作日")
                )
                if day_type in day_type_strengths:
                    strength = float(day_type_strengths[day_type])
            if shadow_bottom_up_total > 0 and group_total > 0:
                target_total = shadow_bottom_up_total * (
                    group_total / shadow_bottom_up_total
                ) ** strength
            else:
                target_total = max(group_total, 0.0)
        elif total_target_mode == "calibrated_bottom_up":
            strength = float(policy.get("group_calibration_strength", 1.0))
            day_type_strengths = policy.get(
                "group_calibration_strength_by_day_type", {}
            )
            if isinstance(day_type_strengths, dict):
                day_type = str(target_context.get("_day_type", "工作日"))
                if day_type in day_type_strengths:
                    strength = float(day_type_strengths[day_type])
            if bottom_up_total > 0 and group_total > 0:
                target_total = bottom_up_total * (
                    group_total / bottom_up_total
                ) ** strength
            else:
                target_total = max(group_total, 0.0)
        else:
            raise ValueError(f"未知 group_total_target_mode：{total_target_mode}")

        selected_frame, min_batch_removed_count = _allocate_and_prune_min_batch(
            selected_frame, target_total, policy
        )
        below_min_batch_count = int(
            (
                selected_frame["pred_qty"] + 1e-9
                < selected_frame["_min_batch_qty"]
            )
            .sum()
        )
        selected_frame["pred_orders"] = (
            selected_frame["pred_qty"]
            * selected_frame["expected_orders_per_qty"]
        )
        selected_frame["pred_rev"] = (
            selected_frame["pred_qty"]
            * selected_frame["expected_rev_per_qty"]
        )
        selected_frame["pred_total_qty"] = float(target_total)
        selected_norms = set(selected_frame["_dish_norm"].astype(str))
        final_capacity = int(len(selected_frame))
        menu_history.setdefault(history_key, []).append(
            (target_date, selected_norms)
        )
        if selection_audit is not None:
            selection_audit.append(
                {
                    "date": target_date,
                    "食堂名称": history_key[0],
                    "餐次": history_key[1],
                    "capacity": target_capacity,
                    "planned_capacity": target_capacity,
                    "selected_count": final_capacity,
                    "candidate_pool": candidate_pool,
                    "hard_excluded": hard_excluded - selected_norms,
                    "below_min_batch_count": below_min_batch_count,
                    "min_batch_removed_count": min_batch_removed_count,
                    "group_total_target_mode": total_target_mode,
                    "shadow_calibration_capacity": shadow_calibration_capacity,
                    "shadow_bottom_up_total": shadow_bottom_up_total,
                    **total_components,
                    **capacity_components,
                }
            )

        for _, row in selected_frame.iterrows():
            library_payload = json.dumps(
                {"菜品类别": row["菜品类别"], "口味": row["口味"], "主食标记": bool(row["主食标记"])},
                ensure_ascii=False,
            )
            inferred_payload = json.dumps(
                {
                    "inferred_main_ingredient": row["inferred_main_ingredient"],
                    "ingredient_family": row["ingredient_family"],
                    "制作方法": row["制作方法"],
                    "相似菜分组": row["相似菜分组"],
                },
                ensure_ascii=False,
            )
            reason = (
                f"近{window}天均销量{row['_recent_qty']:.2f}；长期均销量{row['_long_qty']:.2f}；"
                f"自适应近期权重{row['_adaptive_recent_weight']:.2f}；"
                f"同星期因子{row['_weekday_factor']:.2f}；历史出菜概率{row['_offer_probability']:.2f}；"
                f"收缩后近7天出菜概率{row['_recent_offer_probability']:.2f}；"
                f"销售额稳定性{row['_stability']:.2f}；近7天出菜{int(round(row['_recent7_count']))}次；"
                f"推荐历史重复负荷{row['_generated_repeat_load']:.2f}；"
                f"餐次基础量{total_components['base_total_qty']:.2f}、"
                f"组合因子{total_components['final_combined_factor']:.3f}；"
                f"计划/最终菜品数{target_capacity}/{final_capacity}；"
                f"影子校准菜品数{shadow_calibration_capacity}；"
                f"最小备餐量{row['_min_batch_qty']:.2f}；"
                f"类别{row['菜品类别']}、口味{row['口味']}；通过确定性搭配约束"
            )
            rows.append(
                {
                    "date": target_date,
                    "食堂名称": context["食堂名称"],
                    "餐次": context["餐次"],
                    "菜品名称": row["菜品名称_库"],
                    "pred_qty": float(row["pred_qty"]),
                    "pred_orders": float(row["pred_orders"]),
                    "pred_rev": float(row["pred_rev"]),
                    "score": float(row["score"]),
                    "pred_total_qty": float(row["pred_total_qty"]),
                    "menu_capacity": int(final_capacity),
                    "planned_menu_capacity": int(target_capacity),
                    "expected_orders_per_qty": float(row["expected_orders_per_qty"]),
                    "expected_rev_per_qty": float(row["expected_rev_per_qty"]),
                    "min_batch_qty": float(row["_min_batch_qty"]),
                    "min_batch_source": str(row["_min_batch_source"]),
                    "base_total_qty": float(total_components["base_total_qty"]),
                    "weekday_total_factor": float(total_components["weekday_factor"]),
                    "group_trend_factor": float(total_components["group_trend_factor"]),
                    "group_momentum_factor": float(total_components["group_momentum_factor"]),
                    "calendar_factor": float(total_components["calendar_factor"]),
                    "weather_factor": float(total_components["weather_factor"]),
                    "weather_residual_factor": float(total_components["weather_residual_factor"]),
                    "raw_combined_factor": float(total_components["raw_combined_factor"]),
                    "final_combined_factor": float(total_components["final_combined_factor"]),
                    "factor_clip_applied": bool(total_components["factor_clip_applied"]),
                    "calendar_sample_count": float(total_components["calendar_sample_count"]),
                    "weather_sample_count": float(total_components["weather_sample_count"]),
                    "capacity_state_estimate": float(capacity_components["capacity_state_estimate"]),
                    "capacity_demand_estimate": float(capacity_components["capacity_demand_estimate"]),
                    "capacity_qty_per_dish": float(capacity_components["capacity_qty_per_dish"]),
                    "capacity_state_sample_count": float(capacity_components["capacity_state_sample_count"]),
                    "capacity_weather_factor": float(capacity_components["capacity_weather_factor"]),
                    "capacity_mode": str(capacity_components["capacity_mode"]),
                    "group_total_target_mode": total_target_mode,
                    "shadow_calibration_capacity": int(shadow_calibration_capacity),
                    "shadow_bottom_up_total": float(shadow_bottom_up_total),
                    "calendar_category": str(target_context.get("_calendar_category", "普通日期")),
                    "calendar_state": str(target_context.get("_calendar_state", "unknown")),
                    "day_type": str(target_context.get("_day_type", "unknown")),
                    "weather_state": str(target_context.get("_weather_state", "unknown")),
                    "temperature_band": str(target_context.get("_temperature_band", "unknown")),
                    "rain_band": str(target_context.get("_rain_band", "unknown")),
                    "主要食材": row["主要食材"],
                    "食材大类": row["食材大类"],
                    "制作方法": row["制作方法"],
                    "口味": row["口味"],
                    "菜品类别": row["菜品类别"],
                    "相似菜分组": row["相似菜分组"],
                    "主食标记": bool(row["主食标记"]),
                    "inferred_main_ingredient": row["inferred_main_ingredient"],
                    "ingredient_family": row["ingredient_family"],
                    "attribute_source": row["attribute_source"],
                    "inference_confidence": row["inference_confidence"],
                    "菜品库属性": library_payload,
                    "推断属性": inferred_payload,
                    "library_matched": bool(row["library_matched"]),
                    "_main_hard": False,
                    "_family_hard": False,
                    "_method_hard": False,
                    "_flavor_hard": False,
                    "_similarity_hard": bool(row["_similarity_hard"]),
                    "reason": reason,
                }
            )
    result = pd.DataFrame(rows, columns=_recommendation_columns())
    if not result.empty:
        result["date"] = pd.to_datetime(result["date"])
    return result


def _metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_default(value) if isinstance(value, (np.integer, np.floating)) else value for key, value in metrics.items()}


def evaluate_policy_metrics(
    recommendations: pd.DataFrame,
    actual: pd.DataFrame,
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """Use the fixed evaluator and add the requested per-session normalisation."""

    metrics = evaluate_recommendations(
        recommendations, actual, {"constraints": constraints}
    )
    sessions = int(metrics["evaluated_group_count"])
    metrics["net_benefit_per_session"] = (
        float(metrics["net_benefit"]) / sessions if sessions else 0.0
    )
    return metrics


def build_segment_metrics(
    recommendations: pd.DataFrame,
    actual: pd.DataFrame,
    constraints: dict[str, Any],
    trial_index: int,
) -> list[dict[str, Any]]:
    """Evaluate one trial by month, canteen, meal, weekday, and time phase."""

    actual = actual.copy()
    actual["date"] = pd.to_datetime(actual["date"])
    recs = recommendations.copy()
    recs["date"] = pd.to_datetime(recs["date"])
    partitions: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", actual)]
    for month, subset in actual.groupby(actual["date"].dt.strftime("%Y-%m"), sort=True):
        partitions.append(("month", str(month), subset))
    for canteen, subset in actual.groupby("食堂名称", sort=True):
        partitions.append(("canteen", str(canteen), subset))
    for meal, subset in actual.groupby("餐次", sort=True):
        partitions.append(("meal", str(meal), subset))
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    for weekday, subset in actual.groupby(actual["date"].dt.weekday, sort=True):
        partitions.append(("weekday", weekday_names[int(weekday)], subset))

    context_dimensions = [
        ("day_type", "_day_type"),
        ("calendar_state", "_calendar_state"),
        ("school_break", "是否校历寒暑假"),
        ("school_break_type", "校历假期类型"),
        ("statutory_holiday", "是否法定节假日假期"),
        ("statutory_holiday_name", "法定节假日名称"),
        ("adjusted_workday", "是否调休工作日"),
        ("temperature_band", "_temperature_band"),
        ("rain_band", "_rain_band"),
        ("humidity_state", "_humidity_state"),
        ("wind_state", "_wind_state"),
        ("extreme_weather", "_extreme_weather"),
    ]
    for dimension, column in context_dimensions:
        if column not in actual:
            continue
        for value, subset in actual.groupby(column, sort=True, dropna=False):
            partitions.append((dimension, str(value), subset))

    dates = np.asarray(sorted(actual["date"].unique()))
    for label, phase_dates in zip(["early", "middle", "late"], np.array_split(dates, 3)):
        phase_set = set(pd.to_datetime(phase_dates))
        partitions.append(("training_phase", label, actual.loc[actual["date"].isin(phase_set)]))

    rows: list[dict[str, Any]] = []
    for dimension, segment, subset in partitions:
        keys = subset[GROUP_KEYS].drop_duplicates()
        segment_recs = recs.merge(keys, on=GROUP_KEYS, how="inner")
        metrics = evaluate_policy_metrics(segment_recs, subset, constraints)
        hit_metrics, _ = hit_quantity_diagnostics(segment_recs, subset)
        rows.append(
            {
                "trial_index": int(trial_index),
                "dimension": dimension,
                "segment": segment,
                "segment_start": str(subset["date"].min().date()),
                "segment_end": str(subset["date"].max().date()),
                **_metric_payload(metrics),
                **_metric_payload(hit_metrics),
            }
        )
    return rows


def build_auxiliary_diagnostics(
    recommendations: pd.DataFrame,
    actual: pd.DataFrame,
    selection_audit: list[dict[str, Any]],
    constraints: dict[str, Any],
    segment_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute search diagnostics without changing evaluator.py."""

    selection_metrics, missed = selection_diagnostics(
        recommendations, actual, selection_audit
    )
    capacity_metrics, _ = menu_capacity_diagnostics(
        recommendations, actual, selection_audit
    )
    metrics: dict[str, Any] = {
        **selection_metrics,
        **capacity_metrics,
        **factor_stack_diagnostics(recommendations),
        **hit_quantity_diagnostics(recommendations, actual)[0],
        **prediction_consistency_diagnostics(recommendations),
        **recommendation_repeat_diagnostics(recommendations),
        **soft_diversity_diagnostics(recommendations, constraints),
    }
    segments = pd.DataFrame(segment_rows)
    for dimension in ["month", "meal"]:
        subset = segments.loc[segments["dimension"].eq(dimension)]
        metrics[f"worst_{dimension}_mape"] = (
            float(subset["total_mape"].max()) if not subset.empty else 0.0
        )
        metrics[f"worst_{dimension}_net_per_session"] = (
            float(subset["net_benefit_per_session"].min())
            if not subset.empty
            else 0.0
        )
    return metrics, missed


def write_menu(frame: pd.DataFrame, path: Path) -> None:
    result = frame.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    result.to_csv(path, index=False, encoding="utf-8-sig")


def build_name_audit(
    split_frames: list[tuple[str, pd.DataFrame]], attributes: pd.DataFrame
) -> pd.DataFrame:
    attribute_lookup = attributes.set_index("_dish_norm").to_dict("index")
    rows: list[dict[str, Any]] = []
    for split, frame in split_frames:
        for dish_norm, group in frame.groupby("_dish_norm", sort=False):
            dish_name = str(group["菜品名称"].iloc[0])
            attr = attribute_lookup.get(dish_norm)
            if attr is None:
                inferred = infer_name_attributes(dish_name)
                rows.append(
                    {
                        "record_type": "unmatched_library", "split": split, "菜品名称": dish_name,
                        "normalized_name": dish_norm, "library_matched": False,
                        "inferred_main_ingredient": inferred["inferred_main_ingredient"],
                        "ingredient_family": inferred["ingredient_family"],
                        "inference_confidence": inferred["inference_confidence"],
                        "attribute_source": inferred["attribute_source"], "记录数": int(len(group)),
                    }
                )
            elif str(attr["inference_confidence"]) in {"low"} or str(attr["attribute_source"]) == "unknown":
                rows.append(
                    {
                        "record_type": "low_confidence_inference", "split": split, "菜品名称": dish_name,
                        "normalized_name": dish_norm, "library_matched": True,
                        "inferred_main_ingredient": attr["inferred_main_ingredient"],
                        "ingredient_family": attr["ingredient_family"],
                        "inference_confidence": attr["inference_confidence"],
                        "attribute_source": attr["attribute_source"], "记录数": int(len(group)),
                    }
                )
    columns = [
        "record_type", "split", "菜品名称", "normalized_name", "library_matched",
        "inferred_main_ingredient", "ingredient_family", "inference_confidence",
        "attribute_source", "记录数",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["record_type", "split", "菜品名称"], ignore_index=True
    )

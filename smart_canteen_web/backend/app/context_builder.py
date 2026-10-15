"""Leakage-safe single-request inference context for frozen Trial 18."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .model_loader import frozen_policy


WEEKDAY_NAMES = [
    "星期一",
    "星期二",
    "星期三",
    "星期四",
    "星期五",
    "星期六",
    "星期日",
]


def weekday_name(value: pd.Timestamp) -> str:
    return WEEKDAY_NAMES[pd.Timestamp(value).weekday()]


def build_target_context(
    prediction_date: pd.Timestamp,
    meal: str,
    *,
    school_break: bool,
    school_break_type: str,
    statutory_holiday: bool,
    statutory_holiday_name: str,
    adjusted_workday: bool,
    temperature_c: float,
    humidity_pct: float,
    precipitation_mm: float,
    wind_speed_kmh: float,
) -> dict[str, Any]:
    row = {
        "date": pd.Timestamp(prediction_date),
        "星期": weekday_name(pd.Timestamp(prediction_date)),
        "是否校历寒暑假": int(school_break),
        "校历假期类型": (
            str(school_break_type) if school_break else "非寒暑假"
        ),
        "是否法定节假日假期": int(statutory_holiday),
        "法定节假日名称": (
            str(statutory_holiday_name)
            if statutory_holiday
            else "非节假日"
        ),
        "是否调休工作日": int(adjusted_workday),
        "气温（℃）": float(temperature_c),
        "相对湿度（%）": float(humidity_pct),
        "降水量（毫米）": float(precipitation_mm),
        "10米风速（公里/小时）": float(wind_speed_kmh),
    }
    derived = frozen_policy._derive_context_states(pd.DataFrame([row])).iloc[0]
    columns = frozen_policy.KNOWN_CONTEXT_COLUMNS + [
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
    result = {column: derived[column] for column in columns}
    result["_meal"] = str(meal)
    return result


def _fill_unmatched_attributes(
    features: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    unmatched = features["library_matched"].isna()
    if not unmatched.any():
        return features
    result = features.copy()
    name_lookup = (
        history.drop_duplicates("_dish_norm")
        .set_index("_dish_norm")["菜品名称"]
        .to_dict()
    )
    for index in result.index[unmatched]:
        dish_norm = str(result.at[index, "_dish_norm"])
        dish_name = str(name_lookup.get(dish_norm, dish_norm))
        inferred = frozen_policy.infer_name_attributes(dish_name)
        fallback = {
            "菜品名称_库": dish_name,
            "菜品类别": "unknown",
            "口味": "unknown",
            "主要食材": inferred["inferred_main_ingredient"],
            "食材大类": inferred["ingredient_family"],
            "制作方法": inferred["制作方法"],
            "相似菜分组": inferred["相似菜分组"],
            "主食标记": False,
            "inferred_main_ingredient": inferred[
                "inferred_main_ingredient"
            ],
            "ingredient_family": inferred["ingredient_family"],
            "attribute_source": inferred["attribute_source"],
            "inference_confidence": inferred["inference_confidence"],
            "library_matched": False,
            "_main_hard": inferred["_main_hard"],
            "_family_hard": inferred["_family_hard"],
            "_method_hard": inferred["_method_hard"],
            "_flavor_hard": False,
            "_similarity_hard": inferred["_similarity_hard"],
            "特征来源": "未匹配-仅菜名推断",
        }
        for key, value in fallback.items():
            result.at[index, key] = value
    return result


def build_inference_context(
    *,
    history: pd.DataFrame,
    attributes: pd.DataFrame,
    prediction_date: pd.Timestamp,
    canteen: str,
    meal: str,
    target_context: dict[str, Any],
) -> dict[str, Any]:
    """Build one Trial 18 context without any target-day actual rows."""

    target_date = pd.Timestamp(prediction_date).normalize()
    historical = history.loc[history["date"].lt(target_date)].copy()
    if historical.empty:
        raise ValueError("目标日期以前没有可用真实销售历史")
    if historical["date"].max() >= target_date:
        raise AssertionError("实时推理历史包含目标日期或未来数据")
    weekday = target_date.weekday()
    global_history = historical.loc[historical["餐次"].eq(meal)]
    local_history = global_history.loc[
        global_history["食堂名称"].astype(str).eq(str(canteen))
    ]
    if global_history.empty:
        raise ValueError("目标餐次没有可用历史")

    local_features = frozen_policy._dish_aggregate(
        local_history, target_date, weekday, "l_"
    )
    global_features = frozen_policy._dish_aggregate(
        global_history, target_date, weekday, "g_"
    )
    local_regular = local_history.loc[
        local_history.get(
            "_calendar_category",
            pd.Series("普通日期", index=local_history.index),
        ).eq("普通日期")
    ]
    global_regular = global_history.loc[
        global_history.get(
            "_calendar_category",
            pd.Series("普通日期", index=global_history.index),
        ).eq("普通日期")
    ]
    local_regular_features = frozen_policy._dish_aggregate(
        local_regular, target_date, weekday, "lr_"
    )
    global_regular_features = frozen_policy._dish_aggregate(
        global_regular, target_date, weekday, "gr_"
    )
    features = (
        local_features.join(global_features, how="outer")
        .join(local_regular_features, how="outer")
        .join(global_regular_features, how="outer")
        .reset_index()
    )
    features = attributes.merge(features, on="_dish_norm", how="outer")
    features = _fill_unmatched_attributes(features, global_history)
    local_meta = frozen_policy._session_aggregate(
        local_history, target_date, weekday, "l_"
    )
    global_meta = frozen_policy._session_aggregate(
        global_history, target_date, weekday, "g_"
    )
    history_max_date = pd.Timestamp(historical["date"].max()).normalize()
    forecast_gap_days = max(int((target_date - history_max_date).days), 0)
    return {
        "date": target_date,
        "食堂名称": str(canteen),
        "餐次": str(meal),
        "weekday": weekday,
        "target_context": target_context,
        "features": features,
        "meta": {**global_meta, **local_meta},
        "max_history_date": history_max_date,
        "forecast_gap_days": forecast_gap_days,
        "history_rows_at_day_start": int(len(historical)),
        "history_dates_at_day_start": int(historical["date"].nunique()),
        "day_group_count": 1,
        "actual_appended_after_day_batch": True,
    }

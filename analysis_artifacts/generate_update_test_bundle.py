from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "smart_canteen_web" / "data"
BUILD_ROOT = Path("/private/tmp/smart_canteen_update_20260824_build")
START_DATE = pd.Timestamp("2026-06-11")
END_DATE = pd.Timestamp("2026-08-24")

WEEKDAY_NAMES = {
    0: "星期一",
    1: "星期二",
    2: "星期三",
    3: "星期四",
    4: "星期五",
    5: "星期六",
    6: "星期日",
}

SALES_COLUMNS = [
    "年",
    "月",
    "日",
    "餐次",
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
    "菜品名称",
    "菜品总销量",
    "菜品售卖单位",
    "订单数量",
    "菜品单价",
    "销售额",
    "平均取菜量",
    "食堂名称",
]


def calendar_row(target: pd.Timestamp) -> dict[str, object]:
    summer = target >= pd.Timestamp("2026-07-21")
    dragon_boat = pd.Timestamp("2026-06-19") <= target <= pd.Timestamp(
        "2026-06-21"
    )
    return {
        "日期": target.strftime("%Y-%m-%d"),
        "星期": WEEKDAY_NAMES[target.weekday()],
        "是否校历寒暑假": int(summer),
        "校历假期类型": "暑假" if summer else "非寒暑假",
        "是否法定节假日假期": int(dragon_boat),
        "法定节假日名称": "端午节" if dragon_boat else "非节假日",
        "是否调休工作日": 0,
    }


def build_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame([calendar_row(target) for target in dates])


def build_weather(dates: pd.DatetimeIndex) -> pd.DataFrame:
    historical = pd.read_csv(
        DATA_ROOT / "features" / "weather.csv", encoding="utf-8-sig"
    )
    historical["日期"] = pd.to_datetime(historical["日期"], errors="raise")
    historical = historical.loc[historical["日期"].dt.year.eq(2025)].copy()
    weather_columns = [
        "气温（℃）",
        "相对湿度（%）",
        "降水量（毫米）",
        "10米风速（公里/小时）",
    ]
    historical = (
        historical.set_index("日期")[weather_columns]
        .reindex(pd.date_range("2025-06-11", "2025-08-24", freq="D"))
        .interpolate(method="time", limit_direction="both")
    )
    historical["month_day"] = historical.index.strftime("%m-%d")
    lookup = historical.drop_duplicates("month_day").set_index("month_day")
    rows = []
    for target in dates:
        source = lookup.loc[target.strftime("%m-%d")]
        rows.append(
            {
                "日期": target.strftime("%Y-%m-%d"),
                "气温（℃）": int(round(float(source["气温（℃）"]))),
                "相对湿度（%）": int(round(float(source["相对湿度（%）"]))),
                "降水量（毫米）": int(
                    round(float(source["降水量（毫米）"]))
                ),
                "10米风速（公里/小时）": int(
                    round(float(source["10米风速（公里/小时）"]))
                ),
            }
        )
    return pd.DataFrame(rows)


def _source_date_for_target(
    target: pd.Timestamp, available: list[pd.Timestamp], target_index: int
) -> pd.Timestamp:
    same_weekday = [value for value in available if value.weekday() == target.weekday()]
    recent = same_weekday[-5:]
    if not recent:
        raise RuntimeError(f"{target:%Y-%m-%d} 没有可用同星期历史")
    return recent[target_index % len(recent)]


def build_sales(
    dataset_id: str,
    display_name: str,
    dates: pd.DatetimeIndex,
    weather: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source_path = DATA_ROOT / "datasets" / dataset_id / "sales.csv"
    history = pd.read_csv(source_path, encoding="utf-8-sig")
    history["_date"] = pd.to_datetime(
        {"year": history["年"], "month": history["月"], "day": history["日"]},
        errors="raise",
    )
    history = history.loc[history["_date"].le(pd.Timestamp("2026-06-10"))].copy()
    available_dates = sorted(pd.Timestamp(value) for value in history["_date"].unique())
    weather_lookup = weather.set_index("日期").to_dict("index")
    calendar_lookup = calendar.set_index("日期").to_dict("index")
    rows: list[pd.DataFrame] = []
    source_usage: dict[str, int] = {}
    meal_counts: dict[str, int] = {}

    for target_index, target in enumerate(dates):
        source_date = _source_date_for_target(target, available_dates, target_index)
        source_usage[source_date.strftime("%Y-%m-%d")] = (
            source_usage.get(source_date.strftime("%Y-%m-%d"), 0) + 1
        )
        source_day = history.loc[history["_date"].eq(source_date)].copy()
        date_text = target.strftime("%Y-%m-%d")
        calendar_values = calendar_lookup[date_text]
        weather_values = weather_lookup[date_text]

        day_factor = 1.0
        if target.weekday() >= 5:
            day_factor *= 0.88
        if bool(calendar_values["是否校历寒暑假"]):
            day_factor *= 0.58 if dataset_id == "a_hupan" else 0.52
        if bool(calendar_values["是否法定节假日假期"]):
            day_factor *= 0.72
        day_factor *= 1.0 + 0.035 * math.sin((target_index + 1) * 1.7)

        for meal, meal_frame in source_day.groupby("餐次", sort=False):
            meal_frame = meal_frame.copy()
            meal_counts[str(meal)] = meal_counts.get(str(meal), 0) + len(meal_frame)
            meal_multiplier = {"早餐": 0.96, "午餐": 1.02, "晚餐": 1.0}.get(
                str(meal), 1.0
            )
            original_orders = pd.to_numeric(
                meal_frame["订单数量"], errors="coerce"
            ).fillna(0.0)
            row_noise = [
                1.0 + 0.025 * math.sin((target_index + 1) * (row_no + 3))
                for row_no in range(len(meal_frame))
            ]
            multiplier = pd.Series(row_noise, index=meal_frame.index) * (
                day_factor * meal_multiplier
            )
            new_orders = (original_orders * multiplier).round().clip(lower=0)
            positive = original_orders.gt(0)
            scale = pd.Series(day_factor * meal_multiplier, index=meal_frame.index)
            scale.loc[positive] = new_orders.loc[positive] / original_orders.loc[positive]
            meal_frame["订单数量"] = new_orders.astype(float)
            meal_frame["菜品总销量"] = (
                pd.to_numeric(meal_frame["菜品总销量"], errors="coerce") * scale
            ).round(4)
            meal_frame["销售额"] = (
                pd.to_numeric(meal_frame["销售额"], errors="coerce") * scale
            ).round(2)
            meal_frame["年"] = target.year
            meal_frame["月"] = target.month
            meal_frame["日"] = target.day
            meal_frame["星期"] = calendar_values["星期"]
            for column in [
                "是否校历寒暑假",
                "校历假期类型",
                "是否法定节假日假期",
                "法定节假日名称",
                "是否调休工作日",
            ]:
                meal_frame[column] = calendar_values[column]
            for column in [
                "气温（℃）",
                "相对湿度（%）",
                "降水量（毫米）",
                "10米风速（公里/小时）",
            ]:
                meal_frame[column] = weather_values[column]
            meal_frame["食堂名称"] = display_name
            rows.append(meal_frame[SALES_COLUMNS])

    result = pd.concat(rows, ignore_index=True)
    duplicates = int(
        result.duplicated(["年", "月", "日", "餐次", "菜品名称"]).sum()
    )
    if duplicates:
        raise RuntimeError(f"{dataset_id} 生成了{duplicates}个重复业务键")
    report = {
        "dataset_id": dataset_id,
        "display_name": display_name,
        "rows": int(len(result)),
        "dates": int(result[["年", "月", "日"]].drop_duplicates().shape[0]),
        "date_start": START_DATE.strftime("%Y-%m-%d"),
        "date_end": END_DATE.strftime("%Y-%m-%d"),
        "meal_rows": meal_counts,
        "source_dates_used": source_usage,
        "duplicate_business_keys": duplicates,
    }
    return result, report


def main() -> None:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    calendar = build_calendar(dates)
    weather = build_weather(dates)
    calendar.to_csv(
        BUILD_ROOT / "calendar.csv", index=False, encoding="utf-8-sig"
    )
    weather.to_csv(BUILD_ROOT / "weather.csv", index=False, encoding="utf-8-sig")

    reports: dict[str, object] = {
        "date_start": START_DATE.strftime("%Y-%m-%d"),
        "date_end": END_DATE.strftime("%Y-%m-%d"),
        "date_count": int(len(dates)),
        "weather_rows": int(len(weather)),
        "calendar_rows": int(len(calendar)),
        "synthetic": True,
        "datasets": {},
    }
    for dataset_id, display_name in [
        ("a_hupan", "A湖畔智慧餐厅"),
        ("f_jichuyuan", "F基础园智慧餐厅"),
    ]:
        sales, report = build_sales(
            dataset_id, display_name, dates, weather, calendar
        )
        sales.to_csv(
            BUILD_ROOT / f"{dataset_id}_sales.csv",
            index=False,
            encoding="utf-8-sig",
        )
        reports["datasets"][dataset_id] = report

    (BUILD_ROOT / "generation_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

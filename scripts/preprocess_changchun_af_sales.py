#!/usr/bin/env python3
"""Convert Changchun A/F and Jilin University canteen sales to one schema."""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd


CANTEENS = {
    "A": {
        "name": "长春A食堂",
        "source": Path("1-长春A食堂/长春A食堂菜品销量.xlsx"),
        "mapping": Path("research/数据/长春A食堂_菜品名称转换映射.csv"),
        "filename": "长春A食堂_菜品销量_菜名修正后.csv",
    },
    "F": {
        "name": "长春F食堂",
        "source": Path("6-长春F食堂/长春F食堂菜品销量.xlsx"),
        "mapping": Path("research/数据/长春F食堂_菜品名称转换映射.csv"),
        "filename": "长春F食堂_菜品销量_菜名修正后.csv",
    },
    "hupan": {
        "name": "吉大湖畔智慧餐厅",
        "source": Path("research/数据/菜品销量_吉大湖畔智慧餐厅.csv"),
        "mapping": Path("research/数据/长春A食堂_菜品名称转换映射.csv"),
        "filename": "菜品销量_吉大湖畔智慧餐厅_菜名修正后.csv",
    },
    "jichuyuan": {
        "name": "吉大南校基础园智慧餐厅",
        "source": Path("research/数据/菜品销量_吉大南校基础园智慧餐厅.csv"),
        "mapping": Path("research/数据/长春F食堂_菜品名称转换映射.csv"),
        "filename": "菜品销量_吉大南校基础园智慧餐厅_菜名修正后.csv",
    },
}

DEFAULT_WEATHER = Path("research/数据/长春_2025-01-01至2026-07-12_三餐时段天气.csv")

ACADEMIC_BREAKS = [
    ("2025-01-13", "2025-02-23", "寒假"),
    ("2025-07-07", "2025-08-24", "暑假"),
    ("2026-01-19", "2026-03-08", "寒假"),
    ("2026-07-13", "2026-09-06", "暑假"),
]

# 国务院办公厅公布的放假调休区间，起止日期均包含在假期内。
PUBLIC_HOLIDAYS = [
    ("2025-01-01", "2025-01-01", "元旦"),
    ("2025-01-28", "2025-02-04", "春节"),
    ("2025-04-04", "2025-04-06", "清明节"),
    ("2025-05-01", "2025-05-05", "劳动节"),
    ("2025-05-31", "2025-06-02", "端午节"),
    ("2025-10-01", "2025-10-08", "国庆节、中秋节"),
    ("2026-01-01", "2026-01-03", "元旦"),
    ("2026-02-15", "2026-02-23", "春节"),
    ("2026-04-04", "2026-04-06", "清明节"),
    ("2026-05-01", "2026-05-05", "劳动节"),
    ("2026-06-19", "2026-06-21", "端午节"),
    ("2026-09-25", "2026-09-27", "中秋节"),
    ("2026-10-01", "2026-10-07", "国庆节"),
]

MAKEUP_WORKDAYS = {
    "2025-01-26",
    "2025-02-08",
    "2025-04-27",
    "2025-09-28",
    "2025-10-11",
    "2026-01-04",
    "2026-02-14",
    "2026-02-28",
    "2026-05-09",
    "2026-09-20",
    "2026-10-10",
}

WEATHER_COLUMNS = [
    "气温（℃）",
    "相对湿度（%）",
    "降水量（毫米）",
    "10米风速（公里/小时）",
]

WEEKDAY_NAMES = {
    0: "星期一",
    1: "星期二",
    2: "星期三",
    3: "星期四",
    4: "星期五",
    5: "星期六",
    6: "星期日",
}

OUTPUT_COLUMNS = [
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
    *WEATHER_COLUMNS,
    "菜品名称",
    "菜品总销量",
    "菜品售卖单位",
    "订单数量",
    "菜品单价",
    "销售额",
    "平均取菜量",
    "食堂名称",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canteen",
        choices=["A", "F", "hupan", "jichuyuan", "targets", "all"],
        default="all",
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1], help="项目根目录"
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        help="可选映射 CSV；默认根据食堂自动选择 A组或F组映射",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        help="天气 CSV；默认使用数据目录内的长春三餐时段天气文件",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录；默认使用数据/映射转换结果",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="可选：与该目录内现有修正后 CSV 逐行核对",
    )
    return parser.parse_args()


def preferred_clean_mode(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return np.nan
    text = values.astype(str)
    clean = text.loc[~text.str.contains("�", regex=False)]
    pool = clean if not clean.empty else text
    counts = pool.value_counts(sort=False)
    highest = counts.max()
    winners = set(counts.loc[counts.eq(highest)].index)
    return next(value for value in pool if value in winners)


def convert_quantity(frame: pd.DataFrame) -> pd.Series:
    price_unit = frame["菜品单价"].astype(str).str.extract(r"元/(\d+)(.*)$")
    denominator = pd.to_numeric(price_unit[0], errors="coerce")
    unit = price_unit[1].fillna("")
    gram_like = denominator.gt(1) & unit.map(lambda value: value == "克" or "�" in value)
    result = frame["菜品总销量"].astype(float).copy()
    result.loc[gram_like] = result.loc[gram_like] / denominator.loc[gram_like]
    return result


def load_weather(path: Path) -> pd.DataFrame:
    weather = pd.read_csv(path)
    required = {"日期时间", "餐次", *WEATHER_COLUMNS}
    missing_columns = required.difference(weather.columns)
    if missing_columns:
        raise ValueError(f"天气文件缺少字段：{sorted(missing_columns)}")

    date_time = pd.to_datetime(weather["日期时间"], errors="coerce")
    if date_time.isna().any():
        raise ValueError("天气文件包含无法解析的日期时间")
    weather["年"] = date_time.dt.year.astype("int64")
    weather["月"] = date_time.dt.month.astype("int64")
    weather["日"] = date_time.dt.day.astype("int64")
    weather["餐次"] = weather["餐次"].astype(str).str.strip()
    for column in WEATHER_COLUMNS:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
        if weather[column].isna().any():
            raise ValueError(f"天气文件的 {column} 包含空值或非数值")

    keys = ["年", "月", "日", "餐次"]
    if weather.duplicated(keys).any():
        duplicates = weather.loc[weather.duplicated(keys, keep=False), keys]
        raise ValueError(f"天气文件存在重复日期餐次：\n{duplicates.head(20).to_string(index=False)}")
    return weather[keys + WEATHER_COLUMNS]


def add_date_and_weather_features(
    frame: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(
        {"year": result["年"], "month": result["月"], "day": result["日"]}
    )
    result["星期"] = dates.dt.dayofweek.map(WEEKDAY_NAMES)

    result["校历假期类型"] = "非寒暑假"
    for start, end, break_type in ACADEMIC_BREAKS:
        in_break = dates.between(start, end, inclusive="both")
        result.loc[in_break, "校历假期类型"] = break_type
    result["是否校历寒暑假"] = result["校历假期类型"].ne("非寒暑假").astype("int64")

    result["法定节假日名称"] = "非节假日"
    for start, end, holiday_name in PUBLIC_HOLIDAYS:
        in_holiday = dates.between(start, end, inclusive="both")
        result.loc[in_holiday, "法定节假日名称"] = holiday_name
    result["是否法定节假日假期"] = (
        result["法定节假日名称"].ne("非节假日").astype("int64")
    )
    result["是否调休工作日"] = (
        dates.dt.strftime("%Y-%m-%d").isin(MAKEUP_WORKDAYS).astype("int64")
    )

    result["_输出顺序"] = np.arange(len(result), dtype="int64")
    keys = ["年", "月", "日", "餐次"]
    result = result.merge(weather, on=keys, how="left", sort=False, validate="many_to_one")
    missing_weather = result[WEATHER_COLUMNS].isna().any(axis=1)
    if missing_weather.any():
        missing = result.loc[missing_weather, keys].drop_duplicates()
        raise ValueError(
            "天气文件未覆盖全部销量日期餐次：\n"
            f"{missing.head(20).to_string(index=False)}"
        )
    return result.sort_values("_输出顺序", kind="stable").drop(columns="_输出顺序")


def preprocess_one(
    root: Path, mapping: pd.DataFrame, weather: pd.DataFrame, code: str
) -> pd.DataFrame:
    config = CANTEENS[code]
    source_path = root / config["source"]
    raw = pd.read_excel(source_path) if source_path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(source_path)
    if "日期" in raw.columns:
        date_values = pd.to_datetime(raw["日期"])
        raw["年"] = date_values.dt.year.astype("int64")
        raw["月"] = date_values.dt.month.astype("int64")
        raw["日"] = date_values.dt.day.astype("int64")
    elif {"年", "月", "日"}.issubset(raw.columns):
        raw[["年", "月", "日"]] = raw[["年", "月", "日"]].astype("int64")
    else:
        raise ValueError(f"源文件缺少日期或年/月/日字段：{source_path}")
    raw["食堂名称"] = config["name"]
    raw["_原始顺序"] = np.arange(len(raw), dtype="int64")

    lookup = mapping.loc[mapping["食堂名称"].eq(config["name"])].copy()
    if "餐次" in lookup.columns:
        meal_specific = lookup.loc[lookup["餐次"].fillna("*").astype(str).ne("*")]
        wildcard = lookup.loc[lookup["餐次"].fillna("*").astype(str).eq("*")]
        if meal_specific.duplicated(["旧菜品名称", "原菜品单价", "餐次"]).any():
            raise ValueError("映射表存在重复的食堂+旧菜名+原菜品单价+餐次键")
        if wildcard.duplicated(["旧菜品名称", "原菜品单价"]).any():
            raise ValueError("映射表存在重复的食堂+旧菜名+原菜品单价通用键")
        exact_lookup = meal_specific.set_index(
            ["旧菜品名称", "原菜品单价", "餐次"]
        )["新菜品名称"]
        exact_keys = pd.MultiIndex.from_frame(
            raw[["菜品名称", "菜品单价", "餐次"]].rename(
                columns={"菜品名称": "旧菜品名称", "菜品单价": "原菜品单价"}
            )
        )
        corrected_names = exact_lookup.reindex(exact_keys)
        if corrected_names.isna().any() and not wildcard.empty:
            wildcard_lookup = wildcard.set_index(["旧菜品名称", "原菜品单价"])[
                "新菜品名称"
            ]
            wildcard_keys = pd.MultiIndex.from_frame(
                raw[["菜品名称", "菜品单价"]].rename(
                    columns={"菜品名称": "旧菜品名称", "菜品单价": "原菜品单价"}
                )
            )
            fallback = wildcard_lookup.reindex(wildcard_keys)
            corrected_names = corrected_names.where(corrected_names.notna(), fallback.to_numpy())
    else:
        if lookup.duplicated(["旧菜品名称", "原菜品单价"]).any():
            raise ValueError("映射表存在重复的食堂+旧菜名+原菜品单价键")
        name_by_key = lookup.set_index(["旧菜品名称", "原菜品单价"])["新菜品名称"]
        keys = pd.MultiIndex.from_frame(
            raw[["菜品名称", "菜品单价"]].rename(
                columns={"菜品名称": "旧菜品名称", "菜品单价": "原菜品单价"}
            )
        )
        corrected_names = name_by_key.reindex(keys)
    if corrected_names.isna().any():
        missing = raw.loc[
            corrected_names.isna().to_numpy(), ["菜品名称", "菜品单价"]
        ].drop_duplicates()
        raise ValueError(f"映射表未覆盖全部原始记录：\n{missing.head(20).to_string(index=False)}")
    raw["菜品名称"] = corrected_names.to_numpy()
    raw["菜品总销量"] = convert_quantity(raw)
    raw["_加权取菜量"] = raw["平均取菜量"].fillna(0) * raw["订单数量"].fillna(0)

    group_columns = ["年", "月", "日", "餐次", "菜品名称", "食堂名称"]
    result = (
        raw.groupby(group_columns, sort=False, dropna=False)
        .agg(
            菜品总销量=("菜品总销量", "sum"),
            菜品售卖单位=("菜品售卖单位", preferred_clean_mode),
            订单数量=("订单数量", "sum"),
            菜品单价=("菜品单价", preferred_clean_mode),
            销售额=("销售额", "sum"),
            _加权取菜量=("_加权取菜量", "sum"),
            _原始顺序=("_原始顺序", "min"),
        )
        .reset_index()
    )
    result["平均取菜量"] = np.where(
        result["订单数量"].ne(0), result["_加权取菜量"] / result["订单数量"], np.nan
    )
    result = result.sort_values("_原始顺序", kind="stable")
    result["菜品总销量"] = result["菜品总销量"].round(6)
    result["销售额"] = result["销售额"].round(2)
    result["平均取菜量"] = result["平均取菜量"].round(6)
    result["订单数量"] = result["订单数量"].astype("int64")
    result = add_date_and_weather_features(result, weather)
    return result[OUTPUT_COLUMNS].reset_index(drop=True)


def compare_with_reference(generated: pd.DataFrame, reference_path: Path) -> None:
    reference = pd.read_csv(reference_path)
    if "日期" in reference.columns and not {"年", "月", "日"}.issubset(reference.columns):
        date_values = pd.to_datetime(reference.pop("日期"))
        reference.insert(0, "日", date_values.dt.day.astype("int64"))
        reference.insert(0, "月", date_values.dt.month.astype("int64"))
        reference.insert(0, "年", date_values.dt.year.astype("int64"))
    if list(generated.columns) != list(reference.columns):
        raise AssertionError(f"字段不一致：{list(generated.columns)} != {list(reference.columns)}")
    if len(generated) != len(reference):
        raise AssertionError(f"行数不一致：{len(generated)} != {len(reference)}")

    text_columns = [
        "餐次",
        "星期",
        "校历假期类型",
        "法定节假日名称",
        "菜品名称",
        "菜品售卖单位",
        "菜品单价",
        "食堂名称",
    ]
    numeric_columns = [
        "年",
        "月",
        "日",
        "是否校历寒暑假",
        "是否法定节假日假期",
        "是否调休工作日",
        *WEATHER_COLUMNS,
        "菜品总销量",
        "订单数量",
        "销售额",
        "平均取菜量",
    ]
    for column in text_columns:
        mismatch = generated[column].fillna("").astype(str) != reference[column].fillna("").astype(str)
        if mismatch.any():
            first = int(np.flatnonzero(mismatch.to_numpy())[0])
            raise AssertionError(
                f"{column} 第 {first + 2} 行不一致："
                f"{generated.at[first, column]!r} != {reference.at[first, column]!r}"
            )
    for column in numeric_columns:
        left = pd.to_numeric(generated[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(reference[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0, atol=1e-6, equal_nan=True):
            mismatch = ~np.isclose(left, right, rtol=0, atol=1e-6, equal_nan=True)
            first = int(np.flatnonzero(mismatch)[0])
            raise AssertionError(
                f"{column} 第 {first + 2} 行不一致：{left[first]} != {right[first]}"
            )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir or root / "research" / "数据" / "映射转换结果"
    reference_dir = args.reference_dir.resolve() if args.reference_dir else None
    weather_path = args.weather or root / DEFAULT_WEATHER
    if not weather_path.is_absolute():
        weather_path = root / weather_path
    weather = load_weather(weather_path)

    if args.canteen == "all":
        selected = ["A", "F", "hupan", "jichuyuan"]
    elif args.canteen == "targets":
        selected = ["hupan", "jichuyuan"]
    else:
        selected = [args.canteen]
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_cache = {}
    for code in selected:
        if args.mapping:
            mapping_path = args.mapping if args.mapping.is_absolute() else root / args.mapping
        else:
            mapping_path = root / CANTEENS[code]["mapping"]
        if mapping_path not in mapping_cache:
            mapping_cache[mapping_path] = pd.read_csv(mapping_path)
        mapping = mapping_cache[mapping_path]
        generated = preprocess_one(root, mapping, weather, code)
        output_path = output_dir / CANTEENS[code]["filename"]
        generated.to_csv(output_path, index=False, encoding="utf-8-sig")
        if reference_dir:
            compare_with_reference(generated, reference_dir / CANTEENS[code]["filename"])
            status = "，且已与参考文件逐行对齐"
        else:
            status = ""
        print(f"已生成 {output_path}：{len(generated)} 行{status}")
        del generated
        gc.collect()


if __name__ == "__main__":
    main()

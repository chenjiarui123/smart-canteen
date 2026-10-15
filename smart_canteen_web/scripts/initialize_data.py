#!/usr/bin/env python3
"""Build the two self-contained logical datasets from the existing project."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"

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
STATIC_LIBRARY_COLUMNS = [
    "菜品id",
    "菜品名称",
    "类别",
    "颜色",
    "营养",
    "口味",
    "主要食材",
    "主料",
    "食材大类",
    "主要食材大类",
    "制作方法",
    "烹饪方法",
    "主食标记",
    "是否主食",
    "相似菜分组",
    "相似菜品分组",
    "特征来源",
]
RECOMMENDATION_HISTORY_COLUMNS = [
    "request_id",
    "created_at",
    "date",
    "食堂名称",
    "餐次",
    "菜品名称",
    "pred_qty",
    "pred_orders",
    "pred_rev",
    "score",
    "reason",
]
WEEKDAY_NAMES = [
    "星期一",
    "星期二",
    "星期三",
    "星期四",
    "星期五",
    "星期六",
    "星期日",
]

DATASET_SOURCES = {
    "a_hupan": {
        "display_name": "A湖畔智慧餐厅",
        "aliases": ["长春A食堂", "吉大湖畔智慧餐厅"],
        "sales": [
            "数据/长春A食堂_菜品销量_菜名修正后.csv",
            "数据/菜品销量_吉大湖畔智慧餐厅_菜名修正后.csv",
        ],
        "library": "数据/长春A食堂_菜品库.csv",
    },
    "f_jichuyuan": {
        "display_name": "F基础园智慧餐厅",
        "aliases": ["长春F食堂", "吉大南校基础园智慧餐厅"],
        "sales": [
            "数据/长春F食堂_菜品销量_菜名修正后.csv",
            "数据/菜品销量_吉大南校基础园智慧餐厅_菜名修正后.csv",
        ],
        "library": "数据/长春F食堂_菜品库.csv",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def mode_deterministic(values: pd.Series):
    clean = values.dropna()
    if clean.empty:
        return ""
    counts = clean.astype(str).value_counts()
    best = sorted(counts[counts.eq(counts.max())].index)[0]
    original = clean.loc[clean.astype(str).eq(best)]
    return original.iloc[0]


def numeric_values(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + index * step, 1) for index in range(count + 1)]


def initialise(force: bool) -> None:
    registry_path = DATA_ROOT / "dataset_registry.json"
    if registry_path.exists() and not force:
        print("数据已初始化；如需从源文件重建请显式使用 --force")
        return
    missing = [
        relative
        for item in DATASET_SOURCES.values()
        for relative in item["sales"] + [item["library"]]
        if not (SOURCE_ROOT / relative).exists()
    ]
    if missing:
        raise RuntimeError(
            "源项目数据不存在，不能重建；复制后的成品项目无需运行初始化脚本"
        )

    all_sales: list[pd.DataFrame] = []
    manifest_datasets = {}
    registry = {"datasets": {}}
    for dataset_id, spec in DATASET_SOURCES.items():
        folder = DATA_ROOT / "datasets" / dataset_id
        folder.mkdir(parents=True, exist_ok=True)
        existing_sales = folder / "sales.csv"
        if force and existing_sales.exists():
            try:
                os.chmod(existing_sales, 0o644)
            except OSError:
                pass
        frames = []
        sources = []
        for relative in spec["sales"]:
            source_path = SOURCE_ROOT / relative
            frame = read_csv(source_path)
            if sorted(set(SALES_COLUMNS) - set(frame.columns)):
                raise RuntimeError(f"{relative} 缺少标准销售字段")
            frame = frame[SALES_COLUMNS].copy()
            frame["食堂名称"] = spec["display_name"]
            frames.append(frame)
            sources.append(
                {
                    "file": relative,
                    "sha256": sha256(source_path),
                    "rows": int(len(frame)),
                }
            )
        sales = pd.concat(frames, ignore_index=True)
        sales["_date"] = pd.to_datetime(
            {"year": sales["年"], "month": sales["月"], "day": sales["日"]}
        )
        sales = (
            sales.sort_values(["_date", "餐次", "菜品名称"])
            .drop(columns="_date")
            .reset_index(drop=True)
        )
        write_csv(sales, folder / "sales.csv")
        all_sales.append(sales)

        library_source = SOURCE_ROOT / spec["library"]
        library_raw = read_csv(library_source)
        kept = [
            column
            for column in STATIC_LIBRARY_COLUMNS
            if column in library_raw.columns
        ]
        library = library_raw[kept].copy()
        write_csv(library, folder / "dish_library.csv")
        write_csv(
            pd.DataFrame(columns=SALES_COLUMNS),
            folder / "runtime_actual_sales.csv",
        )
        write_csv(
            pd.DataFrame(columns=RECOMMENDATION_HISTORY_COLUMNS),
            folder / "recommendation_history.csv",
        )
        registry["datasets"][dataset_id] = {
            "display_name": spec["display_name"],
            "directory": f"datasets/{dataset_id}",
            "aliases": spec["aliases"],
        }
        dates = pd.to_datetime(
            {"year": sales["年"], "month": sales["月"], "day": sales["日"]}
        )
        manifest_datasets[dataset_id] = {
            "display_name": spec["display_name"],
            "sources": sources,
            "dish_library_source": {
                "file": spec["library"],
                "sha256": sha256(library_source),
                "source_rows": int(len(library_raw)),
                "static_rows": int(len(library)),
                "ignored_numeric_history_columns": sorted(
                    set(library_raw.columns) - set(kept)
                ),
            },
            "sales_rows": int(len(sales)),
            "sales_sha256": sha256(folder / "sales.csv"),
            "dish_library_sha256": sha256(folder / "dish_library.csv"),
            "sales_date_range": [
                dates.min().strftime("%Y-%m-%d"),
                dates.max().strftime("%Y-%m-%d"),
            ],
            "runtime_actual_sales_added_rows": 0,
            "updated_at": datetime.now().astimezone().isoformat(),
        }

    combined = pd.concat(all_sales, ignore_index=True)
    combined["_date"] = pd.to_datetime(
        {
            "year": combined["年"],
            "month": combined["月"],
            "day": combined["日"],
        }
    )
    weather = (
        combined.groupby("_date", as_index=False)
        .agg(
            **{
                "气温（℃）": ("气温（℃）", "median"),
                "相对湿度（%）": ("相对湿度（%）", "median"),
                "降水量（毫米）": ("降水量（毫米）", "median"),
                "10米风速（公里/小时）": (
                    "10米风速（公里/小时）",
                    "median",
                ),
            }
        )
        .sort_values("_date")
    )
    weather.insert(0, "日期", weather["_date"].dt.strftime("%Y-%m-%d"))
    weather = weather.drop(columns="_date")
    for column in weather.columns[1:]:
        weather[column] = pd.to_numeric(weather[column]).round(1)
    write_csv(weather, DATA_ROOT / "features" / "weather.csv")

    calendar_columns = [
        "是否校历寒暑假",
        "校历假期类型",
        "是否法定节假日假期",
        "法定节假日名称",
        "是否调休工作日",
    ]
    calendar = (
        combined.groupby("_date", as_index=False)
        .agg(**{column: (column, mode_deterministic) for column in calendar_columns})
        .sort_values("_date")
    )
    calendar.insert(0, "日期", calendar["_date"].dt.strftime("%Y-%m-%d"))
    calendar.insert(
        1,
        "星期",
        calendar["_date"].map(lambda value: WEEKDAY_NAMES[value.weekday()]),
    )
    calendar = calendar.drop(columns="_date")
    write_csv(calendar, DATA_ROOT / "features" / "calendar.csv")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ui_options = {
        "meals": ["早餐", "午餐", "晚餐"],
        "yes_no": [
            {"label": "是", "value": True},
            {"label": "否", "value": False},
        ],
        "school_break_types": sorted(
            {
                "寒假",
                "暑假",
                *calendar["校历假期类型"].dropna().astype(str),
            }
            - {"", "非寒暑假", "nan"}
        ),
        "statutory_holiday_names": sorted(
            set(calendar["法定节假日名称"].dropna().astype(str))
            - {"", "非节假日", "nan"}
        ),
        "weather": {
            "temperature_c": {
                "label": "气温（℃）",
                "unit": "℃",
                "values": numeric_values(-50, 50, 0.1),
            },
            "humidity_pct": {
                "label": "相对湿度（%）",
                "unit": "%",
                "values": numeric_values(0, 100, 1),
            },
            "precipitation_mm": {
                "label": "降水量（毫米）",
                "unit": "毫米",
                "values": numeric_values(0, 300, 0.1),
            },
            "wind_speed_kmh": {
                "label": "10米风速（公里/小时）",
                "unit": "公里/小时",
                "values": numeric_values(0, 150, 0.1),
            },
        },
    }
    (DATA_ROOT / "ui_options.json").write_text(
        json.dumps(ui_options, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    now = datetime.now().astimezone().isoformat()
    manifest = {
        "data_version": "1.0.0",
        "created_at": now,
        "updated_at": now,
        "datasets": manifest_datasets,
        "features": {
            "weather_rows": int(len(weather)),
            "calendar_rows": int(len(calendar)),
            "weather_source": "四份初始化销售数据按日期取中位数",
            "calendar_source": "四份初始化销售数据按日期确定性众数",
        },
        "updates": [],
    }
    (DATA_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (DATA_ROOT / "staging").mkdir(exist_ok=True)
    (DATA_ROOT / "backups").mkdir(exist_ok=True)
    (DATA_ROOT / ".locks").mkdir(exist_ok=True)
    (PROJECT_ROOT / "temp_recommendations").mkdir(exist_ok=True)

    # The backend never writes sales.csv. Read-only mode adds a local safeguard
    # on POSIX; Windows portability relies on repository-level enforcement.
    for dataset_id in DATASET_SOURCES:
        sales_path = DATA_ROOT / "datasets" / dataset_id / "sales.csv"
        try:
            os.chmod(sales_path, 0o444)
        except OSError:
            pass
    print(
        json.dumps(
            {
                "status": "complete",
                "datasets": {
                    key: value["sales_rows"]
                    for key, value in manifest_datasets.items()
                },
                "weather_rows": len(weather),
                "calendar_rows": len(calendar),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    initialise(parser.parse_args().force)

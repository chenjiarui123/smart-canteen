from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.dataset_repository import (  # noqa: E402
    RECOMMENDATION_HISTORY_COLUMNS,
    SALES_COLUMNS,
    DatasetRepository,
)
from app.context_builder import WEEKDAY_NAMES  # noqa: E402


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


@pytest.fixture()
def temp_repository(tmp_path: Path) -> DatasetRepository:
    data = tmp_path / "data"
    datasets = {
        "a_hupan": "A湖畔智慧餐厅",
        "f_jichuyuan": "F基础园智慧餐厅",
    }
    registry = {"datasets": {}}
    weather_rows = []
    calendar_rows = []
    for dataset_index, (dataset_id, canteen) in enumerate(datasets.items()):
        folder = data / "datasets" / dataset_id
        sales_rows = []
        for day_index, target in enumerate(
            pd.date_range("2026-01-01", periods=35, freq="D")
        ):
            for meal_index, meal in enumerate(["早餐", "午餐", "晚餐"]):
                for dish_index in range(8):
                    sales_rows.append(
                        {
                            "年": target.year,
                            "月": target.month,
                            "日": target.day,
                            "餐次": meal,
                            "星期": WEEKDAY_NAMES[target.weekday()],
                            "是否校历寒暑假": 0,
                            "校历假期类型": "非寒暑假",
                            "是否法定节假日假期": 0,
                            "法定节假日名称": "非节假日",
                            "是否调休工作日": 0,
                            "气温（℃）": 10.0,
                            "相对湿度（%）": 60.0,
                            "降水量（毫米）": 0.0,
                            "10米风速（公里/小时）": 5.0,
                            "菜品名称": f"{dataset_id}_菜{dish_index}",
                            "菜品总销量": float(
                                30
                                + dataset_index * 10
                                + meal_index * 5
                                + dish_index
                                + day_index % 4
                            ),
                            "菜品售卖单位": "份",
                            "订单数量": float(20 + dish_index),
                            "菜品单价": "5元/份",
                            "销售额": float(150 + dish_index * 10),
                            "平均取菜量": 1.0,
                            "食堂名称": canteen,
                        }
                    )
            if dataset_index == 0:
                weather_rows.append(
                    {
                        "日期": target.strftime("%Y-%m-%d"),
                        "气温（℃）": 10.0,
                        "相对湿度（%）": 60.0,
                        "降水量（毫米）": 0.0,
                        "10米风速（公里/小时）": 5.0,
                    }
                )
                calendar_rows.append(
                    {
                        "日期": target.strftime("%Y-%m-%d"),
                        "星期": WEEKDAY_NAMES[target.weekday()],
                        "是否校历寒暑假": 0,
                        "校历假期类型": "非寒暑假",
                        "是否法定节假日假期": 0,
                        "法定节假日名称": "非节假日",
                        "是否调休工作日": 0,
                    }
                )
        next_day = pd.Timestamp("2026-02-05")
        if dataset_index == 0:
            weather_rows.append(
                {
                    "日期": next_day.strftime("%Y-%m-%d"),
                    "气温（℃）": 10.0,
                    "相对湿度（%）": 60.0,
                    "降水量（毫米）": 0.0,
                    "10米风速（公里/小时）": 5.0,
                }
            )
            calendar_rows.append(
                {
                    "日期": next_day.strftime("%Y-%m-%d"),
                    "星期": WEEKDAY_NAMES[next_day.weekday()],
                    "是否校历寒暑假": 0,
                    "校历假期类型": "非寒暑假",
                    "是否法定节假日假期": 0,
                    "法定节假日名称": "非节假日",
                    "是否调休工作日": 0,
                }
            )
        _write(pd.DataFrame(sales_rows)[SALES_COLUMNS], folder / "sales.csv")
        _write(
            pd.DataFrame(columns=SALES_COLUMNS),
            folder / "runtime_actual_sales.csv",
        )
        library = pd.DataFrame(
            [
                {
                    "菜品id": index,
                    "菜品名称": f"{dataset_id}_菜{index}",
                    "类别": "主食" if index == 0 else "荤菜",
                    "口味": "清淡" if index % 2 == 0 else "咸鲜",
                    "特征来源": "测试静态属性",
                }
                for index in range(8)
            ]
        )
        _write(library, folder / "dish_library.csv")
        _write(
            pd.DataFrame(columns=RECOMMENDATION_HISTORY_COLUMNS),
            folder / "recommendation_history.csv",
        )
        registry["datasets"][dataset_id] = {
            "display_name": canteen,
            "directory": f"datasets/{dataset_id}",
            "aliases": [],
        }
    _write(pd.DataFrame(weather_rows), data / "features" / "weather.csv")
    _write(pd.DataFrame(calendar_rows), data / "features" / "calendar.csv")
    (data / "dataset_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False), encoding="utf-8"
    )
    values = {
        "temperature_c": {"label": "气温（℃）", "unit": "℃", "values": [10.0]},
        "humidity_pct": {"label": "相对湿度（%）", "unit": "%", "values": [60.0]},
        "precipitation_mm": {
            "label": "降水量（毫米）",
            "unit": "毫米",
            "values": [0.0, 1.0],
        },
        "wind_speed_kmh": {"label": "风速", "unit": "公里/小时", "values": [5.0]},
    }
    (data / "ui_options.json").write_text(
        json.dumps(
            {
                "meals": ["早餐", "午餐", "晚餐"],
                "yes_no": [
                    {"label": "是", "value": True},
                    {"label": "否", "value": False},
                ],
                "school_break_types": ["寒假", "暑假"],
                "statutory_holiday_names": ["元旦"],
                "weather": values,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "data_version": "test",
                "updated_at": "2026-01-01T00:00:00+08:00",
                "datasets": {
                    key: {
                        "updated_at": "2026-01-01T00:00:00+08:00",
                        "runtime_actual_sales_added_rows": 0,
                    }
                    for key in datasets
                },
                "updates": [],
            }
        ),
        encoding="utf-8",
    )
    return DatasetRepository(data)

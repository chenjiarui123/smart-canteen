from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from openpyxl import Workbook

from app.dataset_repository import SALES_COLUMNS, safe_filename
from app.preprocessor import (
    parse_upload,
    preprocess_calendar,
    preprocess_sales,
    preprocess_weather,
)


def _new_sales_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "年": 2026,
                "月": 2,
                "日": 5,
                "餐次": "午餐",
                "星期": "星期四",
                "是否校历寒暑假": 0,
                "校历假期类型": "非寒暑假",
                "是否法定节假日假期": 0,
                "法定节假日名称": "非节假日",
                "是否调休工作日": 0,
                "气温（℃）": 10.0,
                "相对湿度（%）": 60.0,
                "降水量（毫米）": 0.0,
                "10米风速（公里/小时）": 5.0,
                "菜品名称": "a_hupan_菜0",
                "菜品总销量": 42.0,
                "菜品售卖单位": "份",
                "订单数量": 30.0,
                "菜品单价": "5元/份",
                "销售额": 210.0,
                "平均取菜量": 1.0,
                "食堂名称": "A湖畔智慧餐厅",
            }
        ]
    )[SALES_COLUMNS]


def test_csv_and_xlsx_upload_and_formula_rejection(temp_repository):
    row = _new_sales_row()
    csv_bytes = row.to_csv(index=False).encode("utf-8-sig")
    parsed = parse_upload("销售.csv", csv_bytes, None)
    assert len(parsed.frame) == 1

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "销售"
    sheet.append(list(row.columns))
    sheet.append(list(row.iloc[0]))
    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()
    parsed_xlsx = parse_upload("销售.xlsx", stream.getvalue(), "销售")
    assert len(parsed_xlsx.frame) == 1

    workbook = Workbook()
    sheet = workbook.active
    sheet["A1"] = "=1+1"
    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()
    with pytest.raises(ValueError, match="公式"):
        parse_upload("公式.xlsx", stream.getvalue(), None)


def test_sales_preprocess_merges_context_and_is_idempotent(temp_repository):
    row = _new_sales_row()
    artifact = preprocess_sales(
        row, "a_hupan", "新增.csv", temp_repository
    )
    assert artifact.can_commit
    assert artifact.summary["valid_rows"] == 1
    assert artifact.processed.iloc[0]["气温（℃）"] == 10.0
    added, ignored, _ = temp_repository.append_runtime_sales(
        "a_hupan", artifact.processed, "新增.csv"
    )
    assert (added, ignored) == (1, 0)
    added_again, ignored_again, _ = temp_repository.append_runtime_sales(
        "a_hupan", artifact.processed, "新增.csv"
    )
    assert (added_again, ignored_again) == (0, 1)


def test_concurrent_same_business_key_is_written_once(temp_repository):
    row = _new_sales_row()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: temp_repository.append_runtime_sales(
                    "a_hupan", row, "并发.csv"
                ),
                range(2),
            )
        )
    assert sum(item[0] for item in results) == 1
    runtime = pd.read_csv(
        temp_repository.dataset_path(
            "a_hupan", "runtime_actual_sales.csv"
        ),
        encoding="utf-8-sig",
    )
    assert len(runtime) == 1


def test_atomic_write_failure_rolls_back(temp_repository, monkeypatch):
    path = temp_repository.dataset_path(
        "a_hupan", "runtime_actual_sales.csv"
    )
    original = path.read_bytes()
    real_replace = os.replace

    def fail_replace(source, destination):
        if str(destination).endswith("runtime_actual_sales.csv"):
            raise OSError("simulated")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        temp_repository.atomic_write_csv(_new_sales_row(), path)
    assert path.read_bytes() == original


def test_cross_platform_filename_sanitisation():
    assert safe_filename('../../A:B*?.csv') == 'A_B__.csv'


def test_weather_and_calendar_preview_never_overwrite_conflicts(
    temp_repository,
):
    weather = pd.DataFrame(
        [
            {
                "日期": "2026-01-01",
                "气温（℃）": 10.0,
                "相对湿度（%）": 60.0,
                "降水量（毫米）": 1.0,
                "10米风速（公里/小时）": 5.0,
            },
            {
                "日期": "2026-02-06",
                "气温（℃）": 10.0,
                "相对湿度（%）": 60.0,
                "降水量（毫米）": 0.0,
                "10米风速（公里/小时）": 5.0,
            },
        ]
    )
    weather_artifact = preprocess_weather(
        weather, "天气.csv", temp_repository
    )
    assert weather_artifact.summary["conflict_dates"] == ["2026-01-01"]
    assert weather_artifact.summary["new_dates"] == ["2026-02-06"]
    assert list(weather_artifact.processed["日期"]) == ["2026-02-06"]

    calendar = pd.DataFrame(
        [
            {
                "日期": "2026-01-01",
                "星期": "星期四",
                "是否校历寒暑假": 1,
                "校历假期类型": "寒假",
                "是否法定节假日假期": 0,
                "法定节假日名称": "非节假日",
                "是否调休工作日": 0,
            },
            {
                "日期": "2026-02-06",
                "星期": "星期五",
                "是否校历寒暑假": 0,
                "校历假期类型": "非寒暑假",
                "是否法定节假日假期": 0,
                "法定节假日名称": "非节假日",
                "是否调休工作日": 0,
            },
        ]
    )
    calendar_artifact = preprocess_calendar(
        calendar, "校历.xlsx", temp_repository
    )
    assert calendar_artifact.summary["conflict_dates"] == ["2026-01-01"]
    assert calendar_artifact.summary["new_dates"] == ["2026-02-06"]
    assert list(calendar_artifact.processed["日期"]) == ["2026-02-06"]

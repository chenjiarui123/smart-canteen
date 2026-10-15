from __future__ import annotations

from datetime import date
from typing import Literal, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import FileResponse

from ..context_builder import weekday_name
from ..dataset_repository import DatasetRepository, sha256_file
from ..history_store import HistoryStore
from ..low_frequency_recommender import load_low_frequency_config
from ..paths import FRONTEND_DIST
from ..recommender import RecommenderService
from ..schemas import (
    CommitRequest,
    CommitResponse,
    ContextResponse,
    HealthResponse,
    PreviewResponse,
    RecommendationFinalizeRequest,
    RecommendationRequest,
    RecommendationResponse,
)
from ..updater import DataUpdateService
from ..weather_display import round_weather_integer, normalized_weather_options


router = APIRouter(prefix="/api")
repository = DatasetRepository()
history_store = HistoryStore(repository)
recommender = RecommenderService(repository, history_store)
updater = DataUpdateService(repository)


@router.get("/health", response_model=HealthResponse)
def health() -> dict:
    statuses = {}
    status = "ok"
    manifest = repository.manifest()
    for dataset_id, item in repository.registry().items():
        files = {}
        for filename in [
            "sales.csv",
            "runtime_actual_sales.csv",
            "dish_library.csv",
            "recommendation_history.csv",
        ]:
            path = repository.dataset_path(dataset_id, filename)
            files[filename] = path.exists()
            if not path.exists():
                status = "degraded"
        expected = manifest["datasets"][dataset_id]
        base_integrity = (
            sha256_file(repository.dataset_path(dataset_id, "sales.csv"))
            == expected["sales_sha256"]
        )
        library_integrity = (
            sha256_file(
                repository.dataset_path(dataset_id, "dish_library.csv")
            )
            == expected["dish_library_sha256"]
        )
        if not base_integrity or not library_integrity:
            status = "degraded"
        statuses[dataset_id] = {
            "display_name": item["display_name"],
            "files": files,
            "read_only_sales_integrity": base_integrity,
            "dish_library_integrity": library_integrity,
        }
    return {
        "status": status,
        "model": recommender.status(),
        "datasets": statuses,
        "frontend_built": (FRONTEND_DIST / "index.html").exists(),
    }


@router.get("/options")
def options() -> dict:
    ui = repository.ui_options()
    festival_names = list(
        load_low_frequency_config()["festival_dish_patterns"].keys()
    )
    statutory_holiday_names = list(
        dict.fromkeys(list(ui["statutory_holiday_names"]) + festival_names)
    )
    datasets = [
        {
            "id": dataset_id,
            "label": item["display_name"],
            "canonical_canteen": item["display_name"],
        }
        for dataset_id, item in repository.registry().items()
    ]
    return {
        "datasets": datasets,
        "meals": ui["meals"],
        "dish_categories": ["全部", "荤菜", "素菜", "主食", "甜品", "水果"],
        "yes_no": ui["yes_no"],
        "school_break_types": ui["school_break_types"],
        "statutory_holiday_names": statutory_holiday_names,
        "weather": normalized_weather_options(ui["weather"]),
    }


@router.get("/context/{context_date}", response_model=ContextResponse)
def context(context_date: date) -> dict:
    date_text = context_date.isoformat()
    weather = repository.read_weather()
    calendar = repository.read_calendar()
    weather_row = weather.loc[weather["日期"].astype(str).eq(date_text)]
    calendar_row = calendar.loc[calendar["日期"].astype(str).eq(date_text)]
    weather_payload = None
    if not weather_row.empty:
        row = weather_row.iloc[0]
        weather_payload = {
            "temperature_c": round_weather_integer(row["气温（℃）"]),
            "humidity_pct": int(round(float(row["相对湿度（%）"]))),
            "precipitation_mm": round_weather_integer(
                row["降水量（毫米）"]
            ),
            "wind_speed_kmh": round_weather_integer(
                row["10米风速（公里/小时）"]
            ),
        }
    calendar_payload = None
    if not calendar_row.empty:
        row = calendar_row.iloc[0]
        calendar_payload = {
            "school_break": bool(int(row["是否校历寒暑假"])),
            "school_break_type": str(row["校历假期类型"]),
            "statutory_holiday": bool(int(row["是否法定节假日假期"])),
            "statutory_holiday_name": str(row["法定节假日名称"]),
            "adjusted_workday": bool(int(row["是否调休工作日"])),
        }
    return {
        "date": context_date,
        "weekday": weekday_name(pd.Timestamp(context_date)),
        "found_weather": weather_payload is not None,
        "found_calendar": calendar_payload is not None,
        "weather": weather_payload,
        "calendar": calendar_payload,
    }


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> dict:
    return recommender.recommend(request)


@router.post("/recommend/finalize", response_model=RecommendationResponse)
def finalize_recommendation(
    request: RecommendationFinalizeRequest,
) -> dict:
    return recommender.finalize(request)


async def _preview(
    kind: str,
    file: UploadFile,
    dataset_id: Optional[str],
    sheet_name: Optional[str],
) -> dict:
    content = await file.read()
    try:
        return updater.preview(
            kind=kind,
            filename=file.filename or "upload",
            content=content,
            dataset_id=dataset_id,
            sheet_name=sheet_name,
        )
    finally:
        await file.close()


@router.post("/data/preview", response_model=PreviewResponse)
async def data_preview(
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    sheet_name: Optional[str] = Form(default=None),
) -> dict:
    return await _preview("sales", file, dataset_id, sheet_name)


@router.post("/data/commit", response_model=CommitResponse)
def data_commit(request: CommitRequest) -> dict:
    if not request.dataset_id:
        raise ValueError("销售数据确认写入必须指定逻辑食堂")
    return updater.commit(request.preview_token, request.dataset_id)


@router.post("/weather/preview", response_model=PreviewResponse)
async def weather_preview(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(default=None),
) -> dict:
    return await _preview("weather", file, None, sheet_name)


@router.post("/weather/commit", response_model=CommitResponse)
def weather_commit(request: CommitRequest) -> dict:
    return updater.commit(request.preview_token, None)


@router.post("/calendar/preview", response_model=PreviewResponse)
async def calendar_preview(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(default=None),
) -> dict:
    return await _preview("calendar", file, None, sheet_name)


@router.post("/calendar/commit", response_model=CommitResponse)
def calendar_commit(request: CommitRequest) -> dict:
    return updater.commit(request.preview_token, None)


@router.get("/recommendations/{recommendation_id}/download")
def download_recommendation(
    recommendation_id: str,
    format: Literal["csv", "md"] = Query(default="csv"),
) -> FileResponse:
    path = history_store.resolve_download(recommendation_id, format)
    media_type = (
        "text/csv; charset=utf-8"
        if format == "csv"
        else "text/markdown; charset=utf-8"
    )
    return FileResponse(
        path=path,
        media_type=media_type,
        filename=path.name,
    )

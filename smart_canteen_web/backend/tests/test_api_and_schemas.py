from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app
from app.schemas import RecommendationRequest


def test_weather_request_rejects_string_values():
    with pytest.raises(ValidationError):
        RecommendationRequest(
            dataset_id="a_hupan",
            canteen="A湖畔智慧餐厅",
            prediction_date=date(2026, 7, 29),
            meal="午餐",
            school_break=False,
            statutory_holiday=False,
            adjusted_workday=False,
            temperature_c="20.0",
            humidity_pct=60,
            precipitation_mm=0,
            wind_speed_kmh=5,
        )


@pytest.mark.parametrize("value", [0, 61, 1.5, "12", None])
def test_low_frequency_count_requires_integer_between_one_and_sixty(value):
    with pytest.raises(ValidationError):
        RecommendationRequest(
            dataset_id="a_hupan",
            canteen="A湖畔智慧餐厅",
            prediction_date=date(2026, 7, 29),
            meal="午餐",
            school_break=False,
            statutory_holiday=False,
            adjusted_workday=False,
            temperature_c=20,
            humidity_pct=60,
            precipitation_mm=0,
            wind_speed_kmh=5,
            low_frequency_count=value,
        )


def test_health_options_context_and_vue_fallback():
    app = create_app()
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    options = client.get("/api/options")
    assert options.status_code == 200
    assert [item["id"] for item in options.json()["datasets"]] == [
        "a_hupan",
        "f_jichuyuan",
    ]
    assert options.json()["dish_categories"] == [
        "全部", "荤菜", "素菜", "主食", "甜品", "水果"
    ]
    for key in ["temperature_c", "precipitation_mm", "wind_speed_kmh"]:
        values = options.json()["weather"][key]["values"]
        assert values == sorted(set(values))
        assert all(isinstance(value, int) for value in values)
    context = client.get("/api/context/2026-01-01")
    assert context.status_code == 200
    assert context.json()["weekday"] == "星期四"
    weather = context.json().get("weather") or {}
    for key in ["temperature_c", "precipitation_mm", "wind_speed_kmh"]:
        if key in weather:
            assert isinstance(weather[key], int)
    assert client.get("/").status_code == 200
    assert client.get("/data-update").status_code == 200
    missing_api = client.get("/api/not-found")
    assert missing_api.status_code == 404
    assert "text/html" not in missing_api.headers.get("content-type", "")

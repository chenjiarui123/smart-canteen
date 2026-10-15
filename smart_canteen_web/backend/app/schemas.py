from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


Meal = Literal["早餐", "午餐", "晚餐"]
LowFrequencyCategory = Literal["全部", "荤菜", "素菜", "主食", "甜品", "水果"]


class RecommendationRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=64)
    canteen: str = Field(min_length=1, max_length=64)
    prediction_date: date
    meal: Meal
    school_break: bool
    school_break_type: Optional[str] = Field(default=None, max_length=32)
    statutory_holiday: bool
    statutory_holiday_name: Optional[str] = Field(default=None, max_length=32)
    adjusted_workday: bool
    temperature_c: float
    humidity_pct: float
    precipitation_mm: float
    wind_speed_kmh: float
    opportunity_refresh: int = Field(default=0, ge=0, le=20)
    low_frequency_category: LowFrequencyCategory = "全部"
    low_frequency_count: int = Field(default=12, strict=True, ge=1, le=60)
    previous_proposal_id: Optional[str] = Field(
        default=None, pattern=r"^[0-9a-f]{32}$"
    )
    selected_low_frequency_dishes: list[str] = Field(
        default_factory=list, max_length=10000
    )

    @field_validator(
        "temperature_c",
        "humidity_pct",
        "precipitation_mm",
        "wind_speed_kmh",
        mode="before",
    )
    @classmethod
    def weather_must_be_json_number(cls, value: Any) -> Any:
        if isinstance(value, (str, bool)) or value is None:
            raise ValueError("天气字段必须使用数值下拉选项")
        return value

    @field_validator("selected_low_frequency_dishes")
    @classmethod
    def retained_names_are_controlled(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value]
        if any(not item or len(item) > 128 for item in cleaned):
            raise ValueError("已选潜力菜包含空值或超长名称")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("已选潜力菜不能重复")
        return cleaned

    @model_validator(mode="after")
    def validate_linked_fields(self) -> "RecommendationRequest":
        if self.school_break and not self.school_break_type:
            raise ValueError("校历寒暑假为“是”时必须选择假期类型")
        if not self.school_break:
            self.school_break_type = "非寒暑假"
        if self.statutory_holiday and not self.statutory_holiday_name:
            raise ValueError("法定节假日为“是”时必须选择节假日名称")
        if not self.statutory_holiday:
            self.statutory_holiday_name = "非节假日"
        return self


class RecommendationItem(BaseModel):
    dish_name: str
    sale_unit: str
    pred_qty: float
    pred_orders: float
    pred_rev: float
    score: float
    dish_category: str
    main_ingredient: str
    cooking_method: str
    flavor: str
    reason: str
    recommendation_reason: str
    recommendation_source: Literal[
        "high_frequency", "low_frequency_opportunity"
    ]
    ordinary_sample_count: Optional[int] = None
    ordinary_avg_qty: Optional[float] = None
    matching_holiday_sample_count: Optional[int] = None
    matching_holiday_avg_qty: Optional[float] = None
    holiday_score_used: Optional[bool] = None
    matched_holiday_name: Optional[str] = None
    context_match_score: Optional[float] = None
    selection_penalty: Optional[float] = None
    suggested_trial_qty: Optional[float] = None


class RecommendationSummary(BaseModel):
    canteen: str
    prediction_date: date
    meal: Meal
    history_start: Optional[date]
    history_end: Optional[date]
    recommended_count: int
    pred_total_qty: float
    pred_total_orders: float
    pred_total_rev: float
    weather_impact: str
    holiday_impact: str
    final_combined_factor: float
    menu_capacity: int
    constraint_relaxed: bool
    cold_start_warning: Optional[str] = None
    forecast_gap_days: int = 0
    forecast_gap_warning: Optional[str] = None
    high_frequency_count: int = 0
    low_frequency_candidate_count: int = 0
    low_frequency_category: LowFrequencyCategory = "全部"
    low_frequency_requested_count: int = 12
    low_frequency_returned_count: int = 0
    low_frequency_shortage: bool = False
    low_frequency_message: Optional[str] = None
    selected_low_frequency_count: int = 0
    total_preparation_qty: float = 0.0
    redistributed_to_total_demand: bool = False
    menu_count_reference_exceeded: bool = False
    selection_status: Literal["待确认", "已确认"] = "待确认"
    adjacent_status: str = "missing_previous"
    adjacent_repeat_rate: Optional[float] = None
    adjacent_reference_source: Optional[str] = None
    adjacent_warning: Optional[str] = None
    replacement_count: int = 0
    replacement_shortage: int = 0
    manual_adjacent_repeat_rate: Optional[float] = None


class RecommendationFinalizeRequest(BaseModel):
    proposal_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    selected_high_frequency: list[str] = Field(max_length=10000)
    selected_low_frequency: list[str] = Field(max_length=10000)
    redistribute_to_total_demand: bool = False

    @field_validator("selected_high_frequency", "selected_low_frequency")
    @classmethod
    def selected_names_are_controlled(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value]
        if any(not item or len(item) > 128 for item in cleaned):
            raise ValueError("菜品选择包含空值或超长名称")
        return cleaned


class RecommendationResponse(BaseModel):
    status: Literal["draft", "finalized"]
    proposal_id: str
    recommendation_id: Optional[str]
    summary: RecommendationSummary
    menu: list[RecommendationItem]
    high_frequency_menu: list[RecommendationItem]
    low_frequency_opportunities: list[RecommendationItem]
    low_frequency_status: Literal["ok", "unavailable"]
    low_frequency_warning: Optional[str] = None
    combination_check: Optional[dict[str, Any]] = None
    excluded_candidates: list[dict[str, str]]
    csv_download_url: Optional[str]
    markdown_download_url: Optional[str]


class ContextResponse(BaseModel):
    date: date
    weekday: str
    found_weather: bool
    found_calendar: bool
    weather: Optional[dict[str, int | float]] = None
    calendar: Optional[dict[str, Any]] = None


class PreviewResponse(BaseModel):
    preview_token: Optional[str] = None
    kind: str
    dataset_id: Optional[str] = None
    requires_sheet_selection: bool = False
    sheet_names: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    can_commit: bool = False


class CommitRequest(BaseModel):
    preview_token: str = Field(min_length=16, max_length=64)
    dataset_id: Optional[str] = Field(default=None, max_length=64)


class CommitResponse(BaseModel):
    status: Literal["success"]
    kind: str
    added_rows: int
    ignored_duplicate_rows: int
    backup_created: bool
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model: dict[str, Any]
    datasets: dict[str, Any]
    frontend_built: bool

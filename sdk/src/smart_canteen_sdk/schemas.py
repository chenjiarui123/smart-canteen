"""Version 1 public contracts. Quantities are in business units, never UI units."""
from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128)]
Meal = Literal["早餐", "午餐", "晚餐"]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Weather(Contract):
    temperature_c: float = Field(ge=-80, le=65)
    humidity_pct: float = Field(ge=0, le=100)
    precipitation_mm: NonNegative
    wind_speed_kmh: NonNegative


class Calendar(Contract):
    school_break: bool
    school_break_type: str = "非寒暑假"
    statutory_holiday: bool
    statutory_holiday_name: str = "非节假日"
    adjusted_workday: bool

    @model_validator(mode="after")
    def check_names(self):
        if self.school_break and self.school_break_type not in {"寒假", "暑假"}:
            raise ValueError("school_break 为 true 时需指定寒假或暑假")
        if self.statutory_holiday and self.statutory_holiday_name in {"", "非节假日"}:
            raise ValueError("法定节假日必须指定名称")
        if not self.school_break:
            self.school_break_type = "非寒暑假"
        if not self.statutory_holiday:
            self.statutory_holiday_name = "非节假日"
        return self


class Opportunities(Contract):
    enabled: bool = True
    category: Literal["全部", "荤菜", "素菜", "主食", "甜品", "水果"] = "全部"
    count: int = Field(default=12, strict=True, ge=1, le=60)
    refresh_seed: int = Field(default=0, strict=True, ge=0, le=2147483647)


class RecommendationRequest(Contract):
    canteen_id: Identifier
    prediction_date: date
    meal: Meal
    weather: Weather
    calendar: Calendar
    opportunities: Opportunities = Field(default_factory=Opportunities)


class Dish(Contract):
    dish_id: Identifier
    dish_name: Identifier
    category: str = "其他"
    flavor: str = "unknown"
    main_ingredient: str | None = None
    ingredient_family: str | None = None
    cooking_method: str | None = None
    similarity_group: str | None = None
    staple: bool | None = None
    attributes_available: bool = True


class SaleRecord(Contract):
    date: date
    canteen_id: Identifier
    meal: Meal
    dish_id: Identifier
    quantity: NonNegative
    unit: Identifier
    order_count: NonNegative
    revenue: NonNegative
    unit_price: str = ""
    average_portion: NonNegative = 1.0


class HistoricalContext(Contract):
    date: date
    canteen_id: Identifier
    meal: Meal | None = None
    weather: Weather
    calendar: Calendar


class MenuHistory(Contract):
    date: date
    canteen_id: Identifier
    meal: Meal
    menu_id: Identifier
    created_at: datetime
    source: Literal["automatic", "finalized"]
    dish_ids: list[Identifier]
    low_frequency_dish_ids: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ids(self):
        if self.created_at.tzinfo is None:
            raise ValueError("created_at 必须包含时区")
        if len(self.dish_ids) != len(set(self.dish_ids)):
            raise ValueError("历史菜单有重复菜品")
        if not set(self.low_frequency_dish_ids) <= set(self.dish_ids):
            raise ValueError("历史潜力菜必须属于历史菜单")
        return self


class DataSnapshot(Contract):
    snapshot_id: Identifier
    canteen_id: Identifier
    canteen_name: Identifier | None = None
    unit_policy: Literal["strict", "legacy_compatibility"] = "strict"
    sales_history: list[SaleRecord] = Field(min_length=1)
    dish_catalog: list[Dish] = Field(min_length=1)
    historical_context: list[HistoricalContext] = Field(min_length=1)
    menu_history: list[MenuHistory] = Field(default_factory=list)


class Selection(Contract):
    high_frequency_dish_ids: list[Identifier]
    low_frequency_dish_ids: list[Identifier] = Field(default_factory=list)
    redistribute_to_total_demand: bool = False


class MenuItem(Contract):
    dish_id: str
    dish_name: str
    unit: str
    quantity: NonNegative
    quantity_kind: Literal["expected_demand", "trial_preparation"]
    expected_demand_qty: NonNegative | None
    suggested_trial_qty: NonNegative | None = None
    reference_orders: NonNegative
    reference_revenue: NonNegative
    score: float
    category: str
    recommendation_source: Literal["high_frequency", "low_frequency_opportunity"]
    reason: str
    model_quantity: NonNegative
    business_units_per_model_unit: float


class PublicProposal(Contract):
    schema_version: Literal["1.0"] = "1.0"
    sdk_version: str = "0.1.0"
    strategy_version: str
    proposal_id: str
    snapshot_id: str
    created_at: datetime
    status: Literal["draft"] = "draft"
    request: RecommendationRequest
    high_frequency_menu: list[MenuItem]
    low_frequency_opportunities: list[MenuItem]
    summary: dict[str, Any]
    audit: dict[str, Any]
    history_events: list[MenuHistory]


class Proposal(PublicProposal):
    state: dict[str, Any] = Field(repr=False)
    state_sha256: str

    def public_dict(self):
        """Frontend payload; keep the complete object in trusted backend storage."""
        return self.model_dump(mode="json", exclude={"state", "state_sha256"})


class FinalResult(Contract):
    schema_version: Literal["1.0"] = "1.0"
    sdk_version: str = "0.1.0"
    strategy_version: str
    proposal_id: str
    snapshot_id: str
    status: Literal["ready"] = "ready"
    menu: list[MenuItem]
    summary: dict[str, Any]
    audit: dict[str, Any]
    history_events: list[MenuHistory]

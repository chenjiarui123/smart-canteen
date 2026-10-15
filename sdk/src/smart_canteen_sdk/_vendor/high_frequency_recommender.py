"""High-frequency strategy adapter with explicit adjacent-day feasibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .dataset_repository import normalize_dish_name
from .model_loader import (
    constraints_copy,
    frozen_evaluator,
    frozen_policy_copy,
    make_recommendations_with_history,
)
from .report_generator import build_excluded_candidates, factor_impact_text
from .active_strategy import active_config, generate
from .recommendation_copy import high_frequency_reason


@dataclass
class HighFrequencyResult:
    recommendations: pd.DataFrame
    summary: dict[str, Any]
    excluded_candidates: list[dict[str, str]]
    selection_audit: dict[str, Any]


class HighFrequencyRecommender:
    """Run the integrity-checked active strategy and decorate its results."""

    def generate(
        self,
        *,
        context: dict[str, Any],
        history: pd.DataFrame,
        target_date: pd.Timestamp,
        canteen: str,
        meal: str,
        initial_menu_history: dict,
        prior_norms: set[str],
        recent7: dict[str, int],
        recent14: dict[str, int],
        sale_units: dict[str, str],
    ) -> HighFrequencyResult:
        selection_audit: list[dict[str, Any]] = []
        recommendations = generate(
            [context],
            active_config(),
            constraints_copy(),
            selection_audit=selection_audit,
            initial_menu_history=initial_menu_history,
        )
        if recommendations.empty:
            audit = selection_audit[0] if selection_audit else {}
            raise ValueError(audit.get("infeasible_reason") or "策略没有生成可用菜单")
        recommendations = recommendations.copy()
        recommendations["_dish_norm"] = recommendations["菜品名称"].map(
            normalize_dish_name
        )
        recommendations["菜品售卖单位"] = recommendations["_dish_norm"].map(
            sale_units
        ).fillna("份")
        calendar_factor = float(recommendations["calendar_factor"].iloc[0])
        weather_factor = float(
            recommendations["weather_residual_factor"].iloc[0]
        )
        past = history.loc[
            history["date"].lt(target_date)
            & history["食堂名称"].eq(canteen)
            & history["餐次"].eq(meal)
        ].drop_duplicates(["date", "_dish_norm"])
        history_counts = past.groupby("_dish_norm").size().to_dict()
        recent_counts = past.loc[
            past["date"].ge(target_date - pd.Timedelta(days=30))
        ].groupby("_dish_norm").size().to_dict()
        for index, row in recommendations.iterrows():
            dish = str(row["_dish_norm"])
            recommendations.at[index, "reason"] = high_frequency_reason(
                recent_count=int(recent_counts.get(dish, 0)),
                history_count=int(history_counts.get(dish, 0)),
                appeared_last=dish in prior_norms,
                recommended7=int(recent7.get(dish, 0)),
                recommended14=int(recent14.get(dish, 0)),
                weather_factor=weather_factor,
                calendar_factor=calendar_factor,
            )
        # The new source label is audit metadata only; no frozen prediction,
        # candidate, score, or selection value is recalculated here.
        recommendations["recommendation_source"] = "high_frequency"
        recommendations["recommendation_reason"] = recommendations["reason"]

        violations = frozen_evaluator.constraint_violations(
            frozen_evaluator.prepare_recommendations(recommendations),
            {"constraints": constraints_copy()},
        )
        if violations["total_constraint_violation_count"] != 0:
            raise RuntimeError("当前策略输出出现硬约束违规")
        allocated = float(recommendations["pred_qty"].sum())
        target_total = float(recommendations["pred_total_qty"].iloc[0])
        if abs(allocated - target_total) > max(
            1e-6, abs(target_total) * 1e-9
        ):
            raise RuntimeError("推荐菜品销量之和不等于餐次预测总量")

        audit = selection_audit[0] if selection_audit else {}
        excluded = build_excluded_candidates(
            audit, recommendations, context["features"]
        )
        local_sessions = int(context["meta"].get("l_session_count", 0))
        cold_warning = None
        if local_sessions < int(frozen_policy_copy()["candidate_warmup_sessions"]):
            cold_warning = (
                f"目标餐次仅有{local_sessions}个历史餐次，候选池处于冷启动回退阶段"
            )
        forecast_gap_days = int(
            context.get(
                "forecast_gap_days",
                max(
                    (
                        pd.Timestamp(target_date).normalize()
                        - pd.Timestamp(history["date"].max()).normalize()
                    ).days,
                    0,
                ),
            )
        )
        recent_window = int(frozen_policy_copy()["recent_window"])
        forecast_gap_warning = None
        if forecast_gap_days > recent_window:
            forecast_gap_warning = (
                f"远期预测：预测日距离历史末日{forecast_gap_days}天，"
                "近期特征不足，已通过自适应近期权重增加长期历史占比"
            )
        summary = {
            "canteen": str(canteen),
            "prediction_date": target_date.date().isoformat(),
            "meal": str(meal),
            "history_start": history["date"].min().date().isoformat(),
            "history_end": history["date"].max().date().isoformat(),
            "recommended_count": int(len(recommendations)),
            "pred_total_qty": allocated,
            "pred_total_orders": float(recommendations["pred_orders"].sum()),
            "pred_total_rev": float(recommendations["pred_rev"].sum()),
            "weather_impact": factor_impact_text("天气", weather_factor),
            "holiday_impact": factor_impact_text(
                "校历/节假日", calendar_factor
            ),
            "final_combined_factor": float(
                recommendations["final_combined_factor"].iloc[0]
            ),
            "menu_capacity": int(
                recommendations["planned_menu_capacity"].iloc[0]
            ),
            "constraint_relaxed": False,
            "cold_start_warning": cold_warning,
            "forecast_gap_days": forecast_gap_days,
            "forecast_gap_warning": forecast_gap_warning,
            "adjacent_status": audit.get("adjacent_status", "missing_previous"),
            "adjacent_repeat_rate": audit.get("adjacent_repeat_rate"),
            "replacement_count": audit.get("replacement_count", 0),
            "replacement_shortage": audit.get("replacement_shortage", 0),
            "adjacent_warning": (
                "缺少前一天同餐次菜单对照，相邻日重复率不可计算。"
                if audit.get("adjacent_status") == "missing_previous" else None
            ),
        }
        return HighFrequencyResult(
            recommendations=recommendations,
            summary=summary,
            excluded_candidates=excluded,
            selection_audit=audit,
        )

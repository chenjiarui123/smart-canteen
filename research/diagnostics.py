#!/usr/bin/env python3
"""Auxiliary diagnostics for policy search; fixed evaluator metrics stay untouched."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from evaluator import GROUP_KEYS, normalize_name, prepare_actual, prepare_recommendations


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _audit_key(row: pd.Series | dict[str, Any]) -> tuple[pd.Timestamp, str, str]:
    return (
        pd.Timestamp(row["date"]),
        str(row["食堂名称"]),
        str(row["餐次"]),
    )


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def selection_diagnostics(
    recommendations: pd.DataFrame,
    actual: pd.DataFrame,
    selection_audit: list[dict[str, Any]],
    high_revenue_quantile: float = 0.8,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Explain fill, candidate recall, and why realised high-revenue dishes were missed."""

    recs = prepare_recommendations(recommendations)
    realised = prepare_actual(actual)
    audit_lookup = {_audit_key(row): row for row in selection_audit}
    selected_lookup = {
        key: set(group["_dish_norm"].astype(str))
        for key, group in recs.groupby(GROUP_KEYS, sort=False)
    }

    fill_rates = []
    for row in selection_audit:
        capacity = int(row.get("capacity", 0))
        fill_rates.append(_safe_ratio(int(row.get("selected_count", 0)), capacity))

    missed_rows: list[dict[str, Any]] = []
    high_total = 0
    high_candidate_hits = 0
    high_selected_hits = 0
    constraint_excluded_revenue = 0.0
    for key, group in realised.groupby(GROUP_KEYS, sort=False):
        key = (pd.Timestamp(key[0]), str(key[1]), str(key[2]))
        audit = audit_lookup.get(key, {})
        candidate_pool = set(audit.get("candidate_pool", set()))
        hard_excluded = set(audit.get("hard_excluded", set()))
        selected = selected_lookup.get(key, set())
        capacity = int(audit.get("capacity", 0))
        selected_count = int(audit.get("selected_count", len(selected)))
        threshold = float(group["actual_rev"].quantile(high_revenue_quantile))
        high_mask = group["actual_rev"].ge(threshold) & group["actual_rev"].gt(0)
        for row in group.to_dict("records"):
            dish = str(row["_dish_norm"])
            actual_rev = float(row["actual_rev"])
            is_high = bool(actual_rev >= threshold and actual_rev > 0)
            if dish in hard_excluded and dish not in selected:
                constraint_excluded_revenue += actual_rev
            if not is_high:
                continue
            high_total += 1
            high_candidate_hits += int(dish in candidate_pool)
            high_selected_hits += int(dish in selected)
            if dish in selected:
                continue
            if dish not in candidate_pool:
                reason = "not_in_candidate_pool"
            elif dish in hard_excluded:
                reason = "hard_constraint"
            elif selected_count >= capacity > 0:
                reason = "capacity"
            else:
                reason = "score_or_soft_penalty"
            missed_rows.append(
                {
                    "date": key[0].strftime("%Y-%m-%d"),
                    "食堂名称": key[1],
                    "餐次": key[2],
                    "菜品名称": row["菜品名称"],
                    "actual_rev": actual_rev,
                    "actual_qty": float(row["actual_qty"]),
                    "candidate_pool": dish in candidate_pool,
                    "hard_excluded": dish in hard_excluded,
                    "selected_count": selected_count,
                    "menu_capacity": capacity,
                    "miss_reason": reason,
                }
            )

    metrics: dict[str, float | int] = {
        "menu_fill_rate": float(np.mean(fill_rates)) if fill_rates else 0.0,
        "min_menu_fill_rate": float(np.min(fill_rates)) if fill_rates else 0.0,
        "high_revenue_candidate_recall": _safe_ratio(high_candidate_hits, high_total),
        "high_revenue_selected_recall": _safe_ratio(high_selected_hits, high_total),
        "constraint_excluded_revenue": float(constraint_excluded_revenue),
        "high_revenue_dish_count": int(high_total),
        "missed_high_revenue_dish_count": int(len(missed_rows)),
    }
    return metrics, pd.DataFrame(missed_rows)


def prediction_consistency_diagnostics(
    recommendations: pd.DataFrame,
) -> dict[str, float]:
    recs = recommendations.copy()
    if recs.empty:
        return {
            "order_consistency_wape": 0.0,
            "revenue_consistency_wape": 0.0,
            "total_allocation_wape": 0.0,
        }
    for column in ["pred_qty", "pred_orders", "pred_rev"]:
        recs[column] = _numeric_series(recs, column)
    expected_orders = recs["pred_qty"] * _numeric_series(
        recs, "expected_orders_per_qty"
    )
    expected_revenue = recs["pred_qty"] * _numeric_series(
        recs, "expected_rev_per_qty"
    )
    metrics = {
        "order_consistency_wape": _safe_ratio(
            float((recs["pred_orders"] - expected_orders).abs().sum()),
            float(recs["pred_orders"].sum()),
        ),
        "revenue_consistency_wape": _safe_ratio(
            float((recs["pred_rev"] - expected_revenue).abs().sum()),
            float(recs["pred_rev"].sum()),
        ),
    }
    if "pred_total_qty" in recs:
        totals = recs.groupby(GROUP_KEYS, as_index=False).agg(
            allocated_qty=("pred_qty", "sum"),
            pred_total_qty=("pred_total_qty", "first"),
        )
        metrics["total_allocation_wape"] = _safe_ratio(
            float((totals["allocated_qty"] - totals["pred_total_qty"]).abs().sum()),
            float(totals["pred_total_qty"].sum()),
        )
    else:
        metrics["total_allocation_wape"] = 0.0
    return metrics


def menu_capacity_diagnostics(
    recommendations: pd.DataFrame,
    actual: pd.DataFrame,
    selection_audit: list[dict[str, Any]],
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Compare the planned/final menu size with the realised positive-sale menu."""

    realised = actual.copy()
    realised["date"] = pd.to_datetime(realised["date"])
    realised["_dish_norm"] = realised["菜品名称"].map(normalize_name)
    actual_qty = pd.to_numeric(
        realised.get("菜品总销量", 0.0), errors="coerce"
    ).fillna(0.0)
    realised = realised.loc[actual_qty.gt(0)].copy()
    realised["actual_qty"] = actual_qty.loc[realised.index]
    context_columns = [
        column
        for column in [
            "_calendar_category",
            "_calendar_state",
            "_day_type",
            "是否调休工作日",
            "是否校历寒暑假",
            "是否法定节假日假期",
        ]
        if column in realised
    ]
    actual_sessions = (
        realised.groupby(GROUP_KEYS, as_index=False)
        .agg(
            actual_menu_count=("_dish_norm", "nunique"),
            actual_total_qty=("actual_qty", "sum"),
            **{
                column: (column, "first")
                for column in context_columns
            },
        )
    )

    recs = recommendations.copy()
    if not recs.empty:
        recs["date"] = pd.to_datetime(recs["date"])
        recs["_dish_norm"] = recs["菜品名称"].map(normalize_name)
        recs["pred_qty"] = _numeric_series(recs, "pred_qty")
        predicted_sessions = (
            recs.groupby(GROUP_KEYS, as_index=False)
            .agg(
                predicted_menu_count=("_dish_norm", "nunique"),
                predicted_total_qty=("pred_qty", "sum"),
                below_min_batch_count=(
                    "pred_qty",
                    lambda values: 0,
                ),
            )
        )
        if "min_batch_qty" in recs:
            below = (
                recs["pred_qty"]
                + 1e-9
                < pd.to_numeric(
                    recs["min_batch_qty"], errors="coerce"
                ).fillna(0.0)
            )
            below_by_session = (
                recs.assign(_below_min_batch=below.astype(int))
                .groupby(GROUP_KEYS, as_index=False)["_below_min_batch"]
                .sum()
                .rename(
                    columns={
                        "_below_min_batch": "below_min_batch_count"
                    }
                )
            )
            predicted_sessions = predicted_sessions.drop(
                columns=["below_min_batch_count"]
            ).merge(below_by_session, on=GROUP_KEYS, how="left")
    else:
        predicted_sessions = pd.DataFrame(
            columns=GROUP_KEYS
            + [
                "predicted_menu_count",
                "predicted_total_qty",
                "below_min_batch_count",
            ]
        )

    audit_rows = []
    for row in selection_audit:
        audit_rows.append(
            {
                "date": pd.Timestamp(row["date"]),
                "食堂名称": str(row["食堂名称"]),
                "餐次": str(row["餐次"]),
                "planned_menu_count": int(
                    row.get(
                        "planned_capacity",
                        row.get("capacity", 0),
                    )
                ),
                "final_audit_menu_count": int(
                    row.get("selected_count", 0)
                ),
                "min_batch_removed_count": int(
                    row.get("min_batch_removed_count", 0)
                ),
                "base_total_qty": float(
                    row.get("base_total_qty", 0.0)
                ),
                "raw_combined_factor": float(
                    row.get("raw_combined_factor", 1.0)
                ),
                "final_combined_factor": float(
                    row.get("final_combined_factor", 1.0)
                ),
                "factor_clip_applied": bool(
                    row.get("factor_clip_applied", False)
                ),
            }
        )
    audit_frame = pd.DataFrame(audit_rows)
    sessions = actual_sessions.merge(
        predicted_sessions, on=GROUP_KEYS, how="left"
    )
    if not audit_frame.empty:
        sessions = sessions.merge(audit_frame, on=GROUP_KEYS, how="left")
    for column in [
        "predicted_menu_count",
        "predicted_total_qty",
        "below_min_batch_count",
        "planned_menu_count",
        "final_audit_menu_count",
        "min_batch_removed_count",
        "base_total_qty",
    ]:
        if column not in sessions:
            sessions[column] = 0.0
        sessions[column] = pd.to_numeric(
            sessions[column], errors="coerce"
        ).fillna(0.0)
    if "raw_combined_factor" not in sessions:
        sessions["raw_combined_factor"] = 1.0
    if "final_combined_factor" not in sessions:
        sessions["final_combined_factor"] = 1.0
    if "factor_clip_applied" not in sessions:
        sessions["factor_clip_applied"] = False

    sessions["menu_count_error"] = (
        sessions["predicted_menu_count"]
        - sessions["actual_menu_count"]
    )
    sessions["menu_count_ape"] = (
        sessions["menu_count_error"].abs()
        / sessions["actual_menu_count"].replace(0, np.nan)
    ).fillna(0.0)
    sessions["total_qty_error"] = (
        sessions["predicted_total_qty"] - sessions["actual_total_qty"]
    )
    calendar_category = sessions.get(
        "_calendar_category",
        pd.Series("普通日期", index=sessions.index),
    ).astype(str)
    holiday_mask = ~calendar_category.isin(["普通日期", "调休工作日"])
    predicted_dishes = float(sessions["predicted_menu_count"].sum())
    metrics: dict[str, float | int] = {
        "menu_count_mae": float(
            sessions["menu_count_error"].abs().mean()
        ),
        "menu_count_mape": float(
            sessions["menu_count_ape"].mean() * 100
        ),
        "menu_count_bias": float(sessions["menu_count_error"].mean()),
        "holiday_menu_count_bias": (
            float(sessions.loc[holiday_mask, "menu_count_error"].mean())
            if holiday_mask.any()
            else 0.0
        ),
        "over_capacity_session_rate": float(
            sessions["menu_count_error"].gt(0).mean()
        ),
        "excess_dish_count": int(
            sessions["menu_count_error"].clip(lower=0).sum()
        ),
        "avg_pred_qty_per_dish": _safe_ratio(
            float(sessions["predicted_total_qty"].sum()),
            predicted_dishes,
        ),
        "below_min_batch_count": int(
            sessions["below_min_batch_count"].sum()
        ),
        "min_batch_removed_count": int(
            sessions["min_batch_removed_count"].sum()
        ),
    }
    return metrics, sessions


def factor_stack_diagnostics(
    recommendations: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarise how often multiple downward factors and clipping are active."""

    if recommendations.empty:
        return {
            "factor_clip_session_rate": 0.0,
            "multi_down_factor_session_rate": 0.0,
            "mean_raw_combined_factor": 1.0,
            "mean_final_combined_factor": 1.0,
            "min_raw_combined_factor": 1.0,
            "min_final_combined_factor": 1.0,
        }
    recs = recommendations.copy()
    recs["date"] = pd.to_datetime(recs["date"])
    sessions = recs.groupby(GROUP_KEYS, as_index=False).first()
    factor_columns = [
        "weekday_total_factor",
        "group_trend_factor",
        "group_momentum_factor",
        "calendar_factor",
        "weather_residual_factor",
    ]
    downward_count = pd.Series(0, index=sessions.index, dtype=int)
    for column in factor_columns:
        if column in sessions:
            downward_count += (
                pd.to_numeric(
                    sessions[column], errors="coerce"
                ).fillna(1.0)
                < 1.0 - 1e-12
            ).astype(int)
    raw = pd.to_numeric(
        sessions.get("raw_combined_factor", 1.0), errors="coerce"
    ).fillna(1.0)
    final = pd.to_numeric(
        sessions.get("final_combined_factor", 1.0), errors="coerce"
    ).fillna(1.0)
    clipped = sessions.get(
        "factor_clip_applied",
        pd.Series(False, index=sessions.index),
    )
    return {
        "factor_clip_session_rate": float(
            pd.Series(clipped).fillna(False).astype(bool).mean()
        ),
        "multi_down_factor_session_rate": float(
            downward_count.ge(2).mean()
        ),
        "mean_raw_combined_factor": float(raw.mean()),
        "mean_final_combined_factor": float(final.mean()),
        "min_raw_combined_factor": float(raw.min()),
        "min_final_combined_factor": float(final.min()),
    }


def state_meal_diagnostics(
    sessions: pd.DataFrame,
    trial_index: int,
) -> list[dict[str, Any]]:
    """Return signed demand/menu biases for every date-state and meal."""

    dimensions = [
        column
        for column in [
            "_calendar_category",
            "_calendar_state",
            "_day_type",
        ]
        if column in sessions
    ]
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        for (state, meal), group in sessions.groupby(
            [dimension, "餐次"], sort=True, dropna=False
        ):
            actual_qty = float(group["actual_total_qty"].sum())
            predicted_qty = float(group["predicted_total_qty"].sum())
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "dimension": dimension,
                    "date_state": str(state),
                    "餐次": str(meal),
                    "session_count": int(len(group)),
                    "actual_total_qty": actual_qty,
                    "predicted_total_qty": predicted_qty,
                    "qty_bias_pct": _safe_ratio(
                        predicted_qty - actual_qty, actual_qty
                    )
                    * 100,
                    "total_mape": float(
                        (
                            group["total_qty_error"].abs()
                            / group["actual_total_qty"].replace(0, np.nan)
                        )
                        .fillna(0.0)
                        .mean()
                        * 100
                    ),
                    "actual_menu_count": float(
                        group["actual_menu_count"].mean()
                    ),
                    "predicted_menu_count": float(
                        group["predicted_menu_count"].mean()
                    ),
                    "menu_count_bias": float(
                        group["menu_count_error"].mean()
                    ),
                }
            )
    return rows


def hit_quantity_diagnostics(
    recommendations: pd.DataFrame,
    actual: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate quantity only where a recommended dish was actually sold."""

    recs = prepare_recommendations(recommendations)
    realised = prepare_actual(actual)
    hits = recs.merge(
        realised[GROUP_KEYS + ["_dish_norm", "actual_qty"]],
        on=GROUP_KEYS + ["_dish_norm"],
        how="inner",
    )
    hits = hits.loc[hits["actual_qty"].gt(0)].copy()
    if hits.empty:
        return {
            "hit_qty_mape": 0.0,
            "hit_qty_wmape": 0.0,
            "hit_qty_median_ape": 0.0,
            "weighted_qty_bias": 0.0,
            "hit_overprediction_rate": 0.0,
            "hit_underprediction_rate": 0.0,
            "hit_item_count": 0,
            "nonhit_pred_qty_share": 0.0,
        }, hits
    hits["qty_error"] = hits["pred_qty"] - hits["actual_qty"]
    hits["abs_qty_error"] = hits["qty_error"].abs()
    hits["ape"] = hits["abs_qty_error"] / hits["actual_qty"]
    hit_keys = set(
        zip(
            hits["date"],
            hits["食堂名称"],
            hits["餐次"],
            hits["_dish_norm"],
        )
    )
    rec_keys = list(
        zip(recs["date"], recs["食堂名称"], recs["餐次"], recs["_dish_norm"])
    )
    nonhit_pred_qty = float(
        recs.loc[[key not in hit_keys for key in rec_keys], "pred_qty"].sum()
    )
    total_pred_qty = float(recs["pred_qty"].sum())
    actual_sum = float(hits["actual_qty"].sum())
    metrics: dict[str, float | int] = {
        "hit_qty_mape": float(hits["ape"].mean() * 100),
        "hit_qty_wmape": _safe_ratio(float(hits["abs_qty_error"].sum()), actual_sum) * 100,
        "hit_qty_median_ape": float(hits["ape"].median() * 100),
        "weighted_qty_bias": _safe_ratio(float(hits["qty_error"].sum()), actual_sum) * 100,
        "hit_overprediction_rate": float(hits["qty_error"].gt(0).mean()),
        "hit_underprediction_rate": float(hits["qty_error"].lt(0).mean()),
        "hit_item_count": int(len(hits)),
        "nonhit_pred_qty_share": _safe_ratio(nonhit_pred_qty, total_pred_qty),
    }
    columns = GROUP_KEYS + [
        "菜品名称",
        "_dish_norm",
        "pred_qty",
        "actual_qty",
        "qty_error",
        "abs_qty_error",
        "ape",
    ]
    return metrics, hits[columns].sort_values("ape", ascending=False).reset_index(drop=True)


def recommendation_repeat_diagnostics(recommendations: pd.DataFrame) -> dict[str, float | int]:
    recs = prepare_recommendations(recommendations)
    if recs.empty:
        return {
            "previous_business_day_repeat_rate": 0.0,
            "repeat_7d_rate": 0.0,
            "repeat_14d_rate": 0.0,
        }
    repeats = Counter()
    eligible = Counter()
    for _, group in recs.groupby(["食堂名称", "餐次"], sort=False):
        history: list[tuple[pd.Timestamp, set[str]]] = []
        for date, day in group.groupby("date", sort=True):
            current = set(day["_dish_norm"].astype(str))
            previous = history[-1][1] if history else set()
            seven = set().union(
                *(dishes for old_date, dishes in history if old_date >= date - pd.Timedelta(days=7))
            ) if history else set()
            fourteen = set().union(
                *(dishes for old_date, dishes in history if old_date >= date - pd.Timedelta(days=14))
            ) if history else set()
            for label, prior in [("prev", previous), ("7d", seven), ("14d", fourteen)]:
                if history:
                    eligible[label] += len(current)
                    repeats[label] += len(current & prior)
            history.append((pd.Timestamp(date), current))
    return {
        "previous_business_day_repeat_rate": _safe_ratio(repeats["prev"], eligible["prev"]),
        "repeat_7d_rate": _safe_ratio(repeats["7d"], eligible["7d"]),
        "repeat_14d_rate": _safe_ratio(repeats["14d"], eligible["14d"]),
    }


def soft_diversity_diagnostics(
    recommendations: pd.DataFrame,
    limits: dict[str, int],
) -> dict[str, int]:
    recs = recommendations.copy()
    totals = {"soft_family_excess": 0, "soft_method_excess": 0, "soft_flavor_excess": 0}
    specs = [
        ("食材大类", "max_same_ingredient_family", "soft_family_excess"),
        ("制作方法", "max_same_cooking_method", "soft_method_excess"),
        ("口味", "max_same_flavor", "soft_flavor_excess"),
    ]
    for _, group in recs.groupby(GROUP_KEYS, sort=False):
        for column, limit_key, metric in specs:
            if column not in group:
                continue
            values = group[column].astype(str)
            values = values.loc[~values.str.lower().isin({"", "unknown", "未知", "nan", "none"})]
            counts = values.value_counts()
            totals[metric] += int((counts - int(limits[limit_key])).clip(lower=0).sum())
    return totals


def transfer_error_diagnostics(
    recommendations: pd.DataFrame,
    actual: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    """MAPE/WAPE/sMAPE by transfer horizon, month, canteen, and meal."""

    recs = prepare_recommendations(recommendations)
    realised = prepare_actual(actual)
    actual_groups = realised.groupby(GROUP_KEYS, as_index=False).agg(actual_qty=("actual_qty", "sum"))
    pred_groups = recs.groupby(GROUP_KEYS, as_index=False).agg(pred_qty=("pred_qty", "sum"))
    groups = actual_groups.merge(pred_groups, on=GROUP_KEYS, how="left").fillna({"pred_qty": 0.0})
    groups["ape"] = (groups["pred_qty"] - groups["actual_qty"]).abs() / groups["actual_qty"].replace(0, np.nan)
    groups["abs_error"] = (groups["pred_qty"] - groups["actual_qty"]).abs()
    groups["smape_item"] = 2 * groups["abs_error"] / (groups["pred_qty"].abs() + groups["actual_qty"].abs()).replace(0, np.nan)
    start = groups["date"].min()
    offset = (groups["date"] - start).dt.days
    groups["horizon"] = np.select(
        [offset.le(6), offset.between(7, 13), offset.between(14, 29)],
        ["first_7_days", "days_8_14", "days_15_30"],
        default="after_30_days",
    )
    partitions: list[tuple[str, str, pd.DataFrame]] = []
    for value, subset in groups.groupby("horizon", sort=False):
        partitions.append(("horizon", str(value), subset))
    for value, subset in groups.groupby(groups["date"].dt.strftime("%Y-%m"), sort=True):
        partitions.append(("month", str(value), subset))
    for value, subset in groups.groupby("食堂名称", sort=True):
        partitions.append(("canteen", str(value), subset))
    for value, subset in groups.groupby("餐次", sort=True):
        partitions.append(("meal", str(value), subset))
    rows = []
    for dimension, segment, subset in partitions:
        rows.append(
            {
                "stage": stage,
                "dimension": dimension,
                "segment": segment,
                "start_date": subset["date"].min().strftime("%Y-%m-%d"),
                "end_date": subset["date"].max().strftime("%Y-%m-%d"),
                "session_count": int(len(subset)),
                "mape": float(subset["ape"].mean() * 100),
                "wape": _safe_ratio(float(subset["abs_error"].sum()), float(subset["actual_qty"].sum())) * 100,
                "smape": float(subset["smape_item"].mean() * 100),
                "signed_bias": _safe_ratio(
                    float(subset["pred_qty"].sum() - subset["actual_qty"].sum()),
                    float(subset["actual_qty"].sum()),
                ) * 100,
            }
        )
    return pd.DataFrame(rows)

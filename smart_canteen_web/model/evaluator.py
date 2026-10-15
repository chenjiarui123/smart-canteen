#!/usr/bin/env python3
"""Fixed, deterministic metrics for the smart-canteen recommendation task.

This module deliberately contains no policy code.  It only joins a recommendation
table to the realised menu for the same (date, canteen, meal, dish) keys and
computes the metrics defined by the project brief.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GROUP_KEYS = ["date", "食堂名称", "餐次"]
DISH_KEY = "菜品名称"
ITEM_KEYS = GROUP_KEYS + [DISH_KEY]


def normalize_name(value: Any) -> str:
    """Apply only the name normalisations allowed by the task definition."""

    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.translate(str.maketrans({"（": "(", "）": ")", "【": "[", "】": "]"}))
    return re.sub(r"\s+", " ", text)


def _with_date(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "date" not in result.columns:
        required = {"年", "月", "日"}
        if not required.issubset(result.columns):
            raise ValueError("数据必须包含 date 或 年/月/日")
        result["date"] = pd.to_datetime(
            {"year": result["年"], "month": result["月"], "day": result["日"]},
            errors="raise",
        )
    else:
        result["date"] = pd.to_datetime(result["date"], errors="raise")
    return result


def prepare_actual(actual: pd.DataFrame) -> pd.DataFrame:
    """Return one realised row per evaluation item key."""

    required = {"食堂名称", "餐次", DISH_KEY, "菜品总销量", "订单数量", "销售额"}
    missing = sorted(required - set(actual.columns))
    if missing:
        raise ValueError(f"实际销售数据缺少字段：{missing}")
    frame = _with_date(actual)
    frame[DISH_KEY] = frame[DISH_KEY].astype(str)
    frame["_dish_norm"] = frame[DISH_KEY].map(normalize_name)
    for column in ["菜品总销量", "订单数量", "销售额"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    grouped = (
        frame.groupby(GROUP_KEYS + ["_dish_norm"], as_index=False, sort=False)
        .agg(
            **{
                DISH_KEY: (DISH_KEY, "first"),
                "actual_qty": ("菜品总销量", "sum"),
                "actual_orders": ("订单数量", "sum"),
                "actual_rev": ("销售额", "sum"),
            }
        )
    )
    return grouped


def prepare_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    required = set(ITEM_KEYS + ["pred_qty", "pred_orders", "pred_rev"])
    missing = sorted(required - set(recommendations.columns))
    if missing:
        raise ValueError(f"推荐结果缺少字段：{missing}")
    frame = _with_date(recommendations)
    frame[DISH_KEY] = frame[DISH_KEY].astype(str)
    frame["_dish_norm"] = frame[DISH_KEY].map(normalize_name)
    for column in ["pred_qty", "pred_orders", "pred_rev"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(frame[column])).any() or (frame[column] < 0).any():
            raise ValueError(f"{column} 必须为有限非负数")
    return frame


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna(False).map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "是"}
    )


def _excess_count(values: pd.Series, limit: int, eligible: pd.Series | None = None) -> int:
    valid = values.notna() & values.astype(str).str.strip().ne("")
    valid &= ~values.astype(str).str.lower().isin({"unknown", "未知", "nan", "none"})
    if eligible is not None:
        valid &= _as_bool(eligible)
    counts = values.loc[valid].astype(str).value_counts()
    return int((counts - int(limit)).clip(lower=0).sum())


def constraint_violations(
    recommendations: pd.DataFrame, config: dict[str, Any]
) -> dict[str, int]:
    """Count excess selections; unknown/low-confidence attributes never bind."""

    limits = config.get("constraints", config)
    totals = {
        "same_ingredient_violation_count": 0,
        "same_ingredient_family_violation_count": 0,
        "same_method_violation_count": 0,
        "same_flavor_violation_count": 0,
        "similarity_group_violation_count": 0,
        "duplicate_dish_violation_count": 0,
    }
    for _, group in recommendations.groupby(GROUP_KEYS, sort=False):
        totals["duplicate_dish_violation_count"] += int(group["_dish_norm"].duplicated().sum())
        specs = [
            ("主要食材", "_main_hard", "max_same_main_ingredient", "same_ingredient_violation_count"),
            ("食材大类", "_family_hard", "max_same_ingredient_family", "same_ingredient_family_violation_count"),
            ("制作方法", "_method_hard", "max_same_cooking_method", "same_method_violation_count"),
            ("口味", "_flavor_hard", "max_same_flavor", "same_flavor_violation_count"),
            ("相似菜分组", "_similarity_hard", "max_same_similarity_group", "similarity_group_violation_count"),
        ]
        for value_col, eligible_col, limit_key, metric_key in specs:
            if value_col not in group.columns:
                continue
            eligible = group[eligible_col] if eligible_col in group.columns else None
            totals[metric_key] += _excess_count(
                group[value_col], int(limits.get(limit_key, 10**9)), eligible
            )
    totals["total_constraint_violation_count"] = int(sum(totals.values()))
    return totals


def _repeat_metrics(recommendations: pd.DataFrame) -> tuple[int, int, float]:
    if recommendations.empty:
        return 0, 0, 0.0
    frame = recommendations.sort_values(GROUP_KEYS + [DISH_KEY]).copy()
    if "主食标记" in frame.columns:
        non_staple = ~_as_bool(frame["主食标记"])
    elif "菜品类别" in frame.columns:
        non_staple = frame["菜品类别"].astype(str).ne("主食")
    else:
        non_staple = pd.Series(True, index=frame.index)
    frame = frame.loc[non_staple].copy()
    if frame.empty:
        return 0, 0, 0.0

    group_dates = (
        recommendations[GROUP_KEYS]
        .drop_duplicates()
        .sort_values(GROUP_KEYS)
    )
    previous_date: dict[tuple[str, str, pd.Timestamp], pd.Timestamp] = {}
    for (canteen, meal), dates in group_dates.groupby(["食堂名称", "餐次"], sort=False):
        ordered = dates["date"].sort_values().tolist()
        for index in range(1, len(ordered)):
            previous_date[(str(canteen), str(meal), ordered[index])] = ordered[index - 1]

    seen = set(
        zip(frame["食堂名称"], frame["餐次"], frame["date"], frame["_dish_norm"])
    )
    eligible = 0
    repeats = 0
    for canteen, meal, current, dish in frame[
        ["食堂名称", "餐次", "date", "_dish_norm"]
    ].itertuples(index=False, name=None):
        prior = previous_date.get((str(canteen), str(meal), current))
        if prior is None:
            continue
        eligible += 1
        repeats += int((canteen, meal, prior, dish) in seen)
    return repeats, eligible, _safe_ratio(repeats, eligible)


def evaluate_recommendations(
    recommendations: pd.DataFrame,
    actual: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, float | int]:
    """Compute all fixed primary and auxiliary metrics."""

    recs = prepare_recommendations(recommendations)
    realised = prepare_actual(actual)

    actual_groups = (
        realised.groupby(GROUP_KEYS, as_index=False, sort=False)
        .agg(
            actual_total_qty=("actual_qty", "sum"),
            actual_total_orders=("actual_orders", "sum"),
            actual_total_rev=("actual_rev", "sum"),
        )
    )
    pred_groups = (
        recs.groupby(GROUP_KEYS, as_index=False, sort=False)
        .agg(
            pred_total_qty=("pred_qty", "sum"),
            pred_total_orders=("pred_orders", "sum"),
            recommended_count=(DISH_KEY, "size"),
        )
    )
    group_eval = actual_groups.merge(pred_groups, on=GROUP_KEYS, how="left")
    group_eval[["pred_total_qty", "pred_total_orders", "recommended_count"]] = group_eval[
        ["pred_total_qty", "pred_total_orders", "recommended_count"]
    ].fillna(0)
    valid_total = group_eval["actual_total_qty"] > 0
    total_mape = float(
        (
            (group_eval.loc[valid_total, "pred_total_qty"] - group_eval.loc[valid_total, "actual_total_qty"])
            .abs()
            .div(group_eval.loc[valid_total, "actual_total_qty"])
        ).mean()
        * 100
    )

    joined = recs.merge(
        realised[GROUP_KEYS + ["_dish_norm", "actual_qty", "actual_orders", "actual_rev"]],
        on=GROUP_KEYS + ["_dish_norm"],
        how="left",
    )
    joined[["actual_qty", "actual_orders", "actual_rev"]] = joined[
        ["actual_qty", "actual_orders", "actual_rev"]
    ].fillna(0.0)

    actual_rev_selected = float(joined["actual_rev"].sum())
    overpred_rev = float((joined["pred_rev"] - joined["actual_rev"]).clip(lower=0).sum())
    net_benefit = actual_rev_selected - 0.5 * overpred_rev
    selected_actual_qty = float(joined["actual_qty"].sum())
    selected_actual_orders = float(joined["actual_orders"].sum())
    qty_wmape = _safe_ratio(
        float((joined["pred_qty"] - joined["actual_qty"]).abs().sum()), selected_actual_qty
    ) * 100
    order_wmape = _safe_ratio(
        float((joined["pred_orders"] - joined["actual_orders"]).abs().sum()),
        selected_actual_orders,
    ) * 100
    total_actual_rev = float(actual_groups["actual_total_rev"].sum())
    revenue_coverage = _safe_ratio(actual_rev_selected, total_actual_rev)
    dish_hit_rate = float((joined["actual_qty"] > 0).mean()) if len(joined) else 0.0
    unmatched_dish_count = (
        int(recs.loc[~_as_bool(recs["library_matched"]), "_dish_norm"].nunique())
        if "library_matched" in recs.columns
        else 0
    )
    repeat_count, repeat_eligible_count, repeat_rate = _repeat_metrics(recs)
    violations = constraint_violations(recs, config)

    metrics: dict[str, float | int] = {
        "total_mape": total_mape,
        "net_benefit": float(net_benefit),
        "qty_wmape": float(qty_wmape),
        "order_wmape": float(order_wmape),
        "revenue_coverage": float(revenue_coverage),
        "dish_hit_rate": float(dish_hit_rate),
        "actual_rev_selected": actual_rev_selected,
        "overpred_rev": overpred_rev,
        "repeat_rate": float(repeat_rate),
        "repeat_count": int(repeat_count),
        "repeat_eligible_count": int(repeat_eligible_count),
        "unmatched_dish_count": unmatched_dish_count,
        "evaluated_group_count": int(len(group_eval)),
        "recommended_item_count": int(len(recs)),
    }
    metrics.update(violations)
    return metrics


def build_failure_cases(
    recommendations: pd.DataFrame, actual: pd.DataFrame
) -> pd.DataFrame:
    """Build an explainable group-level error table without changing metrics."""

    recs = prepare_recommendations(recommendations)
    realised = prepare_actual(actual)
    actual_groups = realised.groupby(GROUP_KEYS, as_index=False).agg(
        actual_total_qty=("actual_qty", "sum"), actual_total_rev=("actual_rev", "sum")
    )
    pred_groups = recs.groupby(GROUP_KEYS, as_index=False).agg(
        pred_total_qty=("pred_qty", "sum"), recommended_count=(DISH_KEY, "size")
    )
    joined = recs.merge(
        realised[GROUP_KEYS + ["_dish_norm", "actual_rev", "actual_qty"]],
        on=GROUP_KEYS + ["_dish_norm"], how="left"
    )
    joined[["actual_rev", "actual_qty"]] = joined[["actual_rev", "actual_qty"]].fillna(0)
    selected = joined.groupby(GROUP_KEYS, as_index=False).agg(
        actual_rev_selected=("actual_rev", "sum"),
        dish_hits=("actual_qty", lambda values: int((values > 0).sum())),
    )
    result = actual_groups.merge(pred_groups, on=GROUP_KEYS, how="left").merge(
        selected, on=GROUP_KEYS, how="left"
    )
    result[["pred_total_qty", "recommended_count", "actual_rev_selected", "dish_hits"]] = result[
        ["pred_total_qty", "recommended_count", "actual_rev_selected", "dish_hits"]
    ].fillna(0)
    result["qty_error"] = result["pred_total_qty"] - result["actual_total_qty"]
    result["abs_pct_error"] = result["qty_error"].abs() / result["actual_total_qty"].replace(0, np.nan)
    result["revenue_coverage"] = result["actual_rev_selected"] / result["actual_total_rev"].replace(0, np.nan)
    result["failure_type"] = np.select(
        [result["qty_error"] > 0, result["qty_error"] < 0],
        ["总量预测过高", "总量预测过低"],
        default="总量预测准确",
    )
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    return result.sort_values(["abs_pct_error", "actual_total_rev"], ascending=[False, False])


def _json_safe(metrics: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = None if not math.isfinite(float(value)) else float(value)
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendations", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    metrics = evaluate_recommendations(
        pd.read_csv(args.recommendations), pd.read_csv(args.actual), config
    )
    payload = json.dumps(_json_safe(metrics), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

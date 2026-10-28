from __future__ import annotations

from typing import Any

import pandas as pd


def factor_impact_text(label: str, factor: float) -> str:
    if factor > 1.005:
        return f"{label}上调（×{factor:.3f}）"
    if factor < 0.995:
        return f"{label}下调（×{factor:.3f}）"
    return f"{label}影响接近中性（×{factor:.3f}）"


def build_excluded_candidates(
    audit: dict[str, Any],
    recommendations: pd.DataFrame,
    feature_frame: pd.DataFrame,
    limit: int = 12,
) -> list[dict[str, str]]:
    names = (
        feature_frame.drop_duplicates("_dish_norm")
        .set_index("_dish_norm")["菜品名称_库"]
        .fillna("")
        .astype(str)
        .to_dict()
    )
    selected = set(recommendations["菜品名称"].astype(str))
    selected_norms = set(
        recommendations.get(
            "_dish_norm", pd.Series(dtype=str)
        ).astype(str)
    )
    hard = set(map(str, audit.get("hard_excluded", set())))
    pool = set(map(str, audit.get("candidate_pool", set())))
    rows: list[dict[str, str]] = []
    for dish in sorted(hard, key=lambda value: (names.get(value, value), value)):
        rows.append(
            {
                "dish_name": names.get(dish, dish),
                "reason": "相似菜分组硬约束冲突",
            }
        )
    remaining = sorted(
        pool - hard - selected_norms,
        key=lambda value: (names.get(value, value), value),
    )
    for dish in remaining:
        name = names.get(dish, dish)
        if name in selected:
            continue
        rows.append(
            {
                "dish_name": name,
                "reason": "未进入容量上限内的高分候选，或受到软搭配惩罚",
            }
        )
        if len(rows) >= limit:
            break
    return rows[:limit]


def build_markdown_report(
    *,
    summary: dict[str, Any],
    recommendations: pd.DataFrame,
    excluded: list[dict[str, str]],
    model_version: str,
    parameter_sha256: str,
    combination_check: dict[str, Any] | None = None,
    low_frequency_config_sha256: str | None = None,
) -> str:
    # Keep display conversion local: model predictions and persisted history
    # count 50 g portions for gram-priced dishes, as in the CSV export.
    display_rows = recommendations.to_dict("records")
    unit_totals: dict[str, dict[str, float]] = {}
    for row in display_rows:
        unit = row.get("菜品售卖单位", "份")
        unit = "份" if pd.isna(unit) else str(unit).strip()
        multiplier = 50.0 if unit == "克" else 1.0
        row["display_unit"] = unit
        row["display_qty"] = float(row["pred_qty"]) * multiplier
        if row.get("recommendation_source") == "low_frequency_opportunity":
            row["display_trial_qty"] = (
                float(row.get("suggested_trial_qty", 0.0) or 0.0) * multiplier
            )
        source = (
            "high_frequency"
            if row.get("recommendation_source") == "high_frequency"
            else "low_frequency_opportunity"
        )
        totals = unit_totals.setdefault(
            unit, {"high_frequency": 0.0, "low_frequency_opportunity": 0.0}
        )
        totals[source] += row["display_qty"]
    lines = [
        f"# {summary['prediction_date']} {summary['meal']}菜单推荐",
        "",
        f"- 逻辑食堂：{summary['canteen']}",
        f"- 使用历史：{summary['history_start']} 至 {summary['history_end']}",
        f"- 距历史末日：{int(summary.get('forecast_gap_days', 0))} 天",
        *(
            [f"- 远期预测提示：{summary['forecast_gap_warning']}"]
            if summary.get("forecast_gap_warning")
            else []
        ),
        f"- 冻结模型：{model_version}",
        f"- 参数哈希：`{parameter_sha256}`",
        *(
            [f"- 低频潜力菜配置哈希：`{low_frequency_config_sha256}`"]
            if low_frequency_config_sha256
            else []
        ),
        f"- 推荐菜品数：{summary['recommended_count']}（参考菜品数 {summary['menu_capacity']}）",
        f"- 当餐预测总需求（模型参考值）：{summary['pred_total_qty']:.2f}",
        f"- 最终备餐量合计（模型参考值）：{summary.get('total_preparation_qty', summary['pred_total_qty']):.2f}",
        f"- 总参考订单数：{summary['pred_total_orders']:.2f}",
        f"- 总参考销售额：{summary['pred_total_rev']:.2f}",
        f"- 天气影响：{summary['weather_impact']}",
        f"- 假期影响：{summary['holiday_impact']}",
        f"- 最终综合修正因子：{summary['final_combined_factor']:.3f}",
        f"- 菜品数量参考值：{summary['menu_capacity']}（不作为潜力菜硬上限）",
        f"- 约束放宽：{'是' if summary['constraint_relaxed'] else '否'}",
        f"- 自动高频菜单相邻日重复率：{format(summary['adjacent_repeat_rate'], '.1%') if summary.get('adjacent_repeat_rate') is not None else '缺少对照，无法计算'}",
        f"- 同餐次合格候选替换数：{summary.get('replacement_count', 0)}",
        f"- 相邻日提示：{summary.get('adjacent_warning') or '无'}",
        "",
        "模型参考值沿用内部计量口径：克制菜按50克份计数，其他菜按各自售卖单位计数。"
        "不同售卖单位不能相加作为实际备餐总量；实际数量见下方分单位合计。",
        "",
        "## 实际备餐量（按售卖单位分组）",
        "",
        "| 单位 | 高频菜单备餐量 | 低频潜力菜试餐量 | 合计 |",
        "|---|---:|---:|---:|",
    ]
    for unit, totals in sorted(unit_totals.items()):
        high_qty = totals["high_frequency"]
        low_qty = totals["low_frequency_opportunity"]
        lines.append(
            f"| {unit} | {high_qty:.2f} | {low_qty:.2f} | {high_qty + low_qty:.2f} |"
        )
    lines.extend([
        "",
        "## 推荐菜单",
        "",
        "备餐量中，高频菜单采用预测量，低频潜力菜采用试餐建议；"
        "若开启按总需求重新分配，则展示重新分配后的最终数量。"
        "低频菜的订单和销售额为试餐量对应的参考估计。",
        "",
        "| 来源 | 菜品 | 单位 | 最终备餐量 | 参考订单 | 参考销售额 | 类别 | 主要食材 | 做法 | 口味 |",
        "|---|---|---|---:|---:|---:|---|---|---|---|",
    ])
    for row in display_rows:
        source = (
            "高频菜单"
            if row.get("recommendation_source") == "high_frequency"
            else "低频潜力菜"
        )
        lines.append(
            f"| {source} | "
            "{菜品名称} | {display_unit} | {display_qty:.2f} | "
            "{pred_orders:.2f} | {pred_rev:.2f} | {菜品类别} | "
            "{主要食材} | {制作方法} | {口味} |".format(**row)
        )
    lines.extend(["", "## 菜品推荐说明", ""])
    for row in display_rows:
        source = (
            "高频菜单"
            if row.get("recommendation_source") == "high_frequency"
            else "低频潜力菜"
        )
        lines.append(f"- **{row['菜品名称']}**（{source}）：{row['reason']}")
        if row.get("recommendation_source") == "low_frequency_opportunity":
            ordinary_count = int(row.get("ordinary_sample_count", 0) or 0)
            matching_count = int(
                row.get("matching_holiday_sample_count", 0) or 0
            )
            lines.append(
                f"  - 普通日期样本 {ordinary_count} 次；匹配节日样本 "
                f"{matching_count} 次；节日销售记录"
                f"{'已使用' if row.get('holiday_score_used') else '未使用'}；"
                f"原始建议试餐量 {row['display_trial_qty']:.2f} {row['display_unit']}"
                "（最终备餐量见推荐菜单）。"
            )
    lines.extend(["", "## 主要被排除候选", ""])
    if excluded:
        for row in excluded:
            lines.append(f"- {row['dish_name']}：{row['reason']}")
    else:
        lines.append("- 本次没有可报告的主要排除候选。")
    if summary.get("cold_start_warning"):
        lines.extend(
            ["", "## 历史提示", "", f"- {summary['cold_start_warning']}"]
        )
    if combination_check:
        lines.extend(
            [
                "",
                "## 最终菜单合并检查",
                "",
                f"- 高频菜：{combination_check['selected_high_frequency_count']} 道",
                f"- 低频潜力菜：{combination_check['selected_low_frequency_count']} 道",
                f"- 总菜品数：{combination_check['selected_count']} 道",
                f"- 菜品数量参考值：{combination_check['reference_menu_count']} 道",
                f"- 当餐预测总需求（模型参考值）：{combination_check['predicted_session_demand_qty']:.2f}",
                f"- 最终备餐量合计（模型参考值）：{combination_check['total_preparation_qty']:.2f}",
                f"- 按总需求重新分配：{'是' if combination_check['redistributed_to_total_demand'] else '否'}",
                f"- 模型参考值守恒：{'是' if combination_check['pred_total_conserved'] else '否（保留高频预测并额外加入试餐量）'}",
                f"- 硬约束违规：{combination_check['total_constraint_violation_count']}",
            ]
        )
        warnings = combination_check.get("soft_warnings", [])
        if warnings:
            lines.append(f"- 搭配风险提示：{'；'.join(warnings)}")
        else:
            lines.append("- 搭配风险提示：无")
        reference_warnings = combination_check.get("reference_warnings", [])
        if reference_warnings:
            lines.append(
                "- 数量/备餐提醒（备餐量差值采用模型参考口径）："
                f"{'；'.join(reference_warnings)}"
            )
        else:
            lines.append("- 数量/备餐提醒：无")
    lines.extend(
        [
            "",
            "> 高频菜单由报告所列版本策略生成；低频潜力菜独立计算，不参与自动高频菜单80%重复率分母。",
            "",
        ]
    )
    return "\n".join(lines)

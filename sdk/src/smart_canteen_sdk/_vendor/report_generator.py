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
        f"- 当餐预测总需求：{summary['pred_total_qty']:.2f}",
        f"- 最终总备餐量：{summary.get('total_preparation_qty', summary['pred_total_qty']):.2f}",
        f"- 总预测订单数：{summary['pred_total_orders']:.2f}",
        f"- 总预测销售额：{summary['pred_total_rev']:.2f}",
        f"- 天气影响：{summary['weather_impact']}",
        f"- 假期影响：{summary['holiday_impact']}",
        f"- 最终综合修正因子：{summary['final_combined_factor']:.3f}",
        f"- 菜品数量参考值：{summary['menu_capacity']}（不作为潜力菜硬上限）",
        f"- 约束放宽：{'是' if summary['constraint_relaxed'] else '否'}",
        f"- 自动高频菜单相邻日重复率：{format(summary['adjacent_repeat_rate'], '.1%') if summary.get('adjacent_repeat_rate') is not None else '缺少对照，无法计算'}",
        f"- 同餐次合格候选替换数：{summary.get('replacement_count', 0)}",
        f"- 相邻日提示：{summary.get('adjacent_warning') or '无'}",
        "",
        "## 推荐菜单",
        "",
        "| 来源 | 菜品 | 单位 | 预测销量 | 预测订单 | 预测销售额 | 类别 | 主要食材 | 做法 | 口味 |",
        "|---|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in recommendations.to_dict("records"):
        source = (
            "高频菜单"
            if row.get("recommendation_source") == "high_frequency"
            else "低频潜力菜"
        )
        lines.append(
            f"| {source} | "
            "{菜品名称} | {菜品售卖单位} | {pred_qty:.2f} | "
            "{pred_orders:.2f} | {pred_rev:.2f} | {菜品类别} | "
            "{主要食材} | {制作方法} | {口味} |".format(**row)
        )
    lines.extend(["", "## 菜品推荐说明", ""])
    for row in recommendations.to_dict("records"):
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
                f"建议试餐量 {float(row.get('suggested_trial_qty', 0.0) or 0.0):.2f}。"
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
                f"- 当餐预测总需求：{combination_check['predicted_session_demand_qty']:.2f}",
                f"- 总备餐量：{combination_check['total_preparation_qty']:.2f}",
                f"- 按总需求重新分配：{'是' if combination_check['redistributed_to_total_demand'] else '否'}",
                f"- 总量守恒：{'是' if combination_check['pred_total_conserved'] else '否（保留高频预测并额外加入试餐量）'}",
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
            lines.append(f"- 数量/备餐提醒：{'；'.join(reference_warnings)}")
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

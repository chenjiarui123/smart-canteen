from __future__ import annotations

from copy import deepcopy

import pandas as pd

from app.history_store import HistoryStore
from app.report_generator import build_markdown_report


def _summary(menu: pd.DataFrame) -> dict:
    return {
        "prediction_date": "2026-02-05",
        "meal": "晚餐",
        "canteen": "A湖畔智慧餐厅",
        "history_start": "2026-01-01",
        "history_end": "2026-02-04",
        "recommended_count": len(menu),
        "menu_capacity": 2,
        "pred_total_qty": 5.75,
        "total_preparation_qty": float(menu["pred_qty"].sum()),
        "pred_total_orders": float(menu["pred_orders"].sum()),
        "pred_total_rev": float(menu["pred_rev"].sum()),
        "weather_impact": "中性",
        "holiday_impact": "中性",
        "final_combined_factor": 1.0,
        "constraint_relaxed": False,
    }


def _row(index: int, unit, qty: float, source: str, trial_qty=0.0) -> dict:
    return {
        "date": pd.Timestamp("2026-02-05"),
        "食堂名称": "A湖畔智慧餐厅",
        "餐次": "晚餐",
        "菜品名称": f"a_hupan_菜{index}",
        "菜品类别": "荤菜",
        "菜品售卖单位": unit,
        "pred_qty": qty,
        "pred_orders": 1.0,
        "pred_rev": 18.125,
        "score": 9.0,
        "recommendation_source": source,
        "suggested_trial_qty": trial_qty,
        "reason": "验收测试",
        "主要食材": "测试食材",
        "制作方法": "蒸",
        "口味": "清淡",
        "ordinary_sample_count": 3,
        "matching_holiday_sample_count": 0,
        "holiday_score_used": False,
    }


def test_report_and_csv_use_same_units_without_changing_model_or_history(
    temp_repository, tmp_path
):
    menu = pd.DataFrame(
        [
            _row(1, " 克 ", 2.5, "high_frequency"),
            _row(2, None, 3.25, "high_frequency"),
            # Final allocation can differ from the original trial suggestion.
            _row(3, "克", 0.8, "low_frequency_opportunity", 1.2),
            _row(4, "个", 2.0, "low_frequency_opportunity", 3.0),
        ]
    )
    original = menu.copy(deep=True)
    summary = _summary(menu)
    check = {
        "selected_high_frequency_count": 2,
        "selected_low_frequency_count": 2,
        "selected_count": 4,
        "reference_menu_count": 2,
        "predicted_session_demand_qty": 5.75,
        "total_preparation_qty": 8.55,
        "redistributed_to_total_demand": False,
        "pred_total_conserved": False,
        "total_constraint_violation_count": 0,
        "reference_warnings": ["当前总备餐量高于当餐预测总需求2.80"],
    }
    original_summary, original_check = deepcopy(summary), deepcopy(check)
    report = build_markdown_report(
        summary=summary,
        recommendations=menu,
        excluded=[],
        model_version="test",
        parameter_sha256="test",
        combination_check=check,
    )

    store = HistoryStore(temp_repository, tmp_path / "recommendations")
    _, csv_path, markdown_path = store.save(
        dataset_id="a_hupan",
        canteen="A湖畔智慧餐厅",
        meal="晚餐",
        prediction_date=pd.Timestamp("2026-02-05"),
        recommendations=menu,
        markdown=report,
    )
    exported = pd.read_csv(csv_path, encoding="utf-8-sig")
    menu_lines = [
        line.split("|")[1:-1]
        for line in report.splitlines()
        if line.startswith(("| 高频菜单 |", "| 低频潜力菜 |"))
    ]
    assert [float(columns[3].strip()) for columns in menu_lines] == [
        125.0, 3.25, 40.0, 2.0
    ]
    assert [float(columns[3].strip()) for columns in menu_lines] == exported[
        "菜品销量"
    ].tolist()
    assert [columns[2].strip() for columns in menu_lines] == ["克", "份", "克", "个"]
    assert "| 克 | 125.00 | 40.00 | 165.00 |" in report
    assert "| 份 | 3.25 | 0.00 | 3.25 |" in report
    assert "| 个 | 0.00 | 2.00 | 2.00 |" in report
    assert "原始建议试餐量 60.00 克" in report
    assert "原始建议试餐量 3.00 个" in report
    assert "高频菜单采用预测量，低频潜力菜采用试餐建议" in report
    assert "当餐预测总需求（模型参考值）：5.75" in report
    assert "最终备餐量合计（模型参考值）：8.55" in report
    assert "不同售卖单位不能相加作为实际备餐总量" in report
    assert "备餐量差值采用模型参考口径" in report
    assert markdown_path.read_text(encoding="utf-8") == report
    pd.testing.assert_frame_equal(menu, original)
    assert summary == original_summary
    assert check == original_check
    history = temp_repository.read_recommendation_history("a_hupan")
    assert history["pred_qty"].tolist() == original["pred_qty"].tolist()
    assert history["pred_rev"].tolist() == original["pred_rev"].tolist()


def test_report_without_unit_column_defaults_to_portions():
    menu = pd.DataFrame([_row(1, "份", 2.5, "high_frequency")]).drop(
        columns="菜品售卖单位"
    )
    original = menu.copy(deep=True)
    report = build_markdown_report(
        summary=_summary(menu),
        recommendations=menu,
        excluded=[],
        model_version="test",
        parameter_sha256="test",
    )
    assert "| 高频菜单 | a_hupan_菜1 | 份 | 2.50 |" in report
    assert "| 份 | 2.50 | 0.00 | 2.50 |" in report
    pd.testing.assert_frame_equal(menu, original)

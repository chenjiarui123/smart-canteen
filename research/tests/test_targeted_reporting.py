import pandas as pd

from targeted_search import evaluate


def test_split_report_retains_previous_day_from_full_run_audit():
    key = {"date": pd.Timestamp("2026-04-01"), "食堂名称": "测试食堂", "餐次": "午餐"}
    actual = pd.DataFrame([{**key, "菜品名称": "测试菜", "_dish_norm": "测试菜",
                            "菜品总销量": 10., "订单数量": 5., "销售额": 20.}])
    recs = pd.DataFrame([{**key, "菜品名称": "测试菜", "pred_qty": 10., "pred_orders": 5.,
                         "pred_rev": 20., "pred_total_qty": 10.}])
    audit = [{**key, "capacity": 1, "planned_capacity": 1, "selected_count": 1,
              "adjacent_status": "not_enforced", "adjacent_repeat_rate": 1.,
              "candidate_pool": {"测试菜"}, "hard_excluded": set()}]
    metrics, sessions, segments = evaluate(recs, actual, audit)
    assert metrics["missing_previous_sessions"] == 0
    assert metrics["repeat_over_80_sessions"] == 1
    assert metrics["total_mape"] == 0
    assert not sessions.missing_previous.any()
    assert all(row["missing_previous_sessions"] == 0 for row in segments)

import copy
import importlib.util
from pathlib import Path

import pandas as pd

import policy
import targeted_policy as targeted


def candidates(n=20):
    return pd.DataFrame([dict(_dish_norm=f"dish{i}", 菜品名称_库=f"dish{i}", score=100-i,
                              相似菜分组=f"group{i}", _similarity_hard=True,
                              主要食材="unknown", 食材大类="unknown", 制作方法="unknown", 口味="unknown",
                              l_offer_count=10, g_offer_count=10, _allocation_weight=10.0,
                              _qty_stability=1.0, _min_batch_qty=1.0,
                              _unit_orders_ratio=.5, _unit_revenue_ratio=2.0,
                              pred_qty=10.0, expected_orders_per_qty=.5, expected_rev_per_qty=2.0)
                         for i in range(n)])


def config():
    import json
    config_path = Path(__file__).resolve().parents[1] / "config.json"
    return dict(json.loads(config_path.read_text())["best_policy"], adjacent_repeat_hard=True)


def context():
    return {"date": pd.Timestamp("2025-09-02"), "meta": {"l_session_count": 40}}


def test_quota_all_categories_exact_floor_and_fixed_count():
    pool = candidates()
    before = pool.iloc[:11].copy()
    c = context()
    out = targeted.enforce_repeat(before, pool, 110, config(), policy.BASE_CONSTRAINTS,
                                   [(pd.Timestamp("2025-09-01"), set(before._dish_norm))], c)
    assert len(out) == 11
    assert len(set(out._dish_norm) & set(before._dish_norm)) <= 8
    assert c["_targeted_audit"]["replacement_count"] >= 3
    assert abs(out.pred_qty.sum()-110) < 1e-9


def test_missing_yesterday_is_not_zero_or_last_business_day():
    pool = candidates()
    c = context()
    targeted.enforce_repeat(pool.iloc[:10], pool, 100, config(), policy.BASE_CONSTRAINTS,
                            [(pd.Timestamp("2025-08-31"), set(pool._dish_norm))], c)
    assert c["_targeted_audit"]["adjacent_status"] == "missing_previous"
    assert c["_targeted_audit"]["adjacent_repeat_rate"] is None


def test_infeasible_does_not_release_smaller_or_over_limit_menu():
    pool = candidates(10)
    c = context()
    out = targeted.enforce_repeat(pool, pool, 100, config(), policy.BASE_CONSTRAINTS,
                                  [(pd.Timestamp("2025-09-01"), set(pool._dish_norm))], c)
    assert out.empty
    assert c["_targeted_audit"]["replacement_shortage"] == 2
    assert c["_targeted_audit"]["unreleased_count"] == 10


def test_unreliable_replacements_and_minimum_batch_rechecked():
    pool = candidates()
    pool.loc[10:, "l_offer_count"] = 1
    c = context()
    out = targeted.enforce_repeat(pool.iloc[:10], pool, 100, config(), policy.BASE_CONSTRAINTS,
                                  [(pd.Timestamp("2025-09-01"), set(pool.iloc[:10]._dish_norm))], c)
    assert out.empty
    pool.loc[10:, "l_offer_count"] = 10
    pool.loc[10:12, "_min_batch_qty"] = 1000
    out = targeted.enforce_repeat(pool.iloc[:10], pool, 100, config(), policy.BASE_CONSTRAINTS,
                                  [(pd.Timestamp("2025-09-01"), set(pool.iloc[:10]._dish_norm))], context())
    assert len(out) == 10
    assert not set(out._dish_norm) & {"dish10", "dish11", "dish12"}
    assert (out.pred_qty + 1e-9 >= out._min_batch_qty).all()


def test_joint_holiday_estimate_no_floor_or_double_multiplication():
    meta = {"l_regular_workday_qty_mean": 1000,
            "l_calendar_state_stats": {"校历:寒假": {"median": 200, "count": 15},
                                       "校历:寒假|法定:春节": {"median": 10, "count": 10}}}
    target = {"_calendar_state": "校历:寒假|法定:春节", "_calendar_category": "校历法定重叠", "_day_type": "工作日"}
    value, count, source = targeted.holiday_estimate(meta, target, {"holiday_shrinkage_tau": 1}, "qty")
    assert 10 < value < 25
    assert count == 10 and source.endswith("校历:寒假|法定:春节")


def test_fast_selection_matches_original_on_synthetic_walk():
    spec = importlib.util.spec_from_file_location("walk", Path(__file__).with_name("test_walk_forward.py"))
    walk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(walk)
    dishes = tuple(f"菜{i}" for i in range(20))
    sales = walk.synthetic_sales(dishes)
    prior = sales.loc[sales.date < pd.Timestamp("2025-01-03")]
    actual = sales.loc[sales.date >= pd.Timestamp("2025-01-03")]
    contexts = policy.build_contexts(prior, actual, walk.synthetic_attributes(dishes))
    cfg = dict(config(), adjacent_repeat_hard=False)
    expected = policy.make_recommendations(contexts, cfg, policy.BASE_CONSTRAINTS)
    result = targeted.make_recommendations(contexts, cfg, policy.BASE_CONSTRAINTS)
    pd.testing.assert_frame_equal(result, expected)

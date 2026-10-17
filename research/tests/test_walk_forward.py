from __future__ import annotations

import inspect
import json
import hashlib
import unittest

import pandas as pd

import adaptive_search
import diagnostics
import policy


def synthetic_sales(dishes: tuple[str, ...] = ("测试菜",)) -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2025-01-01", periods=4, freq="D"):
        for meal in ["午餐", "晚餐"]:
            for index, dish in enumerate(dishes):
                rows.append(
                    {
                        "date": date,
                        "食堂名称": "测试食堂",
                        "餐次": meal,
                        "_dish_norm": dish,
                        "菜品名称": dish,
                        "菜品总销量": 10.0 + date.day + index,
                        "订单数量": 5.0 + index,
                        "销售额": 20.0 + date.day + index,
                        "_orders_per_qty": (5.0 + index)
                        / (10.0 + date.day + index),
                        "_rev_per_qty": (20.0 + date.day + index)
                        / (10.0 + date.day + index),
                    }
                )
    return pd.DataFrame(rows)


def synthetic_attributes(dishes: tuple[str, ...] = ("测试菜",)) -> pd.DataFrame:
    rows = []
    for index, dish in enumerate(dishes):
        rows.append(
            {
                "_dish_norm": dish,
                "菜品名称_库": dish,
                "菜品类别": "荤菜",
                "口味": "咸",
                "主要食材": "猪肉",
                "食材大类": "常规猪肉",
                "制作方法": "炒",
                "相似菜分组": f"分组{index}",
                "主食标记": False,
                "inferred_main_ingredient": "猪肉",
                "ingredient_family": "常规猪肉",
                "attribute_source": "dish_name",
                "inference_confidence": "high",
                "library_matched": True,
                "_main_hard": True,
                "_family_hard": True,
                "_method_hard": True,
                "_flavor_hard": True,
                "_similarity_hard": True,
                "特征来源": "测试",
            }
        )
    return pd.DataFrame(rows)


def current_policy() -> dict:
    records = adaptive_search._load_trials()
    best = adaptive_search._best_record(records)
    if best is None:
        raise AssertionError("缺少自适应最佳实验")
    return dict(best["policy"])


class WalkForwardInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        sales = synthetic_sales()
        self.prior = sales.loc[sales["date"] < pd.Timestamp("2025-01-03")].copy()
        self.actual = sales.loc[sales["date"] >= pd.Timestamp("2025-01-03")].copy()
        self.contexts = policy.build_contexts(
            self.prior, self.actual, synthetic_attributes()
        )

    def test_every_context_uses_strictly_earlier_history(self) -> None:
        for context in self.contexts:
            self.assertLess(context["max_history_date"], context["date"])

    def test_all_meals_share_pre_append_day_snapshot(self) -> None:
        schedule = policy.context_schedule_frame(self.contexts)
        for _, day in schedule.groupby("date"):
            self.assertEqual(day["history_rows_at_day_start"].nunique(), 1)
            self.assertEqual(day["history_dates_at_day_start"].nunique(), 1)
            self.assertEqual(int(day["day_group_count"].iloc[0]), len(day))

    def test_target_context_contains_only_known_calendar_weather_inputs(self) -> None:
        forbidden = {"菜品总销量", "订单数量", "销售额", "平均取菜量"}
        for context in self.contexts:
            self.assertFalse(forbidden.intersection(context["target_context"]))

    def test_recommendation_state_is_recreated(self) -> None:
        selected = current_policy()
        first = policy.make_recommendations(
            self.contexts, selected, policy.BASE_CONSTRAINTS
        )
        second = policy.make_recommendations(
            self.contexts, selected, policy.BASE_CONSTRAINTS
        )
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True), second.reset_index(drop=True)
        )

    def test_final_quantity_revenue_and_total_are_coherent(self) -> None:
        recs = policy.make_recommendations(
            self.contexts, current_policy(), policy.BASE_CONSTRAINTS
        )
        consistency = diagnostics.prediction_consistency_diagnostics(recs)
        self.assertLess(consistency["order_consistency_wape"], 1e-12)
        self.assertLess(consistency["revenue_consistency_wape"], 1e-12)
        self.assertLess(consistency["total_allocation_wape"], 1e-12)

    def test_same_main_ingredient_is_soft_but_similarity_is_hard(self) -> None:
        dishes = ("菜一", "菜二")
        sales = synthetic_sales(dishes)
        prior = sales.loc[sales["date"] < pd.Timestamp("2025-01-03")].copy()
        actual = sales.loc[sales["date"] == pd.Timestamp("2025-01-03")].copy()
        attributes = synthetic_attributes(dishes)
        recs = policy.make_recommendations(
            policy.build_contexts(prior, actual, attributes),
            current_policy(),
            policy.BASE_CONSTRAINTS,
        )
        self.assertEqual(
            recs.groupby(["date", "食堂名称", "餐次"]).size().min(), 2
        )
        conflict_attributes = attributes.copy()
        conflict_attributes["相似菜分组"] = "同一分组"
        conflict_recs = policy.make_recommendations(
            policy.build_contexts(prior, actual, conflict_attributes),
            current_policy(),
            policy.BASE_CONSTRAINTS,
        )
        self.assertEqual(
            conflict_recs.groupby(["date", "食堂名称", "餐次"]).size().max(), 1
        )

    def test_hit_metrics_match_direct_calculation(self) -> None:
        recs = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "食堂名称": "A",
                    "餐次": "午餐",
                    "菜品名称": "菜1",
                    "pred_qty": 12.0,
                    "pred_orders": 6.0,
                    "pred_rev": 24.0,
                },
                {
                    "date": "2025-01-01",
                    "食堂名称": "A",
                    "餐次": "午餐",
                    "菜品名称": "菜2",
                    "pred_qty": 5.0,
                    "pred_orders": 2.0,
                    "pred_rev": 10.0,
                },
            ]
        )
        actual = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "食堂名称": "A",
                    "餐次": "午餐",
                    "菜品名称": "菜1",
                    "菜品总销量": 10.0,
                    "订单数量": 5.0,
                    "销售额": 20.0,
                }
            ]
        )
        metrics, _ = diagnostics.hit_quantity_diagnostics(recs, actual)
        self.assertAlmostEqual(metrics["hit_qty_mape"], 20.0)
        self.assertAlmostEqual(metrics["hit_qty_wmape"], 20.0)
        self.assertEqual(metrics["hit_item_count"], 1)

    def test_selection_uses_one_percent_pareto_guard(self) -> None:
        base = {
            "trial_index": 1,
            "is_new_best": True,
            "revenue_price_compliant": True,
            "total_constraint_violation_count": 0,
            "duplicate_dish_violation_count": 0,
            "total_allocation_wape": 0.0,
            "menu_fill_rate": 1.0,
            "dish_hit_rate": 0.6,
            "high_revenue_candidate_recall": 0.95,
            "revenue_coverage": 0.63,
            "hit_qty_wmape": 29.0,
            "weighted_qty_bias": 1.0,
            "segment_summary": {},
            "total_mape": 12.82,
            "hit_qty_mape": 61.37,
            "net_benefit": 1_700_000.0,
            "failure_reason": "",
        }
        best = dict(
            base,
            trial_index=2,
            total_mape=12.87,
            hit_qty_mape=61.53,
            net_benefit=1_750_000.0,
        )
        drifting = dict(
            best,
            total_mape=12.70,
            hit_qty_mape=62.30,
            net_benefit=1_760_000.0,
        )
        accepted, reason = adaptive_search._select_candidate(
            drifting, best, [], [base, best]
        )
        self.assertFalse(accepted)
        self.assertIn("1%", reason)

    def test_combined_factor_is_clipped_after_all_components(self) -> None:
        selected = current_policy()
        selected.update(
            {
                "recent_window": 7,
                "recent_weight": 0.5,
                "transfer_shrinkage_tau": 0.0,
                "exclude_special_from_regular_history": False,
                "calendar_mode": "none",
                "weather_mode": "none",
                "weekday_clip": [0.5, 1.5],
                "weekday_strength": 1.0,
                "group_trend_strength": 1.0,
                "group_trend_strength_by_meal": {},
                "group_trend_reference_window": 30,
                "group_trend_clip": [0.5, 1.5],
                "group_momentum_strength": 1.0,
                "group_momentum_apply_categories": ["普通日期"],
                "group_momentum_clip": [0.5, 1.5],
                "group_factor_combination_mode": "clipped_multiplicative",
                "group_combined_factor_clip": [0.8, 1.2],
            }
        )
        meta = {
            "l_session_count": 10.0,
            "g_session_count": 0.0,
            "l_qty_mean": 100.0,
            "l_w7_qty_mean": 50.0,
            "l_w7_session_count": 7.0,
            "l_w30_qty_mean": 100.0,
            "l_weekday_qty_mean": 50.0,
            "l_qty_last": 50.0,
            "l_qty_previous": 100.0,
        }
        components = policy._group_total_forecast(
            meta,
            "qty",
            selected,
            {
                "_calendar_category": "普通日期",
                "_day_type": "工作日",
                "_meal": "午餐",
            },
            return_components=True,
        )
        self.assertLess(components["raw_combined_factor"], 0.8)
        self.assertAlmostEqual(components["final_combined_factor"], 0.8)
        self.assertTrue(components["factor_clip_applied"])

    def test_calendar_capacity_uses_state_and_demand_without_factor_stack(self) -> None:
        selected = current_policy()
        selected.update(
            {
                "capacity_mode": "calendar_hierarchical",
                "capacity_state_shrinkage_tau": 0.0,
                "capacity_demand_weight": 0.5,
                "capacity_hierarchical_strength": 1.0,
                "capacity_scale": 1.0,
                "capacity_min": 1,
                "capacity_max": 500,
                "capacity_weather_strength": 0.0,
                "transfer_shrinkage_tau": 0.0,
            }
        )
        entry = {
            "count": 10.0,
            "dish_count_median": 60.0,
            "qty_per_dish_median": 50.0,
        }
        meta = {
            "l_session_count": 20.0,
            "l_capacity_median": 100.0,
            "l_regular_weekend_capacity_median": 100.0,
            "l_regular_weekend_qty_per_dish_median": 50.0,
            "l_qty_per_dish_median": 50.0,
            "l_special_calendar_stats": {"special": entry},
            "l_calendar_category_stats": {"法定节假日": entry},
            "l_calendar_state_stats": {"法定:测试节": entry},
        }
        result = policy._capacity_forecast(
            meta,
            selected,
            {
                "_calendar_category": "法定节假日",
                "_calendar_state": "法定:测试节",
                "_day_type": "周末",
            },
            pred_total_qty=3000.0,
        )
        self.assertEqual(result["planned_capacity"], 60)
        self.assertEqual(result["capacity_weather_factor"], 1.0)

    def test_min_batch_pruning_preserves_total(self) -> None:
        selected = pd.DataFrame(
            [
                {
                    "菜品名称_库": "菜一",
                    "score": 3.0,
                    "_allocation_weight": 3.0,
                    "_qty_stability": 1.0,
                    "_min_batch_qty": 10.0,
                },
                {
                    "菜品名称_库": "菜二",
                    "score": 2.0,
                    "_allocation_weight": 2.0,
                    "_qty_stability": 1.0,
                    "_min_batch_qty": 20.0,
                },
                {
                    "菜品名称_库": "菜三",
                    "score": 1.0,
                    "_allocation_weight": 1.0,
                    "_qty_stability": 1.0,
                    "_min_batch_qty": 30.0,
                },
            ]
        )
        result, removed = policy._allocate_and_prune_min_batch(
            selected,
            60.0,
            {
                "allocation_uncertainty_power": 0.0,
                "min_batch_mode": "dish_q25_prune",
                "capacity_min": 1,
            },
        )
        self.assertGreaterEqual(removed, 1)
        self.assertAlmostEqual(result["pred_qty"].sum(), 60.0)
        self.assertTrue(
            (result["pred_qty"] + 1e-9 >= result["_min_batch_qty"]).all()
        )

    def test_adaptive_runner_has_no_fixed_trial_cap_or_external_sales(self) -> None:
        source = inspect.getsource(adaptive_search)
        self.assertNotIn("MAX_TRIALS", source)
        self.assertNotIn("trial_specs", source)
        self.assertNotIn("菜品销量_吉大湖畔", source)
        records = adaptive_search._load_trials()
        self.assertTrue(records)
        self.assertTrue(all(not row["external_data_read"] for row in records))
        self.assertEqual(
            len({row["evaluation_schedule_sha256"] for row in records}), 1
        )

    def test_fixed_evaluator_hash(self) -> None:
        digest = hashlib.sha256(
            policy.EVALUATOR_PATH.read_bytes()
        ).hexdigest()
        self.assertEqual(
            digest, adaptive_search.FIXED_EVALUATOR_SHA256
        )

    def test_real_training_schedule_starts_after_14_days(self) -> None:
        train = policy.read_sales(policy.TRAIN_PATH)
        warmup, evaluation, first_eval = policy.training_backtest_frames(train, 14)
        self.assertEqual(first_eval, pd.Timestamp("2025-08-04"))
        self.assertEqual(evaluation["date"].min(), pd.Timestamp("2025-08-04"))
        self.assertEqual(evaluation["date"].max(), pd.Timestamp("2025-12-31"))
        self.assertGreaterEqual(warmup["date"].nunique(), 14)


if __name__ == "__main__":
    unittest.main()

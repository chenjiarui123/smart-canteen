from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.history_store import HistoryStore
from app.schemas import RecommendationFinalizeRequest
from app.low_frequency_recommender import LowFrequencyRecommender
from app.recommender import RecommenderService
from app.recommendation_merger import RecommendationMerger, RecommendationProposal
from app.schemas import RecommendationRequest


def _history(holiday_qty: float = 1000.0, ordinary: bool = True) -> pd.DataFrame:
    rows = []
    if ordinary:
        for day, qty in [("2026-05-01", 10.0), ("2026-05-08", 12.0)]:
            rows.append((day, 0, "非节假日", qty))
    rows.append(("2026-06-20", 1, "端午节", holiday_qty))
    result = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(day),
                "食堂名称": "测试食堂",
                "餐次": "午餐",
                "菜品名称": "蜜枣粽子",
                "菜品总销量": qty,
                "菜品售卖单位": "份",
                "订单数量": qty * 0.8,
                "销售额": qty * 6.0,
                "是否法定节假日假期": holiday,
                "法定节假日名称": holiday_name,
                "气温（℃）": 20.0,
                "降水量（毫米）": 0.0,
            }
            for day, holiday, holiday_name, qty in rows
        ]
    )
    result["_dish_norm"] = "蜜枣粽子"
    return result


def _attributes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "_dish_norm": "蜜枣粽子",
                "菜品名称_库": "蜜枣粽子",
                "菜品类别": "主食",
                "_library_dish_category": "主食",
                "主要食材": "糯米",
                "食材大类": "主食",
                "制作方法": "蒸",
                "口味": "甜",
                "相似菜分组": "粽子",
                "主食标记": True,
                "inferred_main_ingredient": "糯米",
                "ingredient_family": "主食",
                "attribute_source": "library",
                "inference_confidence": "high",
                "library_matched": True,
                "_similarity_hard": True,
            }
        ]
    )


def _target(holiday_name: str = "非节假日") -> dict:
    return {
        "法定节假日名称": holiday_name,
        "气温（℃）": 20.0,
        "降水量（毫米）": 0.0,
    }


def _generate(
    history: pd.DataFrame,
    holiday_name: str = "非节假日",
    recommendation_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return LowFrequencyRecommender().generate(
        history=history,
        attributes=_attributes(),
        recommendation_history=(
            recommendation_history
            if recommendation_history is not None
            else pd.DataFrame()
        ),
        target_date=pd.Timestamp("2026-07-01"),
        canteen="测试食堂",
        meal="午餐",
        target_context=_target(holiday_name),
        high_frequency_dishes=set(),
    )


def test_festival_dish_can_enter_ordinary_pool_without_using_holiday_sales():
    ordinary = _generate(_history(holiday_qty=1000.0))
    changed_holiday = _generate(_history(holiday_qty=100000.0))
    assert list(ordinary["菜品名称"]) == ["蜜枣粽子"]
    assert ordinary.iloc[0]["holiday_score_used"] is False or not bool(
        ordinary.iloc[0]["holiday_score_used"]
    )
    assert ordinary.iloc[0]["ordinary_sample_count"] == 2
    assert ordinary.iloc[0]["ordinary_avg_qty"] == 11.0
    assert ordinary.iloc[0]["score"] == changed_holiday.iloc[0]["score"]
    assert ordinary.iloc[0]["pred_qty"] == changed_holiday.iloc[0]["pred_qty"]
    assert "未使用端午节销量" in ordinary.iloc[0]["recommendation_reason"]


def test_only_matching_holiday_can_use_shrunk_holiday_performance():
    dragon_boat = _generate(_history(), "端午节").iloc[0]
    mid_autumn = _generate(_history(), "中秋节").iloc[0]
    assert bool(dragon_boat["holiday_score_used"])
    assert dragon_boat["matched_holiday_name"] == "端午节"
    assert dragon_boat["matching_holiday_sample_count"] == 1
    assert dragon_boat["score"] > mid_autumn["score"]
    assert not bool(mid_autumn["holiday_score_used"])
    assert pd.isna(mid_autumn["matched_holiday_name"])
    assert "未使用端午节销量" in mid_autumn["recommendation_reason"]


def test_holiday_only_dish_is_low_weight_on_ordinary_date():
    result = _generate(_history(ordinary=False)).iloc[0]
    assert result["ordinary_sample_count"] == 0
    assert not bool(result["holiday_score_used"])
    assert result["suggested_trial_qty"] <= 1.0
    assert "样本不足" in result["recommendation_reason"]


def test_formal_low_frequency_selection_progressively_reduces_score():
    baseline = _generate(_history()).iloc[0]
    formal = pd.DataFrame(
        [
            {
                "date": "2026-06-25",
                "食堂名称": "测试食堂",
                "餐次": "午餐",
                "菜品名称": "蜜枣粽子",
                "recommendation_source": "low_frequency_opportunity",
            },
            {
                "date": "2026-06-26",
                "食堂名称": "测试食堂",
                "餐次": "午餐",
                "菜品名称": "蜜枣粽子",
                "recommendation_source": "low_frequency_opportunity",
            },
        ]
    )
    penalised = _generate(_history(), recommendation_history=formal).iloc[0]
    assert penalised["selection_penalty"] < baseline["selection_penalty"]
    assert penalised["score"] < baseline["score"]


@pytest.mark.parametrize("holiday", ["非节假日", "端午节", "中秋节"])
def test_reason_copy_does_not_change_opportunity_selection_or_numbers(monkeypatch, holiday):
    current = _generate(_history(), holiday)
    monkeypatch.setattr("app.low_frequency_recommender.low_frequency_reason", lambda **kwargs: "旧版说明")
    reference = _generate(_history(), holiday)
    columns = [c for c in current.columns if c not in {"reason", "recommendation_reason"}]
    pd.testing.assert_frame_equal(current[columns], reference[columns], check_exact=True)


def test_soft_warning_does_not_show_recommended_limit():
    rows = []
    for index in range(20):
        rows.append(
            {
                "菜品名称": f"测试菜{index}",
                "recommendation_source": (
                    "low_frequency_opportunity" if index == 0 else "high_frequency"
                ),
                "主要食材": "鸡肉",
                "食材大类": "荤菜",
                "制作方法": "炒",
                "口味": "咸",
                "菜品类别": "荤菜",
                "相似菜分组": "unknown",
            }
        )
    warnings = RecommendationMerger._soft_warnings(pd.DataFrame(rows))
    assert warnings
    assert all("超过建议值" not in warning for warning in warnings)
    assert any("共20道" in warning for warning in warnings)


class _FailingLowFrequency:
    def generate(self, **_kwargs):
        raise RuntimeError("simulated low-frequency failure")


class _RotatingLowFrequency:
    def generate(
        self,
        *,
        refresh_nonce=0,
        category="全部",
        candidate_count=12,
        excluded_dishes=None,
        **_kwargs,
    ):
        excluded = set(excluded_dishes or set())
        rows = []
        index = 0
        while len(rows) < min(candidate_count, 3):
            dish = f"潜力测试菜{refresh_nonce}-{index}"
            index += 1
            if dish in excluded:
                continue
            rows.append(
                {
                    "date": pd.Timestamp("2026-02-05"),
                    "食堂名称": "A湖畔智慧餐厅",
                    "餐次": "午餐",
                    "菜品名称": dish,
                    "_dish_norm": dish,
                    "pred_qty": 2.0,
                    "suggested_trial_qty": 2.0,
                    "pred_orders": 1.0,
                    "pred_rev": 10.0,
                    "score": 8.0,
                    "菜品售卖单位": "份",
                    "菜品类别": "荤菜" if category == "全部" else category,
                    "主要食材": "unknown",
                    "食材大类": "unknown",
                    "制作方法": "unknown",
                    "口味": "unknown",
                    "相似菜分组": "unknown",
                    "_similarity_hard": False,
                    "recommendation_source": "low_frequency_opportunity",
                    "recommendation_reason": "测试潜力候选",
                    "reason": "测试潜力候选",
                }
            )
        return pd.DataFrame(rows)


def _request() -> RecommendationRequest:
    return RecommendationRequest(
        dataset_id="a_hupan",
        canteen="A湖畔智慧餐厅",
        prediction_date=date(2026, 2, 5),
        meal="午餐",
        school_break=False,
        statutory_holiday=False,
        adjusted_workday=False,
        temperature_c=10.0,
        humidity_pct=60.0,
        precipitation_mm=0.0,
        wind_speed_kmh=5.0,
    )


def test_low_frequency_failure_never_blocks_frozen_high_frequency(
    temp_repository, tmp_path: Path
):
    service = RecommenderService(
        temp_repository,
        HistoryStore(temp_repository, tmp_path / "recommendations"),
        low_frequency_recommender=_FailingLowFrequency(),
    )
    result = service.recommend(_request())
    assert result["status"] == "draft"
    assert result["low_frequency_status"] == "unavailable"
    assert result["low_frequency_opportunities"] == []
    assert result["high_frequency_menu"]
    assert result["menu"] == result["high_frequency_menu"]


def _category_history_and_attributes() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    attrs = []
    categories = ["荤菜"] * 14 + ["水果"] * 6 + ["素菜"] * 8
    for index, category in enumerate(categories):
        dish = f"{category}候选{index:02d}"
        attrs.append(
            {
                "_dish_norm": dish,
                "菜品名称_库": dish,
                "菜品类别": category,
                "_library_dish_category": category,
            }
        )
        for day, qty in [("2026-05-01", 20 + index), ("2026-05-08", 22 + index)]:
            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "食堂名称": "测试食堂",
                    "餐次": "午餐",
                    "菜品名称": dish,
                    "菜品总销量": float(qty),
                    "菜品售卖单位": "份",
                    "订单数量": float(qty) * 0.8,
                    "销售额": float(qty) * 5,
                    "是否法定节假日假期": 0,
                    "法定节假日名称": "非节假日",
                    "气温（℃）": 20.0,
                    "降水量（毫米）": 0.0,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(attrs)


@pytest.mark.parametrize(
    ("category", "count"),
    [("荤菜", 10), ("水果", 4), ("全部", 20)],
)
def test_category_filter_runs_before_ranking_and_honours_requested_limit(
    category, count
):
    history, attributes = _category_history_and_attributes()
    result = LowFrequencyRecommender().generate(
        history=history,
        attributes=attributes,
        recommendation_history=pd.DataFrame(),
        target_date=pd.Timestamp("2026-07-01"),
        canteen="测试食堂",
        meal="午餐",
        target_context=_target(),
        high_frequency_dishes=set(),
        category=category,
        candidate_count=count,
    )
    assert len(result) <= count
    assert result["菜品名称"].is_unique
    if category != "全部":
        assert set(result["菜品类别"]) == {category}
        assert "其他" not in set(result["菜品类别"])


def test_selected_potential_dish_survives_three_refreshes_and_can_finalize(
    temp_repository, tmp_path: Path
):
    service = RecommenderService(
        temp_repository,
        HistoryStore(temp_repository, tmp_path / "recommendations"),
        low_frequency_recommender=_RotatingLowFrequency(),
    )
    draft = service.recommend(_request())
    selected = draft["low_frequency_opportunities"][0]["dish_name"]
    for refresh in range(1, 4):
        request = _request().model_copy(
            update={
                "opportunity_refresh": refresh,
                "previous_proposal_id": draft["proposal_id"],
                "selected_low_frequency_dishes": [selected],
                "low_frequency_category": "荤菜",
                "low_frequency_count": 3,
            }
        )
        draft = service.recommend(request)
        assert selected not in {
            item["dish_name"] for item in draft["low_frequency_opportunities"]
        }
        proposal = service.proposals.get(draft["proposal_id"])
        assert selected in set(proposal.low_frequency["菜品名称"])
    finalized = service.finalize(
        RecommendationFinalizeRequest(
            proposal_id=draft["proposal_id"],
            selected_high_frequency=[
                item["dish_name"] for item in draft["high_frequency_menu"]
            ],
            selected_low_frequency=[selected],
        )
    )
    assert selected in {item["dish_name"] for item in finalized["menu"]}
    assert len({item["dish_name"] for item in finalized["menu"]}) == len(
        finalized["menu"]
    )


def test_refresh_rejects_retained_opportunity_after_context_change(
    temp_repository, tmp_path: Path
):
    service = RecommenderService(
        temp_repository,
        HistoryStore(temp_repository, tmp_path / "recommendations"),
        low_frequency_recommender=_RotatingLowFrequency(),
    )
    draft = service.recommend(_request())
    selected = draft["low_frequency_opportunities"][0]["dish_name"]
    changed = _request().model_copy(
        update={
            "opportunity_refresh": 1,
            "previous_proposal_id": draft["proposal_id"],
            "selected_low_frequency_dishes": [selected],
            "precipitation_mm": 1.0,
        }
    )
    with pytest.raises(ValueError, match="天气或校历已变化"):
        service.recommend(changed)


def test_concurrent_finalize_consumes_proposal_once(
    temp_repository, tmp_path: Path, monkeypatch
):
    store = HistoryStore(temp_repository, tmp_path / "recommendations")
    service = RecommenderService(
        temp_repository,
        store,
        low_frequency_recommender=_RotatingLowFrequency(),
    )
    draft = service.recommend(_request())
    finalize_request = RecommendationFinalizeRequest(
        proposal_id=draft["proposal_id"],
        selected_high_frequency=[
            item["dish_name"] for item in draft["high_frequency_menu"]
        ],
        selected_low_frequency=[],
    )
    real_save = store.save
    save_calls = 0
    counter_lock = threading.Lock()
    start = threading.Barrier(2)

    def slow_save(**kwargs):
        nonlocal save_calls
        with counter_lock:
            save_calls += 1
        time.sleep(0.2)
        return real_save(**kwargs)

    monkeypatch.setattr(store, "save", slow_save)

    def invoke():
        start.wait()
        try:
            service.finalize(finalize_request)
            return "finalized"
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))

    assert results.count("finalized") == 1
    assert sum("候选方案不存在或已过期" in result for result in results) == 1
    assert save_calls == 1
    history = temp_repository.read_recommendation_history("a_hupan")
    assert history["request_id"].nunique() == 1


def test_merger_reallocates_total_and_checks_combined_menu():
    high = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-01"),
                "食堂名称": "测试食堂",
                "餐次": "午餐",
                "菜品名称": "高频菜",
                "_dish_norm": "高频菜",
                "pred_qty": 100.0,
                "pred_orders": 80.0,
                "pred_rev": 500.0,
                "expected_orders_per_qty": 0.8,
                "expected_rev_per_qty": 5.0,
                "score": 100.0,
                "recommendation_source": "high_frequency",
                "主要食材": "鸡肉",
                "食材大类": "鸡肉",
                "制作方法": "炒",
                "口味": "咸",
                "菜品类别": "荤菜",
                "相似菜分组": "unknown",
                "_similarity_hard": False,
            }
        ]
    )
    low = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-01"),
                "食堂名称": "测试食堂",
                "餐次": "午餐",
                "菜品名称": "潜力菜",
                "_dish_norm": "潜力菜",
                "pred_qty": 20.0,
                "suggested_trial_qty": 20.0,
                "pred_orders": 10.0,
                "pred_rev": 120.0,
                "expected_orders_per_qty": 0.5,
                "expected_rev_per_qty": 6.0,
                "score": 20.0,
                "recommendation_source": "low_frequency_opportunity",
                "主要食材": "糯米",
                "食材大类": "主食",
                "制作方法": "蒸",
                "口味": "甜",
                "菜品类别": "主食",
                "相似菜分组": "unknown",
                "_similarity_hard": False,
            }
        ]
    )
    proposal = RecommendationProposal(
        proposal_id="0" * 32,
        created_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        dataset_id="test",
        canteen="测试食堂",
        meal="午餐",
        target_date=pd.Timestamp("2026-07-01"),
        high_frequency=high,
        low_frequency=low,
        summary={"menu_capacity": 1, "pred_total_qty": 100.0},
        excluded_candidates=[],
    )
    merged, check = RecommendationMerger().merge(
        proposal, ["高频菜"], ["潜力菜"]
    )
    assert set(merged["recommendation_source"]) == {
        "high_frequency",
        "low_frequency_opportunity",
    }
    # Potential dishes are additional trial items: exceeding the reference
    # count is allowed and does not consume the high-frequency slot.
    assert abs(merged["pred_qty"].sum() - 120.0) < 1e-9
    assert not check["pred_total_conserved"]
    assert not check["redistributed_to_total_demand"]
    assert check["menu_count_difference"] == 1
    assert check["reference_warnings"]
    assert check["total_constraint_violation_count"] == 0
    assert check["selected_count"] == 2

    redistributed, redistributed_check = RecommendationMerger().merge(
        proposal, ["高频菜"], ["潜力菜"], True
    )
    assert abs(redistributed["pred_qty"].sum() - 100.0) < 1e-9
    assert redistributed_check["pred_total_conserved"]
    assert redistributed_check["redistributed_to_total_demand"]
    assert redistributed_check["selected_high_frequency_count"] == 1
    assert redistributed_check["selected_low_frequency_count"] == 1

    high_only, high_check = RecommendationMerger().merge(
        proposal, ["高频菜"], []
    )
    assert abs(high_only["pred_qty"].sum() - 100.0) < 1e-9
    assert high_check["selected_low_frequency_count"] == 0

    for redistribute in (False, True):
        with pytest.raises(ValueError, match="至少保留一道高频菜单菜品"):
            RecommendationMerger().merge(
                proposal, [], ["潜力菜"], redistribute
            )

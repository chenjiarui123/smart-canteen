from pathlib import Path

import pandas as pd
import pytest

from app.history_store import HistoryStore
from app.recommender import RecommenderService
from app.schemas import RecommendationRequest, RecommendationFinalizeRequest


def request(day):
    return RecommendationRequest(dataset_id="a_hupan", canteen="A湖畔智慧餐厅", prediction_date=day,
                                 meal="午餐", school_break=False, statutory_holiday=False,
                                 adjusted_workday=False, temperature_c=10, humidity_pct=60,
                                 precipitation_mm=0, wind_speed_kmh=5)


def test_saved_automatic_reference_and_confirmed_priority(temp_repository, tmp_path):
    store = HistoryStore(temp_repository, tmp_path / "out")
    day = pd.Timestamp("2026-02-04")
    store.save_automatic_high_frequency("a_hupan", "A湖畔智慧餐厅", "午餐", day,
                                       pd.DataFrame({"菜品名称": ["自动菜一", "自动菜二"]}))
    dishes, source = store.previous_day_reference("a_hupan", "A湖畔智慧餐厅", "午餐", day + pd.Timedelta(days=1))
    assert source == "saved_high_frequency" and dishes == {"自动菜一", "自动菜二"}
    assert store.previous_day_reference("a_hupan", "A湖畔智慧餐厅", "晚餐", day + pd.Timedelta(days=1))[0] is None
    assert store.previous_day_reference("a_hupan", "A湖畔智慧餐厅", "午餐", day + pd.Timedelta(days=2))[0] is None
    rows = []
    for rid, stamp, dish in [("first", "2026-02-04T10:00:00", "旧版菜"), ("second", "2026-02-04T11:00:00", "新版菜")]:
        rows.append(dict(request_id=rid, created_at=stamp, date="2026-02-04", 食堂名称="A湖畔智慧餐厅",
                         餐次="午餐", 菜品名称=dish, pred_qty=10, pred_orders=5, pred_rev=20, score=1,
                         recommendation_source="high_frequency", reason="test"))
    with temp_repository.write_lock():
        temp_repository.append_recommendation_history_unlocked("a_hupan", pd.DataFrame(rows))
    dishes, source = store.previous_day_reference("a_hupan", "A湖畔智慧餐厅", "午餐", day + pd.Timedelta(days=1))
    assert source == "confirmed" and dishes == {"新版菜"}


def test_manual_override_warns_and_keeps_selection(temp_repository, tmp_path):
    store = HistoryStore(temp_repository, tmp_path / "out")
    service = RecommenderService(temp_repository, store)
    draft = service.recommend(request("2026-02-05"))
    selected = [r["dish_name"] for r in draft["high_frequency_menu"]]
    # An updated previous-day confirmation/reference after draft creation must
    # be checked again at finalize, without denying human selection rights.
    store.save_automatic_high_frequency("a_hupan", "A湖畔智慧餐厅", "午餐", pd.Timestamp("2026-02-04"),
                                       pd.DataFrame({"菜品名称": selected}))
    result = service.finalize(RecommendationFinalizeRequest(proposal_id=draft["proposal_id"],
                              selected_high_frequency=selected, selected_low_frequency=[]))
    assert result["summary"]["manual_adjacent_repeat_rate"] == 1
    assert "超过80%" in result["summary"]["adjacent_warning"]
    assert len(result["menu"]) == len(selected)


def test_api_ready_strategy_refuses_infeasible_without_shrinking(temp_repository, tmp_path):
    from app.active_strategy import active_config
    if not active_config().get("targeted_enabled"):
        pytest.skip("targeted release not frozen yet")
    store = HistoryStore(temp_repository, tmp_path / "out")
    service = RecommenderService(temp_repository, store)
    draft = service.recommend(request("2026-02-05"))
    names = [r["dish_name"] for r in draft["high_frequency_menu"]]
    store.save_automatic_high_frequency("a_hupan", "A湖畔智慧餐厅", "午餐", pd.Timestamp("2026-02-04"), pd.DataFrame({"菜品名称": names}))
    with pytest.raises(ValueError, match="缺少.*替换"):
        service.recommend(request("2026-02-05"))

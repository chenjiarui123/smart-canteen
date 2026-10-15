from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from app.history_store import HistoryStore, MENU_EXPORT_COLUMNS
from app.recommender import RecommenderService
from app.schemas import RecommendationRequest
from app.schemas import RecommendationFinalizeRequest


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


def test_end_to_end_recommendation_is_saved_without_overwrite(
    temp_repository, tmp_path: Path
):
    output_root = tmp_path / "recommendations"
    store = HistoryStore(temp_repository, output_root)
    service = RecommenderService(temp_repository, store)

    first_draft = service.recommend(_request())
    first = service.finalize(
        RecommendationFinalizeRequest(
            proposal_id=first_draft["proposal_id"],
            selected_high_frequency=[
                item["dish_name"]
                for item in first_draft["high_frequency_menu"]
            ],
            selected_low_frequency=[],
        )
    )
    second_draft = service.recommend(_request())
    second = service.finalize(
        RecommendationFinalizeRequest(
            proposal_id=second_draft["proposal_id"],
            selected_high_frequency=[
                item["dish_name"]
                for item in second_draft["high_frequency_menu"]
            ],
            selected_low_frequency=[],
        )
    )

    assert first["recommendation_id"] != second["recommendation_id"]
    assert first["menu"] == second["menu"]
    assert all(
        item["recommendation_source"] == "high_frequency"
        for item in first["menu"]
    )
    folder = output_root / "a_hupan" / "午餐"
    assert (folder / "2026-02-05.csv").exists()
    assert (folder / "2026-02-05.md").exists()
    assert (folder / "2026-02-05_v2.csv").exists()
    assert (folder / "2026-02-05_v2.md").exists()
    assert (folder / "2026-02-05.csv").read_bytes().startswith(
        b"\xef\xbb\xbf"
    )
    exported = pd.read_csv(folder / "2026-02-05.csv", encoding="utf-8-sig")
    assert exported.columns.tolist() == MENU_EXPORT_COLUMNS
    assert exported["菜品id"].notna().all()
    assert exported["菜品名"].is_unique
    history = temp_repository.read_recommendation_history("a_hupan")
    assert set(history["request_id"]) == {
        first["recommendation_id"],
        second["recommendation_id"],
    }
    assert len(history) == len(first["menu"]) + len(second["menu"])


def test_download_export_uses_total_grams_but_history_keeps_model_quantity(
    temp_repository, tmp_path: Path
):
    store = HistoryStore(temp_repository, tmp_path / "recommendations")
    menu = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-02-05"),
                "食堂名称": "A湖畔智慧餐厅",
                "餐次": "晚餐",
                "菜品名称": "a_hupan_菜1",
                "菜品类别": "荤菜",
                "菜品售卖单位": "克",
                "pred_qty": 2.5,
                "pred_orders": 1.0,
                "pred_rev": 18.126,
                "score": 9.0,
                "recommendation_source": "high_frequency",
                "reason": "测试",
            },
            {
                "date": pd.Timestamp("2026-02-05"),
                "食堂名称": "A湖畔智慧餐厅",
                "餐次": "晚餐",
                "菜品名称": "a_hupan_菜2",
                "菜品类别": "素菜",
                "菜品售卖单位": "份",
                "pred_qty": 3.25,
                "pred_orders": 2.0,
                "pred_rev": 12.345,
                "score": 8.0,
                "recommendation_source": "high_frequency",
                "reason": "测试",
            },
        ]
    )
    recommendation_id, csv_path, _ = store.save(
        dataset_id="a_hupan",
        canteen="A湖畔智慧餐厅",
        meal="晚餐",
        prediction_date=pd.Timestamp("2026-02-05"),
        recommendations=menu,
        markdown="# 测试",
    )
    assert recommendation_id
    export = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert export.columns.tolist() == MENU_EXPORT_COLUMNS
    assert export.loc[export["菜品名"].eq("a_hupan_菜1"), "菜品销量"].iloc[0] == 125
    assert export.loc[export["菜品名"].eq("a_hupan_菜2"), "菜品销量"].iloc[0] == 3.25
    assert export["菜品销售额"].tolist() == [18.126, 12.345]
    assert export["菜品id"].astype(str).tolist() == ["1", "2"]

    history = temp_repository.read_recommendation_history("a_hupan")
    assert history["pred_qty"].tolist() == [2.5, 3.25]

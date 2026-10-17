from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from app.context_builder import build_inference_context, build_target_context
from app.dataset_repository import DatasetRepository, normalize_dish_name
from app.high_frequency_recommender import HighFrequencyRecommender
from app.model_loader import (
    MANIFEST,
    constraints_copy,
    frozen_policy,
    frozen_policy_copy,
    make_recommendations_with_history,
)


def _context(repository: DatasetRepository, dataset_id: str, meal: str):
    dataset = repository.dataset(dataset_id)
    history = repository.combined_sales_for_model(dataset_id)
    attributes, _ = frozen_policy.build_library_attributes(
        repository.dataset_path(dataset_id, "dish_library.csv")
    )
    target = pd.Timestamp("2026-02-05")
    external = build_target_context(
        target,
        meal,
        school_break=False,
        school_break_type="非寒暑假",
        statutory_holiday=False,
        statutory_holiday_name="非节假日",
        adjusted_workday=False,
        temperature_c=10.0,
        humidity_pct=60.0,
        precipitation_mm=0.0,
        wind_speed_kmh=5.0,
    )
    return build_inference_context(
        history=history,
        attributes=attributes,
        prediction_date=target,
        canteen=dataset["display_name"],
        meal=meal,
        target_context=external,
    )


def test_two_logical_datasets_never_mix(temp_repository):
    a = temp_repository.combined_sales_for_model("a_hupan")
    f = temp_repository.combined_sales_for_model("f_jichuyuan")
    assert set(a["食堂名称"]) == {"A湖畔智慧餐厅"}
    assert set(f["食堂名称"]) == {"F基础园智慧餐厅"}
    assert set(a["_dish_norm"]).isdisjoint(set(f["_dish_norm"]))


def test_inference_context_strictly_uses_prior_history_and_meals_are_independent(
    temp_repository,
):
    lunch = _context(temp_repository, "a_hupan", "午餐")
    dinner = _context(temp_repository, "a_hupan", "晚餐")
    target = pd.Timestamp("2026-02-05")
    assert lunch["max_history_date"] < target
    assert dinner["max_history_date"] < target
    assert lunch["餐次"] == "午餐"
    assert dinner["餐次"] == "晚餐"
    assert lunch["meta"]["l_session_count"] == dinner["meta"]["l_session_count"]


def test_empty_persistent_history_matches_frozen_policy_exactly(temp_repository):
    context = _context(temp_repository, "a_hupan", "午餐")
    frozen = frozen_policy.make_recommendations(
        [copy.deepcopy(context)],
        frozen_policy_copy(),
        constraints_copy(),
    ).sort_values("菜品名称").reset_index(drop=True)
    adapted = make_recommendations_with_history(
        [copy.deepcopy(context)],
        frozen_policy_copy(),
        constraints_copy(),
        initial_menu_history={},
    ).sort_values("菜品名称").reset_index(drop=True)
    pdt.assert_frame_equal(frozen, adapted, check_exact=True)


def test_high_frequency_adapter_preserves_active_candidates_predictions_and_scores(
    temp_repository,
):
    context = _context(temp_repository, "a_hupan", "午餐")
    from app.active_strategy import active_config, generate
    baseline = generate(
        [copy.deepcopy(context)],
        active_config(),
        constraints_copy(),
        initial_menu_history={},
    ).sort_values("菜品名称").reset_index(drop=True)
    history = temp_repository.combined_sales_for_model("a_hupan")
    result = HighFrequencyRecommender().generate(
        context=copy.deepcopy(context),
        history=history.loc[history["date"].lt(pd.Timestamp("2026-02-05"))],
        target_date=pd.Timestamp("2026-02-05"),
        canteen="A湖畔智慧餐厅",
        meal="午餐",
        initial_menu_history={},
        prior_norms=set(),
        recent7={},
        recent14={},
        sale_units={},
    ).recommendations.sort_values("菜品名称").reset_index(drop=True)
    unchanged_columns = [
        column for column in baseline.columns if column != "reason"
    ]
    pdt.assert_frame_equal(
        baseline[unchanged_columns],
        result[unchanged_columns],
        check_exact=True,
    )
    assert set(result["recommendation_source"]) == {"high_frequency"}


def test_persisted_recommendation_history_changes_repeat_penalty(temp_repository):
    context = _context(temp_repository, "a_hupan", "午餐")
    baseline = make_recommendations_with_history(
        [copy.deepcopy(context)],
        frozen_policy_copy(),
        constraints_copy(),
        initial_menu_history={},
    )
    repeated = set(baseline["菜品名称"].map(lambda value: str(value).lower()))
    seed = {
        ("A湖畔智慧餐厅", "午餐"): [
            (pd.Timestamp("2026-02-04"), repeated),
            (pd.Timestamp("2026-02-03"), repeated),
        ]
    }
    penalised = make_recommendations_with_history(
        [copy.deepcopy(context)],
        frozen_policy_copy(),
        constraints_copy(),
        initial_menu_history=seed,
    )
    assert penalised["score"].sum() < baseline["score"].sum()


def test_far_future_inactivity_is_measured_at_history_end(temp_repository):
    history = temp_repository.combined_sales_for_model("a_hupan").copy()
    shift = pd.Timestamp("2026-06-10") - history["date"].max()
    history["date"] = history["date"] + shift
    target = pd.Timestamp("2026-07-10")
    recent_non_staple = normalize_dish_name("a_hupan_菜1")
    truly_inactive = normalize_dish_name("a_hupan_菜7")

    # The recent non-staple was available on June 9, while the truly inactive
    # dish appeared only on the first historical day (34 days before June 10).
    history = history.loc[
        ~(
            history["_dish_norm"].eq(recent_non_staple)
            & history["date"].eq(pd.Timestamp("2026-06-10"))
        )
        & ~(
            history["_dish_norm"].eq(truly_inactive)
            & history["date"].gt(history["date"].min())
        )
    ].copy()
    attributes, _ = frozen_policy.build_library_attributes(
        temp_repository.dataset_path("a_hupan", "dish_library.csv")
    )
    categories = {
        normalize_dish_name("a_hupan_菜0"): "主食",
        normalize_dish_name("a_hupan_菜1"): "荤菜",
        normalize_dish_name("a_hupan_菜2"): "荤菜",
        normalize_dish_name("a_hupan_菜3"): "荤菜",
        normalize_dish_name("a_hupan_菜4"): "素菜",
        normalize_dish_name("a_hupan_菜5"): "素菜",
        normalize_dish_name("a_hupan_菜6"): "素菜",
        truly_inactive: "荤菜",
    }
    attributes["菜品类别"] = attributes["_dish_norm"].map(categories)
    attributes["主食标记"] = attributes["菜品类别"].eq("主食")
    external = build_target_context(
        target,
        "晚餐",
        school_break=False,
        school_break_type="非寒暑假",
        statutory_holiday=False,
        statutory_holiday_name="非节假日",
        adjusted_workday=False,
        temperature_c=20,
        humidity_pct=60,
        precipitation_mm=0,
        wind_speed_kmh=5,
    )
    context = build_inference_context(
        history=history,
        attributes=attributes,
        prediction_date=target,
        canteen="A湖畔智慧餐厅",
        meal="晚餐",
        target_context=external,
    )
    assert context["max_history_date"] == pd.Timestamp("2026-06-10")
    assert context["forecast_gap_days"] == 30

    dinner = history.loc[history["餐次"].eq("晚餐")]
    result = HighFrequencyRecommender().generate(
        context=context,
        history=history,
        target_date=target,
        canteen="A湖畔智慧餐厅",
        meal="晚餐",
        initial_menu_history={},
        prior_norms=set(),
        recent7={},
        recent14={},
        sale_units=(
            dinner.sort_values("date")
            .drop_duplicates("_dish_norm", keep="last")
            .set_index("_dish_norm")["菜品售卖单位"]
            .to_dict()
        ),
    )
    candidate_pool = set(result.selection_audit["candidate_pool"])
    assert recent_non_staple in candidate_pool
    assert truly_inactive not in candidate_pool
    assert {"荤菜", "素菜", "主食"}.issubset(
        set(result.recommendations["菜品类别"])
    )
    assert result.summary["forecast_gap_days"] == 30
    assert "远期预测" in result.summary["forecast_gap_warning"]
    assert "增加长期历史占比" in result.summary["forecast_gap_warning"]
    recent_reason = result.recommendations.loc[
        result.recommendations["_dish_norm"].eq(recent_non_staple), "reason"
    ].iloc[0]
    assert "近30天同餐次暂无供应记录" in recent_reason
    # June 10 is inside [target - 30 days, target); other dishes have one record.
    other_reasons = result.recommendations.loc[
        ~result.recommendations["_dish_norm"].eq(recent_non_staple), "reason"
    ]
    assert other_reasons.str.contains("近30天同餐次有1次供应记录", regex=False).all()
    assert not result.recommendations["reason"].str.contains("自适应近期权重", regex=False).any()


def test_frozen_model_hashes_and_source_project_unchanged():
    model_root = Path(__file__).resolve().parents[2] / "model"
    for filename, expected in MANIFEST["frozen_source_hashes"].items():
        import hashlib
        actual = hashlib.sha256((model_root / filename).read_bytes()).hexdigest()
        assert actual == expected
    source_root = next(
        (
            parent / "research"
            for parent in Path(__file__).resolve().parents
            if (parent / "research" / "policy.py").is_file()
        ),
        None,
    )
    if source_root is None:
        import pytest

        pytest.skip("research source tree not present (standalone release build)")
    import hashlib

    for filename, expected in MANIFEST["source_project_integrity_baseline"].items():
        path = source_root / filename
        assert path.is_file(), f"integrity baseline file missing: {path}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_frozen_high_frequency_configs_are_byte_unchanged_and_low_is_independent():
    import hashlib

    project_root = Path(__file__).resolve().parents[2]
    model_root = project_root / "model"
    expected = {
        "best_policy.json": "fb129adf7c44354de5027579309d9bdb3a71f2fa4a2cb22af8f78918b5e6687b",
        "constraints.json": "895425ddebe00c473f794258afb9d7e74551fdcaf3293daf150ce0dbaba20f39",
        "policy.py": "40fc1ff02d494059c6db366da37cfca8fac24190993945872313e21a8a3ef9b5",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((model_root / filename).read_bytes()).hexdigest() == digest
    assert (project_root / "backend/app/low_frequency_recommender.py").exists()
    assert (project_root / "backend/app/low_frequency_config.json").exists()

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from .context_builder import build_inference_context, build_target_context
from .dataset_repository import DatasetRepository, normalize_dish_name
from .high_frequency_recommender import HighFrequencyRecommender
from .history_store import HistoryStore
from .low_frequency_recommender import (
    LowFrequencyRecommender,
    SELECTABLE_CATEGORIES,
    load_low_frequency_config,
    low_frequency_config_sha256,
)
from .model_loader import MANIFEST, frozen_policy, model_status
from .recommendation_merger import ProposalStore, RecommendationMerger
from .report_generator import build_markdown_report
from .schemas import RecommendationFinalizeRequest, RecommendationRequest
from .weather_display import normalized_weather_options
from .active_strategy import active_status


LOGGER = logging.getLogger("smart_canteen.recommender")


class RecommenderService:
    """Orchestrate frozen high-frequency and isolated opportunity modules."""

    def __init__(
        self,
        repository: DatasetRepository,
        history_store: HistoryStore,
        low_frequency_recommender: LowFrequencyRecommender | None = None,
    ):
        self.repository = repository
        self.history_store = history_store
        self.high_frequency_recommender = HighFrequencyRecommender()
        self.low_frequency_recommender = (
            low_frequency_recommender or LowFrequencyRecommender()
        )
        config = load_low_frequency_config()
        self.proposals = ProposalStore(config["proposal_ttl_minutes"])
        self.merger = RecommendationMerger()
        self._finalize_lock = threading.RLock()
        self._attribute_cache: dict[str, tuple[int, pd.DataFrame]] = {}

    def _attributes(self, dataset_id: str) -> pd.DataFrame:
        path = self.repository.dataset_path(dataset_id, "dish_library.csv")
        stamp = path.stat().st_mtime_ns
        cached = self._attribute_cache.get(dataset_id)
        if cached and cached[0] == stamp:
            return cached[1].copy()
        attributes, raw = frozen_policy.build_library_attributes(path)
        forbidden = {
            "历史平均销量",
            "历史平均销售额",
            "总销量_处理后",
            "总销量_原始",
            "总销售额",
            "订单数量",
        }
        leaked = forbidden.intersection(attributes.columns)
        if leaked:
            raise RuntimeError("菜品库历史数值进入了模型属性")
        category_column = next(
            (
                column
                for column in ("菜品类别", "类别")
                if column in raw.columns
            ),
            None,
        )
        if category_column is None:
            category_map: dict[str, str] = {}
        else:
            category_rows = raw[["_dish_norm", category_column]].copy()
            category_rows[category_column] = (
                category_rows[category_column]
                .fillna("其他")
                .astype(str)
                .str.strip()
                .where(
                    lambda values: values.isin(SELECTABLE_CATEGORIES),
                    "其他",
                )
            )
            category_map = (
                category_rows.drop_duplicates("_dish_norm", keep="first")
                .set_index("_dish_norm")[category_column]
                .to_dict()
            )
        # Keep the frozen strategy's existing 菜品类别 column untouched.  The
        # extra column is consumed only by the independent opportunity module.
        attributes["_library_dish_category"] = (
            attributes["_dish_norm"].map(category_map).fillna("其他")
        )
        self._attribute_cache[dataset_id] = (stamp, attributes)
        return attributes.copy()

    def _validate_request_options(
        self, request: RecommendationRequest
    ) -> dict[str, Any]:
        dataset = self.repository.dataset(request.dataset_id)
        if request.canteen != dataset["display_name"]:
            raise ValueError("食堂名称与所选逻辑数据集不一致")
        options = self.repository.ui_options()
        if request.meal not in options["meals"]:
            raise ValueError("餐次不在允许选项中")
        if (
            request.school_break
            and request.school_break_type not in options["school_break_types"]
        ):
            raise ValueError("校历假期类型不在允许选项中")
        if (
            request.statutory_holiday
            and request.statutory_holiday_name
            not in (
                list(options["statutory_holiday_names"])
                + list(load_low_frequency_config()["festival_dish_patterns"])
            )
        ):
            raise ValueError("法定节假日名称不在允许选项中")
        weather = {
            "temperature_c": request.temperature_c,
            "humidity_pct": request.humidity_pct,
            "precipitation_mm": request.precipitation_mm,
            "wind_speed_kmh": request.wind_speed_kmh,
        }
        displayed_weather = normalized_weather_options(options["weather"])
        for key, value in weather.items():
            allowed = displayed_weather[key]["values"]
            numeric = float(value)
            if numeric not in allowed:
                raise ValueError(
                    f"{options['weather'][key]['label']}不在允许下拉范围内"
                )
        return dataset

    @staticmethod
    def _context_signature(request: RecommendationRequest) -> tuple[Any, ...]:
        return (
            request.school_break,
            request.school_break_type,
            request.statutory_holiday,
            request.statutory_holiday_name,
            request.adjusted_workday,
            request.temperature_c,
            request.humidity_pct,
            request.precipitation_mm,
            request.wind_speed_kmh,
        )

    @staticmethod
    def _sale_unit_lookup(
        history: pd.DataFrame, target_date: pd.Timestamp
    ) -> dict[str, str]:
        prior = history.loc[history["date"].lt(target_date)].sort_values("date")
        if prior.empty:
            return {}
        return (
            prior.drop_duplicates("_dish_norm", keep="last")
            .set_index("_dish_norm")["菜品售卖单位"]
            .fillna("份")
            .astype(str)
            .to_dict()
        )

    @staticmethod
    def _optional(row: dict[str, Any], key: str, default: Any = None) -> Any:
        value = row.get(key, default)
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
        return value

    @classmethod
    def _menu_items(cls, frame: pd.DataFrame) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for row in frame.to_dict("records"):
            item = {
                "dish_name": str(row["菜品名称"]),
                "sale_unit": str(row.get("菜品售卖单位", "份")),
                "pred_qty": float(row["pred_qty"]),
                "pred_orders": float(row["pred_orders"]),
                "pred_rev": float(row["pred_rev"]),
                "score": float(row["score"]),
                "dish_category": str(row.get("菜品类别", "unknown")),
                "main_ingredient": str(row.get("主要食材", "unknown")),
                "cooking_method": str(row.get("制作方法", "unknown")),
                "flavor": str(row.get("口味", "unknown")),
                "reason": str(row.get("reason", "")),
                "recommendation_reason": str(
                    row.get("recommendation_reason", row.get("reason", ""))
                ),
                "recommendation_source": str(
                    row.get("recommendation_source", "high_frequency")
                ),
                "ordinary_sample_count": cls._optional(
                    row, "ordinary_sample_count"
                ),
                "ordinary_avg_qty": cls._optional(row, "ordinary_avg_qty"),
                "matching_holiday_sample_count": cls._optional(
                    row, "matching_holiday_sample_count"
                ),
                "matching_holiday_avg_qty": cls._optional(
                    row, "matching_holiday_avg_qty"
                ),
                "holiday_score_used": cls._optional(
                    row, "holiday_score_used"
                ),
                "matched_holiday_name": cls._optional(
                    row, "matched_holiday_name"
                ),
                "context_match_score": cls._optional(
                    row, "context_match_score"
                ),
                "selection_penalty": cls._optional(row, "selection_penalty"),
                "suggested_trial_qty": cls._optional(
                    row, "suggested_trial_qty"
                ),
            }
            for numeric in [
                "ordinary_avg_qty",
                "matching_holiday_avg_qty",
                "context_match_score",
                "selection_penalty",
                "suggested_trial_qty",
            ]:
                if item[numeric] is not None:
                    item[numeric] = float(item[numeric])
            for integer in [
                "ordinary_sample_count",
                "matching_holiday_sample_count",
            ]:
                if item[integer] is not None:
                    item[integer] = int(item[integer])
            if item["holiday_score_used"] is not None:
                item["holiday_score_used"] = bool(item["holiday_score_used"])
            items.append(item)
        return items

    def recommend(self, request: RecommendationRequest) -> dict[str, Any]:
        """Create a two-column draft; do not write formal history yet."""

        dataset = self._validate_request_options(request)
        target_date = pd.Timestamp(request.prediction_date)
        context_signature = self._context_signature(request)
        complete_history = self.repository.combined_sales_for_model(
            request.dataset_id
        )
        history = complete_history.loc[
            complete_history["date"].lt(target_date)
        ].copy()
        if history.empty:
            raise ValueError("所选日期以前没有可用真实销售历史")
        if history["date"].max() >= target_date:
            raise AssertionError("实时推荐读取了目标日或未来实际数据")
        if set(history["食堂名称"].astype(str)) != {dataset["display_name"]}:
            raise RuntimeError("逻辑数据集内部食堂名称不一致")

        target_context = build_target_context(
            target_date,
            request.meal,
            school_break=request.school_break,
            school_break_type=request.school_break_type or "非寒暑假",
            statutory_holiday=request.statutory_holiday,
            statutory_holiday_name=request.statutory_holiday_name or "非节假日",
            adjusted_workday=request.adjusted_workday,
            temperature_c=request.temperature_c,
            humidity_pct=request.humidity_pct,
            precipitation_mm=request.precipitation_mm,
            wind_speed_kmh=request.wind_speed_kmh,
        )
        attributes = self._attributes(request.dataset_id)
        context = build_inference_context(
            history=history,
            attributes=attributes,
            prediction_date=target_date,
            canteen=dataset["display_name"],
            meal=request.meal,
            target_context=target_context,
        )
        initial_history = self.history_store.initial_menu_history(
            request.dataset_id,
            dataset["display_name"],
            request.meal,
            target_date,
        )
        reference, reference_source = self.history_store.previous_day_reference(
            request.dataset_id, dataset["display_name"], request.meal, target_date
        )
        if reference is not None:
            key = (dataset["display_name"], request.meal)
            previous_date = target_date - pd.Timedelta(days=1)
            initial_history[key] = [(d, names) for d, names in initial_history.get(key, []) if d != previous_date]
            initial_history[key].append((previous_date, reference))
            initial_history[key].sort(key=lambda item: item[0])
        prior_norms, recent7, recent14 = self.history_store.repeat_counts(
            initial_history,
            dataset["display_name"],
            request.meal,
            target_date,
        )
        high = self.high_frequency_recommender.generate(
            context=context,
            history=history,
            target_date=target_date,
            canteen=dataset["display_name"],
            meal=request.meal,
            initial_menu_history=initial_history,
            prior_norms=prior_norms,
            recent7=recent7,
            recent14=recent14,
            sale_units=self._sale_unit_lookup(history, target_date),
        )

        retained_low = pd.DataFrame()
        if request.selected_low_frequency_dishes:
            if not request.previous_proposal_id:
                raise ValueError("刷新潜力菜时缺少原候选方案编号")
            previous = self.proposals.get(request.previous_proposal_id)
            same_context = (
                previous.dataset_id == request.dataset_id
                and previous.canteen == dataset["display_name"]
                and previous.meal == request.meal
                and pd.Timestamp(previous.target_date).normalize()
                == target_date.normalize()
            )
            if not same_context:
                raise ValueError("已选菜品属于其他食堂、日期或餐次，请重新生成")
            if previous.context_signature != context_signature:
                raise ValueError(
                    "推荐条件中的天气或校历已变化，请重新生成候选后再选择潜力菜"
                )
            requested_norms = {
                normalize_dish_name(name)
                for name in request.selected_low_frequency_dishes
            }
            available_norms = set(
                previous.low_frequency.get(
                    "_dish_norm", pd.Series(dtype=str)
                ).astype(str)
            )
            if not requested_norms.issubset(available_norms):
                raise ValueError("刷新请求包含不在原候选方案中的潜力菜")
            retained_low = previous.low_frequency.loc[
                previous.low_frequency["_dish_norm"].astype(str).isin(
                    requested_norms
                )
            ].copy()

        low_status = "ok"
        low_warning = None
        try:
            low = self.low_frequency_recommender.generate(
                history=history,
                attributes=attributes,
                recommendation_history=self.repository.read_recommendation_history(
                    request.dataset_id
                ),
                target_date=target_date,
                canteen=dataset["display_name"],
                meal=request.meal,
                target_context=target_context,
                high_frequency_dishes=set(
                    high.recommendations["_dish_norm"].astype(str)
                ),
                refresh_nonce=request.opportunity_refresh,
                category=request.low_frequency_category,
                candidate_count=request.low_frequency_count,
                excluded_dishes=set(
                    retained_low.get("_dish_norm", pd.Series(dtype=str)).astype(
                        str
                    )
                ),
            )
        except Exception:
            LOGGER.exception("低频潜力菜模块失败，已降级为高频菜单")
            low = pd.DataFrame()
            low_status = "unavailable"
            low_warning = "潜力菜本次不可用；常用菜品推荐未受影响，可继续确认保存。"

        proposal_low = pd.concat(
            [retained_low, low], ignore_index=True, sort=False
        )
        if not proposal_low.empty:
            proposal_low = proposal_low.drop_duplicates(
                "_dish_norm", keep="first"
            ).reset_index(drop=True)
        returned_count = int(len(low))
        shortage = returned_count < request.low_frequency_count
        if low_status == "unavailable":
            low_message = "潜力菜暂不可用，已选菜品仍保留在当前方案中。"
        elif shortage:
            low_message = (
                f"计划推荐{request.low_frequency_count}道，当前类别仅有"
                f"{returned_count}道合格候选，未降低筛选标准补足。"
            )
        else:
            low_message = (
                f"已按{request.low_frequency_category}类别返回"
                f"{returned_count}道合格潜力菜。"
            )

        summary = dict(high.summary)
        summary["adjacent_reference_source"] = reference_source
        summary["high_frequency_count"] = int(len(high.recommendations))
        summary["low_frequency_candidate_count"] = int(len(low))
        summary["low_frequency_category"] = request.low_frequency_category
        summary["low_frequency_requested_count"] = request.low_frequency_count
        summary["low_frequency_returned_count"] = returned_count
        summary["low_frequency_shortage"] = shortage
        summary["low_frequency_message"] = low_message
        summary["total_preparation_qty"] = float(
            high.recommendations["pred_qty"].sum()
        )
        summary["redistributed_to_total_demand"] = False
        summary["menu_count_reference_exceeded"] = bool(
            len(high.recommendations) > int(summary["menu_capacity"])
        )
        summary["selection_status"] = "待确认"
        proposal = self.proposals.add(
            dataset_id=request.dataset_id,
            canteen=dataset["display_name"],
            meal=request.meal,
            target_date=target_date,
            high_frequency=high.recommendations,
            low_frequency=proposal_low,
            summary=summary,
            excluded_candidates=high.excluded_candidates,
            context_signature=context_signature,
        )
        high_items = self._menu_items(high.recommendations)
        low_items = self._menu_items(low)
        self.history_store.save_automatic_high_frequency(
            request.dataset_id, dataset["display_name"], request.meal, target_date, high.recommendations
        )
        return {
            "status": "draft",
            "proposal_id": proposal.proposal_id,
            "recommendation_id": None,
            "summary": summary,
            # Backward-compatible alias: the original high-frequency menu.
            "menu": high_items,
            "high_frequency_menu": high_items,
            "low_frequency_opportunities": low_items,
            "low_frequency_status": low_status,
            "low_frequency_warning": low_warning,
            "combination_check": None,
            "excluded_candidates": high.excluded_candidates,
            "csv_download_url": None,
            "markdown_download_url": None,
        }

    def finalize(
        self, request: RecommendationFinalizeRequest
    ) -> dict[str, Any]:
        with self._finalize_lock:
            return self._finalize_locked(request)

    def _finalize_locked(
        self, request: RecommendationFinalizeRequest
    ) -> dict[str, Any]:
        proposal = self.proposals.get(request.proposal_id)
        combined, combination_check = self.merger.merge(
            proposal,
            request.selected_high_frequency,
            request.selected_low_frequency,
            request.redistribute_to_total_demand,
        )
        summary = dict(proposal.summary)
        reference, source = self.history_store.previous_day_reference(
            proposal.dataset_id, proposal.canteen, proposal.meal, proposal.target_date
        )
        names = set(combined["菜品名称"].map(normalize_dish_name))
        manual_rate = len(names & reference) / len(names) if names and reference else None
        summary["manual_adjacent_repeat_rate"] = manual_rate
        if manual_rate is not None and manual_rate > .8:
            warning = f"人工确认菜单与前一天同餐次重复率为{manual_rate:.1%}，超过80%；保留您的选菜结果。"
            combination_check.setdefault("reference_warnings", []).append(warning)
            summary["adjacent_warning"] = warning
        summary.update(
            {
                "recommended_count": int(len(combined)),
                # Keep the frozen meal-demand forecast distinct from the
                # reference preparation sum when opportunity dishes are added.
                "pred_total_qty": float(
                    combination_check["predicted_session_demand_qty"]
                ),
                "pred_total_orders": float(combined["pred_orders"].sum()),
                "pred_total_rev": float(combined["pred_rev"].sum()),
                "total_preparation_qty": float(
                    combination_check["total_preparation_qty"]
                ),
                "redistributed_to_total_demand": bool(
                    combination_check["redistributed_to_total_demand"]
                ),
                "menu_count_reference_exceeded": bool(
                    combination_check["menu_count_difference"] > 0
                ),
                "high_frequency_count": int(
                    combined["recommendation_source"].eq("high_frequency").sum()
                ),
                "low_frequency_candidate_count": int(
                    len(proposal.low_frequency)
                ),
                "selected_low_frequency_count": int(
                    combined["recommendation_source"].eq(
                        "low_frequency_opportunity"
                    ).sum()
                ),
                "selection_status": "已确认",
            }
        )
        markdown = build_markdown_report(
            summary=summary,
            recommendations=combined,
            excluded=proposal.excluded_candidates,
            model_version=active_status()["model_version"],
            parameter_sha256=active_status()["parameter_sha256"],
            combination_check=combination_check,
            low_frequency_config_sha256=low_frequency_config_sha256(),
        )
        recommendation_id, _, _ = self.history_store.save(
            dataset_id=proposal.dataset_id,
            canteen=proposal.canteen,
            meal=proposal.meal,
            prediction_date=proposal.target_date,
            recommendations=combined,
            markdown=markdown,
        )
        self.proposals.discard(request.proposal_id)
        final_items = self._menu_items(combined)
        return {
            "status": "finalized",
            "proposal_id": request.proposal_id,
            "recommendation_id": recommendation_id,
            "summary": summary,
            "menu": final_items,
            "high_frequency_menu": [
                item for item in final_items
                if item["recommendation_source"] == "high_frequency"
            ],
            "low_frequency_opportunities": [
                item for item in final_items
                if item["recommendation_source"] == "low_frequency_opportunity"
            ],
            "low_frequency_status": "ok",
            "low_frequency_warning": None,
            "combination_check": combination_check,
            "excluded_candidates": proposal.excluded_candidates,
            "csv_download_url": (
                f"/api/recommendations/{recommendation_id}/download?format=csv"
            ),
            "markdown_download_url": (
                f"/api/recommendations/{recommendation_id}/download?format=md"
            ),
        }

    def status(self) -> dict[str, Any]:
        status = active_status()
        status.update(
            {
                "high_frequency_strategy": "targeted_repeat_holiday" if status["strategy_modified"] else "frozen_unchanged",
                "low_frequency_module": "independent",
                "low_frequency_config_sha256": low_frequency_config_sha256(),
            }
        )
        return status

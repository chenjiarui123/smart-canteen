"""Lightweight proposal store and final high/low menu merger."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .dataset_repository import normalize_dish_name
from .model_loader import constraints_copy, frozen_evaluator


@dataclass
class RecommendationProposal:
    proposal_id: str
    created_at: datetime
    dataset_id: str
    canteen: str
    meal: str
    target_date: pd.Timestamp
    high_frequency: pd.DataFrame
    low_frequency: pd.DataFrame
    summary: dict[str, Any]
    excluded_candidates: list[dict[str, str]]


class ProposalStore:
    def __init__(self, ttl_minutes: int = 30):
        self.ttl = timedelta(minutes=int(ttl_minutes))
        self._items: dict[str, RecommendationProposal] = {}
        self._lock = threading.RLock()

    def _purge(self) -> None:
        cutoff = datetime.now(timezone.utc) - self.ttl
        expired = [
            key for key, value in self._items.items()
            if value.created_at < cutoff
        ]
        for key in expired:
            self._items.pop(key, None)

    def add(
        self,
        *,
        dataset_id: str,
        canteen: str,
        meal: str,
        target_date: pd.Timestamp,
        high_frequency: pd.DataFrame,
        low_frequency: pd.DataFrame,
        summary: dict[str, Any],
        excluded_candidates: list[dict[str, str]],
    ) -> RecommendationProposal:
        with self._lock:
            self._purge()
            proposal_id = uuid.uuid4().hex
            proposal = RecommendationProposal(
                proposal_id=proposal_id,
                created_at=datetime.now(timezone.utc),
                dataset_id=dataset_id,
                canteen=canteen,
                meal=meal,
                target_date=pd.Timestamp(target_date),
                high_frequency=high_frequency.copy(deep=True),
                low_frequency=low_frequency.copy(deep=True),
                summary=dict(summary),
                excluded_candidates=list(excluded_candidates),
            )
            self._items[proposal_id] = proposal
            return proposal

    def get(self, proposal_id: str) -> RecommendationProposal:
        if not (
            len(proposal_id) == 32
            and all(char in "0123456789abcdef" for char in proposal_id)
        ):
            raise ValueError("候选方案编号无效")
        with self._lock:
            self._purge()
            proposal = self._items.get(proposal_id)
            if proposal is None:
                raise ValueError("候选方案不存在或已过期，请重新生成")
            return proposal

    def discard(self, proposal_id: str) -> None:
        with self._lock:
            self._items.pop(proposal_id, None)


class RecommendationMerger:
    def __init__(self):
        self._finalize_lock = threading.RLock()

    @staticmethod
    def _select(
        frame: pd.DataFrame, names: list[str], source_label: str
    ) -> pd.DataFrame:
        requested = [normalize_dish_name(value) for value in names]
        if len(requested) != len(set(requested)):
            raise ValueError(f"{source_label}选择中存在重复菜品")
        if not requested:
            return frame.iloc[0:0].copy()
        if "_dish_norm" not in frame.columns:
            raise ValueError(f"{source_label}本次没有可选菜品")
        available = set(frame.get("_dish_norm", pd.Series(dtype=str)).astype(str))
        invalid = sorted(set(requested) - available)
        if invalid:
            raise ValueError(f"{source_label}选择包含不在本次候选中的菜品")
        order = {dish: index for index, dish in enumerate(requested)}
        selected = frame.loc[frame["_dish_norm"].astype(str).isin(requested)].copy()
        if selected.empty:
            return selected
        selected["_selection_order"] = selected["_dish_norm"].map(order)
        return selected.sort_values("_selection_order").drop(
            columns="_selection_order"
        )

    @staticmethod
    def _ensure_constraint_columns(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        text_defaults = {
            "主要食材": "unknown",
            "食材大类": "unknown",
            "制作方法": "unknown",
            "口味": "unknown",
            "菜品类别": "unknown",
            "相似菜分组": "unknown",
        }
        for column, default in text_defaults.items():
            if column not in result.columns:
                result[column] = default
            result[column] = result[column].fillna(default)
        # Trial 18 treats ingredient/family/method/flavour as diversity-soft
        # mechanisms and similarity groups as the pairwise hard constraint.
        for column in ["_main_hard", "_family_hard", "_method_hard", "_flavor_hard"]:
            result[column] = False
        if "_similarity_hard" not in result.columns:
            result["_similarity_hard"] = False
        result["_similarity_hard"] = result["_similarity_hard"].fillna(False).astype(bool)
        return result

    @staticmethod
    def _soft_warnings(frame: pd.DataFrame) -> list[str]:
        low = frame.loc[
            frame.get(
                "recommendation_source", pd.Series("", index=frame.index)
            ).astype(str).eq("low_frequency_opportunity")
        ]
        if low.empty:
            return []
        constraints = constraints_copy()
        specs = [
            ("主要食材", "max_same_main_ingredient", "主要食材"),
            ("食材大类", "max_same_ingredient_family", "食材大类"),
            ("制作方法", "max_same_cooking_method", "制作方法"),
            ("口味", "max_same_flavor", "口味"),
        ]
        warnings: list[str] = []
        unknown = {"", "unknown", "未知", "nan", "none"}
        for column, limit_key, label in specs:
            values = frame[column].astype(str)
            counts = values.loc[~values.str.lower().isin(unknown)].value_counts()
            low_values = set(
                low[column].astype(str).loc[
                    ~low[column].astype(str).str.lower().isin(unknown)
                ]
            )
            for value, count in counts.items():
                limit = int(constraints[limit_key])
                if value in low_values and int(count) > limit:
                    warnings.append(f"{label}“{value}”共{int(count)}道")
        return warnings[:8]

    @staticmethod
    def _allocate_total(
        high: pd.DataFrame,
        low: pd.DataFrame,
        target_total: float,
    ) -> pd.DataFrame:
        combined = pd.concat([high, low], ignore_index=True, sort=False)
        if combined.empty:
            raise ValueError("最终菜单至少选择一道菜")
        if low.empty and len(high) > 0:
            original_total = float(high["pred_qty"].sum())
            if abs(original_total - target_total) <= max(1e-6, abs(target_total) * 1e-9):
                return combined

        high_count = len(high)
        if len(low):
            low_values = low.get("suggested_trial_qty", low["pred_qty"])
            low_qty = pd.to_numeric(
                low_values, errors="coerce"
            ).fillna(0.0).clip(lower=0.0)
        else:
            low_qty = pd.Series(dtype=float)
        low_budget = min(float(low_qty.sum()), target_total * (0.4 if high_count else 1.0))
        allocations = pd.Series(0.0, index=combined.index, dtype=float)
        if len(low):
            if float(low_qty.sum()) > 0:
                allocations.iloc[high_count:] = low_qty * (
                    low_budget / float(low_qty.sum())
                )
            else:
                allocations.iloc[high_count:] = low_budget / len(low)
        remaining = max(target_total - float(allocations.sum()), 0.0)
        if high_count:
            high_weights = pd.to_numeric(high["pred_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
            if float(high_weights.sum()) <= 0:
                high_weights = pd.Series(1.0, index=high.index)
            allocations.iloc[:high_count] = high_weights.to_numpy() * (
                remaining / float(high_weights.sum())
            )
        elif len(low) and target_total > 0:
            current = float(allocations.sum())
            if current > 0:
                allocations *= target_total / current
            else:
                allocations[:] = target_total / len(allocations)
        difference = target_total - float(allocations.sum())
        allocations.iloc[-1] += difference
        combined["pred_qty"] = allocations.clip(lower=0.0)
        orders_ratio = pd.to_numeric(
            combined.get("expected_orders_per_qty", 0.0), errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        rev_ratio = pd.to_numeric(
            combined.get("expected_rev_per_qty", 0.0), errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        combined["pred_orders"] = combined["pred_qty"] * orders_ratio
        combined["pred_rev"] = combined["pred_qty"] * rev_ratio
        combined["pred_total_qty"] = float(target_total)
        return combined

    @staticmethod
    def _keep_reference_quantities(
        high: pd.DataFrame,
        low: pd.DataFrame,
        target_total: float,
    ) -> pd.DataFrame:
        """Keep frozen high predictions and add low-frequency trial amounts."""

        combined = pd.concat([high, low], ignore_index=True, sort=False)
        if combined.empty:
            raise ValueError("最终菜单至少选择一道菜")
        combined["pred_qty"] = pd.to_numeric(
            combined["pred_qty"], errors="raise"
        ).clip(lower=0.0)
        combined["pred_orders"] = pd.to_numeric(
            combined["pred_orders"], errors="raise"
        ).clip(lower=0.0)
        combined["pred_rev"] = pd.to_numeric(
            combined["pred_rev"], errors="raise"
        ).clip(lower=0.0)
        # This field remains the frozen meal-demand forecast.  The sum of dish
        # preparation quantities may be higher when trial dishes are added.
        combined["pred_total_qty"] = float(target_total)
        return combined

    def merge(
        self,
        proposal: RecommendationProposal,
        selected_high_frequency: list[str],
        selected_low_frequency: list[str],
        redistribute_to_total_demand: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        with self._finalize_lock:
            high = self._select(
                proposal.high_frequency,
                selected_high_frequency,
                "高频菜单",
            )
            low = self._select(
                proposal.low_frequency,
                selected_low_frequency,
                "潜力菜",
            )
            combined_names = list(high["_dish_norm"].astype(str)) + list(
                low.get("_dish_norm", pd.Series(dtype=str)).astype(str)
            )
            if len(combined_names) != len(set(combined_names)):
                raise ValueError("最终菜单不能包含重复菜品")
            reference_menu_count = int(proposal.summary["menu_capacity"])
            target_total = float(proposal.summary["pred_total_qty"])
            combined = (
                self._allocate_total(high, low, target_total)
                if redistribute_to_total_demand
                else self._keep_reference_quantities(high, low, target_total)
            )
            combined = self._ensure_constraint_columns(combined)
            prepared = frozen_evaluator.prepare_recommendations(combined)
            violations = frozen_evaluator.constraint_violations(
                prepared, {"constraints": constraints_copy()}
            )
            if violations["total_constraint_violation_count"]:
                raise ValueError("最终菜单存在重复菜品或相似菜分组硬冲突，请调整选择")
            allocated = float(combined["pred_qty"].sum())
            conserved = bool(
                np.isclose(allocated, target_total, rtol=1e-9, atol=1e-6)
            )
            if redistribute_to_total_demand and not conserved:
                raise RuntimeError("最终菜单备餐量未保持餐次预测总量守恒")
            count_difference = int(len(combined) - reference_menu_count)
            preparation_difference = float(allocated - target_total)
            reference_warnings: list[str] = []
            if count_difference > 0:
                reference_warnings.append(
                    f"最终菜品数比冻结容量参考值多{count_difference}道；"
                    "潜力菜为额外试餐项，不占用高频菜名额"
                )
            elif count_difference < 0:
                reference_warnings.append(
                    f"最终菜品数比冻结容量参考值少{abs(count_difference)}道"
                )
            if not redistribute_to_total_demand and abs(preparation_difference) > 1e-6:
                direction = "高于" if preparation_difference > 0 else "低于"
                reference_warnings.append(
                    f"当前总备餐量{direction}当餐预测总需求"
                    f"{abs(preparation_difference):.2f}；该差异仅作备餐提醒"
                )
            check = {
                **violations,
                "soft_warnings": self._soft_warnings(combined),
                "reference_warnings": reference_warnings,
                "pred_total_conserved": conserved,
                "redistributed_to_total_demand": bool(
                    redistribute_to_total_demand
                ),
                "menu_capacity": reference_menu_count,
                "reference_menu_count": reference_menu_count,
                "menu_count_difference": count_difference,
                "selected_count": int(len(combined)),
                "selected_high_frequency_count": int(len(high)),
                "selected_low_frequency_count": int(len(low)),
                "total_preparation_qty": allocated,
                "predicted_session_demand_qty": target_total,
                "preparation_qty_difference": preparation_difference,
            }
            return combined.reset_index(drop=True), check

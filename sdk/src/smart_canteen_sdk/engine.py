"""Portable, stateless facade around the deployed recommendation strategy."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ._vendor.active_strategy import active_status
from ._vendor.context_builder import build_inference_context
from ._vendor.dataset_repository import normalize_dish_name
from ._vendor.high_frequency_recommender import HighFrequencyRecommender
from ._vendor.low_frequency_recommender import LowFrequencyRecommender
from ._vendor.recommendation_merger import RecommendationMerger, RecommendationProposal
from .data import menu_context, prepare, unit_scale, validate_snapshot
from .errors import CanteenError
from .schemas import DataSnapshot, FinalResult, MenuHistory, MenuItem, Proposal, RecommendationRequest, Selection

LOGGER = logging.getLogger(__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def pack_frame(frame):
    return json.loads(frame.to_json(orient="split", date_format="iso", double_precision=15))


def unpack_frame(value):
    frame = pd.DataFrame(value["data"], columns=value["columns"])
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def items(frame, identities):
    result = []
    for row in frame.to_dict("records"):
        unit = str(row.get("菜品售卖单位", "份"))
        scale = unit_scale(unit)
        qty = float(row["pred_qty"])
        high = row["recommendation_source"] == "high_frequency"
        result.append(MenuItem(
            dish_id=identities[normalize_dish_name(row["菜品名称"])],
            dish_name=str(row["菜品名称"]), unit=unit, quantity=qty * scale,
            quantity_kind="expected_demand" if high else "trial_preparation",
            expected_demand_qty=qty * scale if high else None,
            suggested_trial_qty=None if high else qty * scale,
            reference_orders=float(row["pred_orders"]), reference_revenue=float(row["pred_rev"]),
            score=float(row["score"]), category=str(row.get("菜品类别", "其他")),
            recommendation_source=row["recommendation_source"],
            reason=str(row.get("recommendation_reason", row.get("reason", ""))),
            model_quantity=qty, business_units_per_model_unit=scale,
        ))
    return result


def totals(menu):
    quantities, demand = {}, {}
    for item in menu:
        quantities[item.unit] = quantities.get(item.unit, 0.0) + item.quantity
        if item.expected_demand_qty is not None:
            demand[item.unit] = demand.get(item.unit, 0.0) + item.expected_demand_qty
    return {"selected_count": len(menu), "quantities_by_unit": quantities,
            "expected_demand_by_unit": demand,
            "reference_total_orders": sum(i.reference_orders for i in menu),
            "reference_total_revenue": sum(i.reference_revenue for i in menu),
            "expected_high_frequency_revenue": sum(i.reference_revenue for i in menu if i.expected_demand_qty is not None)}


def history_event(request, menu, source, menu_id, now):
    return MenuHistory(date=request.prediction_date, canteen_id=request.canteen_id,
                       meal=request.meal, menu_id=menu_id, created_at=now, source=source,
                       dish_ids=[i.dish_id for i in menu],
                       low_frequency_dish_ids=[i.dish_id for i in menu if i.recommendation_source == "low_frequency_opportunity"])


class CanteenEngine:
    """No database, CSV, network, proposal cache or business-data writes."""

    def __init__(self, strategy_version: str | None = None):
        root = Path(__file__).parent
        bundle = json.loads((root / "BUNDLE_MANIFEST.json").read_text())
        for path, expected in bundle.items():
            if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
                raise CanteenError("MODEL_INTEGRITY_ERROR", f"SDK 算法资源校验失败: {path}")
        self._bundle_sha256 = digest(bundle)
        self._info = active_status()
        if strategy_version is not None and strategy_version != self._info["model_version"]:
            raise CanteenError("UNSUPPORTED_STRATEGY", "当前安装包不包含指定策略版本")

    def get_model_info(self):
        return {**self._info, "sdk_version": "0.1.0", "schema_version": "1.0",
                "bundle_sha256": self._bundle_sha256,
                "quantity_convention": "business units; gram model scale is 50",
                "persistence": "caller managed"}

    def validate_data(self, data):
        data = DataSnapshot.model_validate(data)
        validate_snapshot(data)
        return {"valid": True, "snapshot_id": data.snapshot_id,
                "sales_rows": len(data.sales_history), "dish_count": len(data.dish_catalog)}

    def recommend(self, request, data) -> Proposal:
        request = RecommendationRequest.model_validate(request)
        data = DataSnapshot.model_validate(data)
        history, attrs, target_context = prepare(data, request)
        canteen = data.canteen_name or data.canteen_id
        day = pd.Timestamp(request.prediction_date)
        context = build_inference_context(history=history, attributes=attrs, prediction_date=day,
                                          canteen=canteen, meal=request.meal, target_context=target_context)
        initial, prior, recent7, recent14, formal, reference, reference_source = menu_context(data, request)
        units = history.sort_values("date").drop_duplicates("_dish_norm", keep="last").set_index("_dish_norm")["菜品售卖单位"].to_dict()
        try:
            high = HighFrequencyRecommender().generate(
                context=context, history=history, target_date=day, canteen=canteen,
                meal=request.meal, initial_menu_history=initial, prior_norms=prior,
                recent7=recent7, recent14=recent14, sale_units=units)
        except ValueError as exc:
            raise CanteenError("INFEASIBLE_MENU", str(exc)) from exc
        low = pd.DataFrame()
        low_status = "disabled" if not request.opportunities.enabled else "ok"
        warnings = [v for k, v in high.summary.items() if k.endswith("warning") and v]
        past_units = {}
        for s in data.sales_history:
            if s.date < request.prediction_date:
                past_units.setdefault(s.dish_id, set()).add(s.unit)
        changed_units = sorted(k for k, v in past_units.items() if len(v) > 1)
        if changed_units:
            warnings.append("旧数据兼容模式：部分菜品历史售卖单位变化，保留旧模型行为；数量统计不代表已完成历史单位清洗")
        if request.opportunities.enabled:
            try:
                low = LowFrequencyRecommender().generate(
                    history=history, attributes=attrs, recommendation_history=formal,
                    target_date=day, canteen=canteen, meal=request.meal,
                    target_context=target_context, high_frequency_dishes=set(high.recommendations["_dish_norm"]),
                    refresh_nonce=request.opportunities.refresh_seed, category=request.opportunities.category,
                    candidate_count=request.opportunities.count)
            except Exception:
                LOGGER.exception("Potential dish recommendation failed")
                low_status = "unavailable"
                warnings.append("潜力菜模块不可用，已返回常用菜单")
        identities = {normalize_dish_name(d.dish_name): d.dish_id for d in data.dish_catalog}
        high_items, low_items = items(high.recommendations, identities), items(low, identities)
        now, proposal_id = datetime.now(timezone.utc), uuid4().hex
        state = {"high": pack_frame(high.recommendations), "low": pack_frame(low),
                 "summary": high.summary, "identities": identities,
                 "previous_reference": sorted(reference) if reference else [],
                 "reference_source": reference_source,
                 "request": request.model_dump(mode="json"), "snapshot_id": data.snapshot_id,
                 "bundle_sha256": self._bundle_sha256, "proposal_id": proposal_id}
        return Proposal(strategy_version=self._info["model_version"], proposal_id=proposal_id,
                        snapshot_id=data.snapshot_id, created_at=now, request=request,
                        high_frequency_menu=high_items, low_frequency_opportunities=low_items,
                        summary={**totals(high_items), "menu_capacity": high.summary["menu_capacity"],
                                 "low_frequency_count": len(low_items)},
                        audit={"history_start": high.summary["history_start"], "history_end": high.summary["history_end"],
                               "excluded_nonpast_sales_rows": sum(s.date >= request.prediction_date for s in data.sales_history),
                               "constraint_violation_count": 0,
                               "adjacent_status": high.summary["adjacent_status"],
                               "adjacent_repeat_rate": high.summary["adjacent_repeat_rate"],
                               "adjacent_reference_source": reference_source,
                               "low_frequency_status": low_status, "warnings": warnings,
                               "unit_policy": data.unit_policy, "legacy_mixed_unit_dish_ids": changed_units,
                               "excluded_candidates": high.excluded_candidates,
                               "legacy_model_summary": high.summary},
                        history_events=[history_event(request, high_items, "automatic", proposal_id, now)],
                        state=state, state_sha256=digest(state))

    def finalize(self, proposal, selection) -> FinalResult:
        proposal = Proposal.model_validate(proposal)
        selection = Selection.model_validate(selection)
        state = proposal.state
        if digest(state) != proposal.state_sha256:
            raise CanteenError("PROPOSAL_CORRUPTED", "候选方案状态校验失败，请从后端保存的完整方案恢复")
        if state["bundle_sha256"] != self._bundle_sha256 or proposal.strategy_version != self._info["model_version"]:
            raise CanteenError("PROPOSAL_VERSION_MISMATCH", "候选方案与当前算法版本不一致")
        if (state["request"] != proposal.request.model_dump(mode="json") or
            state["snapshot_id"] != proposal.snapshot_id or state["proposal_id"] != proposal.proposal_id):
            raise CanteenError("PROPOSAL_CORRUPTED", "候选方案标识或请求已变化")
        high, low = unpack_frame(state["high"]), unpack_frame(state["low"])
        reverse = {v: k for k, v in state["identities"].items()}
        try:
            selected_high = [reverse[i] for i in selection.high_frequency_dish_ids]
            selected_low = [reverse[i] for i in selection.low_frequency_dish_ids]
        except KeyError as exc:
            raise CanteenError("INVALID_SELECTION", "选择包含未知菜品 ID") from exc
        internal = RecommendationProposal(proposal.proposal_id, proposal.created_at, proposal.request.canteen_id,
                                          state["summary"]["canteen"], proposal.request.meal,
                                          pd.Timestamp(proposal.request.prediction_date), high, low, state["summary"], [])
        merger = RecommendationMerger()
        try:
            combined, check = merger.merge(internal, selected_high, selected_low, False)
            if selection.redistribute_to_total_demand:
                # Preserve a separate budget for every sale unit; never exchange grams and portions.
                budgets = high.groupby("菜品售卖单位")["pred_qty"].sum().to_dict()
                selected_units = set(combined["菜品售卖单位"])
                if any(value > 0 and unit not in selected_units for unit, value in budgets.items()):
                    raise CanteenError("UNIT_BUDGET_INFEASIBLE", "某单位有原预测总量但未保留任何该单位菜品")
                if not selected_units <= set(budgets):
                    raise CanteenError("UNIT_BUDGET_INFEASIBLE", "新增单位没有原预测总量，不能跨单位分配")
                groups = []
                for unit, group in combined.groupby("菜品售卖单位", sort=False):
                    h = group.loc[group.recommendation_source.eq("high_frequency")].reset_index(drop=True)
                    l = group.loc[group.recommendation_source.eq("low_frequency_opportunity")].reset_index(drop=True)
                    groups.append(merger._allocate_total(h, l, float(budgets[unit])))
                combined = pd.concat(groups, ignore_index=True)
                check.update(redistributed_to_total_demand=True, pred_total_conserved=True,
                             total_preparation_qty=float(combined.pred_qty.sum()), preparation_qty_difference=0.0)
                check["reference_warnings"] = [w for w in check["reference_warnings"] if "总备餐量" not in w]
        except CanteenError:
            raise
        except ValueError as exc:
            raise CanteenError("INVALID_SELECTION", str(exc)) from exc
        menu = items(combined, state["identities"])
        reference = set(state["previous_reference"])
        names = set(combined["菜品名称"].map(normalize_dish_name))
        rate = len(names & reference) / len(names) if reference and names else None
        warnings = list(check.get("reference_warnings", [])) + list(check.get("soft_warnings", []))
        warnings.extend(proposal.audit.get("warnings", []))
        if rate is not None and rate > 0.8:
            warnings.append(f"人工菜单与前一天同餐次重复率为 {rate:.1%}，超过80%")
        return FinalResult(strategy_version=proposal.strategy_version, proposal_id=proposal.proposal_id,
                           snapshot_id=proposal.snapshot_id, menu=menu, summary=totals(menu),
                           audit={"constraint_violation_count": check["total_constraint_violation_count"],
                                  "redistributed_to_total_demand": selection.redistribute_to_total_demand,
                                  "manual_adjacent_repeat_rate": rate, "warnings": warnings,
                                  "legacy_model_check": check},
                           history_events=[history_event(proposal.request, menu, "finalized", uuid4().hex, datetime.now(timezone.utc))])

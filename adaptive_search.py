#!/usr/bin/env python3
"""Factor/capacity phase: one-hypothesis-at-a-time training-only search.

This runner deliberately has no pre-generated trial list and no maximum trial
count.  The caller chooses one hypothesis after inspecting all prior outcomes.
External validation/test sales are never opened here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import policy
from diagnostics import (
    hit_quantity_diagnostics,
    menu_capacity_diagnostics,
    state_meal_diagnostics,
)


ROOT = Path(__file__).resolve().parent
TRIALS_PATH = ROOT / "trials.jsonl"
SELECTION_CORRECTIONS_PATH = ROOT / "selection_corrections.jsonl"
SEGMENTS_PATH = ROOT / "trial_segment_metrics.csv"
DIAGNOSTICS_PATH = ROOT / "trial_diagnostics.csv"
STATE_MEAL_PATH = ROOT / "state_meal_diagnostics.csv"
BEST_CONFIG_PATH = ROOT / "config.json"
BEST_MENU_PATH = ROOT / "recommended_menu.csv"
BEST_HIT_FAILURES_PATH = ROOT / "failure_cases.csv"
MISSED_REVENUE_PATH = ROOT / "missed_high_revenue_dishes.csv"
VERSIONS_DIR = ROOT / "versions"
ARCHIVED_BASELINE_CONFIG = (
    ROOT
    / "archive/factor_capacity_phase0_trial108_20260727/adaptive_config.json"
)
PHASE_NAME = "factor_capacity_v1"
FIXED_EVALUATOR_SHA256 = (
    "4f5cd9494b40ea021744640bc86551febb4f019eb4a668ddc37f20fcea9fcffc"
)
MATERIAL_OBJECTIVE_IMPROVEMENT = 0.0005
PARETO_DEGRADATION_LIMIT = 0.01
REDUNDANT_FINAL_POLICY_KEYS = {
    "allocation_offer_power",       # fixed at zero before allocation
    "item_qty_mean_weight",         # pure log-mean quantity won the audit
    "item_robust_statistic",        # frozen core directly uses log-mean
    "prediction_consistency_mode",  # robust branch is coherent by construction
}
_BACKTEST_CACHE: tuple[
    pd.DataFrame,
    list[dict[str, Any]],
    pd.Timestamp,
    str,
    str,
] | None = None


def _revenue_price_compliant(policy_config: dict[str, Any]) -> bool:
    """Whether output revenue is quantity times a history-derived unit value."""

    return str(policy_config.get("item_quantity_mode", "")) == "robust_conditional"


def _json_default(value: Any) -> Any:
    return policy._json_default(value)


def _simplified_policy(policy_config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in policy_config.items()
        if key not in REDUNDANT_FINAL_POLICY_KEYS
    }


def _load_trials() -> list[dict[str, Any]]:
    if not TRIALS_PATH.exists():
        return []
    with TRIALS_PATH.open(encoding="utf-8") as handle:
        records = [
            json.loads(line) for line in handle if line.strip()
        ]
    if SELECTION_CORRECTIONS_PATH.exists():
        with SELECTION_CORRECTIONS_PATH.open(
            encoding="utf-8"
        ) as handle:
            corrections = {
                int(row["trial_index"]): row
                for row in (
                    json.loads(line)
                    for line in handle
                    if line.strip()
                )
            }
        for record in records:
            correction = corrections.get(int(record["trial_index"]))
            if correction is None:
                continue
            record["is_new_best"] = bool(
                correction["is_new_best"]
            )
            record["failure_reason"] = str(
                correction.get("failure_reason", "")
            )
            record["selection_reason"] = str(
                correction["selection_reason"]
            )
            record["selection_correction"] = correction
    return records


def _archived_baseline_policy() -> dict[str, Any]:
    """Load Trial 108 and make all new phase controls explicit."""

    config = json.loads(
        ARCHIVED_BASELINE_CONFIG.read_text(encoding="utf-8")
    )
    baseline = copy.deepcopy(config["best_policy"])
    baseline.update(
        {
            "group_factor_combination_mode": "multiplicative",
            "group_combined_factor_clip": [0.65, 1.45],
            "holiday_weather_residual_strength": 1.0,
            "capacity_mode": "weekday_median",
            "capacity_state_shrinkage_tau": 5.0,
            "capacity_demand_weight": 0.5,
            "capacity_hierarchical_strength": 1.0,
            "capacity_weather_strength": 0.0,
            "capacity_min": 1,
            "capacity_max": 10000,
            # Estimate batches in Trial 1 for a meaningful diagnostic baseline,
            # but do not enforce them until a dedicated experiment.
            "min_batch_mode": "estimate_only",
            "min_batch_dish_min_count": 3.0,
            "min_batch_scale": 1.0,
            "group_total_target_mode": "calibrated_bottom_up",
        }
    )
    return baseline


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig" if not path.exists() else "utf-8",
    )


def _context_input_sha256(contexts: list[dict[str, Any]]) -> str:
    rows = []
    for context in contexts:
        target = context.get("target_context", {})
        rows.append(
            {
                "date": str(pd.Timestamp(context["date"]).date()),
                "食堂名称": context["食堂名称"],
                "餐次": context["餐次"],
                **{key: str(target.get(key, "")) for key in sorted(target)},
            }
        )
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _backtest_assets() -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
    pd.Timestamp,
    str,
    str,
]:
    """Build the fixed expanding-window schedule once per search process."""

    global _BACKTEST_CACHE
    if _BACKTEST_CACHE is None:
        train = policy.read_sales(policy.TRAIN_PATH)
        attributes, _ = policy.build_library_attributes()
        warmup, evaluation, first_eval_date = (
            policy.training_backtest_frames(train)
        )
        contexts = policy.build_contexts(
            warmup, evaluation, attributes
        )
        _BACKTEST_CACHE = (
            evaluation,
            contexts,
            first_eval_date,
            policy.context_schedule_sha256(contexts),
            _context_input_sha256(contexts),
        )
    return _BACKTEST_CACHE


def _segment_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['dimension']}::{row['segment']}"
        summary[key] = {
            "session_count": int(row["evaluated_group_count"]),
            "total_mape": float(row["total_mape"]),
            "hit_qty_mape": float(row.get("hit_qty_mape", 0.0)),
            "net_benefit_per_session": float(row["net_benefit_per_session"]),
        }
    return summary


def _segment_regressions(
    candidate: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
) -> list[str]:
    checked_dimensions = {
        "month",
        "meal",
        "day_type",
        "calendar_state",
        "school_break_type",
        "statutory_holiday_name",
        "adjusted_workday",
        "temperature_band",
        "rain_band",
        "extreme_weather",
    }
    regressions = []
    for key, current in candidate.items():
        dimension = key.split("::", 1)[0]
        prior = reference.get(key)
        if dimension not in checked_dimensions or prior is None:
            continue
        if min(int(current["session_count"]), int(prior["session_count"])) < 5:
            continue
        allowed = float(prior["total_mape"]) * 1.10 + 0.5
        if float(current["total_mape"]) > allowed:
            regressions.append(
                f"{key}: {current['total_mape']:.3f}>{prior['total_mape']:.3f}"
            )
    return regressions


def _guard_failure(
    candidate: dict[str, Any],
    best: dict[str, Any],
    segment_regressions: list[str],
    changed_parameters: dict[str, Any],
) -> str:
    if not bool(candidate.get("revenue_price_compliant", False)):
        return "pred_rev未与pred_qty和历史单位收入保持一致"
    if int(candidate["total_constraint_violation_count"]) != 0:
        return "硬约束违规"
    if int(candidate.get("duplicate_dish_violation_count", 0)) != 0:
        return "存在重复菜名"
    if float(candidate.get("total_allocation_wape", 0.0)) > 1e-9:
        return "单品预测量之和未与餐次预测总量一致"
    guards = [
        ("dish_hit_rate", 0.02, "菜品命中率"),
        ("high_revenue_candidate_recall", 0.02, "高营收候选召回"),
    ]
    for key, tolerance, label in guards:
        if float(candidate.get(key, 0.0)) + tolerance < float(best.get(key, 0.0)):
            return f"{label}明显下降"
    best_coverage = float(best.get("revenue_coverage", 0.0))
    if float(candidate.get("revenue_coverage", 0.0)) < best_coverage * 0.98:
        return "收入覆盖率明显下降"
    capacity_keys = {
        "capacity_mode",
        "capacity_state_shrinkage_tau",
        "capacity_demand_weight",
        "capacity_hierarchical_strength",
        "capacity_weather_strength",
        "capacity_weather_shrinkage_tau",
        "capacity_weather_factor_clip",
        "capacity_scale",
        "min_batch_mode",
        "min_batch_dish_min_count",
        "min_batch_scale",
    }
    is_capacity_experiment = bool(
        capacity_keys.intersection(changed_parameters)
    )
    if is_capacity_experiment:
        best_menu_mae = float(best.get("menu_count_mae", math.inf))
        candidate_menu_mae = float(
            candidate.get("menu_count_mae", math.inf)
        )
        best_batch = int(best.get("below_min_batch_count", 0))
        candidate_batch = int(
            candidate.get("below_min_batch_count", 0)
        )
        if candidate_menu_mae > best_menu_mae * 1.01 + 1e-9:
            return "菜单数量MAE未改善"
        if candidate_batch > best_batch:
            return "低于最小备餐批量的菜品数增加"
        improved = (
            candidate_menu_mae < best_menu_mae - 1e-9
            or candidate_batch < best_batch
        )
        if not improved:
            return "容量/最小批量诊断没有实际改善"
    if segment_regressions:
        return "分组明显退化：" + "; ".join(segment_regressions[:3])
    return ""


def _select_candidate(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
    segment_regressions: list[str],
    records: list[dict[str, Any]],
    changed_parameters: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    changed_parameters = changed_parameters or {}
    if best is None:
        baseline_valid = (
            int(candidate["total_constraint_violation_count"]) == 0
            and int(candidate.get("duplicate_dish_violation_count", 0))
            == 0
            and float(candidate.get("total_allocation_wape", 0.0))
            <= 1e-9
            and bool(candidate.get("revenue_price_compliant", False))
        )
        if baseline_valid:
            return True, ""
        return False, "基线未通过硬约束、总量守恒或收入一致性检查"
    guard = _guard_failure(
        candidate, best, segment_regressions, changed_parameters
    )
    if guard:
        return False, guard
    capacity_keys = {
        "capacity_mode",
        "capacity_state_shrinkage_tau",
        "capacity_demand_weight",
        "capacity_hierarchical_strength",
        "capacity_weather_strength",
        "capacity_weather_shrinkage_tau",
        "capacity_weather_factor_clip",
        "capacity_scale",
        "min_batch_mode",
        "min_batch_dish_min_count",
        "min_batch_scale",
    }
    if records and capacity_keys.intersection(changed_parameters):
        phase_baseline = records[0]
        if abs(
            float(candidate.get("holiday_menu_count_bias", 0.0))
        ) > abs(
            float(
                phase_baseline.get(
                    "holiday_menu_count_bias", math.inf
                )
            )
        ) + 1e-9:
            return False, "节假日菜单数量偏差未优于阶段基线"

    total_change = (
        float(best["total_mape"]) - float(candidate["total_mape"])
    ) / max(float(best["total_mape"]), 1e-12)
    hit_change = (
        float(best["hit_qty_mape"])
        - float(candidate["hit_qty_mape"])
    ) / max(float(best["hit_qty_mape"]), 1e-12)
    net_change = (
        float(candidate["net_benefit"]) - float(best["net_benefit"])
    ) / max(abs(float(best["net_benefit"])), 1e-12)
    changes = {
        "total_mape": total_change,
        "hit_qty_mape": hit_change,
        "net_benefit": net_change,
    }
    if min(changes.values()) < -PARETO_DEGRADATION_LIMIT - 1e-12:
        return (
            False,
            "Pareto容忍失败（任一其他主要目标相对恶化超过1%）："
            + ", ".join(
                f"{key}={value * 100:+.3f}%"
                for key, value in changes.items()
            ),
        )
    shadow_invariance_upgrade = (
        changed_parameters.get("group_total_target_mode", {}).get("to")
        == "independent_shadow_calibrated"
        and min(changes.values()) >= -1e-10
    )
    if shadow_invariance_upgrade:
        return (
            True,
            "结构性等价保留：影子菜单使餐次总量不再依赖实际容量，"
            "三个主要目标未发生可测退化",
        )
    if max(changes.values()) < MATERIAL_OBJECTIVE_IMPROVEMENT:
        return (
            False,
            "三个主要目标均无明确改善（明确改善阈值0.05%）："
            + ", ".join(
                f"{key}={value * 100:+.3f}%"
                for key, value in changes.items()
            ),
        )

    # Fixed all-history anchors prevent a chain of individually tolerated
    # regressions from drifting more than 1% away from any achieved frontier.
    valid_history = [
        row
        for row in records
        if int(row.get("total_constraint_violation_count", 1)) == 0
        and bool(row.get("revenue_price_compliant", False))
        and bool(row.get("is_new_best", False))
    ]
    total_anchor = min(
        [float(row["total_mape"]) for row in valid_history]
        + [float(candidate["total_mape"])]
    )
    hit_anchor = min(
        [float(row["hit_qty_mape"]) for row in valid_history]
        + [float(candidate["hit_qty_mape"])]
    )
    net_anchor = max(
        [float(row["net_benefit"]) for row in valid_history]
        + [float(candidate["net_benefit"])]
    )
    if float(candidate["total_mape"]) > total_anchor * 1.01 + 1e-12:
        return False, "相对历史最优total_mape漂移超过1%"
    if float(candidate["hit_qty_mape"]) > hit_anchor * 1.01 + 1e-12:
        return False, "相对历史最优hit_qty_mape漂移超过1%"
    if float(candidate["net_benefit"]) < net_anchor * 0.99 - 1e-12:
        return False, "相对历史最优net_benefit漂移超过1%"
    return (
        True,
        "Pareto保留："
        + ", ".join(
            f"{key}={value * 100:+.3f}%"
            for key, value in changes.items()
        )
        + "；其他主要目标相对恶化均未超过1%",
    )


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((record for record in reversed(records) if record["is_new_best"]), None)


def _parent_policy(
    records: list[dict[str, Any]], parent: str
) -> tuple[dict[str, Any], str]:
    if parent == "best":
        best = _best_record(records)
        if best is not None:
            return copy.deepcopy(best["policy"]), f"trial_{best['trial_index']}"
        return _archived_baseline_policy(), "archived_phase0_trial108"
    if parent == "baseline":
        return _archived_baseline_policy(), "archived_phase0_trial108"
    index = int(parent)
    match = next((record for record in records if int(record["trial_index"]) == index), None)
    if match is None:
        raise ValueError(f"找不到父实验 Trial {index}")
    return copy.deepcopy(match["policy"]), f"trial_{index}"


def run_one(
    hypothesis: str,
    experiment_type: str,
    changes: dict[str, Any],
    parent: str,
) -> dict[str, Any]:
    if not hypothesis.strip():
        raise ValueError("hypothesis 不能为空")
    records = _load_trials()
    trial_index = len(records) + 1
    parent_policy, parent_label = _parent_policy(records, parent)
    candidate_policy = copy.deepcopy(parent_policy)
    changed_parameters = {
        key: {"from": candidate_policy.get(key), "to": value}
        for key, value in changes.items()
        if candidate_policy.get(key) != value
    }
    if records and not changed_parameters:
        raise ValueError("参数与父策略完全相同，不能计为有效实验")
    candidate_policy.update(copy.deepcopy(changes))

    evaluator_hash = policy.sha256_file(policy.EVALUATOR_PATH)
    if evaluator_hash != FIXED_EVALUATOR_SHA256:
        raise RuntimeError(
            "固定 evaluator.py 的 SHA256 已变化，拒绝运行实验"
        )
    (
        evaluation,
        contexts,
        first_eval_date,
        schedule_hash,
        context_hash,
    ) = _backtest_assets()
    audit: list[dict[str, Any]] = []
    recs = policy.make_recommendations(
        contexts,
        candidate_policy,
        policy.BASE_CONSTRAINTS,
        selection_audit=audit,
    )
    metrics = policy.evaluate_policy_metrics(recs, evaluation, policy.BASE_CONSTRAINTS)
    segment_rows = policy.build_segment_metrics(
        recs, evaluation, policy.BASE_CONSTRAINTS, trial_index
    )
    diagnostics, missed = policy.build_auxiliary_diagnostics(
        recs,
        evaluation,
        audit,
        policy.BASE_CONSTRAINTS,
        segment_rows,
    )
    metrics.update(diagnostics)
    metrics["revenue_price_compliant"] = _revenue_price_compliant(candidate_policy)
    candidate_segments = _segment_summary(segment_rows)
    best = _best_record(records)
    reference_segments = best.get("segment_summary", {}) if best else {}
    regressions = _segment_regressions(candidate_segments, reference_segments)
    accepted, selection_reason = _select_candidate(
        metrics,
        best,
        regressions,
        records,
        changed_parameters,
    )
    failure_reason = "" if accepted else selection_reason
    record: dict[str, Any] = {
        "trial_index": trial_index,
        "phase": PHASE_NAME,
        "experiment_type": experiment_type,
        "hypothesis": hypothesis,
        "search_parent": parent_label,
        "changed_parameters": changed_parameters,
        **policy._metric_payload(metrics),
        "is_new_best": bool(accepted),
        "failure_reason": failure_reason,
        "selection_reason": selection_reason,
        "pareto_degradation_limit": PARETO_DEGRADATION_LIMIT,
        "material_objective_improvement": MATERIAL_OBJECTIVE_IMPROVEMENT,
        "segment_regressions": regressions,
        "segment_summary": candidate_segments,
        "policy": candidate_policy,
        "evaluator_sha256": evaluator_hash,
        "policy_sha256": policy.sha256_file(policy.ROOT / "policy.py"),
        "diagnostics_sha256": policy.sha256_file(policy.ROOT / "diagnostics.py"),
        "evaluation_schedule_sha256": schedule_hash,
        "context_input_sha256": context_hash,
        "first_eval_date": str(first_eval_date.date()),
        "last_eval_date": str(evaluation["date"].max().date()),
        "evaluation_date_count": int(evaluation["date"].nunique()),
        "trial_initial_state": "fresh",
        "external_data_read": False,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    _append_jsonl(TRIALS_PATH, record)
    _append_csv(SEGMENTS_PATH, segment_rows)
    _, capacity_sessions = menu_capacity_diagnostics(
        recs, evaluation, audit
    )
    state_rows = state_meal_diagnostics(
        capacity_sessions, trial_index
    )
    _append_csv(STATE_MEAL_PATH, state_rows)
    diagnostic_row = {
        key: value
        for key, value in record.items()
        if not isinstance(value, (dict, list))
    }
    _append_csv(DIAGNOSTICS_PATH, [diagnostic_row])

    if accepted:
        version_dir = VERSIONS_DIR / f"trial_{trial_index}"
        version_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(policy.ROOT / "policy.py", version_dir / "policy.py")
        shutil.copy2(policy.ROOT / "diagnostics.py", version_dir / "diagnostics.py")
        shutil.copy2(policy.EVALUATOR_PATH, version_dir / "evaluator.py")
        policy.write_json(version_dir / "trial.json", record)
        policy.write_json(
            BEST_CONFIG_PATH,
            {
                "phase": PHASE_NAME,
                "best_trial_index": trial_index,
                "best_policy": candidate_policy,
                "best_metrics": policy._metric_payload(metrics),
                "evaluation_schedule_sha256": schedule_hash,
                "context_input_sha256": context_hash,
                "evaluator_sha256": evaluator_hash,
                "external_evaluated": False,
            },
        )
        policy.write_menu(recs, BEST_MENU_PATH)
        _, hit_cases = hit_quantity_diagnostics(recs, evaluation)
        hit_cases.to_csv(BEST_HIT_FAILURES_PATH, index=False, encoding="utf-8-sig")
        missed.to_csv(
            version_dir / "missed_high_revenue_dishes.csv",
            index=False,
            encoding="utf-8-sig",
        )
        missed.to_csv(
            MISSED_REVENUE_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        capacity_sessions.to_csv(
            version_dir / "menu_capacity_sessions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(state_rows).to_csv(
            version_dir / "state_meal_diagnostics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(
        json.dumps(
            {
                "trial_index": trial_index,
                "experiment_type": experiment_type,
                "is_new_best": accepted,
                "failure_reason": failure_reason,
                "selection_reason": selection_reason,
                "total_mape": metrics["total_mape"],
                "hit_qty_mape": metrics["hit_qty_mape"],
                "hit_qty_wmape": metrics["hit_qty_wmape"],
                "weighted_qty_bias": metrics["weighted_qty_bias"],
                "net_benefit": metrics["net_benefit"],
                "dish_hit_rate": metrics["dish_hit_rate"],
                "menu_fill_rate": metrics["menu_fill_rate"],
                "menu_count_mae": metrics["menu_count_mae"],
                "holiday_menu_count_bias": metrics[
                    "holiday_menu_count_bias"
                ],
                "below_min_batch_count": metrics[
                    "below_min_batch_count"
                ],
                "revenue_coverage": metrics["revenue_coverage"],
                "total_allocation_wape": metrics["total_allocation_wape"],
                "hard_violations": metrics["total_constraint_violation_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return record


def sync_best_outputs() -> dict[str, Any]:
    """Rebuild all derived training-only files for the retained policy."""

    if BEST_CONFIG_PATH.exists():
        current_config = json.loads(
            BEST_CONFIG_PATH.read_text(encoding="utf-8-sig")
        )
        if bool(current_config.get("external_evaluated", False)):
            raise RuntimeError(
                "Trial 18 已完成外部评估；禁止重新同步训练产物并覆盖冻结状态"
            )
    records = _load_trials()
    best = _best_record(records)
    if best is None:
        raise RuntimeError("没有可同步的最佳实验")
    evaluation, contexts, _, schedule_hash, context_hash = (
        _backtest_assets()
    )
    audit: list[dict[str, Any]] = []
    recs = policy.make_recommendations(
        contexts,
        copy.deepcopy(best["policy"]),
        policy.BASE_CONSTRAINTS,
        selection_audit=audit,
    )
    metrics = policy.evaluate_policy_metrics(recs, evaluation, policy.BASE_CONSTRAINTS)
    segment_rows = policy.build_segment_metrics(
        recs, evaluation, policy.BASE_CONSTRAINTS, int(best["trial_index"])
    )
    diagnostics, missed = policy.build_auxiliary_diagnostics(
        recs, evaluation, audit, policy.BASE_CONSTRAINTS, segment_rows
    )
    metrics.update(diagnostics)
    metrics["revenue_price_compliant"] = _revenue_price_compliant(best["policy"])
    for key in ["total_mape", "hit_qty_mape", "net_benefit"]:
        if not math.isclose(
            float(metrics[key]), float(best[key]), rel_tol=1e-10, abs_tol=1e-8
        ):
            raise RuntimeError(
                f"同步复现失败：{key}={metrics[key]}，记录值={best[key]}"
            )
    policy.write_json(
        BEST_CONFIG_PATH,
        {
            "phase": PHASE_NAME,
            "best_trial_index": int(best["trial_index"]),
            "best_policy": _simplified_policy(best["policy"]),
            "best_metrics": policy._metric_payload(metrics),
            "evaluation_schedule_sha256": schedule_hash,
            "context_input_sha256": context_hash,
            "evaluator_sha256": policy.sha256_file(policy.EVALUATOR_PATH),
            "external_evaluated": False,
        },
    )
    policy.write_menu(recs, BEST_MENU_PATH)
    _, hit_cases = hit_quantity_diagnostics(recs, evaluation)
    hit_cases.to_csv(BEST_HIT_FAILURES_PATH, index=False, encoding="utf-8-sig")
    missed.to_csv(
        MISSED_REVENUE_PATH, index=False, encoding="utf-8-sig"
    )
    _, capacity_sessions = menu_capacity_diagnostics(
        recs, evaluation, audit
    )
    state_rows = state_meal_diagnostics(
        capacity_sessions, int(best["trial_index"])
    )
    pd.DataFrame(state_rows).to_csv(
        ROOT / "best_state_meal_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "best_trial_index": int(best["trial_index"]),
        "total_mape": metrics["total_mape"],
        "hit_qty_mape": metrics["hit_qty_mape"],
        "net_benefit": metrics["net_benefit"],
        "reproduced": True,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-best", action="store_true")
    parser.add_argument("--hypothesis")
    parser.add_argument(
        "--type",
        dest="experiment_type",
        choices=["diagnostic", "structural", "parameter", "ablation", "simplification"],
    )
    parser.add_argument("--changes", default="{}", help="JSON object")
    parser.add_argument("--parent", default="best", help="best, baseline, or trial index")
    args = parser.parse_args()
    if args.sync_best:
        sync_best_outputs()
        return
    if not args.hypothesis or not args.experiment_type:
        parser.error("实验必须同时提供 --hypothesis 和 --type")
    changes = json.loads(args.changes)
    if not isinstance(changes, dict):
        raise ValueError("--changes 必须是 JSON object")
    run_one(args.hypothesis, args.experiment_type, changes, args.parent)


if __name__ == "__main__":
    main()

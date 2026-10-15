#!/usr/bin/env python3
"""Finalize current training-only artifacts without opening external sales."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import adaptive_search
import policy
from diagnostics import menu_capacity_diagnostics


ROOT = Path(__file__).resolve().parent


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=policy._json_default,
                )
                + "\n"
            )


def main() -> None:
    records = adaptive_search._load_trials()
    best = adaptive_search._best_record(records)
    if best is None:
        raise RuntimeError("没有已保留策略")
    if policy.sha256_file(policy.EVALUATOR_PATH) != (
        adaptive_search.FIXED_EVALUATOR_SHA256
    ):
        raise RuntimeError("固定 evaluator.py 已变化")
    config = json.loads(
        (ROOT / "config.json").read_text(encoding="utf-8-sig")
    )
    external_result_path = (
        ROOT / "trial18_external_evaluation.json"
    )
    external_evaluated = bool(
        config.get("external_evaluated", False)
        and external_result_path.exists()
    )
    external_result = (
        json.loads(
            external_result_path.read_text(encoding="utf-8-sig")
        )
        if external_evaluated
        else {}
    )

    summary_columns = [
        "trial_index",
        "experiment_type",
        "is_new_best",
        "hypothesis",
        "selection_reason",
        "failure_reason",
        "total_mape",
        "hit_qty_mape",
        "hit_qty_wmape",
        "net_benefit",
        "menu_count_mae",
        "menu_count_mape",
        "menu_count_bias",
        "holiday_menu_count_bias",
        "over_capacity_session_rate",
        "excess_dish_count",
        "below_min_batch_count",
        "menu_fill_rate",
        "high_revenue_candidate_recall",
        "nonhit_pred_qty_share",
        "total_constraint_violation_count",
    ]
    pd.DataFrame(
        [
            {column: row.get(column) for column in summary_columns}
            for row in records
        ]
    ).to_csv(ROOT / "summary.csv", index=False, encoding="utf-8-sig")
    _write_jsonl(ROOT / "all_trials.jsonl", records)
    report_rows = [
        {
            "stage": "training_expanding_window",
            **{
                key: config["best_metrics"].get(key)
                for key in [
                    "total_mape",
                    "hit_qty_mape",
                    "hit_qty_wmape",
                    "qty_wmape",
                    "order_wmape",
                    "net_benefit",
                    "revenue_coverage",
                    "dish_hit_rate",
                    "menu_count_mae",
                    "menu_count_mape",
                    "holiday_menu_count_bias",
                    "below_min_batch_count",
                    "total_constraint_violation_count",
                ]
            },
        }
    ]
    if external_evaluated:
        for split in ["validation", "test"]:
            metrics = external_result[f"{split}_metrics"]
            report_rows.append(
                {
                    "stage": f"frozen_{split}",
                    **{
                        key: metrics.get(key)
                        for key in report_rows[0]
                        if key != "stage"
                    },
                }
            )
    pd.DataFrame(report_rows).to_csv(
        ROOT / "trial18_train_validation_test_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train = policy.read_sales(policy.TRAIN_PATH)
    attributes, _ = policy.build_library_attributes()
    name_audit = policy.build_name_audit(
        [("training", train)], attributes
    )
    name_audit.to_csv(
        ROOT / "unmatched_dishes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _, evaluation, _ = policy.training_backtest_frames(train)
    recs = pd.read_csv(
        ROOT / "recommended_menu.csv",
        encoding="utf-8-sig",
    )
    _, capacity_sessions = menu_capacity_diagnostics(
        recs, evaluation, []
    )
    capacity_sessions.to_csv(
        ROOT / "best_menu_capacity_sessions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline = records[0]
    comparison_keys = [
        "total_mape",
        "hit_qty_mape",
        "net_benefit",
        "menu_count_mae",
        "menu_count_mape",
        "menu_count_bias",
        "holiday_menu_count_bias",
        "over_capacity_session_rate",
        "excess_dish_count",
        "avg_pred_qty_per_dish",
        "below_min_batch_count",
        "menu_fill_rate",
        "high_revenue_candidate_recall",
        "nonhit_pred_qty_share",
    ]
    comparison_rows = []
    for key in comparison_keys:
        before = float(baseline.get(key, 0.0))
        after = float(best.get(key, 0.0))
        comparison_rows.append(
            {
                "metric": key,
                "phase_baseline": before,
                "retained_trial_18": after,
                "absolute_change": after - before,
                "relative_change_pct": (
                    (after - before) / abs(before) * 100
                    if before != 0
                    else 0.0
                ),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        ROOT / "factor_capacity_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    accepted = [
        int(row["trial_index"])
        for row in records
        if bool(row.get("is_new_best", False))
    ]
    best_index = int(best["trial_index"])
    consecutive_failures = len(
        [
            row
            for row in records
            if int(row["trial_index"]) > best_index
            and not bool(row.get("is_new_best", False))
        ]
    )
    experiment_types = sorted(
        {str(row["experiment_type"]) for row in records}
    )
    stop_audit = {
        "phase": adaptive_search.PHASE_NAME,
        "status": (
            "frozen_external_evaluated"
            if external_evaluated
            else "frozen_training_only"
        ),
        "best_trial_index": best_index,
        "accepted_trial_indices": accepted,
        "completed_trial_count": len(records),
        "explored_experiment_types": experiment_types,
        "consecutive_full_failures_after_best": consecutive_failures,
        "stop_reason": (
            "已探索因子裁剪、日历/天气交互、影子总量、层级容量、"
            "需求/状态容量消融、容量天气、最小批量及核心模块消融；"
            "Trial 19-21连续完整回测未刷新Pareto最佳。"
        ),
        "external_data_read": external_evaluated,
        "external_test_rerun": external_evaluated,
        "external_evaluated_trial_index": (
            18 if external_evaluated else None
        ),
        "external_evaluation_lock": (
            "trial18_external_evaluation.lock.json"
            if external_evaluated
            else None
        ),
        "evaluator_sha256": adaptive_search.FIXED_EVALUATOR_SHA256,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    policy.write_json(ROOT / "stop_audit.json", stop_audit)

    version_dir = ROOT / "versions" / f"trial_{best_index}"
    version_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "policy.py",
        "diagnostics.py",
        "adaptive_search.py",
        "evaluator.py",
        "config.json",
        "recommended_menu.csv",
        "best_state_meal_diagnostics.csv",
        "factor_capacity_comparison.csv",
        "README.md",
    ]:
        shutil.copy2(ROOT / filename, version_dir / filename)
    if external_evaluated:
        for filename in [
            "trial18_freeze_manifest.json",
            "trial18_external_evaluation.lock.json",
            "trial18_external_evaluation.json",
            "trial18_external_segment_metrics.csv",
            "trial18_external_state_meal_diagnostics.csv",
            "trial18_train_validation_test_summary.csv",
        ]:
            shutil.copy2(ROOT / filename, version_dir / filename)
    policy.write_json(version_dir / "trial.json", best)
    recs["date"] = pd.to_datetime(recs["date"])
    allocation = recs.groupby(
        policy.GROUP_KEYS, as_index=False
    ).agg(
        allocated_qty=("pred_qty", "sum"),
        pred_total_qty=("pred_total_qty", "first"),
    )
    max_conservation_error = float(
        (
            allocation["allocated_qty"]
            - allocation["pred_total_qty"]
        )
        .abs()
        .max()
    )
    required_artifacts = [
        "policy.py",
        "config.json",
        "evaluator.py",
        "trials.jsonl",
        "all_trials.jsonl",
        "summary.csv",
        "failure_cases.csv",
        "unmatched_dishes.csv",
        "recommended_menu.csv",
        "sample_efficiency.png",
        "README.md",
    ]
    if external_evaluated:
        required_artifacts.extend(
            [
                "trial18_freeze_manifest.json",
                "trial18_external_evaluation.lock.json",
                "trial18_external_evaluation.json",
                "trial18_validation_recommended_menu.csv",
                "trial18_test_recommended_menu.csv",
                "trial18_external_segment_metrics.csv",
                "trial18_external_state_meal_diagnostics.csv",
                "trial18_train_validation_test_summary.csv",
            ]
        )
    artifact_audit = {
        "best_trial_index": best_index,
        "required_artifacts": {
            filename: (ROOT / filename).exists()
            for filename in required_artifacts
        },
        "recommendation_row_count": int(len(recs)),
        "evaluated_session_count": int(len(allocation)),
        "max_total_conservation_absolute_error": max_conservation_error,
        "duplicate_menu_dish_count": int(
            recs.duplicated(
                policy.GROUP_KEYS + ["菜品名称"]
            ).sum()
        ),
        "total_constraint_violation_count": int(
            best["total_constraint_violation_count"]
        ),
        "evaluator_sha256": policy.sha256_file(
            policy.EVALUATOR_PATH
        ),
        "policy_sha256": policy.sha256_file(ROOT / "policy.py"),
        "external_data_read": external_evaluated,
        "external_test_rerun": external_evaluated,
        "external_evaluated_trial_index": (
            18 if external_evaluated else None
        ),
    }
    policy.write_json(
        ROOT / "final_artifact_audit.json", artifact_audit
    )
    print(
        json.dumps(
            {
                "best_trial_index": best_index,
                "trial_count": len(records),
                "consecutive_failures": consecutive_failures,
                "external_data_read": external_evaluated,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the one permitted Trial 18 validation/test evaluation.

This entry point is deliberately separate from adaptive_search.py.  It creates
an exclusive Trial 18 lock before opening the external sales file and refuses
to run again once that lock exists.  The previous Trial 108 lock is preserved.
"""

from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import adaptive_search
import policy
from diagnostics import (
    hit_quantity_diagnostics,
    menu_capacity_diagnostics,
    state_meal_diagnostics,
)
from evaluator import build_failure_cases


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
EXTERNAL_PATH = (
    ROOT / "数据/菜品销量_吉大湖畔智慧餐厅_菜名修正后.csv"
)
VALIDATION_START = pd.Timestamp("2026-01-01")
VALIDATION_END = pd.Timestamp("2026-03-31")
TEST_START = pd.Timestamp("2026-04-01")

MANIFEST_PATH = ROOT / "trial18_freeze_manifest.json"
LOCK_PATH = ROOT / "trial18_external_evaluation.lock.json"
RESULT_PATH = ROOT / "trial18_external_evaluation.json"
VALIDATION_MENU_PATH = (
    ROOT / "trial18_validation_recommended_menu.csv"
)
TEST_MENU_PATH = ROOT / "trial18_test_recommended_menu.csv"
SEGMENTS_PATH = ROOT / "trial18_external_segment_metrics.csv"
STATE_MEAL_PATH = (
    ROOT / "trial18_external_state_meal_diagnostics.csv"
)
FAILURE_PATH = ROOT / "trial18_external_failure_cases.csv"
HIT_FAILURE_PATH = (
    ROOT / "trial18_external_hit_failure_cases.csv"
)
MISSED_REVENUE_PATH = (
    ROOT / "trial18_external_missed_high_revenue.csv"
)
NAME_AUDIT_PATH = ROOT / "trial18_external_name_audit.csv"
PRIOR_LOCK_PATH = ROOT / "adaptive_test_evaluation.lock.json"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _write_lock(payload: dict[str, Any]) -> None:
    policy.write_json(LOCK_PATH, payload)


def _subset_audit(
    rows: list[dict[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if pd.Timestamp(row["date"]) >= start
        and (
            end is None
            or pd.Timestamp(row["date"]) <= end
        )
    ]


def _evaluate(
    split: str,
    recs: pd.DataFrame,
    actual: pd.DataFrame,
    audit: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
]:
    metrics = policy.evaluate_policy_metrics(
        recs, actual, policy.BASE_CONSTRAINTS
    )
    segments = policy.build_segment_metrics(
        recs,
        actual,
        policy.BASE_CONSTRAINTS,
        trial_index=18,
    )
    auxiliary, missed = policy.build_auxiliary_diagnostics(
        recs,
        actual,
        audit,
        policy.BASE_CONSTRAINTS,
        segments,
    )
    metrics.update(auxiliary)
    metrics["split"] = split
    _, hit_cases = hit_quantity_diagnostics(recs, actual)
    _, capacity_sessions = menu_capacity_diagnostics(
        recs, actual, audit
    )
    state_rows = state_meal_diagnostics(
        capacity_sessions, trial_index=18
    )
    return metrics, segments, state_rows, hit_cases, missed


def run_once() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise RuntimeError(
            "trial18_external_evaluation.lock.json 已存在；"
            "Trial 18 外部评估禁止重复运行"
        )
    config = json.loads(
        (ROOT / "config.json").read_text(encoding="utf-8-sig")
    )
    if int(config.get("best_trial_index", -1)) != 18:
        raise RuntimeError("当前冻结策略不是 Trial 18")
    if bool(config.get("external_evaluated", False)):
        raise RuntimeError("当前配置已标记完成外部评估")
    evaluator_hash = policy.sha256_file(policy.EVALUATOR_PATH)
    if evaluator_hash != adaptive_search.FIXED_EVALUATOR_SHA256:
        raise RuntimeError("固定 evaluator.py 哈希不一致")

    manifest = {
        "status": "frozen",
        "phase": adaptive_search.PHASE_NAME,
        "best_trial_index": 18,
        "frozen_at": _now(),
        "policy_sha256": policy.sha256_file(ROOT / "policy.py"),
        "diagnostics_sha256": policy.sha256_file(
            ROOT / "diagnostics.py"
        ),
        "evaluator_sha256": evaluator_hash,
        "config_sha256": policy.sha256_file(ROOT / "config.json"),
        "evaluation_schedule_sha256": config[
            "evaluation_schedule_sha256"
        ],
        "validation_date_range": [
            str(VALIDATION_START.date()),
            str(VALIDATION_END.date()),
        ],
        "test_start": str(TEST_START.date()),
        "prior_trial108_lock_preserved": PRIOR_LOCK_PATH.exists(),
        "prior_trial108_lock_sha256": (
            policy.sha256_file(PRIOR_LOCK_PATH)
            if PRIOR_LOCK_PATH.exists()
            else None
        ),
        "authorization": (
            "用户明确要求在当前 Trial 18 上补充验证集和测试集"
            "指标结果分析"
        ),
    }
    policy.write_json(MANIFEST_PATH, manifest)
    started = {
        "status": "started",
        "started_at": _now(),
        "best_trial_index": 18,
        "freeze_manifest_sha256": policy.sha256_file(
            MANIFEST_PATH
        ),
        "frozen_config_sha256": manifest["config_sha256"],
        "external_data_read": False,
        "external_data_read_count": 0,
    }
    with LOCK_PATH.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(started, ensure_ascii=False, indent=2)
            + "\n"
        )

    try:
        external = policy.read_sales(EXTERNAL_PATH)
        started.update(
            {
                "external_data_read": True,
                "external_data_read_count": 1,
                "external_data_read_at": _now(),
            }
        )
        _write_lock(started)

        validation = external.loc[
            external["date"].between(
                VALIDATION_START, VALIDATION_END
            )
        ].copy()
        test = external.loc[
            external["date"].ge(TEST_START)
        ].copy()
        if validation.empty or test.empty:
            raise RuntimeError("验证集或测试集为空")
        if pd.Timestamp(validation["date"].min()) != VALIDATION_START:
            raise RuntimeError("验证集起始日期不符合冻结定义")
        if pd.Timestamp(test["date"].min()) != TEST_START:
            raise RuntimeError("测试集起始日期不符合冻结定义")

        train = policy.read_sales(policy.TRAIN_PATH)
        attributes, _ = policy.build_library_attributes()
        evaluation_actual = external.loc[
            external["date"].ge(VALIDATION_START)
        ].copy()
        contexts = policy.build_contexts(
            train, evaluation_actual, attributes
        )
        selection_audit: list[dict[str, Any]] = []
        all_recs = policy.make_recommendations(
            contexts,
            copy.deepcopy(config["best_policy"]),
            policy.BASE_CONSTRAINTS,
            selection_audit=selection_audit,
        )
        validation_recs = all_recs.loc[
            all_recs["date"].between(
                VALIDATION_START, VALIDATION_END
            )
        ].copy()
        test_recs = all_recs.loc[
            all_recs["date"].ge(TEST_START)
        ].copy()

        (
            validation_metrics,
            validation_segments,
            validation_state,
            _,
            _,
        ) = _evaluate(
            "validation",
            validation_recs,
            validation,
            _subset_audit(
                selection_audit,
                VALIDATION_START,
                VALIDATION_END,
            ),
        )
        (
            test_metrics,
            test_segments,
            test_state,
            test_hit_cases,
            test_missed,
        ) = _evaluate(
            "test",
            test_recs,
            test,
            _subset_audit(selection_audit, TEST_START),
        )

        for split, recs, metrics in [
            ("validation", validation_recs, validation_metrics),
            ("test", test_recs, test_metrics),
        ]:
            if int(
                metrics["total_constraint_violation_count"]
            ) != 0:
                raise RuntimeError(f"{split} 出现硬约束违规")
            if recs.duplicated(
                policy.GROUP_KEYS + ["菜品名称"]
            ).any():
                raise RuntimeError(f"{split} 出现重复菜品")
            if float(metrics["total_allocation_wape"]) > 1e-9:
                raise RuntimeError(f"{split} 未保持总量守恒")

        policy.write_menu(
            validation_recs, VALIDATION_MENU_PATH
        )
        policy.write_menu(test_recs, TEST_MENU_PATH)
        build_failure_cases(test_recs, test).to_csv(
            FAILURE_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        test_hit_cases.to_csv(
            HIT_FAILURE_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        test_missed.to_csv(
            MISSED_REVENUE_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        policy.build_name_audit(
            [("trial18_validation", validation), ("trial18_test", test)],
            attributes,
        ).to_csv(
            NAME_AUDIT_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"split": "validation", **row}
                for row in validation_segments
            ]
            + [
                {"split": "test", **row}
                for row in test_segments
            ]
        ).to_csv(
            SEGMENTS_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"split": "validation", **row}
                for row in validation_state
            ]
            + [
                {"split": "test", **row}
                for row in test_state
            ]
        ).to_csv(
            STATE_MEAL_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        result = {
            "status": "complete",
            "completed_at": _now(),
            "phase": adaptive_search.PHASE_NAME,
            "best_trial_index": 18,
            "freeze_manifest_sha256": policy.sha256_file(
                MANIFEST_PATH
            ),
            "frozen_config_sha256": manifest["config_sha256"],
            "validation_metrics": policy._metric_payload(
                validation_metrics
            ),
            "test_metrics": policy._metric_payload(test_metrics),
            "validation_date_range": [
                str(validation["date"].min().date()),
                str(validation["date"].max().date()),
            ],
            "test_date_range": [
                str(test["date"].min().date()),
                str(test["date"].max().date()),
            ],
            "validation_menu_sha256": policy.sha256_file(
                VALIDATION_MENU_PATH
            ),
            "test_menu_sha256": policy.sha256_file(
                TEST_MENU_PATH
            ),
            "external_data_read_count": 1,
            "prior_trial108_lock_overwritten": False,
        }
        policy.write_json(RESULT_PATH, result)
        completed_lock = {
            **result,
            "external_data_read": True,
            "result_sha256": policy.sha256_file(RESULT_PATH),
        }
        _write_lock(completed_lock)

        config.update(
            {
                "external_evaluated": True,
                "external_evaluated_trial_index": 18,
                "external_evaluation_lock": LOCK_PATH.name,
                "external_evaluated_at": result["completed_at"],
            }
        )
        policy.write_json(ROOT / "config.json", config)
        stop_path = ROOT / "stop_audit.json"
        if stop_path.exists():
            stop = json.loads(
                stop_path.read_text(encoding="utf-8-sig")
            )
            stop.update(
                {
                    "external_data_read": True,
                    "external_test_rerun": True,
                    "external_evaluated_trial_index": 18,
                    "external_evaluation_lock": LOCK_PATH.name,
                }
            )
            policy.write_json(stop_path, stop)

        version_dir = PROJECT / "versions/trial_18"
        version_dir.mkdir(parents=True, exist_ok=True)
        for path in [
            ROOT / "config.json",
            MANIFEST_PATH,
            LOCK_PATH,
            RESULT_PATH,
            SEGMENTS_PATH,
            STATE_MEAL_PATH,
        ]:
            shutil.copy2(path, version_dir / path.name)
        print(
            json.dumps(
                {
                    "best_trial_index": 18,
                    "validation_total_mape": validation_metrics[
                        "total_mape"
                    ],
                    "test_total_mape": test_metrics["total_mape"],
                    "test_hit_qty_mape": test_metrics[
                        "hit_qty_mape"
                    ],
                    "test_net_benefit": test_metrics[
                        "net_benefit"
                    ],
                    "external_data_read_count": 1,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return result
    except Exception as exc:
        failed = {
            **started,
            "status": "failed",
            "failed_at": _now(),
            "error": repr(exc),
        }
        _write_lock(failed)
        raise


if __name__ == "__main__":
    run_once()

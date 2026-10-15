"""Final artifact/selection verification, not a forecast or model-selection step."""
import json
import sys
import numpy as np
import pandas as pd
import pytest
import policy
from phase_data import ROOT,OUT,DATASETS
from phase_release import verify_freeze
from targeted_search import dump


def run():
    frozen=verify_freeze();assert set(frozen["selected"].values())=={"baseline"}
    records=[json.loads(p.read_text()) for p in (OUT/"experiments").glob("*/record.json")]
    assert all(r["status"]=="complete" for r in records)
    assert not any(v.get("gate",{}).get("eligible") for r in records for v in r["datasets"].values())
    retest={ds:json.loads((OUT/"retest"/ds/"record.json").read_text()) for ds in DATASETS}
    assert all(r["status"]=="complete" for r in retest.values())
    for ds,r in retest.items():
        for n,h in r["sha256"].items():assert policy.sha256_file(OUT/"retest"/ds/n)==h
        assert r["results"]["full_existing_interval"]["repeat_over_80_sessions"]==0
    integrity=json.loads((OUT/"fixed_menu_integrity_audit.json").read_text());assert len(integrity["experiments"])==20
    exports=json.loads((OUT/"demand_preparation_export_manifest.json").read_text())
    for e in exports["exports"]:assert policy.sha256_file(OUT/e["source"])==e["source_sha256"] and policy.sha256_file(OUT/e["export"])==e["export_sha256"]
    artifacts=["阶段与单菜结构实验报告.md","REPRODUCE.md","clarified_experiment_metrics.csv","clarified_over_under_metrics.csv","all_segments.csv","all_raw_segments.csv","combinations_ablations.csv",
        "risk_interval_segments.csv","interval_reference_comparison.csv","holiday_count_volume_changes.csv","chronological_selection_stability.csv","frozen_retest_summary.csv","retest_date_type_coverage.csv","retest_monthly_metrics.csv","current_tests.json","diagnostic_field_clarification.json"]
    dump(OUT/"DELIVERY.json",dict(status="complete",selected=frozen["selected"],experiment_records=len(records),scored_dataset_trials=32,skipped_raw_entry_risk_trials=2,
        test_replay_runs={ds:1 for ds in DATASETS},retest_sessions={ds:r["results"]["full_existing_interval"]["session_count"] for ds,r in retest.items()},
        unchanged_web_default=True,unchanged_original_evaluator=True,unchanged_registered_data=True,target_30_percent_achieved=False,
        fixed_menu_audited_trials=len(integrity["experiments"]),max_unit_total_error=max(r["max_unit_error"] for r in integrity["experiments"]),
        unresolved=["2025 summer start calendar boundary unknown","Supply quantity/stockout/leftover process absent","No qualified new recorded-sales forecast","Legacy pre-merge cache tests intentionally still reject current fingerprints"],
        reporting_correction="Disambiguated total versus hit over/under quantities after freeze; no primary metric, prediction, selection or evaluator change",
        environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,pytest=pytest.__version__),
        artifact_sha256={name:policy.sha256_file(OUT/name) for name in artifacts},postfreeze_report_only_sources={n:policy.sha256_file(ROOT/n) for n in ["phase_retest_report.py","phase_metric_clarification.py","phase_finish.py"]}))
    print("Verified complete delivery",OUT,flush=True)


if __name__=="__main__":run()

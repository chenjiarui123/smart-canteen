"""Audit saved development experiments and freeze a decision, never tune/test."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import policy
from diagnostics import hit_quantity_diagnostics
from structural_data import ROOT, OUT, DATASETS, verify_protected
from structural_experiments import gates, load_assets
from targeted_search import dump


ABLATIONS = {
    "A_B_E_F": {"separate_preparation": "B_E_F", "shares": "A_E_F", "competition": "A_B_F", "reliability": "A_B_E"},
    "A_C_F": {"separate_preparation": "C_F", "peer_borrowing": "A_F_baseline_anchor", "reliability": "A_C"},
    "guarded_A_C_F": {"admission_guard": "A_C_F", "separate_preparation": "guarded_C_F", "peer_borrowing": "guarded_A_F_baseline", "reliability": "guarded_A_C"},
}


def adjacent(menu):
    menus = {k: set(g["菜品名称"].map(policy.normalize_name)) for k, g in menu.groupby(policy.GROUP_KEYS)}
    rows = []
    for (day, canteen, meal), names in menus.items():
        previous = menus.get((day-pd.Timedelta(days=1), canteen, meal))
        repeated = len(names & previous) if previous else None
        rows.append(dict(date=day, 食堂名称=canteen, 餐次=meal, recommended_count=len(names),
                         repeated_count=repeated, limit=len(names)*4//5,
                         adjacent_repeat_rate=repeated/len(names) if repeated is not None else None,
                         status="missing_previous" if previous is None else "checked"))
    result = pd.DataFrame(rows)
    assert not (result.repeated_count > result.limit).any()
    return result


def main():
    protected = verify_protected()
    records = {p.parent.name: json.loads(p.read_text()) for p in (OUT/"experiments").glob("*/record.json")}
    assert all(r["status"] == "complete" for r in records.values())
    for name, r in records.items():
        assert not r["test_sales_used"]
        for source, sha in r["source_hashes"].items():
            assert policy.sha256_file(OUT/"experiments"/name/"source"/source) == sha, (name, source)
        if "expert" in r:
            assert policy.sha256_file(OUT/"experiments"/r["expert"]/"record.json") == r["expert_record_sha256"]
    ablation_rows = []
    for full, removals in ABLATIONS.items():
        for component, reduced in removals.items():
            expected = dict(records[full]["config"])
            expected.pop(component)
            assert expected == records[reduced]["config"], (full, component, reduced)
            ablation_rows.append(dict(full=full, removed=component, reduced=reduced, exact_one_component=True))
    dump(OUT/"ablation_audit.json", ablation_rows)
    summary, checks, segment_rows, repeat_summary, inherited_boxes = [], [], [], {}, []
    for dataset in DATASETS:
        baseline, features, actual, _ = load_assets(dataset)
        development = OUT/("development_"+dataset)
        schedule = pd.read_csv(development/"schedule.csv", parse_dates=["date", "max_history_date"])
        assert schedule.max_history_date.lt(schedule.date).all()
        assert schedule.groupby("date").max_history_date.nunique().eq(1).all()
        for field in ["history_max_date", "structural_history_max_date"]:
            assert pd.to_datetime(features[field]).lt(features.date).all()
        warmup = pd.read_pickle(development/"warmup.pkl")
        assert actual.date.min() == warmup.date.min()+pd.Timedelta(days=14)
        assert warmup.date.max() < actual.date.min()
        assert actual.date.max() <= pd.Timestamp("2025-12-31")
        assert baseline.groupby(policy.GROUP_KEYS).ngroups == actual.groupby(policy.GROUP_KEYS).ngroups
        rep = adjacent(baseline)
        rep.to_csv(development/"adjacent_final_audit.csv", index=False)
        base_audit = json.loads((development/"baseline_audit.json").read_text())
        assert sum(a["adjacent_status"] == "missing_previous" for a in base_audit) == rep.status.eq("missing_previous").sum()
        repeat_summary[dataset] = dict(sessions=len(rep), max_repeat=float(rep.adjacent_repeat_rate.max()),
            mean_repeat=float(rep.adjacent_repeat_rate.mean()), over_80=0,
            missing_previous=int(rep.status.eq("missing_previous").sum()),
            infeasible=sum(a["adjacent_status"] == "infeasible" for a in base_audit))
        old_hits = hit_quantity_diagnostics(baseline, actual)[1]
        hitkeys = policy.GROUP_KEYS+["_dish_norm"]
        old_hit_keys = set(map(tuple, old_hits[hitkeys].to_numpy()))
        unit_features = features.set_index(hitkeys).sale_unit
        for name, r in sorted(records.items()):
            if dataset not in r["datasets"]:
                continue
            folder = OUT/"experiments"/name/dataset
            menu = pd.read_pickle(folder/"menu.pkl.gz")
            keep = [c for c in baseline if c not in {"pred_qty", "pred_orders", "pred_rev"}]
            pd.testing.assert_frame_equal(menu[keep], baseline[keep], check_exact=True)
            assert np.isfinite(menu[["pred_qty", "pred_orders", "pred_rev"]]).all().all()
            assert menu[["pred_qty", "pred_orders", "pred_rev"]].ge(0).all().all()
            np.testing.assert_allclose(menu.pred_orders, menu.pred_qty*menu.expected_orders_per_qty, rtol=1e-12, atol=1e-8)
            np.testing.assert_allclose(menu.pred_rev, menu.pred_qty*menu.expected_rev_per_qty, rtol=1e-12, atol=1e-8)
            np.testing.assert_allclose(menu.pred_qty, menu.expected_demand_qty, rtol=0, atol=0)
            np.testing.assert_allclose(menu.minimum_batch_surplus_risk, (menu.suggested_preparation_qty-menu.pred_qty).clip(lower=0), rtol=1e-12, atol=1e-8)
            def totals(frame, by_unit=False):
                frame = frame.assign(_dish_norm=frame["菜品名称"].map(policy.normalize_name))
                if by_unit:
                    frame = frame.join(unit_features.rename("sale_unit"), on=hitkeys, validate="one_to_one")
                return frame.groupby(policy.GROUP_KEYS+(["sale_unit"] if by_unit else [])).pred_qty.sum()
            errors = []
            for by_unit in [False, True]:
                b, n = totals(baseline, by_unit), totals(menu, by_unit)
                np.testing.assert_allclose(b, n, rtol=1e-12, atol=1e-8)
                errors.append(float((b-n).abs().max()))
            if name == "final_verified_control":
                reference = "baseline" if dataset == "a_hupan" else "baseline_f"
                pd.testing.assert_frame_equal(menu, pd.read_pickle(OUT/"experiments"/reference/dataset/"menu.pkl.gz"), check_exact=True)
            audits = pd.read_csv(folder/"allocation_audit.csv")
            conflict = audits.status.str.contains("conflict")
            inherited = audits.loc[(audits.historical_box_violations > 0) & ~conflict]
            for row in inherited.to_dict("records"):
                mask = (menu.date.eq(pd.Timestamp(row["date"])) & menu["食堂名称"].eq(row["食堂名称"]) & menu["餐次"].eq(row["餐次"]))
                # Identity controls can inherit baseline values beyond the new
                # historical cap. Permit only a fully unchanged original meal,
                # and surface this original audit-label limitation explicitly.
                np.testing.assert_array_equal(menu.loc[mask, "pred_qty"], baseline.loc[mask, "pred_qty"])
                inherited_boxes.append(dict(dataset=dataset, trial=name, date=row["date"], meal=row["餐次"],
                    count=int(row["historical_box_violations"]), reason="unchanged baseline exceeds new experimental historical cap; original status label was allocated"))
            # Existing solver tolerance plus CSV round-trip machine precision;
            # this is not a change to any pre-registered selection threshold.
            csv_roundoff = 8*np.finfo(float).eps
            assert (audits.new_hhi <= audits.baseline_hhi*1.15+1e-12+csv_roundoff).all()
            assert audits.top_share_increase.le(.03+1e-12+csv_roundoff).all()
            if r["config"].get("separate_preparation"):
                assert menu.suggested_preparation_qty.ge(menu.min_batch_qty-1e-8).all()
            decisions = folder/"admission_decisions.json"
            if decisions.exists():
                for d in json.loads(decisions.read_text()):
                    if d["previous_max_date"] is not None:
                        assert pd.Timestamp(d["previous_max_date"]) < pd.Timestamp(d["date"])
            # Recompute original primary and hit metrics; no evaluator edits.
            metrics = policy.evaluate_policy_metrics(menu, actual, policy.BASE_CONSTRAINTS)
            hm, hits = hit_quantity_diagnostics(menu, actual)
            metrics.update(hm)
            assert set(map(tuple, hits[hitkeys].to_numpy())) == old_hit_keys
            saved = json.loads((folder/"metrics.json").read_text())
            for key in ["total_mape", "hit_qty_mape", "net_benefit", "hit_qty_wmape", "weighted_qty_bias", "nonhit_pred_qty_share"]:
                np.testing.assert_allclose(metrics[key], saved[key], rtol=1e-12, atol=1e-8)
            segments = pd.read_csv(folder/"session_segments.csv")
            review = gates(dataset, saved, segments)
            if name not in {"baseline", "baseline_f"}:
                recorded_review = r["datasets"][dataset]["gate"]
                for field in ["eligible", "reasons", "thresholds_fixed_before_baseline"]:
                    assert review[field] == recorded_review[field]
                assert len(review["subgroup_failures"]) == len(recorded_review["subgroup_failures"])
                for current, recorded in zip(review["subgroup_failures"], recorded_review["subgroup_failures"]):
                    assert (current["dimension"], current["segment"]) == (recorded["dimension"], recorded["segment"])
                    for field in ["old", "new", "permitted_increase_pp"]:
                        np.testing.assert_allclose(current[field], recorded[field], rtol=1e-14, atol=1e-12)
                assert not review["eligible"], "An eligible candidate requires a new selection review"
            summary.append(dict(dataset=dataset, trial=name, hypothesis=r["hypothesis"],
                **{k:v for k,v in saved.items() if not isinstance(v, dict)},
                eligible=review["eligible"], failure_reasons="; ".join(review["reasons"])))
            segment_rows.append(segments.assign(dataset=dataset, trial=name))
            checks.append(dict(dataset=dataset, trial=name, rows=len(menu), sessions=menu.groupby(policy.GROUP_KEYS).ngroups,
                exact_menu_metadata=True, same_positive_hit_set=True, session_total_error=errors[0], unit_total_error=errors[1],
                original_metrics_recomputed=True, source_snapshot_verified=True, explicit_conflict_sessions=int(conflict.sum()),
                inherited_identity_box_sessions=len(inherited)))
            if name in {"final_verified_control", "A_C_F", "guarded_A_C_F"}:
                menu.to_csv(folder/"menu.csv.gz", index=False, compression="gzip")
                hits.to_csv(folder/"hits.csv.gz", index=False, compression="gzip")
            print("audited", dataset, name, flush=True)
    pd.DataFrame(summary).to_csv(OUT/"all_experiment_metrics.csv", index=False)
    pd.concat(segment_rows, ignore_index=True).to_csv(OUT/"all_session_segments.csv", index=False)
    decision = dict(selected={d:{"identity":True} for d in DATASETS}, accepted_improvement=False,
        reason="No candidate passes immutable development gates; retain deployed forecast, do not enter phase 2",
        hit_mape_target_30_reached=False, experiment_records=len(records),
        distinct_structural_directions=["conditional shares (flat/hierarchical/calendar-weather)", "menu competition", "out-of-sample reliability", "peer residual borrowing", "past-performance admission"],
        completed_ablations=ablation_rows, final_exact_replay="final_verified_control",
        fixed_trial_limit=None, stop_condition_met=True,
        existing_2026_retest="not performed in this round; selected forecast is unchanged and test cannot inform selection",
        web_default_unchanged=True, protected_hashes_verified=True)
    dump(OUT/"selection_and_stop_audit.json", decision)
    dump(OUT/"inherited_baseline_box_exceptions.json", inherited_boxes)
    dump(OUT/"final_requirement_audit.json", dict(status="passed_with_documented_limitations", checks=checks,
        repeat=repeat_summary, data_and_protocol_protection=protected["protected"],
        phase2="not entered: no eligible phase1 model", test="2026 quantities not loaded in search/release",
        limitations=["30% target unmet", "A lacks winter and overlapping holiday development observations",
            "4 inherited F minimum-batch conflicts remain in retained identity forecast",
            "Unchanged baseline may exceed new experimental historical cap; full meal equality checked and exceptions separately enumerated",
            "guard runtime equality tested on two cold days only; not a deployed guard validation",
            "no independent unseen test used; development segment metrics are not confidence intervals"] ))
    frozen = OUT/"frozen"
    frozen.mkdir(exist_ok=False)
    dump(frozen/"structural_config.json", decision["selected"])
    sources = sorted(set([p.name for p in ROOT.glob("structural_*.py")]+[
        "structural_protocol.json", "structural_combinations.json", "structural_peer_trials.json", "structural_peer_ablations.json", "structural_final_control.json",
        "policy.py", "evaluator.py", "diagnostics.py", "allocation_policy.py", "allocation_features.py", "allocation_prepare.py", "targeted_search.py", "targeted_policy.py"]))
    for name in sources:
        shutil.copy2(ROOT/name, frozen/name)
    shutil.copytree(OUT/"baseline_source", frozen/"baseline_source")
    for name in ["effective_baseline_config.json", "preregistered_protocol.json", "selection_and_stop_audit.json", "final_requirement_audit.json"]:
        shutil.copy2(OUT/name, frozen/name)
    dump(frozen/"manifest.json", dict(frozen_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        version="structural_20260912_retained_baseline", decision=decision,
        files={str(p.relative_to(frozen)):policy.sha256_file(p) for p in frozen.rglob("*") if p.is_file()},
        execution="Project-root entry points and existing datasets; snapshot is not a standalone SDK"))
    verify_protected()
    print("AUDIT AND FREEZE COMPLETE", repeat_summary, flush=True)


if __name__ == "__main__":
    main()

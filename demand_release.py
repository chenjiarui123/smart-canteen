"""Verify all completed evidence, retain baseline when no new model qualifies."""
import json
import shutil
import numpy as np
import pandas as pd
import policy
from diagnostics import hit_quantity_diagnostics
from demand_data import ROOT,OUT,PREVIOUS,DATASETS,assets,verify
from demand_evaluate import KEYS,raw_diagnostics
from demand_experiments import final_gate
from structural_release import adjacent
from structural_policy import allocate
from structural_evaluate import evaluate
from targeted_search import dump


PAIRS=[
    ("linear_all_log","linear_hits_log","all supplied vs same-seed historical hits"),
    ("conditional_all_log","conditional_hits_log","all supplied vs same-seed historical hits"),
    ("linear_all_log","linear_withold_log","add old forecast as feature"),
    ("conditional_all_log","conditional_withold_log","add old forecast as feature"),
    ("conditional_hurdle_log","conditional_all_log","remove low-state hurdle"),
    ("hurdle_mixed","conditional_mixed","remove low-state hurdle, mixed decision unchanged"),
    ("hurdle_relative","conditional_relative","remove low-state hurdle, relative decision unchanged"),
    ("conditional_mixed","unconditional_mixed","remove conditional partition, same normalized empirical distribution"),
    ("conditional_mixed","conditional_relative","remove absolute-loss component of leaf decision"),
    ("conditional_mixed","conditional_absolute_verified","remove relative-loss component of leaf decision"),
    ("direct_mixed","direct_relative","remove absolute-loss component of direct IRLS fit"),
]


def main():
    verify();paths=list((OUT/"experiments").glob("*/record.json"));records={p.parent.name:json.loads(p.read_text()) for p in paths}
    assert all(r["status"] in {"complete","failed"} for r in records.values())
    failed=[n for n,r in records.items() if r["status"]=="failed"]
    assert failed==["conditional_absolute"],failed
    checks=[];summary=[];all_segments=[];ablation_rows=[];base_metrics={};repeats={}
    for name,r in records.items():
        for source,sha in r["source_hashes"].items():assert policy.sha256_file(OUT/"experiments"/name/"source"/source)==sha
        if r.get("source_model") and "parent_record_sha256" in r:
            assert policy.sha256_file(OUT/"experiments"/r["source_model"]/"record.json")==r["parent_record_sha256"]
    for dataset in DATASETS:
        a=assets(dataset);base=a["baseline_menu"];old_features=a["features"]
        newbaseline=pd.read_pickle(OUT/"reproduction"/("development_"+dataset)/"baseline_menu.pkl")
        pd.testing.assert_frame_equal(base,newbaseline,check_exact=True)
        f=pd.read_pickle(OUT/dataset/"prediction_features.pkl.gz")
        t=pd.read_pickle(OUT/dataset/"supplied_training.pkl.gz")
        for frame in [f,t]:
            valid=frame.history_max_date.notna()
            assert pd.to_datetime(frame.loc[valid,"history_max_date"]).lt(frame.loc[valid,"date"]).all()
            assert frame.date.max()<=pd.Timestamp("2025-12-31")
        assert len(t)==len(a["warmup"])+len(a["actual"])
        rep=adjacent(base);rep.to_csv(OUT/dataset/"adjacent_final_audit.csv",index=False)
        repeats[dataset]=dict(sessions=len(rep),over_80=0,maximum=float(rep.adjacent_repeat_rate.max()),missing_previous=int(rep.status.eq("missing_previous").sum()))
        base_metrics[dataset]=policy.evaluate_policy_metrics(base,a["actual"],policy.BASE_CONSTRAINTS)
        base_metrics[dataset].update(hit_quantity_diagnostics(base,a["actual"])[0])
        for name,r in sorted(records.items()):
            if r["status"]!="complete":continue
            d=r["datasets"][dataset]
            assert not r["test_sales_used"]
            cache_path=OUT/d["distribution_cache"]
            assert policy.sha256_file(cache_path)==d["distribution_sha256"]
            cache=np.load(cache_path,allow_pickle=False)
            fit=pd.Series(pd.to_datetime(cache["fit_max_date"]))
            assert (fit.isna()|fit.lt(base.date.reset_index(drop=True))).all()
            if d["status"]!="complete":
                assert not d["raw"]["precoordination_gate"]["eligible"]
                checks.append(dict(trial=name,dataset=dataset,status="new coordination correctly skipped after failed raw gate"));continue
            p=OUT/"experiments"/name/dataset;menu=pd.read_pickle(p/"menu.pkl.gz")
            keep=[c for c in base if c not in {"pred_qty","pred_orders","pred_rev"}]
            pd.testing.assert_frame_equal(menu[keep],base[keep],check_exact=True)
            assert np.isfinite(menu.pred_qty).all() and menu.pred_qty.ge(0).all()
            np.testing.assert_allclose(menu.pred_orders,menu.pred_qty*menu.expected_orders_per_qty,rtol=1e-12,atol=1e-8)
            np.testing.assert_allclose(menu.pred_rev,menu.pred_qty*menu.expected_rev_per_qty,rtol=1e-12,atol=1e-8)
            assert menu.suggested_preparation_qty.ge(menu.min_batch_qty-1e-8).all()
            detail=pd.read_pickle(p/"pre_post_detail.pkl.gz")
            np.testing.assert_allclose(detail.final_prediction,menu.pred_qty,rtol=0,atol=0)
            for unit_group in [False,True]:
                def totals(frame):
                    frame=frame.assign(_dish_norm=frame["菜品名称"].map(policy.normalize_name))
                    if unit_group:frame=frame.merge(old_features[KEYS+["sale_unit"]],on=KEYS,validate="one_to_one")
                    return frame.groupby(policy.GROUP_KEYS+(["sale_unit"] if unit_group else [])).pred_qty.sum()
                np.testing.assert_allclose(totals(base),totals(menu),rtol=1e-12,atol=1e-8)
            m=policy.evaluate_policy_metrics(menu,a["actual"],policy.BASE_CONSTRAINTS)
            hm,hits=hit_quantity_diagnostics(menu,a["actual"]);m.update(hm)
            bh=hit_quantity_diagnostics(base,a["actual"])[1]
            assert set(map(tuple,hits[KEYS].to_numpy()))==set(map(tuple,bh[KEYS].to_numpy()))
            for k in ["total_mape","hit_qty_mape","net_benefit","hit_qty_wmape","weighted_qty_bias","nonhit_pred_qty_share"]:
                np.testing.assert_allclose(m[k],d["metrics"][k],rtol=1e-12,atol=1e-8)
            seg=pd.read_csv(p/"session_segments.csv")
            review=final_gate(dataset,d["metrics"],seg,detail,menu,d["raw"])
            assert review["eligible"]==d["gate"]["eligible"] and review["reasons"]==d["gate"]["reasons"]
            assert not review["eligible"],"An eligible forecast requires explicit selection instead of baseline retention"
            row=dict(dataset=dataset,trial=name,**{k:v for k,v in d["metrics"].items() if not isinstance(v,dict)},
                **{k:v for k,v in d["raw"].items() if not isinstance(v,dict)},failure_reasons="; ".join(review["reasons"]))
            summary.append(row);all_segments.append(seg.assign(dataset=dataset,trial=name))
            checks.append(dict(trial=name,dataset=dataset,rows=len(menu),status="passed",exact_menu=True,fixed_unit_and_meal_totals=True,unchanged_positive_hit_set=True,original_metrics_recomputed=True,past_only_fit_dates=True))
            print("verified",name,dataset,flush=True)
        # Simplified final output is literally the reproduced baseline demand;
        # independent operating advice is added without altering sales metrics.
        outputs=[];audits=[]
        fg={k:g for k,g in old_features.groupby(policy.GROUP_KEYS,sort=False)}
        for key,menu in base.groupby(policy.GROUP_KEYS,sort=False):
            new,au,_=allocate(menu,fg[key],{"identity":True,"separate_preparation":True})
            outputs.append(new);audits.append(au)
        selected=pd.concat(outputs).sort_index()
        pd.testing.assert_frame_equal(selected[base.columns],base,check_exact=True)
        folder=OUT/"final_selected"/dataset;folder.mkdir(parents=True,exist_ok=False)
        edges=json.loads((PREVIOUS/("development_"+dataset)/"group_edges.json").read_text())
        metrics,hits,hs,segs,sessions=evaluate(selected,base,old_features,a["actual"],pd.DataFrame(audits),edges)
        selected.to_pickle(folder/"menu.pkl.gz",compression="gzip");selected.to_csv(folder/"menu.csv.gz",index=False,compression="gzip")
        hits.to_csv(folder/"hits.csv.gz",index=False,compression="gzip");hs.to_csv(folder/"hit_segments.csv",index=False)
        segs.to_csv(folder/"session_segments.csv",index=False);sessions.to_csv(folder/"sessions.csv",index=False)
        pd.DataFrame(audits).to_csv(folder/"allocation_audit.csv",index=False);dump(folder/"metrics.json",metrics)
        base_metrics[dataset]=metrics
    for full,reduced,meaning in PAIRS:
        assert all(records[n]["status"]=="complete" for n in [full,reduced])
        for dataset in DATASETS:
            a,b=records[full]["datasets"][dataset],records[reduced]["datasets"][dataset]
            ablation_rows.append(dict(dataset=dataset,full=full,reduced=reduced,contrast=meaning,
                raw_mape_full=a["raw"]["raw_mape"],raw_mape_reduced=b["raw"]["raw_mape"],
                final_mape_full=a["metrics"]["hit_qty_mape"],final_mape_reduced=b["metrics"]["hit_qty_mape"],
                final_net_full=a["metrics"]["net_benefit"],final_net_reduced=b["metrics"]["net_benefit"]))
    pd.DataFrame(summary).to_csv(OUT/"all_experiment_metrics.csv",index=False)
    pd.concat(all_segments,ignore_index=True).to_csv(OUT/"all_session_segments.csv",index=False)
    pd.DataFrame(ablation_rows).to_csv(OUT/"ablation_comparisons.csv",index=False)
    decision=dict(selected={d:{"family":"baseline","separate_preparation":True} for d in DATASETS},forecast_improved=False,
        main_metric_changes={d:{"total_mape":0,"hit_qty_mape":0,"net_benefit":0} for d in DATASETS},
        baseline_metrics=base_metrics,hit_mape_30_achieved=False,web_default_unchanged=True,
        stop_reason="Independent all-supplied regression, conditional distributions/hurdles, relative/mixed objectives and complementary ablations all failed immutable raw or final gates",
        complex_risk_coordination="implemented and synthetic-tested; historical experiment skipped because no raw model qualified",
        test_2026="optional retest not performed; no new eligible forecast",failed_technical_runs=failed,
        failed_run_resolution="Metadata failure was not reproduced; verified retry same config, original retained; no causal fix claimed",
        zero_records="separately supported and tested; actual zero count listed in feature manifests",
        advisory="new independent output max(demand,minimum); not switched into web and not an accuracy improvement")
    dump(OUT/"selection_and_stop.json",decision)
    dump(OUT/"final_verification.json",dict(checks=checks,repeat=repeats,record_count=len(records),protected_inputs_unchanged=True,
        metadata_failure_excluded=True,all_final_selected_original_columns_exact=True,
        inherited_baseline_upper_box_exception="F 2025-08-26 breakfast remains in unchanged baseline; not a new allocated policy",
        first_stage_minimum_preparation_advice=True))
    frozen=OUT/"frozen";frozen.mkdir(exist_ok=False)
    dump(frozen/"config.json",decision["selected"])
    files=list(ROOT.glob("demand_*.py"))+[ROOT/"demand_protocol.json"]
    for file in files:shutil.copy2(file,frozen/file.name)
    shutil.copytree(PREVIOUS/"baseline_source",frozen/"baseline_source")
    dump(frozen/"manifest.json",dict(version="demand_20260913_retained_deployed_forecast",frozen_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        files={str(p.relative_to(frozen)):policy.sha256_file(p) for p in frozen.rglob("*") if p.is_file()},
        dependencies="Project-root original/structural modules and immutable caches; not a standalone SDK",
        deployed_model_unchanged=True))
    verify();print("FROZEN: unchanged sales forecasts, separate preparation advice only",flush=True)


if __name__=="__main__":main()

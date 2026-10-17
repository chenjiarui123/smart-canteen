"""Post-freeze derived session/interval exports; no forecasts or selection."""
import json
import numpy as np
import pandas as pd
import policy
from demand_data import ROOT,OUT,verify
from demand_evaluate import auc
from targeted_search import dump


def interval_metrics(g):
    observed=g.loc[g.actual_qty.notna()]
    if observed.empty:return {"observed_items":0}
    y=observed.actual_qty
    low=y.le(observed.low_threshold)
    width=observed.upper80-observed.lower80
    score=width+10*(observed.lower80-y).clip(lower=0)+10*(y-observed.upper80).clip(lower=0)
    return dict(observed_items=len(observed),low_events=int(low.sum()),
        coverage=float((y.ge(observed.lower80)&y.le(observed.upper80)).mean()),
        width_mean=float(width.mean()),width_over_actual=float(width.sum()/y.sum()) if y.sum()>0 else None,
        interval_score_mean=float(score.mean()),low_auc=auc(low,observed.low_risk_probability),
        low_brier=float(((observed.low_risk_probability-low)**2).mean()),
        final_point_outside_raw_interval=float((observed.final_prediction.lt(observed.lower80)|observed.final_prediction.gt(observed.upper80)).mean()))


def main():
    verify();interval_rows=[];session_files=[]
    for record in (OUT/"experiments").glob("*/record.json"):
        r=json.loads(record.read_text())
        if r["status"]!="complete":continue
        for dataset,d in r["datasets"].items():
            if d["status"]!="complete":continue
            folder=record.parent/dataset
            detail=pd.read_pickle(folder/"pre_post_detail.pkl.gz")
            sessions=pd.read_csv(folder/"sessions.csv",parse_dates=["date"])
            additional=[]
            for key,g in detail.groupby(policy.GROUP_KEYS,sort=False):
                h=g.loc[g.actual_qty>0]
                additional.append(dict(zip(policy.GROUP_KEYS,key))|dict(
                    recommended_count=len(g),old_uncoordinated_total=float(g.old_date_conditioned.sum()),
                    independent_uncoordinated_total=float(g.raw_prediction.sum()),final_total=float(g.final_prediction.sum()),
                    suggested_preparation_total=float(g.preparation_prediction.sum()),
                    positive_hit_count=len(h),hit_rate=float(g.actual_qty.notna().mean()),
                    raw_hit_mape=float(((h.raw_prediction-h.actual_qty).abs()/h.actual_qty).mean()*100) if len(h) else None,
                    final_hit_mape=float(((h.final_prediction-h.actual_qty).abs()/h.actual_qty).mean()*100) if len(h) else None,
                    **interval_metrics(g)))
            out=sessions.merge(pd.DataFrame(additional),on=policy.GROUP_KEYS,validate="one_to_one")
            np.testing.assert_allclose(out.final_total,out.baseline_total,rtol=1e-12,atol=1e-8)
            out["raw_total_error"]=out.independent_uncoordinated_total-out.actual_total
            out.to_csv(folder/"pre_post_sessions.csv",index=False)
            session_files.append(str((folder/"pre_post_sessions.csv").relative_to(ROOT)))
            for dimension in ["month","餐次","date_type","frequency_group","sample_group","volume_segment"]:
                for name,g in detail.groupby(dimension,observed=True):
                    interval_rows.append(dict(dataset=dataset,trial=r["name"],dimension=dimension,segment=str(name),**interval_metrics(g)))
    pd.DataFrame(interval_rows).to_csv(OUT/"interval_risk_segments.csv",index=False)
    frozen=OUT/"frozen";m=json.loads((frozen/"manifest.json").read_text())
    assert all(policy.sha256_file(frozen/n)==sha for n,sha in m["files"].items())
    files=list(ROOT.glob("demand_*.py"))+list(ROOT.glob("tests/test_demand*.py"))+[
        ROOT/"INDEPENDENT_DEMAND.md",ROOT/"demand_protocol.json",OUT/"独立单菜需求模型优化报告.md",OUT/"final_verification.json",
        OUT/"selection_and_stop.json",OUT/"test_verification.json",OUT/"all_experiment_metrics.csv",OUT/"ablation_comparisons.csv",OUT/"interval_risk_segments.csv",frozen/"manifest.json"]
    dump(OUT/"delivery_manifest.json",dict(status="complete_with_no_accepted_forecast_improvement",frozen_source_hashes_verified=True,
        data_library_evaluator_protocol_unchanged=True,full_pre_post_session_exports=session_files,
        files={str(p.relative_to(ROOT)):policy.sha256_file(p) for p in files},
        interval_interpretation="Raw conditional intervals; post-coordination point may be outside them. Not recalibrated joint intervals."))
    verify();print("Delivery complete:",len(session_files),"pre/post session files;",len(interval_rows),"interval/risk segments",flush=True)


if __name__=="__main__":main()

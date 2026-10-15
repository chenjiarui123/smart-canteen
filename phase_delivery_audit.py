"""Post-development evidence audit; never modifies strategies or raw data."""
import json
import numpy as np
import pandas as pd
import policy
from phase_data import OUT,DATASETS,verify
from evaluator import prepare_actual
from diagnostics import hit_quantity_diagnostics
from targeted_search import dump
from phase_features import KEYS


def run():
    verify();integrity=[];chronological=[];intervals=[];count_changes=[]
    for ds in DATASETS:
        folder=OUT/("development_"+ds);baseline=pd.read_pickle(folder/"baseline_menu.pkl");actual=pd.read_pickle(folder/"actual.pkl")
        base_features=pd.read_pickle(folder/"base_features.pkl").set_index(KEYS)
        cut="2025-10-31" if ds=="a_hupan" else "2025-09-30"
        for name,path in [("baseline",folder/"baseline_menu.pkl")]+[(p.parent.parent.name,p) for p in (OUT/"experiments").glob("*/"+ds+"/menu.pkl.gz")]:
            menu=pd.read_pickle(path)
            for label,mask in [("early_middle_selection",actual.date.le(cut)),("late_stability",actual.date.gt(cut))]:
                a=actual.loc[mask];p=menu.loc[menu.date.isin(a.date.unique())]
                metrics=policy.evaluate_policy_metrics(p,a,policy.BASE_CONSTRAINTS);metrics.update(hit_quantity_diagnostics(p,a)[0])
                chronological.append(dict(dataset=ds,trial=name,scope=label,**metrics))
            if name=="baseline":continue
            r=json.loads((path.parent.parent/"record.json").read_text())
            if r["kind"]=="fixed_menu_item":
                keep=[c for c in baseline if c not in {"pred_qty","pred_orders","pred_rev"}]
                pd.testing.assert_frame_equal(menu[keep],baseline[keep],check_exact=True)
                idx=pd.MultiIndex.from_frame(menu[policy.GROUP_KEYS].assign(_dish_norm=menu["菜品名称"].map(policy.normalize_name)))
                units=base_features.loc[idx].sale_unit.to_numpy();b=baseline.assign(sale_unit=units);m=menu.assign(sale_unit=units)
                keys=policy.GROUP_KEYS+["sale_unit"]
                error=(b.groupby(keys).pred_qty.sum()-m.groupby(keys).pred_qty.sum()).abs().max()
                assert error<1e-8
                detail=pd.read_pickle(path.parent/"pre_post_detail.pkl.gz")
                assert (detail.model_fit_max_date.isna()|detail.model_fit_max_date.lt(detail.date)).all()
                np.testing.assert_allclose(menu.pred_orders,menu.pred_qty*menu.expected_orders_per_qty,rtol=1e-12,atol=1e-8)
                np.testing.assert_allclose(menu.pred_rev,menu.pred_qty*menu.expected_rev_per_qty,rtol=1e-12,atol=1e-8)
                integrity.append(dict(dataset=ds,trial=name,menu_metadata_identical=True,max_unit_error=float(error),max_fit_date=str(detail.model_fit_max_date.max()),test_sales_used=False))
                distribution=np.load(path.parent/"distribution.npz")
                lo,hi=np.quantile(distribution["reference"],[.1,.9],axis=1);valid=detail.actual_qty.notna().to_numpy()
                width=detail.upper80-detail.lower80;reference_width=hi-lo
                intervals.append(dict(dataset=ds,trial=name,width_ratio_to_unconditional_reference=float(width.loc[valid].sum()/max(reference_width[valid].sum(),1e-8)),
                    reference_coverage=float(((detail.actual_qty.to_numpy()[valid]>=lo[valid])&(detail.actual_qty.to_numpy()[valid]<=hi[valid])).mean())))
            elif (path.parent/"session_forecasts.csv").exists():
                sessions=pd.read_csv(path.parent/"sessions.csv");old=pd.read_csv(folder/"full_sessions.csv")
                for dim in ["date_type","phase_family"]:
                    joined=sessions.merge(old[[*policy.GROUP_KEYS,"pred_total","pred_count"]],on=policy.GROUP_KEYS,suffixes=("","_baseline"))
                    for group,g in joined.groupby(dim):
                        count_changes.append(dict(dataset=ds,trial=name,dimension=dim,segment=group,sessions=len(g),baseline_qty=float(g.pred_total_baseline.sum()),pred_qty=float(g.pred_total.sum()),
                            qty_change_percent=float(100*(g.pred_total.sum()/max(g.pred_total_baseline.sum(),1e-8)-1)),baseline_count=float(g.pred_count_baseline.mean()),pred_count=float(g.pred_count.mean()),
                            count_change_percent=float(100*(g.pred_count.mean()/max(g.pred_count_baseline.mean(),1e-8)-1)),actual_qty=float(g.actual_total.sum()),actual_count=float(g.actual_count.mean()),
                            overprediction_qty=float(g.overprediction_qty.sum()),underprediction_qty=float(g.underprediction_qty.sum())))
    pd.DataFrame(chronological).to_csv(OUT/"chronological_selection_stability.csv",index=False)
    pd.DataFrame(intervals).to_csv(OUT/"interval_reference_comparison.csv",index=False)
    pd.DataFrame(count_changes).to_csv(OUT/"holiday_count_volume_changes.csv",index=False)
    dump(OUT/"fixed_menu_integrity_audit.json",dict(status="passed",experiments=integrity,protected_data_and_evaluator_unchanged=True))
    print("Delivery audit passed",len(integrity),"fixed-menu dataset trials",flush=True)


if __name__=="__main__":run()

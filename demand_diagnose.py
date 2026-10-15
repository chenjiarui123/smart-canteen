"""Historical stage and risk diagnostics only; never a prediction branch."""
import json
import numpy as np
import pandas as pd
import policy
from demand_data import OUT,PREVIOUS,DATASETS,assets,verify
from demand_evaluate import auc,KEYS
from structural_evaluate import hit_stats
from evaluator import prepare_actual
from targeted_search import dump


def main():
    verify();stage_rows=[];shift_rows=[];risk_rows=[]
    for dataset in DATASETS:
        a=assets(dataset)
        f=pd.read_pickle(OUT/dataset/"prediction_features.pkl.gz")
        training=pd.read_pickle(OUT/dataset/"supplied_training.pkl.gz")
        h=f.merge(prepare_actual(a["actual"])[KEYS+["actual_qty"]],on=KEYS,how="inner")
        h=h.loc[h.actual_qty>0]
        trace=pd.read_pickle(PREVIOUS/("development_"+dataset)/"baseline_stages.pkl.gz")
        h=h.merge(trace[KEYS+["initial_history","date_conditioned","total_allocated","minimum_processed","repeat_final"]],on=KEYS,validate="one_to_one")
        h["volume_segment"]=np.where(h.actual_qty<=h.extreme_threshold,"极低记录量",np.where(h.actual_qty<=h.low_threshold,"低记录量","正常记录量"))
        h["sample_group"]=np.where(h.sample_count>=10,"至少10次历史","不足10次历史")
        h["frequency_group"]=np.where(h.offer_frequency>=.5,"高频","低频")
        for before,after in [("initial_history","date_conditioned"),("date_conditioned","total_allocated"),("total_allocated","minimum_processed"),("minimum_processed","repeat_final")]:
            common=h.loc[h[before].notna()&h[after].notna()]
            for dim in ["overall","volume_segment","sample_group","frequency_group","date_type","餐次"]:
                for name,g in ([("全部",common)] if dim=="overall" else common.groupby(dim)):
                    old=hit_stats(g.assign(pred_qty=g[before]));new=hit_stats(g.assign(pred_qty=g[after]))
                    stage_rows.append(dict(dataset=dataset,dimension=dim,segment=str(name),before=before,after=after,same_positive_hits=len(g),
                        old_mape=old["hit_qty_mape"],new_mape=new["hit_qty_mape"],old_wmape=old["hit_qty_wmape"],new_wmape=new["hit_qty_wmape"]))
        comparison=f.merge(training[KEYS+["competitor_count"]],on=KEYS,how="inner",suffixes=("_predicted_menu","_supplied_menu"))
        for meal,g in comparison.groupby("餐次"):
            shift_rows.append(dict(dataset=dataset,meal=meal,paired_supplied_hits=len(g),
                historical_training_menu_count_mean=float(g.competitor_count_supplied_menu.mean()),
                generated_target_menu_count_mean=float(g.competitor_count_predicted_menu.mean()),
                mean_absolute_count_difference=float((g.competitor_count_supplied_menu-g.competitor_count_predicted_menu).abs().mean())))
        for path in (OUT/"experiments").glob("*/"+dataset+"/pre_post_detail.pkl.gz"):
            d=pd.read_pickle(path);d=d.loc[d.actual_qty.notna()]
            low=d.actual_qty.le(d.low_threshold)
            risk_rows.append(dict(dataset=dataset,trial=path.parent.parent.name,hits=len(d),low_events=int(low.sum()),
                learned_low_auc=auc(low,d.low_risk_probability),historical_size_only_auc=auc(low,-d.historical_scale),
                learned_brier=float(((d.low_risk_probability-low)**2).mean()),past_quantile_prior_brier=float(((.1-low.astype(float))**2).mean())))
    pd.DataFrame(stage_rows).to_csv(OUT/"baseline_paired_stage_diagnostics.csv",index=False)
    pd.DataFrame(shift_rows).to_csv(OUT/"training_prediction_menu_shift.csv",index=False)
    pd.DataFrame(risk_rows).to_csv(OUT/"risk_discrimination_controls.csv",index=False)
    dump(OUT/"diagnostic_notes.json",dict(volume_definition="recorded outcome relative to past meal/unit q01 and q10; retrospective groups only",
        risk_control="historical-size-only AUC distinguishes genuine temporal risk information from easy dish-size ranking",
        supply_shift="training uses observed historical menus, prediction uses locked generated menus; paired count gap measured",
        target_quantity_used_for_prediction=False))
    print(pd.DataFrame(risk_rows).to_string(index=False),flush=True)


if __name__=="__main__":main()

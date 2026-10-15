"""Paired stage transitions: identical dishes on each side of every comparison."""
import json
import pandas as pd
import policy
from structural_data import OUT,DATASETS
from structural_evaluate import hit_stats
from targeted_search import dump


def main():
    rows=[]
    for dataset in DATASETS:
        folder=OUT/("development_"+dataset)
        frame=pd.read_pickle(folder/"stage_actual_diagnostics.pkl.gz")
        valid=frame.actual_qty.gt(0)
        for before,after in [("initial_history","date_conditioned"),("same_total_without_stability","total_allocated"),
                ("total_allocated","minimum_processed"),("minimum_processed","repeat_final")]:
            common=frame.loc[valid & frame[before].notna() & frame[after].notna()].copy()
            for scope,g in [("all_same_stage_dishes",common),("final_menu_subset",common.loc[common.repeat_final.notna()])]:
                old=hit_stats(g.assign(pred_qty=g[before]));new=hit_stats(g.assign(pred_qty=g[after]))
                rows.append({"dataset":dataset,"before":before,"after":after,"scope":scope,"matched_hits":len(g),
                    "old_mape":old["hit_qty_mape"],"new_mape":new["hit_qty_mape"],"delta_pp":new["hit_qty_mape"]-old["hit_qty_mape"]})
        selected=frame.loc[frame.repeat_final.notna() & valid].copy()
        selected["floor_ape"]=(selected.min_batch_qty-selected.actual_qty).clip(lower=0)/selected.actual_qty*100
        selected["final_ape"]=(selected.repeat_final-selected.actual_qty).abs()/selected.actual_qty*100
        selected.nlargest(50,"final_ape").to_csv(folder/"worst_final_dish_stage_cases.csv",index=False)
        dump(folder/"minimum_demand_diagnostics.json",{"unconditional_minimum_only_mape_bound":float(selected.floor_ape.mean()),
            "demand_below_operating_minimum_if_actual_known":int((selected.actual_qty<selected.min_batch_qty).sum()),
            "note":"Truth is used only for retrospective diagnosis, never prediction or rule selection by target sales."})
    pd.DataFrame(rows).to_csv(OUT/"paired_stage_changes.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":main()

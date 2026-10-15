"""Same-item, same-outcome stage diagnostics for the rice-merged baseline."""
import argparse
import numpy as np
import pandas as pd
from evaluator import prepare_actual
from structural_evaluate import hit_stats
from phase_item_experiments import load
from phase_data import OUT,DATASETS
from phase_evaluate import evaluate
from targeted_search import dump
import json

KEYS=["date","食堂名称","餐次","_dish_norm"]


def run(dataset):
    a,f,t=load(dataset);folder=OUT/("development_"+dataset)
    trace=pd.read_pickle(folder/"baseline_stages.pkl.gz")
    joined=trace.merge(prepare_actual(a["actual"])[KEYS+["actual_qty"]],on=KEYS,how="left")
    joined=joined.merge(f[KEYS+["low_threshold","extreme_threshold","historical_scale","own_drop_threshold","offer_frequency","sample_count","date_type","phase_family"]],on=KEYS,how="left",suffixes=("","_feature"))
    joined["volume_segment"]=np.where(joined.actual_qty<=joined.extreme_threshold,"extreme_low",np.where(joined.actual_qty<=joined.low_threshold,"low","normal"))
    joined["own_drop"]=joined.actual_qty/joined.historical_scale<=joined.own_drop_threshold
    joined["frequency_group"]=np.where(joined.offer_frequency>=.5,"high","low")
    joined["sample_group"]=np.where(joined.sample_count>=10,"sufficient","sparse")
    rows=[]
    stages=["initial_history","date_conditioned","total_allocated","minimum_processed","repeat_final"]
    for first,second in zip(stages[:-1],stages[1:]):
        common=joined.loc[joined.actual_qty.gt(0)&joined[first].notna()&joined[second].notna()]
        for dim in ["overall","volume_segment","own_drop","frequency_group","sample_group","date_type","餐次"]:
            for name,g in ([("all",common)] if dim=="overall" else common.groupby(dim)):
                old=hit_stats(g.assign(pred_qty=g[first]));new=hit_stats(g.assign(pred_qty=g[second]))
                rows.append(dict(before=first,after=second,dimension=dim,segment=str(name),same_items=len(g),old_mape=old["hit_qty_mape"],new_mape=new["hit_qty_mape"],old_wmape=old["hit_qty_wmape"],new_wmape=new["hit_qty_wmape"]))
    pd.DataFrame(rows).to_csv(folder/"same_item_stage_diagnosis.csv",index=False);joined.to_csv(folder/"stage_actual_detail.csv.gz",index=False)
    audit=json.loads((folder/"baseline_audit.json").read_text());metrics,sessions,segments,hits=evaluate(dataset,a["baseline_menu"],a["actual"],audit,f)
    dump(folder/"full_metrics.json",metrics);segments.to_csv(folder/"full_segments.csv",index=False);sessions.to_csv(folder/"full_sessions.csv",index=False)
    hits["ape"]=abs(hits.pred_qty-hits.actual_qty)/hits.actual_qty
    hits.nlargest(100,"ape").to_csv(folder/"baseline_failures.csv",index=False)
    # Compare observed historical menu sizes versus known recommended sizes,
    # without using this diagnostic to construct target-time features.
    diagnostic=pd.concat([t.assign(source="historical_supplied"),f.assign(source="baseline_recommended")],ignore_index=True)
    diagnostic.groupby(["source","餐次","date_type"]).agg(rows=("competitor_count","size"),competitors_mean=("competitor_count","mean"),competitor_ratio_mean=("category_count_ratio","mean")).to_csv(folder/"menu_structure_shift.csv")
    print(dataset,pd.DataFrame(rows).query("dimension=='overall'").to_string(index=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("dataset",choices=DATASETS);run(p.parse_args().dataset)

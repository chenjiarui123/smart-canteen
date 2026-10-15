"""Fresh current-data rolling baseline; no pre-merge prediction cache."""
import argparse
import gzip
import json
import pickle
import shutil
import time
import pandas as pd
import policy
from phase_data import ROOT, OUT, DATA, DATASETS, read_development, verify
from allocation_prepare import load_baseline
from allocation_features import OfferedHistory
from structural_trace import BaselineTrace
from targeted_search import dump, date_type, evaluate


def prepare(dataset):
    verify()
    folder=OUT/("development_"+dataset);folder.mkdir(exist_ok=False)
    cache=folder/"contexts";cache.mkdir()
    config=json.loads((OUT/"effective_baseline_config.json").read_text())
    module,_,_=load_baseline();tracer=BaselineTrace(module)
    sales=read_development(dataset)
    warmup,actual,first=policy.training_backtest_frames(sales,14)
    attributes,_=policy.build_library_attributes(DATA/"datasets"/dataset/"dish_library.csv")
    for name,value in [("warmup",warmup),("actual",actual),("attributes",attributes)]:value.to_pickle(folder/(name+".pkl"))
    coverage=dict(status="running",dataset=dataset,warmup_days=14,first=str(first),locked_2026_sales_loaded=False,
                  source_hashes={n:policy.sha256_file(ROOT/n) for n in ["phase_baseline.py","allocation_features.py","structural_trace.py"]})
    dump(folder/"coverage.json",coverage)
    for n in coverage["source_hashes"]:shutil.copy2(ROOT/n,folder/n)
    offered=OfferedHistory();offered.append(warmup);history=warmup.copy();menu_history={}
    menus=[];features=[];traces=[];audits=[];schedules=[];start=time.time()
    for i,(day,today) in enumerate(actual.groupby("date",sort=True)):
        contexts=policy.build_contexts(history,today,attributes)
        assert all(c["max_history_date"]<day for c in contexts)
        # Cached inputs contain historical sales only. Target metadata is known
        # calendar/weather and meal identity, never a target dish list/outcome.
        with gzip.open(cache/(str(day.date())+".pkl.gz"),"wb") as handle:pickle.dump(contexts,handle)
        menu,audit,trace=tracer.generate(contexts,config,menu_history)
        if i==0:
            direct=module.make_recommendations(contexts,config,policy.BASE_CONSTRAINTS,initial_menu_history=menu_history)
            pd.testing.assert_frame_equal(menu,direct,check_exact=True)
        for c in contexts:
            selected=menu.loc[menu["餐次"].eq(c["餐次"])]
            if not selected.empty:
                f=offered.features(selected,c["target_context"])
                f["date_type"]=date_type(dict(c["target_context"],date=day));features.append(f)
                menu_history.setdefault((c["食堂名称"],c["餐次"]),[]).append((day,set(selected["菜品名称"].map(policy.normalize_name))))
        menus.append(menu);traces.append(trace);audits.extend(audit);schedules.append(policy.context_schedule_frame(contexts))
        offered.append(today);history=pd.concat([history,today],ignore_index=True)
        if i%20==0:print(dataset,"fresh baseline",i+1,"/",actual.date.nunique(),flush=True)
    menu=pd.concat(menus,ignore_index=True);trace=pd.concat(traces,ignore_index=True);f=pd.concat(features,ignore_index=True)
    trace_columns=policy.GROUP_KEYS+["_dish_norm","initial_history","date_conditioned","stability_weight","qty_stability","adaptive_recent_weight","min_batch_qty"]
    f=f.merge(trace[trace_columns],on=policy.GROUP_KEYS+["_dish_norm"],validate="one_to_one")
    for name,value in [("baseline_menu",menu),("base_features",f)]:value.to_pickle(folder/(name+".pkl"))
    menu.to_csv(folder/"baseline_menu.csv",index=False)
    trace.to_pickle(folder/"baseline_stages.pkl.gz",compression="gzip");trace.to_csv(folder/"baseline_stages.csv.gz",index=False)
    pd.concat(schedules).to_csv(folder/"schedule.csv",index=False)
    menu.groupby(policy.GROUP_KEYS).agg(final_total=("pred_qty","sum"),final_count=("菜品名称","size")).to_csv(folder/"locked_final_totals.csv")
    dump(folder/"baseline_audit.json",audits)
    metrics,sessions,segments=evaluate(menu,actual,audits)
    dump(folder/"metrics.json",metrics);sessions.to_csv(folder/"sessions.csv",index=False);pd.DataFrame(segments).to_csv(folder/"segments.csv",index=False)
    coverage.update(status="complete",seconds=time.time()-start,trace_first_day_exactly_equal=True,menu_rows=len(menu),
                    sha256={p.name:policy.sha256_file(p) for p in folder.glob("*.pkl*")},context_sha256={p.name:policy.sha256_file(p) for p in cache.iterdir()})
    dump(folder/"coverage.json",coverage);verify()
    print(dataset,"READY",metrics,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("dataset",choices=DATASETS);prepare(p.parse_args().dataset)

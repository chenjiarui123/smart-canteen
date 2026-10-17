"""Reproduce current cleaned development baseline, locking final totals and menu."""
import argparse
import json
import shutil
import pandas as pd
import policy
from structural_data import ROOT,OUT,WEB,DATASETS,sales_subset,verify_protected
from structural_trace import BaselineTrace
from allocation_prepare import load_baseline
from allocation_features import OfferedHistory
from targeted_search import dump,date_type


def prior_development_menu(dataset):
    path=ROOT/"outputs/cleaned_retest_20260912"/dataset/"recommended_menu.csv"
    if not path.exists():
        return None
    dates=pd.to_datetime(pd.read_csv(path,usecols=["date"])["date"])
    excluded=set(dates.index[dates>pd.Timestamp("2025-12-31")]+1)
    return pd.read_csv(path,skiprows=lambda i:i in excluded,parse_dates=["date"])


def prepare(dataset):
    verify_protected()
    folder=OUT/("development_"+dataset)
    folder.mkdir(exist_ok=False)
    config=json.loads((OUT/"effective_baseline_config.json").read_text())
    module,_,_=load_baseline()
    tracer=BaselineTrace(module)
    sales=sales_subset(dataset)
    warmup,actual,first=policy.training_backtest_frames(sales,14)
    attributes,_=policy.build_library_attributes(WEB/"datasets"/dataset/"dish_library.csv")
    warmup.to_pickle(folder/"warmup.pkl")
    actual.to_pickle(folder/"actual.pkl")
    attributes.to_pickle(folder/"attributes.pkl")
    coverage={"status":"running","dataset":dataset,"data_start":str(sales.date.min().date()),"development_start":str(first.date()),
        "development_end":"2025-12-31","warmup_days":14,"sessions":actual.groupby(policy.GROUP_KEYS).ngroups,
        "date_type_sessions":actual.assign(date_type=actual.apply(date_type,axis=1)).drop_duplicates(policy.GROUP_KEYS).groupby("date_type").size().to_dict(),
        "locked_2026_sales_loaded":False,"runtime_history_end_inactivity":True}
    dump(folder/"coverage.json",coverage)
    offered=OfferedHistory()
    offered.append(warmup)
    history=warmup.copy()
    menu_history={}
    menus,features,trace_rows,audits,schedules=[],[],[],[],[]
    for i,(day,today) in enumerate(actual.groupby("date",sort=True)):
        contexts=policy.build_contexts(history,today,attributes)
        assert all(c["max_history_date"]<day for c in contexts)
        menu,audit,trace=tracer.generate(contexts,config,menu_history)
        if i==0:
            direct=module.make_recommendations(contexts,config,policy.BASE_CONSTRAINTS,initial_menu_history=menu_history)
            pd.testing.assert_frame_equal(menu,direct,check_exact=True)
        for c in contexts:
            meal=c["餐次"]
            selected=menu.loc[menu["餐次"].eq(meal)]
            if not selected.empty:
                f=offered.features(selected,c["target_context"])
                f["date_type"]=date_type(today.loc[today["餐次"].eq(meal)].iloc[0])
                features.append(f)
                menu_history.setdefault((c["食堂名称"],meal),[]).append((pd.Timestamp(day),set(selected["菜品名称"].map(policy.normalize_name))))
        menus.append(menu)
        trace_rows.append(trace)
        audits.extend(audit)
        schedules.append(policy.context_schedule_frame(contexts))
        # All three meals have completed prediction before today's observations.
        offered.append(today)
        history=pd.concat([history,today],ignore_index=True)
        if i%20==0:
            print(dataset,"baseline",i+1,"/",actual.date.nunique(),flush=True)
    menu=pd.concat(menus,ignore_index=True)
    trace=pd.concat(trace_rows,ignore_index=True)
    base_features=pd.concat(features,ignore_index=True)
    trace_columns=policy.GROUP_KEYS+["_dish_norm","initial_history","date_conditioned","stability_weight","qty_stability","adaptive_recent_weight"]
    base_features=base_features.merge(trace[trace_columns],on=policy.GROUP_KEYS+["_dish_norm"],validate="one_to_one")
    base_features["uncalibrated_qty"]=base_features.date_conditioned
    base_features["baseline_qty_stability"]=base_features.qty_stability
    base_features["baseline_recent_weight"]=base_features.adaptive_recent_weight
    menu.to_pickle(folder/"baseline_menu.pkl")
    menu.to_csv(folder/"baseline_menu.csv",index=False)
    base_features.to_pickle(folder/"base_features.pkl")
    trace.to_pickle(folder/"baseline_stages.pkl.gz",compression="gzip")
    trace.to_csv(folder/"baseline_stages.csv.gz",index=False,compression="gzip")
    pd.concat(schedules).to_csv(folder/"schedule.csv",index=False)
    totals=menu.groupby(policy.GROUP_KEYS).agg(final_total=("pred_qty","sum"),final_count=("菜品名称","size"))
    totals.to_csv(folder/"locked_final_totals.csv")
    dump(folder/"baseline_audit.json",audits)
    prior=prior_development_menu(dataset)
    if prior is not None:
        pd.testing.assert_frame_equal(menu.reset_index(drop=True),prior.reset_index(drop=True),check_exact=False,rtol=1e-12,atol=1e-8)
    coverage.update(status="complete",trace_first_day_exactly_equal=True,prior_cleaned_development_reproduced=prior is not None,
        source_hashes={n:policy.sha256_file(ROOT/n) for n in ["structural_data.py","structural_trace.py","structural_prepare.py","allocation_features.py"]},
        baseline_sha256=policy.sha256_file(folder/"baseline_menu.pkl"),trace_rows=len(trace),menu_rows=len(menu))
    for name in coverage["source_hashes"]:
        shutil.copy2(ROOT/name,folder/name)
    dump(folder/"coverage.json",coverage)
    verify_protected()
    print(dataset,"baseline ready",len(menu),len(trace),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",choices=DATASETS)
    prepare(parser.parse_args().dataset)

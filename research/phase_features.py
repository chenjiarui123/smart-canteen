"""Pre-day item and category records with own-relative decline labels."""
import argparse
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import policy
from phase_data import OUT, DATASETS, verify
from phase_calendar import PhaseCalendar
from demand_features import RecordedHistory
from targeted_search import dump

KEYS=policy.GROUP_KEYS+["_dish_norm"]
RELATIVE_FEATURES=["last_relative","recent_relative","recent_change","variation","log_age","log_n","frequency",
    "prior_meal_change","prior_meal_trend","state_relative","weather_relative","temperature","rain",
    "weekday_sin","weekday_cos","school","winter","legal","overlap","makeup","log_competitors",
    "school_start_distance","school_end_distance","legal_start_distance","legal_end_distance",
    "school_start_unknown","school_end_unknown","prior_own_drop_rate","category_count_ratio"]


def assets(dataset):
    verify();folder=OUT/("development_"+dataset)
    manifest=json.loads((folder/"coverage.json").read_text());assert manifest["status"]=="complete"
    for name,sha in manifest["sha256"].items():assert policy.sha256_file(folder/name)==sha,name
    return {n:pd.read_pickle(folder/(n+".pkl")) for n in ["baseline_menu","base_features","warmup","actual","attributes"]}


class PhaseHistory:
    def __init__(self,attributes,calendar):
        self.base=RecordedHistory(attributes);self.calendar=calendar
        self.attrs=attributes.set_index("_dish_norm").to_dict("index")
        self.ratios=defaultdict(list);self.group_ratios=defaultdict(list);self.categories=defaultdict(list)

    def features(self,menu,target,units):
        f=self.base.features(menu,target,units);day=pd.Timestamp(menu.date.iloc[0]);phase=self.calendar.features(day)
        for col,value in phase.items():f[col]=value
        f["last_relative"]=np.expm1(f.log_last)/f.historical_scale
        f["recent_relative"]=np.expm1(f.log_recent3)/f.historical_scale
        f["state_relative"]=np.expm1(f.log_state_mean)/f.historical_scale
        f["weather_relative"]=np.expm1(f.log_weather_mean)/f.historical_scale
        f["dish_category"]=[str(self.attrs.get(n,{}).get("菜品类别","unknown")) for n in f._dish_norm]
        thresholds=[];priors=[];scales=[];count_ratios=[];cat_context=[]
        for r in f.to_dict("records"):
            key=(r["餐次"],r["_dish_norm"],r["sale_unit"]);group=(r["餐次"],r["sale_unit"])
            own=np.asarray(self.ratios[key],float);pool=np.asarray(self.group_ratios[group],float)
            # Own-history ratio q10, sample-size pooled towards same-meal/unit
            # ratio q10. No target-day quantity or fixed absolute low threshold.
            parent=float(np.quantile(pool,.1)) if len(pool) else np.nan
            threshold=(len(own)*float(np.quantile(own,.1))+10*parent)/(len(own)+10) if len(own) else parent
            prior=((own<=threshold).sum()+10*float((pool<=threshold).mean() if len(pool) else .1))/(len(own)+10)
            records=self.categories[(r["餐次"],r["sale_unit"],r["dish_category"])]
            scales.append(float(np.median([x[1] for x in records])) if records else r["historical_scale"]*r["competitor_count"])
            typical_count=float(np.median([x[2] for x in records])) if records else r["competitor_count"]
            count_ratios.append(r["competitor_count"]/max(typical_count,1))
            values=np.asarray([x[1] for x in records],float);scale=max(scales[-1],1e-6)
            cat_context.append(dict(category_last_relative=float(values[-1]/scale) if len(values) else 1.,
                category_recent_relative=float(values[-3:].mean()/scale) if len(values) else 1.,
                category_variation=float(np.log1p(values.std()/max(values.mean(),1.))) if len(values) else 0.,
                category_log_n=float(np.log1p(len(values))),category_log_age=float(np.log1p((day-records[-1][0]).days)) if records else np.log1p(365)))
            thresholds.append(threshold);priors.append(prior)
        f["own_drop_threshold"]=thresholds;f["prior_own_drop_rate"]=priors
        f["category_historical_scale"]=scales;f["category_count_ratio"]=count_ratios
        f=pd.concat([f,pd.DataFrame(cat_context)],axis=1)
        return f

    def observe(self,actual,features):
        if not actual.date.eq(actual.date.iloc[0]).all():raise ValueError("one complete observation day")
        for r in features.to_dict("records"):
            ratio=float(r["target_y"])/max(r["historical_scale"],1e-6)
            if r["sample_count"]>=3:
                self.ratios[(r["餐次"],r["_dish_norm"],r["sale_unit"])].append(ratio)
                self.group_ratios[(r["餐次"],r["sale_unit"])].append(ratio)
        for key,g in features.groupby(["餐次","sale_unit","dish_category"]):
            self.categories[key].append((actual.date.iloc[0],float(g.target_y.sum()),len(g),g.phase.iloc[0]))
        self.base.append(actual)


def prepare(dataset):
    a=assets(dataset);folder=OUT/("development_"+dataset);calendar=PhaseCalendar();h=PhaseHistory(a["attributes"],calendar)
    all_actual=pd.concat([a["warmup"],a["actual"]],ignore_index=True);menus={d:g for d,g in a["baseline_menu"].groupby("date")}
    old=a["base_features"].set_index(KEYS);predictions=[];training=[];schedule=[]
    for i,(day,today) in enumerate(all_actual.groupby("date",sort=True)):
        targets={str(meal):g[policy.KNOWN_CONTEXT_COLUMNS+["_calendar_state","_temperature_band","_rain_band"]].iloc[0].to_dict() for meal,g in today.groupby("餐次")}
        selected=set();pred=menus.get(day)
        if pred is not None:
            for meal,menu in pred.groupby("餐次",sort=False):
                index=pd.MultiIndex.from_frame(menu[policy.GROUP_KEYS].assign(_dish_norm=menu["菜品名称"].map(policy.normalize_name)))
                o=old.loc[index];f=h.features(menu,targets[str(meal)],o.sale_unit.to_numpy())
                f["old_date_conditioned"]=o.date_conditioned.to_numpy();f["old_final_qty"]=menu.pred_qty.to_numpy()
                f["old_initial_history"]=o.initial_history.to_numpy();predictions.append(f)
                selected.update((str(meal),n) for n in f._dish_norm)
        observations=[]
        for meal,g in today.groupby("餐次",sort=False):
            f=h.features(g,targets[str(meal)],g["菜品售卖单位"].to_numpy())
            f["target_y"]=g["菜品总销量"].to_numpy();f["was_old_hit"]=[(str(meal),n) in selected for n in f._dish_norm]
            f["warmup"]=pred is None;f["own_ratio"]=f.target_y/f.historical_scale
            f["own_drop"]=f.own_ratio<=f.own_drop_threshold;observations.append(f)
        observed=pd.concat(observations,ignore_index=True);training.append(observed)
        schedule.append(dict(date=day,history_max_date=h.base.max_date,forecast_rows=0 if pred is None else len(pred),observed_rows=len(today),all_meals_predicted_before_observation=True))
        h.observe(today,observed)
        if i%40==0:print(dataset,"own-relative features",i+1,flush=True)
    t=pd.concat(training,ignore_index=True);p=pd.concat(predictions,ignore_index=True)
    for f in [t,p]:assert (f.history_max_date.isna()|pd.to_datetime(f.history_max_date).lt(f.date)).all()
    assert len(t)==len(all_actual)
    t.to_pickle(folder/"phase_training.pkl.gz",compression="gzip");p.to_pickle(folder/"phase_prediction.pkl.gz",compression="gzip")
    pd.DataFrame(schedule).to_csv(folder/"phase_feature_schedule.csv",index=False)
    dump(folder/"phase_feature_manifest.json",dict(training_rows=len(t),prediction_rows=len(p),zero_records=int(t.target_y.eq(0).sum()),
        hash={n:policy.sha256_file(folder/n) for n in ["phase_training.pkl.gz","phase_prediction.pkl.gz"]},
        source=policy.sha256_file(__import__("pathlib").Path(__file__)),threshold="past own-ratio q10, shrunk with 10 same-meal/unit observations; no history means unknown threshold and no drop label",
        target_menu="baseline recommended",historical_menu="completed supplied observations",relative_features=RELATIVE_FEATURES))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("dataset",choices=DATASETS);prepare(p.parse_args().dataset)

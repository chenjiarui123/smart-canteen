"""Past-only features for every supplied record and the locked target menu.

Training rows are constructed in the completed-day observation phase, using
the same pre-day sales history. Historical menu counts are deliberately kept
separate from target counts (which always come from the generated menu).
"""
from collections import defaultdict
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import policy
from demand_data import OUT,PREVIOUS,assets,verify
from targeted_search import dump,date_type


NUMERIC=["log_mean","log_median","log_last","log_last2","log_recent3","log_recent7","log_state_mean","log_weather_mean",
         "recent_change","variation","log_age","log_n","frequency","state_fraction","weather_fraction",
         "prior_meal_change","prior_meal_trend","temperature","rain","weekday_sin","weekday_cos","month_sin","month_cos",
         "school","winter","legal","overlap","makeup","school_elapsed","since_school_end","no_history"]
FEATURES=NUMERIC+["category_"+str(i) for i in range(8)]+["method_"+str(i) for i in range(6)]+["log_competitors"]


class RecordedHistory:
    def __init__(self,attributes):
        self.attributes=attributes.set_index("_dish_norm").to_dict("index")
        self.items=defaultdict(list);self.groups=defaultdict(list);self.meals=defaultdict(list)
        self.school_events={};self.max_date=None;self.cache={}

    def append(self,actual):
        if actual.empty:return
        assert actual.date.nunique()==1
        date=pd.Timestamp(actual.date.iloc[0])
        if self.max_date is not None and self.max_date>=date:raise ValueError("Append completed days once, in date order")
        for r in actual.to_dict("records"):
            key=(str(r["餐次"]),str(r["_dish_norm"]),str(r["菜品售卖单位"]))
            self.items[key].append((date,float(r["菜品总销量"]),str(r["_calendar_state"]),str(r["_temperature_band"]),str(r["_rain_band"])))
            self.groups[(key[0],key[2])].append(float(r["菜品总销量"]))
        for (meal,unit),g in actual.groupby(["餐次","菜品售卖单位"]):
            self.meals[(str(meal),str(unit))].append((date,float(g["菜品总销量"].sum())))
        for meal,g in actual.groupby("餐次"):
            flag=policy._flag(g["是否校历寒暑假"].iloc[0])
            old=self.school_events.get(str(meal))
            if old is None:self.school_events[str(meal)]=(flag,date,None)
            elif flag!=old[0]:self.school_events[str(meal)]=(flag,date,date if not flag else old[2])
        self.max_date=date;self.cache={}

    def group(self,meal,unit):
        key=(meal,unit)
        if key not in self.cache:
            q=np.array(self.groups[key],float);positive=q[q>0]
            v=np.quantile(positive,[.01,.1,.5,.99]) if len(positive) else np.array([0.,0.,1.,1.])
            self.cache[key]=(q,v)
        return self.cache[key]

    def features(self,menu,target,units):
        date=pd.Timestamp(menu.date.iloc[0])
        if self.max_date is not None and self.max_date>=date:raise ValueError("Target sales cannot enter history")
        categories=[self.attributes.get(policy.normalize_name(n),{}).get("菜品类别","unknown") for n in menu["菜品名称"]]
        counts=defaultdict(int)
        for u,c in zip(units,categories):counts[(str(u),str(c))]+=1
        rows=[]
        for (_,r),unit,category in zip(menu.iterrows(),units,categories):
            name=policy.normalize_name(r["菜品名称"]);meal=str(r["餐次"]);unit=str(unit)
            records=self.items[(meal,name,unit)]
            q=np.array([v[1] for v in records],float)
            group,group_q=self.group(meal,unit)
            positive=q[q>0]
            median=float(np.median(positive)) if len(positive) else float(group_q[2])
            mean=float(q.mean()) if len(q) else median
            state=str(target.get("_calendar_state",""));temp=str(target.get("_temperature_band",""));rain=str(target.get("_rain_band",""))
            same=np.array([v[1] for v in records if v[2]==state],float)
            weather=np.array([v[1] for v in records if v[3]==temp and v[4]==rain],float)
            meal_history=self.meals[(meal,unit)]
            volumes=np.array([v[1] for v in meal_history],float)
            recent=volumes[-3:].mean() if len(volumes) else 1.
            longer=volumes[-28:].mean() if len(volumes) else 1.
            school=policy._flag(target.get("是否校历寒暑假",False));legal=policy._flag(target.get("是否法定节假日假期",False))
            event=self.school_events.get(meal)
            elapsed=(date-event[1]).days if event and event[0]==school else 0
            since_end=(date-event[2]).days if event and event[2] is not None else 365
            x=dict(log_mean=np.log1p(mean),log_median=np.log1p(median),log_last=np.log1p(q[-1] if len(q) else median),
                log_last2=np.log1p(q[-2] if len(q)>1 else mean),log_recent3=np.log1p(q[-3:].mean() if len(q) else mean),
                log_recent7=np.log1p(q[-7:].mean() if len(q) else mean),
                log_state_mean=np.log1p((same.sum()+8*mean)/(len(same)+8)),
                log_weather_mean=np.log1p((weather.sum()+12*mean)/(len(weather)+12)),
                recent_change=np.log1p(q[-3:].mean() if len(q) else mean)-np.log1p(mean),
                variation=np.log1p(q.std()/max(mean,1.)) if len(q) else 0.,log_age=np.log1p((date-records[-1][0]).days if records else 365),
                log_n=np.log1p(len(q)),frequency=len(records)/max(len(meal_history),1),state_fraction=len(same)/max(len(q),1),
                weather_fraction=len(weather)/max(len(q),1),prior_meal_change=np.log1p(volumes[-1] if len(volumes) else 1)-np.log1p(longer),
                prior_meal_trend=np.log1p(recent)-np.log1p(longer),temperature=float(target.get("气温（℃）",15))/20,
                rain=np.log1p(max(float(target.get("降水量（毫米）",0)),0)),weekday_sin=np.sin(date.weekday()*2*np.pi/7),weekday_cos=np.cos(date.weekday()*2*np.pi/7),
                month_sin=np.sin(date.month*2*np.pi/12),month_cos=np.cos(date.month*2*np.pi/12),school=float(school),winter=float("寒" in str(target.get("校历假期类型",""))),
                legal=float(legal),overlap=float(school and legal),makeup=float(policy._flag(target.get("是否调休工作日",False))),
                school_elapsed=np.log1p(elapsed) if school else 0.,since_school_end=np.log1p(since_end),no_history=float(not records),log_competitors=np.log1p(counts[(unit,str(category))]))
            attrs=self.attributes.get(name,{})
            for field,value,size in [("category",str(category),8),("method",str(attrs.get("制作方法","unknown")),6)]:
                b=int.from_bytes(hashlib.sha256(value.encode()).digest()[:4],"little")%size
                for j in range(size):x[field+"_"+str(j)]=float(j==b)
            rows.append(dict(date=date,食堂名称=r["食堂名称"],餐次=meal,_dish_norm=name,菜品名称=r["菜品名称"],sale_unit=unit,
                history_max_date=self.max_date,low_threshold=float(group_q[1]),extreme_threshold=float(group_q[0]),
                historical_scale=max(median,1e-6),q99=float(np.quantile(positive,.99)) if len(positive) else float(group_q[3]),
                sample_count=len(q),offer_frequency=x["frequency"],date_type=date_type(dict(target,date=date)),
                competitor_count=counts[(unit,str(category))],**x))
        return pd.DataFrame(rows)


def prepare(dataset):
    verify();a=assets(dataset)
    folder=OUT/dataset;folder.mkdir(exist_ok=False)
    history=RecordedHistory(a["attributes"])
    all_actual=pd.concat([a["warmup"],a["actual"]],ignore_index=True)
    pred_groups={d:g for d,g in a["baseline_menu"].groupby("date")}
    units=a["features"].set_index(policy.GROUP_KEYS+["_dish_norm"]).sale_unit
    trace=pd.read_pickle(PREVIOUS/("development_"+dataset)/"baseline_stages.pkl.gz").set_index(policy.GROUP_KEYS+["_dish_norm"])
    old_estimates=trace.date_conditioned
    records=[];predictions=[];schedule=[]
    for i,(date,day) in enumerate(all_actual.groupby("date",sort=True)):
        today=pred_groups.get(date)
        day_context={str(meal):g[policy.KNOWN_CONTEXT_COLUMNS+["_calendar_state","_temperature_band","_rain_band"]].iloc[0].to_dict() for meal,g in day.groupby("餐次")}
        selected=set()
        if today is not None:
            for key,menu in today.groupby(policy.GROUP_KEYS,sort=False):
                idx=pd.MultiIndex.from_frame(menu[policy.GROUP_KEYS].assign(_dish_norm=menu["菜品名称"].map(policy.normalize_name)))
                f=history.features(menu,day_context[str(key[2])],units.loc[idx].to_numpy())
                f["old_date_conditioned"]=old_estimates.loc[pd.MultiIndex.from_frame(f[policy.GROUP_KEYS+["_dish_norm"]])].to_numpy()
                f["old_final_qty"]=menu.pred_qty.to_numpy()
                predictions.append(f)
                selected.update((str(key[2]),n) for n in f._dish_norm)
        # Observation phase only: historical supplied-menu counts become known
        # after all current predictions. No current target quantity in features.
        for meal,g in day.groupby("餐次",sort=False):
            f=history.features(g,day_context[str(meal)],g["菜品售卖单位"].to_numpy())
            idx=pd.MultiIndex.from_frame(f[policy.GROUP_KEYS+["_dish_norm"]])
            f["old_date_conditioned"]=old_estimates.reindex(idx).to_numpy()
            f["target_y"]=g["菜品总销量"].to_numpy()
            f["was_old_hit"]=[(str(meal),n) in selected for n in f._dish_norm]
            f["warmup"]=today is None
            records.append(f)
        schedule.append(dict(date=date,history_max_date=history.max_date,forecast_rows=0 if today is None else len(today),observed_rows=len(day),all_predictions_before_observe=True))
        history.append(day)
        if i%40==0:print(dataset,"features",i+1,"of",all_actual.date.nunique(),flush=True)
    training=pd.concat(records,ignore_index=True);prediction=pd.concat(predictions,ignore_index=True)
    for frame in [training,prediction]:
        valid=frame.history_max_date.notna()
        assert pd.to_datetime(frame.loc[valid,"history_max_date"]).lt(frame.loc[valid,"date"]).all()
        assert frame.date.max()<=pd.Timestamp("2025-12-31")
    assert len(training)==len(all_actual)
    training.to_pickle(folder/"supplied_training.pkl.gz",compression="gzip")
    prediction.to_pickle(folder/"prediction_features.pkl.gz",compression="gzip")
    pd.DataFrame(schedule).to_csv(folder/"feature_schedule.csv",index=False)
    dump(folder/"feature_manifest.json",dict(training_rows=len(training),prediction_rows=len(prediction),zero_record_count=int(training.target_y.eq(0).sum()),
        predicted_menu_source="locked baseline",training_menu_source="completed supplied records, never used for current predictions",
        history_features="same pre-day snapshot for predictions and subsequent training observation",features=FEATURES,
        sha256={n:policy.sha256_file(folder/n) for n in ["supplied_training.pkl.gz","prediction_features.pkl.gz"]},
        source_sha256=policy.sha256_file(__import__("pathlib").Path(__file__))))
    verify()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("dataset");prepare(p.parse_args().dataset)

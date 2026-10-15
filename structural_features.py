"""Past-menu conditional shares and competition, within unchanged unit groups.

Observed historical menus may train features; target actual menus may not.
Missing historical records are never fabricated as zero demand.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import policy
from structural_data import OUT,DATASETS,verify_protected
from targeted_search import dump


class StructuralHistory:
    def __init__(self,attributes):
        self.attributes=attributes.set_index("_dish_norm").to_dict("index")
        self.items=defaultdict(list)
        self.categories=defaultdict(list)
        self.max_date=None

    def append(self,actual):
        if actual.empty:
            return
        if self.max_date is not None and actual.date.min()<=self.max_date:
            raise ValueError("Append strictly later completed days only")
        frame=actual.copy()
        frame["category"]=[self.attributes.get(n,{}).get("菜品类别","unknown") for n in frame._dish_norm]
        for (date,canteen,meal,unit),g in frame.groupby(policy.GROUP_KEYS+["菜品售卖单位"],sort=False):
            total=float(g["菜品总销量"].sum())
            if total<=0:
                continue
            for category,cg in g.groupby("category",sort=False):
                category_total=float(cg["菜品总销量"].sum())
                if category_total<=0:
                    continue
                state=str(cg._calendar_state.iloc[0])
                weather=str(cg._temperature_band.iloc[0])+"|"+str(cg._rain_band.iloc[0])
                self.categories[(str(meal),str(unit),str(category))].append((pd.Timestamp(date),state,weather,category_total/total))
                competition=int(cg._dish_norm.nunique())
                for r in cg.to_dict("records"):
                    qty=float(r["菜品总销量"])
                    if qty<=0:
                        continue # Explicit zeros remain in original scoring/data.
                    self.items[(str(meal),str(r["_dish_norm"]),str(unit))].append(
                        (pd.Timestamp(date),state,weather,qty/total,qty/category_total,float(competition)))
        self.max_date=pd.Timestamp(actual.date.max())

    @staticmethod
    def conditional(values,state,weather,index,fallback):
        if not values:
            return float(fallback),0,0,0
        def geometric(records):
            return float(np.exp(np.log(np.maximum([r[index] for r in records],1e-12)).mean()))
        # Never boost ordinary days from holiday-only observations: exact state
        # is the evidence tier, with baseline as safe fallback when absent.
        exact=[r for r in values if r[1]==state]
        value=float(fallback)
        if exact:
            weight=len(exact)/(len(exact)+8.)
            value=np.exp((1-weight)*np.log(max(value,1e-12))+weight*np.log(geometric(exact)))
        joint=[r for r in exact if r[2]==weather]
        if joint:
            weight=len(joint)/(len(joint)+12.)
            value=np.exp((1-weight)*np.log(max(value,1e-12))+weight*np.log(geometric(joint)))
        return float(value),len(values),len(exact),len(joint)

    def augment(self,menu,features):
        date=pd.Timestamp(menu.date.iloc[0])
        if self.max_date is not None and self.max_date>=date:
            raise ValueError("Target actual data cannot enter structural features")
        f=features.set_index("_dish_norm").loc[menu["菜品名称"].map(policy.normalize_name)].reset_index()
        categories=menu["菜品类别"].astype(str).to_numpy()
        f["dish_category"]=categories
        units=f.sale_unit.astype(str).to_numpy()
        b=menu.pred_qty.to_numpy(float)
        unit_total={unit:float(b[units==unit].sum()) for unit in set(units)}
        category_total={(unit,category):float(b[(units==unit)&(categories==category)].sum()) for unit,category in set(zip(units,categories))}
        counts={(unit,category):int(((units==unit)&(categories==category)).sum()) for unit,category in category_total}
        rows=[]
        for i,r in enumerate(f.to_dict("records")):
            unit=str(r["sale_unit"])
            category=str(r["dish_category"])
            meal=str(r["餐次"])
            state=str(r["calendar_state"])
            weather=str(r["temperature_band"])+"|"+str(r["rain_band"])
            total=unit_total[unit]
            cat_total=category_total[(unit,category)]
            own_prior=b[i]/total if total else 0.
            within_prior=b[i]/cat_total if cat_total else 0.
            cat_prior=cat_total/total if total else 0.
            records=self.items[(meal,r["_dish_norm"],unit)]
            cat_records=self.categories[(meal,unit,category)]
            share,n,ns,nw=self.conditional(records,state,weather,3,own_prior)
            within,_,_,_=self.conditional(records,state,weather,4,within_prior)
            cat_share,cn,cs,cw=self.conditional(cat_records,state,weather,3,cat_prior)
            exact=[v for v in records if v[1]==state]
            # Competing supply is learned from completed historical menus only.
            # Ridge and clipping constrain low-information correlation; this is
            # an association model, not a claimed causal demand elasticity.
            slope=0.
            average_log_count=np.log(max(counts[(unit,category)],1))
            if len(exact)>=4:
                x=np.log([v[5] for v in exact]); y=np.log([v[3] for v in exact])
                dx=x-x.mean(); dy=y-y.mean()
                slope=float(np.clip(np.dot(dx,dy)/(np.dot(dx,dx)+10.),-1.,1.))
                slope*=len(exact)/(len(exact)+20.)
                average_log_count=float(x.mean())
            count=counts[(unit,category)]
            competition_factor=float(np.exp(slope*(np.log(count)-average_log_count)))
            rows.append({**r,"baseline_unit_total":total,"baseline_category_total":cat_total,
                "unit_share":share,"within_category_share":within,"category_share":cat_share,
                "share_n":n,"state_share_n":ns,"weather_share_n":nw,"category_share_n":cn,
                "category_state_n":cs,"category_weather_n":cw,"generated_category_count":count,
                "competition_slope":slope,"history_log_competitor_count":average_log_count,
                "competition_factor":competition_factor,"structural_history_max_date":self.max_date})
        return pd.DataFrame(rows)


def prepare(dataset):
    verify_protected()
    folder=OUT/("development_"+dataset)
    assert json.loads((folder/"coverage.json").read_text())["status"]=="complete"
    if (folder/"features.pkl").exists():
        raise FileExistsError("Do not overwrite prepared feature evidence")
    menus=pd.read_pickle(folder/"baseline_menu.pkl")
    base=pd.read_pickle(folder/"base_features.pkl")
    actual=pd.read_pickle(folder/"actual.pkl")
    history=StructuralHistory(pd.read_pickle(folder/"attributes.pkl"))
    history.append(pd.read_pickle(folder/"warmup.pkl"))
    groups={k:g for k,g in base.groupby(policy.GROUP_KEYS,sort=False)}
    outputs=[]
    mg={d:g for d,g in menus.groupby("date",sort=False)}
    for i,(day,today) in enumerate(actual.groupby("date",sort=True)):
        for key,menu in mg.get(day,menus.iloc[0:0]).groupby(policy.GROUP_KEYS,sort=False):
            outputs.append(history.augment(menu,groups[key]))
        history.append(today)
        if i%40==0:
            print(dataset,"structural features",i+1,flush=True)
    features=pd.concat(outputs,ignore_index=True)
    assert (pd.to_datetime(features.structural_history_max_date)<features.date).all()
    features.to_pickle(folder/"features.pkl")
    # Reporting thresholds are fitted once from development historical features.
    # This uses only development history, never any target actual quantity.
    dump(folder/"group_edges.json",{"volume":float(features["mean"].median()),"frequency":float(features.offer_frequency.median()),"source":"2025 historical feature values; diagnostic bins only"})
    finish_manifest(dataset)
    print(dataset,"features ready",len(features),flush=True)


def finish_manifest(dataset):
    folder=OUT/("development_"+dataset)
    features=pd.read_pickle(folder/"features.pkl")
    assert (pd.to_datetime(features.structural_history_max_date)<features.date).all()
    assert not features.duplicated(policy.GROUP_KEYS+["_dish_norm"]).any()
    assert len(features)==len(pd.read_pickle(folder/"baseline_menu.pkl"))
    dump(folder/"feature_manifest.json",{"rows":len(features),"sha256":policy.sha256_file(folder/"features.pkl"),
        "source_sha256":policy.sha256_file(Path(__file__)),"future_dates":0,"target_menu_source":"frozen baseline only","unit_totals":"separate exact original unit labels; no conversion"})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",choices=DATASETS)
    parser.add_argument("--finish-manifest",action="store_true")
    args=parser.parse_args()
    finish_manifest(args.dataset) if args.finish_manifest else prepare(args.dataset)

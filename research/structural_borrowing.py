"""Hierarchical residual-demand learning across related dishes, no neural net.

Each meal/unit group fits a ridge model from PREVIOUS predicted-positive-hit
residuals. Shared category/ingredient/method/calendar/price effects supply a prior;
dish-specific remaining residuals are strongly sample-shrunk toward that prior.
"""
from collections import defaultdict
import hashlib
import numpy as np
import pandas as pd


class PeerResidualModel:
    DIM=64
    def __init__(self):
        self.models={}
        self.dishes=defaultdict(list)
        self.max_date=None
        self.coefficient_cache={}

    @staticmethod
    def bucket(value,size):
        return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:4],"little")%size

    def design(self,menu,features):
        f=features.reset_index(drop=True)
        m=menu.reset_index(drop=True)
        n=len(f)
        x=np.zeros((n,self.DIM))
        b=np.maximum(f.baseline_qty.to_numpy(float),1e-6)
        x[:,0]=1
        x[:,1]=np.log1p(b)/5
        x[:,2]=np.log1p(f.sample_count.to_numpy(float))/5
        x[:,3]=np.log1p(f.n14.to_numpy(float))/3
        x[:,4]=np.clip(np.log(np.maximum(f.q25.to_numpy(float),1e-6)/b),-3,3)/3
        x[:,5]=np.clip(np.log(np.maximum(f["median"].to_numpy(float),1e-6)/b),-3,3)/3
        x[:,6]=np.log1p(f["std"].to_numpy(float)/np.maximum(f["mean"].to_numpy(float),1))/2
        x[:,7]=np.log1p(m.expected_rev_per_qty.to_numpy(float))/3
        x[:,8]=np.log1p(f.generated_category_count.to_numpy(float))/4
        x[:,9]=(f.n14.to_numpy(float)==0).astype(float)
        for j,r in f.iterrows():
            for value,offset,size in [(r.dish_category,10,12),(m.loc[j,"主要食材"],22,12),
                    (m.loc[j,"制作方法"],34,8),(r.calendar_state,42,12),
                    (r.temperature_band,54,6),(r.rain_band,60,4)]:
                x[j,offset+self.bucket(value,size)]=1.
        return x

    def predict(self,menu,f):
        date=pd.Timestamp(f.date.iloc[0])
        if self.max_date is not None and self.max_date>=date:
            raise ValueError("Peer model cannot train on target/future observations")
        x=self.design(menu,f)
        logs=[]
        for i,r in enumerate(f.to_dict("records")):
            key=(r["餐次"],r["sale_unit"])
            prediction=0.
            if key in self.models:
                gram,rhs,count=self.models[key]
                cache=self.coefficient_cache.get(key)
                if cache is None or cache[0]!=count:
                    # Strong prior remains effective even after one recent sale.
                    coefficient=np.linalg.solve(gram+20*np.eye(self.DIM),rhs)
                    self.coefficient_cache[key]=(count,coefficient)
                else:coefficient=cache[1]
                prediction=float(x[i]@coefficient)
            own=self.dishes[(key,r["_dish_norm"],r["calendar_state"])][-60:]
            n=len(own)
            if own:prediction+=n/(n+20.)*float(np.median(own))
            if r["calendar_state"] in {"普通工作日","普通周末","调休工作日"} and r["state_share_n"]==0:
                prediction=min(prediction,0.)
            logs.append(float(np.clip(prediction,np.log(.5),np.log(1.5))))
        return f.baseline_qty.to_numpy(float)*np.exp(logs),x

    def observe_day(self,features,design,estimates,actual):
        if features.empty:return
        date=pd.Timestamp(features.date.iloc[0])
        if not features.date.eq(date).all() or not actual.date.eq(date).all():
            raise ValueError("Peer observations must be from the just-predicted day")
        if self.max_date is not None and self.max_date>=date:
            raise ValueError("Observe each peer training day once")
        from evaluator import prepare_actual
        truth=prepare_actual(actual)
        lookup={(r["食堂名称"],r["餐次"],r["_dish_norm"]):r["actual_qty"] for r in truth.to_dict("records")}
        for i,r in enumerate(features.to_dict("records")):
            value=lookup.get((r["食堂名称"],r["餐次"],r["_dish_norm"]))
            if value is None or value<=0:continue
            key=(r["餐次"],r["sale_unit"])
            if key not in self.models:self.models[key]=(np.zeros((self.DIM,self.DIM)),np.zeros(self.DIM),0)
            gram,rhs,count=self.models[key]
            y=float(np.clip(np.log(value/max(r["baseline_qty"],1e-6)),-np.log(4),np.log(4)))
            gram+=np.outer(design[i],design[i]);rhs+=design[i]*y
            self.models[key]=(gram,rhs,count+1)
            previous_coefficient=self.coefficient_cache.get(key,(0,np.zeros(self.DIM)))[1]
            residual=float(np.clip(y-float(design[i]@previous_coefficient),-np.log(4),np.log(4)))
            self.dishes[(key,r["_dish_norm"],r["calendar_state"])].append(residual)
        self.max_date=date

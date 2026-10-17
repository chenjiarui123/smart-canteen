"""Own-relative recorded-sales models: bounded partitions and drop states.

No absolute item-size features in the decline classifier. No missing-supply
zeros. Model refits weekly, with all prior records or a registered hits ablation.
"""
import numpy as np
import pandas as pd
from phase_features import RELATIVE_FEATURES
from demand_models import GRID, decision


class Partition:
    def fit(self,x,y,depth=0):
        self.quantiles=np.quantile(y,GRID);self.feature=None
        if depth>=2 or len(y)<160:return self
        best=0;chosen=None
        for j in range(x.shape[1]):
            for cut in np.unique(np.quantile(x[:,j],[.25,.5,.75])):
                mask=x[:,j]<=cut;n=int(mask.sum());m=len(mask)-n
                if min(n,m)<80:continue
                gain=n*m/len(y)*(y[mask].mean()-y[~mask].mean())**2
                if gain>best:best=gain;chosen=j,float(cut),mask
        if chosen:
            self.feature,self.cut,mask=chosen
            self.left=Partition().fit(x[mask],y[mask],depth+1);self.right=Partition().fit(x[~mask],y[~mask],depth+1)
        return self

    def predict(self,x):
        if self.feature is None:return np.tile(self.quantiles,(len(x),1))
        mask=x[:,self.feature]<=self.cut;result=np.empty((len(x),len(GRID)))
        result[mask]=self.left.predict(x[mask]);result[~mask]=self.right.predict(x[~mask]);return result


class RelativeModel:
    def __init__(self,mode="relative",phase=True):self.mode=mode;self.phase=phase

    def design(self,f):
        cols=RELATIVE_FEATURES if self.phase else [c for c in RELATIVE_FEATURES if not c.startswith(("school_","legal_"))]
        x=f[cols].to_numpy(float)
        # Fixed feature transforms, not outcome clipping: signed distances to
        # boundaries can exceed a year and are compressed monotonically.
        for j,c in enumerate(cols):
            if c.endswith("distance"):x[:,j]=np.sign(x[:,j])*np.log1p(abs(x[:,j]))
            elif c.endswith("relative") or c=="category_count_ratio":x[:,j]=np.log1p(np.maximum(x[:,j],0))
        return np.nan_to_num(x,nan=0,posinf=20,neginf=-20)

    def fit(self,f):
        self.max_date=f.date.max();self.n=len(f);x=self.design(f)
        ratio=f.target_y.to_numpy(float)/f.historical_scale.to_numpy(float);y=np.log1p(ratio)
        self.tree=Partition().fit(x,y);self.reference=np.quantile(y,GRID)
        self.mean=x.mean(axis=0);self.std=np.maximum(x.std(axis=0),.1)
        z=np.column_stack([np.ones(len(x)),(x-self.mean)/self.std])
        label=(ratio<=f.own_drop_threshold.to_numpy(float)).astype(float)
        valid=f.own_drop_threshold.notna().to_numpy() & f.sample_count.ge(3).to_numpy()
        self.coef=np.zeros(z.shape[1]);self.components=None
        if valid.sum()>=160 and len(np.unique(label[valid]))==2:
            xv=z[valid];yv=label[valid];self.coef[0]=np.log((yv.sum()+1)/(len(yv)-yv.sum()+1))
            penalty=10*np.eye(z.shape[1]);penalty[0,0]=.01
            for _ in range(8):
                prob=1/(1+np.exp(-np.clip(xv@self.coef,-25,25)))
                self.coef-=np.linalg.solve(xv.T@(xv*np.maximum(prob*(1-prob),1e-5)[:,None])+penalty,xv.T@(prob-yv)+penalty@self.coef)
            self.components=[]
            for state in [1,0]:
                mask=valid & (label==state)
                self.components.append(Partition().fit(x[mask],y[mask]))
        return self

    def predict(self,f):
        assert self.max_date<f.date.min()
        x=self.design(f);scale=f.historical_scale.to_numpy(float)[:,None]
        samples=np.expm1(self.tree.predict(x))*scale;reference=np.expm1(self.reference)[None,:]*scale
        risk=(samples<=f.own_drop_threshold.to_numpy(float)[:,None]*scale).mean(axis=1)
        if self.mode=="drop" and self.components is not None:
            z=np.column_stack([np.ones(len(x)),(x-self.mean)/self.std]);risk=1/(1+np.exp(-np.clip(z@self.coef,-25,25)))
            low,normal=[np.expm1(t.predict(x))*scale for t in self.components]
            for i,p in enumerate(risk):
                values=np.concatenate([low[i],normal[i]]);weight=np.r_[np.full(len(GRID),p/len(GRID)),np.full(len(GRID),(1-p)/len(GRID))]
                order=np.argsort(values);samples[i]=np.interp(GRID,np.cumsum(weight[order]),values[order])
        return samples,reference,risk


def walk(f,training,config):
    samples=np.empty((len(f),len(GRID)));reference=np.empty_like(samples);risk=np.empty(len(f));fit_dates=np.empty(len(f),object);counts=np.empty(len(f),int)
    models={};fit_days={}
    for i,(day,index) in enumerate(f.groupby("date",sort=True).indices.items()):
        before=training.loc[training.date<day]
        if config.get("hits_only"):before=before.loc[before.warmup|before.was_old_hit]
        for key,local in f.iloc[index].groupby(["餐次","sale_unit"],sort=False).indices.items():
            ids=index[local];target=f.iloc[ids];t=before.loc[before["餐次"].eq(key[0]) & before.sale_unit.eq(key[1])]
            if len(t)>=160 and config.get("mode")!="typical":
                if key not in models or (day-fit_days[key]).days>=7:
                    models[key]=RelativeModel(config.get("mode","relative"),config.get("phase",True)).fit(t);fit_days[key]=day
                model=models[key];samples[ids],reference[ids],risk[ids]=model.predict(target);counts[ids]=model.n;fit_dates[ids]=model.max_date
            else:
                # Empirical own relative reference is trained from supplied
                # records, never the old policy's point prediction.
                ratio=t.target_y.to_numpy(float)/t.historical_scale.to_numpy(float)
                dist=np.quantile(ratio,GRID) if len(ratio) else np.ones(len(GRID))
                samples[ids]=target.historical_scale.to_numpy(float)[:,None]*dist;reference[ids]=samples[ids]
                risk[ids]=(dist[None,:]<=target.own_drop_threshold.to_numpy(float)[:,None]).mean(axis=1)
                counts[ids]=len(t);fit_dates[ids]=t.date.max() if len(t) else pd.NaT
        if i%60==0:print(config,"independent prediction",i+1,flush=True)
    point=f.historical_scale.to_numpy(float).copy() if config.get("mode")=="typical" else decision(samples,config.get("objective","log"),f.historical_scale.to_numpy(float))
    return point,samples,reference,risk,pd.to_datetime(fit_dates),counts


def hierarchy(f,training,point,phase=True):
    """Direct category-level recorded-volume model, not static category shares.

    Learn category response to currently known menu competitor counts. Historical
    categories/counts come from completed actual menus; prediction uses only
    baseline recommended categories/counts. Units never pooled in quantities.
    """
    def cat_features(frame):
        frame=frame.copy()
        for c in ["last_relative","recent_relative","variation","log_n","log_age"]:frame[c]=frame["category_"+c]
        frame["recent_change"]=np.log1p(frame.recent_relative)-np.log(2.)
        frame["frequency"]=1.;frame["state_relative"]=1.;frame["weather_relative"]=1.;frame["prior_own_drop_rate"]=.1
        frame["own_drop_threshold"]=np.nan
        return frame
    group=["date","餐次","sale_unit","dish_category"]
    category=training.groupby(group,sort=False).first().reset_index()
    totals=training.groupby(group,sort=False).target_y.sum().to_numpy()
    category["target_y"]=totals;category["historical_scale"]=category.category_historical_scale
    category=cat_features(category)
    result=point.copy();rows=[];models={};fit_days={}
    for day,index in f.groupby("date",sort=True).indices.items():
        before=category.loc[category.date<day]
        for key,local in f.iloc[index].groupby(["餐次","sale_unit"],sort=False).indices.items():
            ids=index[local];target=f.iloc[ids];t=before.loc[before["餐次"].eq(key[0]) & before.sale_unit.eq(key[1])]
            cats=target.groupby("dish_category",sort=False).first().reset_index();cats["historical_scale"]=cats.category_historical_scale
            cats=cat_features(cats)
            if len(t)>=160:
                if key not in models or (day-fit_days[key]).days>=7:models[key]=RelativeModel("relative",phase).fit(t);fit_days[key]=day
                dist,_,_=models[key].predict(cats);estimate=decision(dist,"log",cats.historical_scale.to_numpy(float))
            else:estimate=cats.historical_scale.to_numpy(float)
            for j,c in enumerate(cats.dish_category):
                members=ids[target.dish_category.to_numpy()==c];weights=point[members]
                result[members]=estimate[j]*weights/max(weights.sum(),1e-12)
                rows.append(dict(date=day,餐次=key[0],sale_unit=key[1],category=c,category_prediction=float(estimate[j]),recommended_competitors=len(members),
                    historical_scale=float(cats.historical_scale.iloc[j]),count_ratio=float(cats.category_count_ratio.iloc[j]),history_max_date=t.date.max()))
    return result,pd.DataFrame(rows)

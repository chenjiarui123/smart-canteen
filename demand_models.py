"""Independent recorded-sales regression and conditional distributions.

No baseline-relative multiplier/correction bounds. All training quantities are
retained, including recorded zeros; no missing supply is synthesised as zero.
"""
import numpy as np
import pandas as pd
from demand_features import FEATURES

GRID=(np.arange(41)+.5)/41


def decision(samples,objective,scale):
    samples=np.maximum(np.asarray(samples,float),0)
    if objective=="log":return np.expm1(np.log1p(samples).mean(axis=1))
    if objective=="mean":return samples.mean(axis=1)
    if objective=="absolute":return np.median(samples,axis=1)
    if objective not in {"relative","mixed"}:raise ValueError(objective)
    weights=np.divide(1.,samples,out=np.zeros_like(samples),where=samples>0)
    # Zeros contribute separately to the absolute component. They are never
    # substituted into a MAPE denominator or removed from stored observations.
    if objective=="mixed":weights=.5*weights+.5/np.maximum(np.asarray(scale)[:,None],1e-6)
    else:
        empty=weights.sum(axis=1)==0
        weights[empty]=1.
    order=np.argsort(samples,axis=1)
    y=np.take_along_axis(samples,order,axis=1);w=np.take_along_axis(weights,order,axis=1)
    index=(np.cumsum(w,axis=1)<w.sum(axis=1)[:,None]/2).sum(axis=1)
    return y[np.arange(len(y)),np.minimum(index,y.shape[1]-1)]


class ShallowPartition:
    """Depth-two least-squares residual partition, fixed grid, no search."""
    def fit(self,x,residual,depth=0):
        self.values=np.quantile(residual,GRID)
        self.mean=float(residual.mean())
        self.feature=None;self.left=None;self.right=None
        if depth>=2 or len(x)<80:return self
        best=0.;choice=None
        # Only readable numerical/context variables define the partition.
        for j in [0,2,8,9,10,11,12,13,15,16,17,18,23,24,25,26,27,28,30,len(FEATURES)-1]+list(range(len(FEATURES),x.shape[1])):
            for threshold in np.unique(np.quantile(x[:,j],[.25,.5,.75])):
                left=x[:,j]<=threshold;n=int(left.sum());m=len(left)-n
                if min(n,m)<40:continue
                gain=n*m/len(left)*(residual[left].mean()-residual[~left].mean())**2
                if gain>best:best=float(gain);choice=(j,float(threshold),left)
        if choice is not None:
            self.feature,self.threshold,left=choice
            self.left=ShallowPartition().fit(x[left],residual[left],depth+1)
            self.right=ShallowPartition().fit(x[~left],residual[~left],depth+1)
        return self

    def predict(self,x):
        if self.feature is None:return np.tile(self.values,(len(x),1))
        left=x[:,self.feature]<=self.threshold
        out=np.empty((len(x),len(GRID)))
        out[left]=self.left.predict(x[left]);out[~left]=self.right.predict(x[~left])
        return out


class IndependentGroup:
    def __init__(self,family="linear",with_old=False):
        self.family=family;self.with_old=with_old

    def raw_design(self,f):
        x=f[FEATURES].to_numpy(float)
        if self.with_old:
            old=f.old_date_conditioned.to_numpy(float)
            missing=~np.isfinite(old)
            old=np.where(missing,np.expm1(f.log_median.to_numpy(float)),old)
            x=np.column_stack([x,np.log1p(np.maximum(old,0)),missing.astype(float)])
        return np.nan_to_num(x,nan=0,posinf=10,neginf=-10)

    @staticmethod
    def ridge(x,y):
        penalty=10*np.eye(x.shape[1]);penalty[0,0]=1e-8
        return np.linalg.solve(x.T@x+penalty,x.T@y)

    def fit(self,f):
        self.fit_max_date=pd.Timestamp(f.date.max());self.training_rows=len(f)
        raw=self.raw_design(f);self.center=raw.mean(axis=0);self.spread=np.maximum(raw.std(axis=0),.1)
        x=np.column_stack([np.ones(len(raw)),(raw-self.center)/self.spread])
        y=np.log1p(f.target_y.to_numpy(float))
        self.coefficient=self.ridge(x,y)
        residual=y-x@self.coefficient
        self.unconditional=np.quantile(residual,GRID)
        self.partition=None
        if self.family in {"relative_linear","mixed_linear"}:
            ratio=f.target_y.to_numpy(float)/np.maximum(f.historical_scale.to_numpy(float),1e-6)
            objective_weights=np.divide(1.,ratio,out=np.ones_like(ratio),where=ratio>0)
            if self.family=="mixed_linear":objective_weights=.5*objective_weights+.5
            coefficient=self.ridge(x,ratio)
            for _ in range(8):
                # Smooth IRLS approximates positive-record MAPE (or equal
                # relative/scale-normalised absolute loss). Recorded zeros use
                # a separate unit-weight absolute term, never 1/0.
                weights=objective_weights/np.maximum(np.abs(ratio-x@coefficient),.01)
                weights/=max(weights.mean(),1e-12)
                coefficient=self.ridge(x*np.sqrt(weights)[:,None],ratio*np.sqrt(weights))
            self.loss_coefficient=coefficient
            errors=ratio-x@coefficient
            self.ratio_errors=np.quantile(errors,GRID)-np.median(errors)
        if self.family=="tree":self.partition=ShallowPartition().fit(raw,residual)
        if self.family in {"conditional","conditional_hurdle"}:
            # Direct distribution of recorded sales / own historical scale.
            # No old-policy anchor and no linear extrapolation across unseen
            # calendar support. Scaling is a unit-preserving target transform.
            normalized=np.log1p(f.target_y.to_numpy(float)/np.maximum(f.historical_scale.to_numpy(float),1e-6))
            self.normalized_reference=np.quantile(normalized,GRID)
            self.partition=ShallowPartition().fit(raw,normalized)
        if self.family in {"hurdle","conditional_hurdle"}:
            low=f.target_y.to_numpy(float)<=f.low_threshold.to_numpy(float)
            self.prevalence=float(low.mean());self.risk_coef=np.zeros(x.shape[1])
            self.risk_coef[0]=np.log((low.sum()+1)/(len(low)-low.sum()+1))
            penalty=10*np.eye(x.shape[1]);penalty[0,0]=.01
            for _ in range(8):
                p=1/(1+np.exp(-np.clip(x@self.risk_coef,-25,25)))
                h=x.T@(x*np.maximum(p*(1-p),1e-5)[:,None])+penalty
                gradient=x.T@(p-low)+penalty@self.risk_coef
                self.risk_coef-=np.linalg.solve(h,gradient)
            self.components=[]
            for mask in [low,~low]:
                if self.family=="conditional_hurdle":
                    selected=mask if mask.any() else np.ones(len(mask),bool)
                    self.components.append(ShallowPartition().fit(raw[selected],normalized[selected]))
                    continue
                elif mask.sum()>=40:
                    coefficient=self.ridge(x[mask],y[mask]);errors=y[mask]-x[mask]@coefficient
                else:coefficient=self.coefficient;errors=residual[mask] if mask.any() else residual
                self.components.append((coefficient,np.quantile(errors,GRID)))
        return self

    def predict(self,f):
        if self.fit_max_date>=pd.Timestamp(f.date.min()):raise ValueError("Fit uses current/future day")
        raw=self.raw_design(f);x=np.column_stack([np.ones(len(raw)),(raw-self.center)/self.spread])
        center=x@self.coefficient
        reference=np.maximum(np.expm1(np.clip(center[:,None]+self.unconditional,-20,25)),0)
        errors=self.partition.predict(raw) if self.partition is not None else np.tile(self.unconditional,(len(x),1))
        samples=np.maximum(np.expm1(np.clip(center[:,None]+errors,-20,25)),0)
        if self.family in {"relative_linear","mixed_linear"}:
            samples=np.maximum((x@self.loss_coefficient)[:,None]+self.ratio_errors,0)*f.historical_scale.to_numpy(float)[:,None]
            reference=samples.copy()
        if self.family in {"conditional","conditional_hurdle"}:
            scale=f.historical_scale.to_numpy(float)[:,None]
            samples=np.expm1(self.partition.predict(raw))*scale
            reference=np.expm1(self.normalized_reference)[None,:]*scale
        probability=(samples<=f.low_threshold.to_numpy(float)[:,None]).mean(axis=1)
        if self.family in {"hurdle","conditional_hurdle"}:
            probability=1/(1+np.exp(-np.clip(x@self.risk_coef,-25,25)))
            if self.family=="conditional_hurdle":
                low,high=[np.expm1(component.predict(raw))*f.historical_scale.to_numpy(float)[:,None] for component in self.components]
            else:low,high=[np.maximum(np.expm1(np.clip((x@coef)[:,None]+r,-20,25)),0) for coef,r in self.components]
            for i,p in enumerate(probability):
                ys=np.concatenate([low[i],high[i]]);ws=np.concatenate([np.full(len(GRID),p/len(GRID)),np.full(len(GRID),(1-p)/len(GRID))])
                order=np.argsort(ys);cdf=np.cumsum(ws[order]);samples[i]=np.interp(GRID,cdf,ys[order])
        return samples,reference,probability


class IndependentForecaster:
    def __init__(self,config):
        self.config=dict(config);self.models={};self.last_fit={}

    def predict(self,features,completed_training):
        date=pd.Timestamp(features.date.iloc[0])
        if not features.date.eq(date).all():raise ValueError("Predict one complete day")
        if not completed_training.empty and completed_training.date.max()>=date:raise ValueError("Training must precede target day")
        training=completed_training
        if self.config.get("hits_only"):
            training=training.loc[training.warmup | training.was_old_hit]
        out=np.empty((len(features),len(GRID)));ref=np.empty_like(out);risk=np.empty(len(features));counts=np.empty(len(features),int)
        fit_dates=np.empty(len(features),object)
        for key,index in features.groupby(["餐次","sale_unit"],sort=False).indices.items():
            f=features.iloc[index]
            t=training.loc[training["餐次"].eq(key[0]) & training.sale_unit.eq(key[1])]
            if len(t)>=80:
                if key not in self.models or (date-self.last_fit[key]).days>=7:
                    model=IndependentGroup(self.config.get("family","linear"),self.config.get("with_old",False)).fit(t)
                    self.models[key]=model;self.last_fit[key]=date
                model=self.models[key]
                out[index],ref[index],risk[index]=model.predict(f)
                counts[index]=model.training_rows;fit_dates[index]=model.fit_max_date
            else:
                # Historical fallback, independent of the old forecast.
                scale=f.historical_scale.to_numpy(float)
                out[index]=scale[:,None];ref[index]=out[index];risk[index]=(scale<=f.low_threshold.to_numpy(float)).astype(float)
                counts[index]=len(t);fit_dates[index]=t.date.max() if len(t) else pd.NaT
        point=decision(out,self.config.get("objective","log"),features.historical_scale.to_numpy(float))
        return point,out,ref,risk,counts,fit_dates

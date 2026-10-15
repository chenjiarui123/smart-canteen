"""Opt-in locked-menu interface; no web default registration or data mutation."""
import json
import numpy as np
import pandas as pd
import policy
from demand_data import OUT
from demand_models import IndependentForecaster,decision
from demand_evaluate import KEYS
from demand_coordination import coordinate


class LockedMenuPredictor:
    def __init__(self,dataset,config=None):
        if dataset not in {"a_hupan","f_jichuyuan"}:raise ValueError("Unknown canteen dataset")
        self.config=dict(config) if config is not None else json.loads((OUT/"frozen/config.json").read_text())[dataset]
        self.model=IndependentForecaster(self.config);self.last_date=None

    def predict_day(self,baseline_menu,old_features,features,completed_training):
        if baseline_menu.empty or baseline_menu.date.nunique()!=1:raise ValueError("One complete nonempty target day required")
        date=pd.Timestamp(baseline_menu.date.iloc[0])
        if self.last_date is not None and date<=self.last_date:raise ValueError("Predict days once in chronological order")
        if baseline_menu["食堂名称"].nunique()!=1:raise ValueError("One canteen per runtime")
        if not completed_training.empty and completed_training.date.max()>=date:raise ValueError("Training contains target/future sales")
        if {"actual_qty","target_y","hit"}&set(features.columns):raise ValueError("Target feature frame must not contain actual outcomes")
        index=pd.MultiIndex.from_frame(baseline_menu[policy.GROUP_KEYS].assign(_dish_norm=baseline_menu["菜品名称"].map(policy.normalize_name)))
        f=features.set_index(KEYS).loc[index].reset_index()
        old=old_features.set_index(KEYS).loc[index].reset_index()
        if not f.date.eq(date).all() or pd.to_datetime(f.history_max_date).ge(date).any():raise ValueError("Features cross prediction date")
        self.last_date=date
        if self.config.get("family")=="baseline":
            new=baseline_menu.copy()
            new["expected_demand_qty"]=new.pred_qty
            new["suggested_preparation_qty"]=np.maximum(new.pred_qty,new.min_batch_qty)
            new["minimum_batch_surplus_risk"]=new.suggested_preparation_qty-new.pred_qty
            return new,{"strategy":"unchanged_deployed_forecast","forecast_improvement_claimed":False,"intervals":None,
                        "note":"Preparation advice is separate; no new forecast or interval was accepted"}
        point,samples,reference,risk,counts,fit_dates=self.model.predict(f,completed_training)
        if self.config.get("unconditional"):
            samples=reference;point=decision(samples,self.config.get("objective","log"),f.historical_scale.to_numpy(float))
            risk=(samples<=f.low_threshold.to_numpy(float)[:,None]).mean(axis=1)
        results=[];audits=[]
        for _,idx in baseline_menu.groupby(policy.GROUP_KEYS,sort=False).indices.items():
            new,a=coordinate(baseline_menu.iloc[idx],old.iloc[idx].reset_index(drop=True),f.iloc[idx],point[idx],samples[idx],
                             self.config.get("coordination","projection"),self.config.get("risk_objective","mixed"),self.config.get("narrow_box",False))
            results.append(new);audits.append(a)
        result=pd.concat(results).sort_index()
        detail=f[KEYS].copy();detail["raw_prediction"]=point;detail["low_risk_probability"]=risk
        detail["lower80"],detail["upper80"]=np.quantile(samples,[.1,.9],axis=1)
        detail["training_rows"]=counts;detail["model_fit_max_date"]=fit_dates
        return result,{"experimental":True,"coordination":audits,"conditional_forecast_diagnostics":detail}

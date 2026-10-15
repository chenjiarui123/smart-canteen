"""Opt-in structural demand predictor with distinct preparation advice.

Phase one preserves the saved final baseline menu AND each unit-group total.
The unchanged evaluator continues to score demand forecasts against actual sales.
Preparation quantities are evaluated separately, never substituted into that score.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
import policy
from allocation_policy import bounded_projection


class ForecastReliability:
    """Past out-of-sample residuals, not historical demand volatility."""
    def __init__(self):
        self.items=defaultdict(list)
        self.groups=defaultdict(list)
        self.max_date=None

    def factors(self,f):
        date=pd.Timestamp(f.date.iloc[0])
        if self.max_date is not None and self.max_date>=date:
            raise ValueError("Reliability uses only previous-day forecasts")
        factors=[]
        for r in f.to_dict("records"):
            group=(r["餐次"],r["sale_unit"],r["dish_category"],r["calendar_state"])
            key=(r["餐次"],r["_dish_norm"],r["sale_unit"],r["calendar_state"])
            pool=self.groups[group][-200:]
            prior=float(np.mean(pool)) if pool else .25
            observations=self.items[key][-60:]
            n=len(observations)
            variance=(sum(observations)+10*prior)/(n+10)
            factors.append(float(np.clip(variance/max(prior,.01),.25,4.)))
        return np.array(factors)

    def observe_day(self,features,estimates,actual):
        if features.empty:
            return
        date=pd.Timestamp(features.date.iloc[0])
        if not features.date.eq(date).all() or not actual.date.eq(date).all():
            raise ValueError("Observe exactly the just-predicted day")
        if self.max_date is not None and self.max_date>=date:
            raise ValueError("Observe each completed day once")
        from evaluator import prepare_actual
        observed=prepare_actual(actual)
        lookup={(r["食堂名称"],r["餐次"],r["_dish_norm"]):r["actual_qty"] for r in observed.to_dict("records")}
        for r,estimate in zip(features.to_dict("records"),estimates):
            value=lookup.get((r["食堂名称"],r["餐次"],r["_dish_norm"]))
            if value is None or value<=0:
                continue # Missing supply is not a zero-demand observation.
            residual=float(np.clip((value-estimate)/max(estimate,1.),-3.,3.)**2)
            group=(r["餐次"],r["sale_unit"],r["dish_category"],r["calendar_state"])
            key=(r["餐次"],r["_dish_norm"],r["sale_unit"],r["calendar_state"])
            self.groups[group].append(residual)
            self.items[key].append(residual)
        self.max_date=date


def initial_estimate(f,config):
    b=f.baseline_qty.to_numpy(float)
    estimate=b.copy()
    if config.get("remove_volatility_discount"):
        estimate=f.date_conditioned.to_numpy(float).copy()
    share_mode=config.get("shares")
    if share_mode:
        units=f.sale_unit.astype(str).to_numpy()
        categories=f.dish_category.astype(str).to_numpy()
        for unit in set(units):
            mask=units==unit
            total=float(b[mask].sum())
            if share_mode=="flat":
                weights=f.unit_share.to_numpy(float)[mask]
                estimate[mask]=total*weights/max(weights.sum(),1e-12)
            elif share_mode=="hierarchical":
                cats=sorted(set(categories[mask]))
                category_weights=np.array([f.loc[mask & (categories==c),"category_share"].iloc[0] for c in cats],float)
                category_weights/=max(category_weights.sum(),1e-12)
                for category,weight in zip(cats,category_weights):
                    cm=mask & (categories==category)
                    item_weights=f.loc[cm,"within_category_share"].to_numpy(float)
                    estimate[cm]=total*weight*item_weights/max(item_weights.sum(),1e-12)
            else:
                raise ValueError("Unknown share model")
    if config.get("competition"):
        estimate*=f.competition_factor.to_numpy(float)
    return estimate


def allocate(menu,features,config,reliability=None,estimate_override=None):
    if menu.empty:
        return menu.copy(),{},np.array([])
    if menu.groupby(policy.GROUP_KEYS).ngroups!=1:
        raise ValueError("One session required")
    names=menu["菜品名称"].map(policy.normalize_name)
    if names.duplicated().any():
        raise ValueError("Duplicate baseline menu dish")
    f=features.set_index("_dish_norm").loc[names].reset_index()
    date=pd.Timestamp(menu.date.iloc[0])
    if not f.date.eq(date).all() or pd.to_datetime(f.structural_history_max_date).ge(date).any():
        raise ValueError("Misaligned or nonhistorical features")
    if not f["食堂名称"].eq(menu["食堂名称"].iloc[0]).all() or not f["餐次"].eq(menu["餐次"].iloc[0]).all():
        raise ValueError("Session mismatch")
    b=menu.pred_qty.to_numpy(float)
    np.testing.assert_allclose(b,f.baseline_qty,rtol=1e-12,atol=1e-8)
    estimate=initial_estimate(f,config)
    if estimate_override is not None:
        estimate=np.asarray(estimate_override,float)
        if estimate.shape!=b.shape or not np.isfinite(estimate).all() or (estimate<0).any():
            raise ValueError("Invalid historical demand override")
    weights=np.maximum(estimate,1e-6)
    reliability_factor=np.ones(len(menu))
    if config.get("reliability"):
        if reliability is None:
            raise ValueError("Prequential reliability state required")
        reliability_factor=reliability.factors(f)
        weights=np.maximum(estimate,1.)**2*reliability_factor
    minimum=menu.min_batch_qty.to_numpy(float)
    lower=np.zeros_like(b) if config.get("separate_preparation") else minimum.copy()
    upper=np.minimum(np.maximum(f.q99.to_numpy(float)*2,0),b*2)
    unknown=f.sale_unit.astype(str).str.contains("�",regex=False).to_numpy() | f.sale_unit.astype(str).isin(["","unknown","nan"]).to_numpy()
    lower[unknown]=b[unknown]
    upper[unknown]=b[unknown]
    units=f.sale_unit.astype(str).to_numpy()
    q=b.copy()
    unit_conflicts=[]
    for unit in sorted(set(units)):
        mask=units==unit
        total=float(b[mask].sum())
        if (upper[mask]<lower[mask]).any() or lower[mask].sum()>total+1e-8 or upper[mask].sum()<total-1e-8:
            unit_conflicts.append({"unit":unit,"total":total,"lower_sum":float(lower[mask].sum()),"upper_sum":float(upper[mask].sum()),"reason":"fixed_unit_total_outside_unchanged_box"})
            continue
        if config.get("identity"):
            continue
        x=bounded_projection(estimate[mask],weights[mask],lower[mask],upper[mask],total)
        if x is None:
            raise AssertionError("Feasible bounds should project")
        q[mask]=x
    # Conservative full-session fallback: do not make conflicted meals look good
    # by silently moving that unit's demand into another type of sale unit.
    status="allocated"
    if unit_conflicts:
        q=b.copy()
        status="explicit_bounds_conflict_baseline_fallback"
    total=float(b.sum())
    blend=1.
    def concentration_ok(x):
        if total<=0:return True
        return float(((x/total)**2).sum())<=float(((b/total)**2).sum())*1.15+1e-12 and float(x.max()/total)<=float(b.max()/total)+.03+1e-12
    if not concentration_ok(q):
        lo,hi=0.,1.
        for _ in range(50):
            mid=(lo+hi)/2
            if concentration_ok(b+mid*(q-b)):lo=mid
            else:hi=mid
        blend=lo
        q=b+blend*(q-b)
        if (q<lower-1e-8).any() or (q>upper+1e-8).any():
            q=b.copy();blend=0.
            status="explicit_concentration_box_conflict_baseline_fallback"
    new=menu.copy()
    new["pred_qty"]=q
    new["pred_orders"]=q*menu.expected_orders_per_qty.to_numpy(float)
    new["pred_rev"]=q*menu.expected_rev_per_qty.to_numpy(float)
    # The original baseline's preparation forecast remains its original output.
    # Separation adds minimum-batch operating advice without changing demand.
    preparation=np.maximum(q,minimum) if config.get("separate_preparation") else q.copy()
    new["expected_demand_qty"]=q
    new["suggested_preparation_qty"]=preparation
    new["minimum_batch_surplus_risk"]=np.maximum(preparation-q,0)
    new["demand_before_coordination"]=estimate
    new["historical_error_adjustment_weight"]=reliability_factor
    audit={"date":date,"食堂名称":menu["食堂名称"].iloc[0],"餐次":menu["餐次"].iloc[0],"status":status,
        "conflicts":unit_conflicts,"baseline_total":total,"demand_total":float(q.sum()),"preparation_total":float(preparation.sum()),
        "sum_error":float(q.sum()-total),"max_unit_total_error":max(abs(float(q[units==u].sum()-b[units==u].sum())) for u in set(units)),
        "below_minimum_demand_items":int((q<minimum-1e-8).sum()),"below_minimum_preparation_items":int((preparation<minimum-1e-8).sum()),
        "baseline_below_minimum_items":int((b<minimum-1e-8).sum()),"surplus_risk_qty":float((preparation-q).sum()),
        "blend":blend,"baseline_hhi":float(((b/total)**2).sum()) if total else 0.,"new_hhi":float(((q/total)**2).sum()) if total else 0.,
        "top_share_increase":float((q.max()-b.max())/total) if total else 0.,"historical_box_violations":int(((q<lower-1e-8)|(q>upper+1e-8)).sum())}
    assert np.isfinite(q).all() and (q>=0).all() and abs(q.sum()-total)<max(1e-8,total*1e-12)
    assert audit["max_unit_total_error"]<max(1e-8,total*1e-12)
    return new,audit,estimate

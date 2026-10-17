"""Optional fixed-menu, fixed-total allocator. No web registration or actual input.

Features must come from OfferedHistory before target-date observations are added.
Baseline metadata and unit ratios are immutable. Infeasible historical envelopes
fall back to baseline with a visible reason, never to a surplus-absorbing dish.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd
import policy


def bounded_projection(estimate, weights, lower, upper, total):
    """Minimise sum((x-estimate)^2/weights), subject to box and sum constraints."""
    estimate, weights, lower, upper = [np.asarray(x, dtype=float) for x in (estimate, weights, lower, upper)]
    if not all(np.isfinite(x).all() for x in (estimate, weights, lower, upper)):
        raise ValueError("Nonfinite allocation inputs")
    if (weights <= 0).any() or (lower < 0).any() or (upper < lower).any():
        raise ValueError("Invalid allocation bounds or weights")
    if total < lower.sum()-1e-8 or total > upper.sum()+1e-8:
        return None
    lo = float(np.min((lower-estimate)/weights))-1
    hi = float(np.max((upper-estimate)/weights))+1
    for _ in range(75):
        mid = (lo+hi)/2
        x = np.clip(estimate+mid*weights, lower, upper)
        if x.sum() < total:
            lo = mid
        else:
            hi = mid
    x = np.clip(estimate+(lo+hi)/2*weights, lower, upper)
    assert abs(x.sum()-total) < max(1e-8, total*1e-12)
    return x


def demand_estimate(features, config):
    statistic = config.get("statistic", "mean")
    value = features[statistic].to_numpy(float).copy()
    recent = config.get("recent_days")
    if recent:
        recent_stat = statistic if statistic in {"mean", "median", "logmean"} else "mean"
        n = features[f"n{recent}"].to_numpy(float)
        w = n/(n+float(config.get("recent_tau", 8)))
        value = (1-w)*value + w*features[f"w{recent}_{recent_stat}"].to_numpy(float)
    # Context is a joint-state conditional estimate, not multiplied factors.
    for kind in config.get("context", []):
        stat = statistic if statistic in {"mean", "median", "logmean"} else "mean"
        n = features[kind+"_n"].to_numpy(float)
        w = n/(n+float(config.get("context_tau", 8)))
        value = (1-w)*value + w*features[kind+"_"+stat].to_numpy(float)
    if config.get("baseline_blend", 0):
        mix = float(config["baseline_blend"])
        value = (1-mix)*value+mix*features.baseline_qty.to_numpy(float)
    if config.get("stability_power", 0):
        # Past quantity variability, not offer probability or predicted hit rate.
        stability=features.baseline_qty_stability.to_numpy(float)
        value *= np.maximum(stability,1e-6)**float(config["stability_power"])
    return np.maximum(value, 1e-8)


def allocate_session(menu, features, config, bias_factors=None):
    """Return same rows/order, with only pred_qty/orders/rev changed, plus audit."""
    if menu.empty:
        return menu.copy(), {"status":"empty_baseline", "sum_error":0.}
    if menu.groupby(policy.GROUP_KEYS).ngroups != 1:
        raise ValueError("allocate_session accepts exactly one meal")
    names = menu["菜品名称"].map(policy.normalize_name)
    if names.duplicated().any():
        raise ValueError("Duplicate baseline dish")
    f = features.set_index("_dish_norm").loc[names].reset_index()
    date = pd.Timestamp(menu.date.iloc[0])
    if not f.date.eq(date).all() or not f["餐次"].eq(menu["餐次"].iloc[0]).all() or not f["食堂名称"].eq(menu["食堂名称"].iloc[0]).all():
        raise ValueError("Feature/menu session mismatch")
    if pd.to_datetime(f.history_max_date).ge(date).any():
        raise ValueError("Future or same-day history is forbidden")
    b = menu.pred_qty.to_numpy(float)
    if not np.allclose(b, f.baseline_qty, rtol=1e-12, atol=1e-8):
        raise ValueError("Feature/baseline quantity mismatch")
    total = float(b.sum())
    lower = menu.min_batch_qty.to_numpy(float)
    baseline_minimum_violations=int((b < lower-1e-8).sum())
    estimate = demand_estimate(f,config)
    if bias_factors is not None:
        bias = np.asarray(bias_factors, float)
        if bias.shape != b.shape or not np.isfinite(bias).all() or (bias <= 0).any():
            raise ValueError("Invalid historical bias")
        estimate *= bias
    upper = np.minimum(np.maximum(f.q99.to_numpy(float)*float(config.get("historical_cap",2.)), lower),
                       np.maximum(b*float(config.get("change_cap",2.)), lower))
    lower = np.maximum(lower, b*float(config.get("change_floor",.25)))
    uncertain_unit = f.sale_unit.astype(str).str.contains("�", regex=False).to_numpy() | f.sale_unit.astype(str).isin(["", "unknown", "nan"]).to_numpy()
    # Do not infer a conversion for malformed historical unit labels.
    lower[uncertain_unit] = b[uncertain_unit]
    upper[uncertain_unit] = b[uncertain_unit]
    weight = np.maximum(estimate,1e-6)**float(config.get("projection_power",1.))
    if config.get("sample_weight"):
        # More reliable observations resist additive adjustments.
        n = f.sample_count.to_numpy(float)
        weight /= np.sqrt(1+np.minimum(n,60))
    audit = {"date":date,"食堂名称":menu["食堂名称"].iloc[0],"餐次":menu["餐次"].iloc[0],
             "baseline_total":total,"estimate_total":float(estimate.sum()),
             "lower_sum":float(lower.sum()),"upper_sum":float(upper.sum()),
             "upper_shortfall":max(0.,total-float(upper.sum())),"status":"allocated"}
    audit["uncertain_unit_frozen_items"] = int(uncertain_unit.sum())
    if total < menu.min_batch_qty.sum()-1e-8:
        x=b.copy()
        audit["status"]="infeasible_baseline_minimum"
    elif config.get("identity"):
        x = b.copy()
        audit["status"] = "baseline"
    else:
        x = bounded_projection(estimate,weight,lower,upper,total)
        if x is None:
            x = b.copy()
            audit["status"] = "infeasible_bounds_baseline_fallback"
    blend = 1.
    if total > 0:
        def concentration_ok(q):
            return ((q/total)**2).sum() <= ((b/total)**2).sum()*float(config.get("hhi_ratio",1.15))+1e-12 and q.max()/total <= b.max()/total+float(config.get("top_share_add",.03))+1e-12
        if not concentration_ok(x):
            lo,hi = 0.,1.
            for _ in range(50):
                mid=(lo+hi)/2
                if concentration_ok(b+mid*(x-b)):
                    lo=mid
                else:
                    hi=mid
            blend=lo
            x=b+blend*(x-b)
            # Blending may leave the historical box when baseline was outside;
            # don't silently treat that as a valid constrained allocation.
            if (x>upper+1e-8).any() or (x<lower-1e-8).any():
                x=b.copy()
                blend=0.
                audit["status"]="concentration_bounds_conflict_baseline_fallback"
    output = menu.copy()
    output["pred_qty"] = x
    output["pred_orders"] = x*menu.expected_orders_per_qty.to_numpy(float)
    output["pred_rev"] = x*menu.expected_rev_per_qty.to_numpy(float)
    audit.update(sum_error=float(x.sum()-total),concentration_blend=blend,
                 allocation_feasible=audit["status"] not in {"infeasible_baseline_minimum", "infeasible_bounds_baseline_fallback", "concentration_bounds_conflict_baseline_fallback"},
                 baseline_minimum_violations=baseline_minimum_violations,
                 minimum_total_shortfall=max(0.,float(menu.min_batch_qty.sum())-total),
                 baseline_hhi=float(((b/total)**2).sum()) if total else 0.,
                 new_hhi=float(((x/total)**2).sum()) if total else 0.,
                 baseline_top_share=float(b.max()/total) if total else 0.,
                 new_top_share=float(x.max()/total) if total else 0.,
                 min_batch_violations=int((x<menu.min_batch_qty.to_numpy(float)-1e-8).sum()),
                 historical_upper_violations=int((x>upper+1e-8).sum()),
                 max_change_ratio=float(np.max(np.divide(x,b,out=np.ones_like(x),where=b>0))))
    assert np.isfinite(x).all() and (x>=0).all()
    assert abs(x.sum()-total) < max(1e-8,total*1e-12)
    return output, audit


class HistoricalBias:
    """Rolling residual ratios, updated only after a whole predicted day ends."""
    def __init__(self, tau=10., mode="dish", strength=1.):
        self.records=defaultdict(list)
        self.max_date=None
        self.tau=tau
        self.mode=mode
        self.strength=strength

    def key(self,r):
        return (str(r["食堂名称"]),str(r["餐次"]),str(r["_dish_norm"]) if self.mode=="dish" else str(r["calendar_state"]))

    def factors(self,features):
        date=pd.Timestamp(features.date.iloc[0])
        if self.max_date is not None and self.max_date>=date:
            raise ValueError("Bias history includes target date")
        result=[]
        for r in features.to_dict("records"):
            values=self.records[self.key(r)][-60:]
            n=len(values)
            result.append(float(np.exp(self.strength*n/(n+self.tau)*np.median(values))) if n else 1.)
        return np.clip(result,.5,2.)

    def observe(self,features,estimates,actual_positive):
        if features.empty:
            return
        date=pd.Timestamp(features.date.iloc[0])
        if not features.date.eq(date).all() or not actual_positive.date.eq(date).all():
            raise ValueError("Bias observations must belong to the just-predicted date")
        lookup={(r["食堂名称"],r["餐次"],r["_dish_norm"]):r["actual_qty"] for r in actual_positive.to_dict("records") if r["actual_qty"]>0}
        for r,estimate in zip(features.to_dict("records"),estimates):
            value=lookup.get((r["食堂名称"],r["餐次"],r["_dish_norm"]))
            if value is not None and estimate>0:
                self.records[self.key(r)].append(float(np.log(value/estimate)))
        self.max_date=max(date,self.max_date) if self.max_date is not None else date

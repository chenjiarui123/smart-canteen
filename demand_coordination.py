"""Expected absolute/relative error allocation under immutable group totals."""
import numpy as np
import policy
from structural_policy import allocate as old_projection


def marginal_allocate(samples,upper,total,scale,objective="mixed"):
    """Exact piecewise-linear expected-loss allocation before concentration cap."""
    n=len(upper);q=np.zeros(n)
    if np.any(upper<0) or upper.sum()<total-1e-8:return None
    segments=[]
    for i in range(n):
        y=np.sort(np.maximum(samples[i],0))
        weights=np.divide(1.,y,out=np.zeros_like(y),where=y>0)
        if objective=="mixed":weights=.5*weights+.5/max(scale[i],1e-6)
        weights/=len(y)
        points=np.unique(np.concatenate([[0.],y[(y>0)&(y<upper[i])],[upper[i]]]))
        for lo,hi in zip(points[:-1],points[1:]):
            slope=float(2*weights[y<=lo].sum()-weights.sum())
            segments.append((slope,i,float(hi-lo)))
    remaining=float(total)
    for slope,i,length in sorted(segments,key=lambda r:(r[0],r[1])):
        take=min(remaining,length);q[i]+=take;remaining-=take
        if remaining<=1e-10:break
    if remaining>1e-8:return None
    return q


def coordinate(menu,old_features,new_features,point,samples,method="projection",objective="mixed",narrow=False):
    projected,audit,_=old_projection(menu,old_features,{"separate_preparation":True},estimate_override=point)
    if method=="projection":return projected,audit
    b=menu.pred_qty.to_numpy(float);units=old_features.sale_unit.to_numpy(str)
    upper=np.minimum(old_features.q99.to_numpy(float)*2,b*2)
    if narrow:upper=np.minimum(upper,np.quantile(samples,.9,axis=1))
    q=b.copy();failures=[]
    for unit in sorted(set(units)):
        mask=units==unit
        x=marginal_allocate(samples[mask],upper[mask],b[mask].sum(),new_features.historical_scale.to_numpy(float)[mask],objective)
        if x is None:failures.append({"unit":unit,"reason":"conditional_box_cannot_hold_fixed_total","upper_sum":float(upper[mask].sum()),"total":float(b[mask].sum())})
        else:q[mask]=x
    status="expected_risk_allocated"
    if failures:q=b.copy();status="explicit_conditional_bounds_conflict_baseline_fallback"
    total=b.sum();blend=1.
    def safe(x):
        return total<=0 or ((x/total)**2).sum()<=((b/total)**2).sum()*1.15+1e-12 and x.max()/total<=b.max()/total+.03+1e-12
    if not safe(q):
        lo,hi=0.,1.
        for _ in range(50):
            mid=(lo+hi)/2
            if safe(b+mid*(q-b)):lo=mid
            else:hi=mid
        blend=lo;q=b+blend*(q-b)
        if (q>upper+1e-8).any():q=b.copy();status="explicit_concentration_box_conflict_baseline_fallback";failures.append({"reason":"concentration_and_box_conflict"})
    new=menu.copy();new["pred_qty"]=q
    new["pred_orders"]=q*new.expected_orders_per_qty;new["pred_rev"]=q*new.expected_rev_per_qty
    minimum=menu.min_batch_qty.to_numpy(float)
    new["expected_demand_qty"]=q;new["suggested_preparation_qty"]=np.maximum(q,minimum)
    new["minimum_batch_surplus_risk"]=np.maximum(minimum-q,0)
    new["demand_before_coordination"]=point;new["historical_error_adjustment_weight"]=1.
    audit.update(status=status,conflicts=failures,demand_total=float(q.sum()),preparation_total=float(np.maximum(q,minimum).sum()),sum_error=float(q.sum()-total),
        max_unit_total_error=max(abs(float(q[units==u].sum()-b[units==u].sum())) for u in set(units)),
        below_minimum_demand_items=int((q<minimum-1e-8).sum()),below_minimum_preparation_items=0,surplus_risk_qty=float(np.maximum(minimum-q,0).sum()),
        blend=blend,new_hhi=float(((q/total)**2).sum()) if total else 0.,top_share_increase=float((q.max()-b.max())/total) if total else 0.,historical_box_violations=int((q>upper+1e-8).sum()),
        objective=objective,conditional_narrow_box=narrow,concentration_method="convex blend with baseline; not claimed global risk optimum with HHI")
    np.testing.assert_allclose(q.sum(),total,rtol=1e-12,atol=1e-8)
    return new,audit
